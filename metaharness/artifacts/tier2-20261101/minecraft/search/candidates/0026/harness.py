"""Candidate 0026: checkpointed survival progression for real-time Minecraft.

The controller is a deterministic milestone compiler plus a postcondition-driven
executor.  It does not select arbitrary commands every tick: an operation is
compiled from the audited inventory and current exposed candidates, then a small
transaction state machine waits for its observable result.  Jev is reserved for
immediate threat triage; Astra may set only a bounded macro preference.
"""
from __future__ import annotations

import json
import math
import re

from metaharness.core.contract import BudgetExceeded, ModelUnavailable
from metaharness.models.jev import Choice, Noul


FOOD_ITEMS = {
    "apple", "bread", "baked_potato", "carrot", "sweet_berries", "melon_slice",
    "mushroom_stew", "beetroot", "potato", "beef", "porkchop", "mutton",
    "chicken", "rabbit", "cod", "salmon", "cooked_beef", "cooked_porkchop",
    "cooked_mutton", "cooked_chicken", "cooked_rabbit", "cooked_cod", "cooked_salmon",
    "dried_kelp", "wheat",
}
FOOD_BLOCKS = {"sweet_berry_bush", "hay_block", "melon", "wheat", "carrots", "potatoes", "beetroots"}
WOOD_ENDINGS = ("_log", "_stem")
WOODEN_PICKS = ("wooden_pickaxe", "stone_pickaxe", "iron_pickaxe", "diamond_pickaxe", "netherite_pickaxe")
STONE_PICKS = {"stone_pickaxe", "iron_pickaxe", "diamond_pickaxe", "netherite_pickaxe"}
WEAPONS = ("stone_sword", "iron_sword", "diamond_sword", "netherite_sword", "stone_axe", "iron_axe")
GROUND = {
    "grass_block", "dirt", "coarse_dirt", "rooted_dirt", "podzol", "mycelium",
    "mud", "clay", "stone", "cobblestone", "deepslate", "cobbled_deepslate",
    "tuff", "andesite", "diorite", "granite", "netherrack", "blackstone",
}
STONE = {"stone", "cobblestone", "deepslate", "cobbled_deepslate"}
IRON_ORES = {"iron_ore", "deepslate_iron_ore"}
COAL_ORES = {"coal_ore", "deepslate_coal_ore"}
HAZARDS = {"lava", "fire", "soul_fire", "magma_block", "cactus", "powder_snow", "campfire", "soul_campfire"}
HOSTILES = {
    "zombie", "skeleton", "creeper", "spider", "drowned", "husk", "stray", "witch",
    "phantom", "enderman", "slime", "magma_cube", "blaze", "ghast", "piglin",
    "piglin_brute", "zombified_piglin", "hoglin", "wither_skeleton", "ender_dragon",
    "shulker", "silverfish", "warden",
}
FAIL_WORDS = ("failed", "failure", "error", "unreachable", "timeout", "rejected", "illegal", "not enough", "not in reach", "no path", "busy", "invalid", "denied")
ID_RE = re.compile(r"^[bs]:-?\d+,-?\d+,-?\d+$")


def num(value, default=0.0):
    try:
        result = float(value)
        return result if math.isfinite(result) else default
    except (TypeError, ValueError, OverflowError):
        return default


def name(value):
    if isinstance(value, dict):
        value = value.get("name", value.get("item", value.get("type", value.get("block", value.get("id", "")))))
    return str(value or "").removeprefix("minecraft:")


def xyz(value):
    if isinstance(value, dict):
        if value.get("position") is not None:
            value = value.get("position")
        elif all(k in value for k in ("x", "y", "z")):
            value = [value.get("x"), value.get("y"), value.get("z")]
    if not isinstance(value, (tuple, list)) or len(value) < 3:
        return None
    p = tuple(num(v, math.inf) for v in value[:3])
    return p if all(math.isfinite(v) for v in p) else None


def dist(a, b):
    pa, pb = xyz(a), xyz(b)
    if pa is None or pb is None:
        return 999.0
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(pa, pb)))


def rows(obs, prefix=None):
    result = []
    raw_rows = obs.get("candidates", [])
    if not isinstance(raw_rows, list):
        return result
    here = xyz((obs.get("self") or {}).get("position"))
    for raw in raw_rows:
        if not isinstance(raw, dict):
            continue
        ident = str(raw.get("id", ""))
        if not ID_RE.fullmatch(ident) or (prefix and not ident.startswith(prefix + ":")):
            continue
        row = dict(raw)
        row["_id"] = ident
        row["_name"] = name(raw.get("name", raw.get("block", "")))
        row["_pos"] = xyz(raw) or tuple(float(v) for v in ident[2:].split(","))
        row["_distance"] = num(raw.get("distance"), dist(here, row["_pos"]))
        result.append(row)
    return result


def inventory(obs):
    """Use personal audited inventory; never count a cursor or a container."""
    source = obs.get("inventory")
    if source in (None, {}, []):
        source = obs.get("inventory_authoritative")
    if isinstance(source, dict) and isinstance(source.get("items"), (dict, list)):
        source = source.get("items")
    answer = {}
    if isinstance(source, dict):
        for key, raw in source.items():
            item = name(key)
            raw_count = raw.get("count", 0) if isinstance(raw, dict) else raw
            count = max(0, int(num(raw_count)))
            if item and count:
                answer[item] = answer.get(item, 0) + count
    elif isinstance(source, list):
        for raw in source:
            if not isinstance(raw, dict):
                continue
            item = name(raw)
            count = max(0, int(num(raw.get("count", 0))))
            if item and count:
                answer[item] = answer.get(item, 0) + count
    if not answer:
        client = obs.get("inventory_client")
        if isinstance(client, dict):
            for key, raw in client.items():
                item = name(key)
                count = max(0, int(num(raw.get("count", 0) if isinstance(raw, dict) else raw)))
                if item and count:
                    answer[item] = count
    return answer


def inv_sig(inv):
    return tuple(sorted((str(k), int(v)) for k, v in inv.items() if int(v) > 0))


def has_task(obs):
    value = obs.get("task")
    return value not in (None, "", {})


def feedback(obs):
    vals = [obs.get(k) for k in ("action_feedback", "action_result", "last_result", "feedback")]
    try:
        return json.dumps(vals, sort_keys=True, default=str)
    except Exception:
        return repr(vals)


def failed_feedback(text):
    low = str(text).lower()
    return any(word in low for word in FAIL_WORDS) and not any(word in low for word in ("success", "succeeded", " ok"))


def position_of_self(obs):
    me = obs.get("self") if isinstance(obs.get("self"), dict) else {}
    return xyz(me.get("position")) or xyz(me)


def cobble_count(inv):
    return int(inv.get("cobblestone", 0) + inv.get("cobbled_deepslate", 0))


def plank_count(inv):
    return sum(int(v) for k, v in inv.items() if k.endswith("_planks"))


def wood_count(inv):
    return sum(int(v) for k, v in inv.items() if k.endswith(WOOD_ENDINGS))


def food_count(inv):
    return sum(int(v) for k, v in inv.items() if k in FOOD_ITEMS)


def food_choice(inv):
    order = (
        "cooked_beef", "cooked_porkchop", "cooked_mutton", "cooked_chicken", "bread",
        "baked_potato", "carrot", "apple", "sweet_berries", "melon_slice", "beef",
        "porkchop", "mutton", "chicken", "rabbit", "cod", "salmon", "potato", "wheat",
    )
    return next((x for x in order if inv.get(x)), "")


class Controller:
    def __init__(self, ctx):
        self.ctx = ctx
        self.allowed = set()
        self.last_action = "noop"
        self.tx = None
        self.work = None
        self.bad = {}
        self.visited = {}
        self.visited_cells = set()
        self.completed = set()
        self.last_feedback = ""
        self.table_hint = None
        self.furnace_hint = None
        self.place_wait_until = -1.0
        self.table_return_attempted = False
        self.furnace_started = {}
        self.mine_steps = 0
        self.mine_direction = None
        self.phase = "opening"
        self.last_progress = 0.0
        self.last_mark = None
        self.last_attack = -100.0
        self.focus = "balanced"
        self.astra_future = None
        self.astra_session = None
        self.astra_count = 0
        self.astra_next = 55.0
        self.jev_future = None
        self.jev_started = -100.0
        self.jev_seq = None
        self.jev_episode = None
        self.jev_signature = None
        self.jev_actions = ()
        self.jev_committed = False
        self.jev_cooldown_until = -1.0
        self.jev_last_hazard = None

    def now(self):
        return num(self.ctx.clock.now())

    def log(self, event, **fields):
        self.ctx.log(event, **fields)

    def legal(self, action):
        verb = str(action or "").split(" ", 1)[0]
        return (not self.allowed or verb in self.allowed) and bool(action)

    def add(self, action, purpose=""):
        if not action or not self.legal(action):
            return None
        return {"action": str(action), "purpose": str(purpose)}

    def is_bad(self, key, now):
        return now < self.bad.get(str(key), -1.0)

    def suppress(self, key, now, seconds=120.0, reason=""):
        if not key:
            return
        key = str(key)
        self.bad[key] = max(self.bad.get(key, 0.0), now + seconds)
        self.log("goal_cooldown", key=key, seconds=seconds, reason=reason)

    def mark_progress(self, obs, inv, now):
        me = obs.get("self") if isinstance(obs.get("self"), dict) else {}
        progress = obs.get("progress") if isinstance(obs.get("progress"), dict) else {}
        mark = (inv_sig(inv), str(me.get("dimension", "")), bool(me.get("dead")),
                round(num(obs.get("score"), num(progress.get("score"), 0)), 2),
                tuple(sorted(str(x) for x in (obs.get("advancements") or []))))
        if self.last_mark is None:
            self.last_mark = mark
            self.last_progress = now
        elif mark != self.last_mark:
            self.last_mark = mark
            self.last_progress = now
            self.log("milestone_state_changed", phase=self.phase)

    def candidates(self, obs):
        return rows(obs, "b"), rows(obs, "s")

    def table_rows(self, blocks, here, radius=4.8):
        found = [r for r in blocks if r["_name"] == "crafting_table" and r["_distance"] <= radius]
        found.sort(key=lambda r: r["_distance"])
        return found

    def furnace_rows(self, blocks):
        found = [r for r in blocks if r["_name"] == "furnace"]
        found.sort(key=lambda r: r["_distance"])
        return found

    def compile_place_support(self, blocks, here):
        if here is None:
            return None
        floor_y = math.floor(here[1]) - 1
        fx, fz = math.floor(here[0]), math.floor(here[2])
        supports = []
        for row in blocks:
            if row["_name"] not in GROUND | {"sand", "gravel", "terracotta", "oak_planks", "crafting_table", "furnace", "chest"}:
                continue
            if row.get("in_reach") is False or row.get("visible") is False:
                continue
            q = row.get("_pos")
            if q is None or math.floor(q[1]) != floor_y:
                continue
            if math.floor(q[0]) == fx and math.floor(q[2]) == fz:
                continue
            if row["_name"] in {"sand", "gravel"}:
                continue
            supports.append(row)
        supports.sort(key=lambda x: x["_distance"])
        return supports[0] if supports else None

    def closest_resource(self, blocks, names=None, suffixes=(), now=None, require_reach=False):
        now = self.now() if now is None else now
        candidates = []
        for row in blocks:
            item = row["_name"]
            matches = (names is None or item in names) and (not suffixes or any(item.endswith(s) for s in suffixes))
            if not matches or not row.get("id") or row.get("harvestable") is False:
                continue
            if item in HAZARDS or item in {"air", "water", "lava", "bubble_column"}:
                continue
            if row.get("visible") is False and not row.get("in_reach"):
                continue
            if require_reach and not row.get("in_reach"):
                continue
            if row["_id"] in self.completed or self.is_bad(row["_id"], now):
                continue
            candidates.append(row)
        candidates.sort(key=lambda r: (0 if r.get("in_reach") else 1, 0 if r.get("visible") else 1, r["_distance"]))
        return candidates[0] if candidates else None

    def stand_for(self, stands, target, now, max_distance=48.0):
        choices = []
        for row in stands:
            q = row.get("_pos")
            if q is None or row["_distance"] < 0.8 or row["_distance"] > max_distance:
                continue
            if self.is_bad("navigate " + row["_id"], now) or self.is_bad(row["_id"], now):
                continue
            if now - self.visited.get(row["_id"], -10000.0) < 8.0:
                continue
            goal_d = dist(q, target) if target is not None else 0.0
            if target is not None and goal_d > 13.0:
                continue
            cell = (math.floor(q[0] / 5), math.floor(q[2] / 5))
            penalty = 20.0 if cell in self.visited_cells else 0.0
            score = (goal_d * 3.0 if target is not None else 0.0) + min(row["_distance"], 22.0) * 0.25 + penalty
            choices.append((score, row))
        choices.sort(key=lambda x: x[0])
        return choices[0][1] if choices else None

    def explore_action(self, stands, now, target=None):
        row = self.stand_for(stands, target, now)
        if row:
            return self.add("navigate " + row["_id"], "move to a currently listed safe stand")
        return None

    def plank_item_for_log(self, inv):
        source = next((k for k, v in inv.items() if v and k.endswith(WOOD_ENDINGS)), "")
        if source.endswith("_log"):
            return source[:-4] + "_planks"
        if source.endswith("_stem"):
            return source[:-5] + "_planks"
        return ""

    def feasible(self, item, inv):
        planks = plank_count(inv)
        sticks = inv.get("stick", 0)
        cobble = cobble_count(inv)
        if item.endswith("_planks"):
            base = item[:-7]
            return 4 * (inv.get(base + "_log", 0) + inv.get(base + "_stem", 0))
        if item == "stick":
            return (planks // 2) * 4
        if item == "crafting_table":
            return planks // 4
        if item == "wooden_pickaxe":
            return min(planks // 3, sticks // 2)
        if item == "stone_pickaxe":
            return min(cobble // 3, sticks // 2)
        if item == "stone_sword":
            return min(planks // 2, sticks)
        if item == "furnace":
            return cobble // 8
        if item == "shield":
            return min(inv.get("iron_ingot", 0), planks // 6)
        return 0

    def craft(self, item, desired, inv, table=None):
        outputs = min(max(1, int(desired)), int(self.feasible(item, inv)))
        if outputs < 1:
            return None
        action = "craft %s %d" % (item, outputs)
        if table is not None:
            action += " " + str(table["_id"])
        return action

    def set_work(self, kind, row, tool=None, purpose=""):
        self.work = {
            "kind": kind, "stage": "approach" if not row.get("in_reach") else ("equip" if tool else "gather"),
            "target": row["_id"], "name": row["_name"], "position": row.get("_pos"),
            "tool": tool, "started": self.now(), "purpose": purpose,
            "approaches": 0, "start_pos": position_of_self(self._last_obs),
            "before_inv": inv_sig(inventory(self._last_obs)),
            "before_cobble": cobble_count(inventory(self._last_obs)),
        }
        self.log("work_started", kind=kind, target=row["_id"], block=row["_name"], stage=self.work["stage"])

    def begin_step(self, row, obs):
        here = position_of_self(obs)
        q = row.get("_pos")
        if here is not None and q is not None:
            dx, dz = q[0] - here[0], q[2] - here[2]
            if abs(dx) + abs(dz) <= 1.6 and abs(dx) + abs(dz) >= 0.7:
                self.mine_direction = (int(round(dx)), int(round(dz)))
        self.set_work("step", row, "wooden_pickaxe", "open one observed adjacent cell")
        self.work["stage"] = "approach" if not row.get("in_reach") else "equip"
        self.work["start_pos"] = here
        self.work["before_cobble"] = cobble_count(inventory(obs))

    def step_target(self, obs, blocks):
        here = position_of_self(obs)
        if here is None or math.floor(here[1]) <= 42:
            return None
        py = math.floor(here[1])
        fx, fz = math.floor(here[0]), math.floor(here[2])
        choices = []
        for row in blocks:
            if row["_name"] not in GROUND or row["_name"] in {"sand", "gravel"}:
                continue
            if row.get("harvestable") is False or (row.get("visible") is False and not row.get("in_reach")):
                continue
            if row["_id"] in self.completed or self.is_bad(row["_id"], self.now()):
                continue
            q = row.get("_pos")
            if q is None:
                continue
            dx, dz = q[0] - here[0], q[2] - here[2]
            horizontal = abs(dx) + abs(dz)
            if horizontal < 0.72 or horizontal > 1.55:
                continue
            level = math.floor(q[1])
            if level not in {py - 1, py}:
                continue
            if math.floor(q[0]) == fx and math.floor(q[2]) == fz and level <= py - 1:
                continue
            tier = 0 if level == py - 1 else 1
            direction_penalty = 0
            if self.mine_direction is not None:
                vx, vz = self.mine_direction
                forward = dx * vx + dz * vz
                side = abs(dx * vz - dz * vx)
                direction_penalty = 0 if forward > 0 and side < 1.8 else 1
            choices.append((tier, direction_penalty, horizontal, row["_distance"], row))
        choices.sort(key=lambda x: x[:4])
        return choices[0][4] if choices else None

    def item_drop(self, obs, target_pos):
        source = obs.get("entities", obs.get("nearby_entities", []))
        if not isinstance(source, list):
            return None
        found = []
        here = position_of_self(obs)
        for raw in source:
            if not isinstance(raw, dict) or name(raw.get("name", raw.get("type", ""))) not in {"item", "item_stack", "item_entity"}:
                continue
            if raw.get("dead"):
                continue
            p = xyz(raw)
            if p is None:
                continue
            d = dist(here, p)
            if d <= 10.0 and (target_pos is None or dist(p, target_pos) <= 8.0):
                found.append((d, p))
        found.sort(key=lambda x: x[0])
        return found[0][1] if found else None

    def work_action(self, obs, blocks, stands, inv, now):
        job = self.work
        if not job:
            return None
        if now - job["started"] > (70.0 if job["kind"] != "step" else 85.0):
            self.suppress(job.get("target"), now, 150.0, "work_deadline")
            self.work = None
            return None
        target_id = job["target"]
        row = next((r for r in blocks if r["_id"] == target_id), None)
        stage = job["stage"]
        if job["kind"] == "step":
            here = position_of_self(obs)
            q = job.get("position")
            if row is None and stage not in {"move", "pickup"} and not job.get("mining_started"):
                self.suppress(target_id, now, 90.0, "step target left before dig")
                self.work = None
                return None
            if stage == "pickup":
                if cobble_count(inv) > int(job.get("before_cobble", 0)):
                    job["stage"] = "move"
                else:
                    drop = self.item_drop(obs, q)
                    if drop:
                        return self.add("goto %.2f %.2f %.2f" % drop, "collect a real mined drop")
                    if now - job.get("stage_at", now) > 8.0:
                        job["stage"] = "move"
                    else:
                        return None
            if stage == "move" or job["stage"] == "move":
                if q is None or here is None:
                    self.work = None
                    return None
                # A vanished block plus a nearby position change is the verified
                # step condition.  Resource acquisition is separately counted.
                if dist(here, q) <= 1.35 and dist(here, job.get("start_pos")) >= 0.55:
                    self.completed.add(target_id)
                    self.mine_steps += 1
                    self.log("opened_cell_entered", target=target_id, block=job.get("name"), steps=self.mine_steps, y=round(here[1], 2))
                    self.work = None
                    return None
                if row is not None:
                    self.suppress(target_id, now, 100.0, "mined cell not open")
                    self.work = None
                    return None
                return self.add("goto %.2f %.2f %.2f" % tuple(q), "enter the observed cell after it was mined")
            if row is None and stage not in {"move", "pickup"}:
                job["stage"] = "move"
                return None
        else:
            # A missing target is a pickup only after a real gather/dig was
            # posted. Exposure-list churn before mining must not fake a drop.
            if row is None and stage not in {"pickup", "done"}:
                if job.get("mining_started"):
                    job["stage"] = "pickup"
                    job["stage_at"] = now
                    stage = "pickup"
                else:
                    missing_since = job.setdefault("missing_since", now)
                    if now - missing_since < 2.0 and int(job.get("approaches", 0)) < 2:
                        stand = self.stand_for(stands, job.get("position"), now, 24.0)
                        if stand:
                            job["approaches"] = int(job.get("approaches", 0)) + 1
                            return self.add("navigate " + stand["_id"], "reacquire a last-seen exposed resource from a safe stand")
                        return None
                    self.suppress(target_id, now, 100.0, "target left observation before mining")
                    self.work = None
                    return None
            if job["stage"] == "pickup":
                if inv_sig(inv) != job.get("before_inv", ()):
                    self.completed.add(target_id)
                    self.log("resource_inventory_delta", target=target_id, block=job.get("name"))
                    self.work = None
                    return None
                drop = self.item_drop(obs, job.get("position"))
                if drop:
                    return self.add("goto %.2f %.2f %.2f" % drop, "walk to an observed dropped item")
                if now - job.get("stage_at", now) > 12.0:
                    self.suppress(target_id, now, 120.0, "drop_not_collected")
                    self.work = None
                    return None
                return None
        if stage == "approach":
            if row is None:
                self.suppress(target_id, now, 120.0, "target_left_observation")
                self.work = None
                return None
            if row.get("in_reach"):
                job["stage"] = "equip" if job.get("tool") else "gather"
                return None
            if int(job.get("approaches", 0)) >= 2:
                self.suppress(target_id, now, 120.0, "approach_limit")
                self.work = None
                return None
            target_pos = row.get("_pos")
            stand = self.stand_for(stands, target_pos, now, 24.0)
            job["approaches"] += 1
            if stand:
                return self.add("navigate " + stand["_id"], "approach exposed work target")
            if row.get("visible") and row.get("_distance", 999) <= 24:
                return self.add("navigate " + target_id, "approach the listed block interaction position")
            self.suppress(target_id, now, 120.0, "no_safe_approach")
            self.work = None
            return None
        if stage == "equip":
            tool = str(job.get("tool") or "")
            me = obs.get("self") if isinstance(obs.get("self"), dict) else {}
            if tool and name(me.get("held_item")) != tool:
                return self.add("equip %s hand" % tool, "equip the required observed tool")
            job["stage"] = "look" if job["kind"] == "step" else "gather"
            return None
        if stage == "look":
            q = row.get("_pos") if row else job.get("position")
            if q is None:
                self.work = None
                return None
            job["stage"] = "dig"
            return self.add("look_at %.2f %.2f %.2f" % tuple(q), "face the exposed block before direct mining")
        if stage == "gather":
            if row is None:
                job["stage"] = "pickup"
                job["stage_at"] = now
                job["before_inv"] = inv_sig(inv)
                return None
            if not row.get("in_reach"):
                job["stage"] = "approach"
                return None
            job["before_inv"] = inv_sig(inv)
            job["mining_started"] = True
            return self.add("gather %s 1" % target_id, job.get("purpose", "collect one observed resource"))
        if stage == "dig":
            if row is None or not row.get("in_reach"):
                job["stage"] = "approach"
                return None
            q = row.get("_pos")
            if q is None:
                self.work = None
                return None
            job["before_inv"] = inv_sig(inv)
            job["stage"] = "await_dig"
            job["mining_started"] = True
            return self.add("dig %d %d %d" % tuple(int(round(x)) for x in q), "mine one observed adjacent cell")
        if stage == "await_dig":
            if row is None:
                job["stage"] = "pickup" if job.get("name") in STONE else "move"
                job["stage_at"] = now
                return None
            return None
        if stage == "done":
            self.work = None
        return None

    def active_limit(self, action):
        verb = str(action).split()[0]
        return {"gather": 29.0, "navigate": 17.0, "goto": 17.0, "dig": 14.0,
                "craft": 9.0, "equip": 7.0, "place": 10.0, "look_at": 2.0,
                "furnace_start": 10.0, "furnace_take": 10.0, "eat": 7.0,
                "sleep": 8.0, "attack": 3.0}.get(verb, 7.0)

    def post(self, action, obs, purpose=""):
        if self.ctx.done:
            return
        action = str(action or "noop").strip() or "noop"
        if not self.legal(action):
            action = "noop"
        verb = action.split()[0]
        if action == self.last_action and verb not in {"noop", "controls", "attack", "shield", "cancel"}:
            action = "noop"
            verb = "noop"
        self.ctx.act(action)
        self.last_action = action
        now = self.now()
        if action == "cancel":
            self.tx = None
            self.work = None
            self.log("posted", action=action, purpose=purpose, phase=self.phase)
            return
        if verb in {"noop", "controls", "shield", "wait"}:
            if action != "noop":
                self.log("posted", action=action, purpose=purpose, phase=self.phase)
            return
        me = obs.get("self") if isinstance(obs.get("self"), dict) else {}
        parts = action.split()
        self.tx = {
            "action": action, "verb": verb, "started": now,
            "before_inv": inv_sig(inventory(obs)), "before_pos": position_of_self(obs),
            "before_held": name(me.get("held_item")), "dimension": me.get("dimension"),
            "feedback": feedback(obs), "purpose": purpose,
        }
        if verb in {"navigate", "goto"} and len(parts) >= 2:
            self.visited[parts[1]] = now
            ident = parts[1]
            if ident.startswith("s:"):
                try:
                    q = tuple(int(v) for v in ident[2:].split(","))
                    self.visited_cells.add((q[0] // 5, q[2] // 5))
                except (TypeError, ValueError):
                    pass
        if verb == "furnace_start" and len(parts) > 1:
            self.furnace_started[parts[1]] = now
        self.log("posted", action=action, purpose=purpose, phase=self.phase)

    def action_success(self, tx, obs, inv, now):
        verb = tx["verb"]
        action = tx["action"]
        before = tx["before_inv"]
        after = inv_sig(inv)
        me = obs.get("self") if isinstance(obs.get("self"), dict) else {}
        if verb == "respawn":
            return not bool(me.get("dead"))
        if verb == "equip":
            wanted = name(action.split()[1] if len(action.split()) > 1 else "")
            return name(me.get("held_item")) == wanted
        if verb == "gather":
            if after != before:
                return True
            target_id = self.work.get("target") if self.work else (action.split()[1] if len(action.split()) > 1 else "")
            blocks, _ = self.candidates(obs)
            return bool(target_id and not any(r["_id"] == target_id for r in blocks))
        if verb in {"craft", "furnace_start", "furnace_take", "eat", "sleep", "trade", "toss"}:
            if after != before:
                return True
            if verb == "eat" and num(me.get("food"), 20) > tx.get("food_before", -1):
                return True
            if verb == "sleep" and me.get("bed_position"):
                return True
            return False
        if verb == "place":
            parts = action.split()
            item = name(parts[1]) if len(parts) > 1 else ""
            before_map = dict(before)
            if before_map.get(item, 0) > inv.get(item, 0):
                if len(parts) > 2 and parts[2].startswith("b:"):
                    try:
                        support = tuple(float(v) for v in parts[2][2:].split(","))
                        if len(support) == 3 and parts[3] == "up":
                            placed = (support[0], support[1] + 1.0, support[2])
                            if item == "crafting_table":
                                self.table_hint = placed
                                self.table_return_attempted = False
                            elif item == "furnace":
                                self.furnace_hint = placed
                            self.place_wait_until = now + 7.0
                    except (TypeError, ValueError, IndexError):
                        pass
                return True
            blocks, _ = self.candidates(obs)
            return any(r["_name"] == item for r in blocks)
        if verb in {"goto", "navigate"}:
            p = position_of_self(obs)
            moved = dist(p, tx.get("before_pos")) >= 0.55
            if moved:
                return True
            parts = action.split()
            if len(parts) > 1 and parts[1].startswith(("b:", "s:")):
                target = next((r for r in rows(obs) if r["_id"] == parts[1]), None)
                return bool(target and target.get("in_reach"))
            return False
        if verb == "dig":
            parts = action.split()
            if len(parts) >= 4:
                target = tuple(float(v) for v in parts[1:4])
                blocks, _ = self.candidates(obs)
                if not any(dist(r.get("_pos"), target) < 0.3 for r in blocks):
                    return True
                changes = obs.get("block_changes")
                if isinstance(changes, list):
                    for row in changes:
                        if isinstance(row, dict) and dist(row.get("position"), target) < 0.3:
                            return True
            return False
        if verb == "look_at":
            return not has_task(obs) and now - tx["started"] >= 0.55
        if verb == "attack":
            return not has_task(obs) and now - tx["started"] >= 0.9
        return not has_task(obs) and now - tx["started"] >= 0.5

    def complete_tx(self, obs, inv, now):
        tx = self.tx
        if tx is None:
            return
        action, verb = tx["action"], tx["verb"]
        job = self.work
        if verb == "gather" and job:
            if inv_sig(inv) != tx["before_inv"]:
                if job.get("kind") == "step":
                    job["stage"] = "move"
                    self.log("stone_drop_acquired", target=job.get("target"), cobble=cobble_count(inv))
                else:
                    self.completed.add(str(job.get("target")))
                    self.log("resource_collected", target=job.get("target"), kind=job.get("kind"))
                    self.work = None
            else:
                blocks, _ = self.candidates(obs)
                if not any(r["_id"] == job.get("target") for r in blocks):
                    job["stage"] = "pickup"
                    job["stage_at"] = now
        elif verb == "dig" and job and job.get("kind") == "step":
            blocks, _ = self.candidates(obs)
            if not any(r["_id"] == job.get("target") for r in blocks):
                job["stage"] = "pickup" if job.get("name") in STONE else "move"
                job["stage_at"] = now
        elif verb == "equip" and job and job.get("stage") == "equip":
            job["stage"] = "look" if job.get("kind") == "step" else "gather"
        elif verb == "look_at" and job and job.get("stage") == "dig":
            job["stage"] = "dig"
        elif verb in {"navigate", "goto"} and job and job.get("stage") == "approach":
            job["approaches"] = int(job.get("approaches", 0))
        elif verb == "furnace_start":
            parts = action.split()
            if len(parts) > 1:
                self.furnace_started[parts[1]] = now
        self.log("transaction_verified", action=action, phase=self.phase, elapsed=round(now - tx["started"], 2))
        self.tx = None

    def update_tx(self, obs, inv, now):
        tx = self.tx
        if tx is None:
            return
        current_feedback = feedback(obs)
        if current_feedback != tx.get("feedback") and failed_feedback(current_feedback):
            key = self.work.get("target") if self.work else tx["action"]
            self.suppress(key, now, 150.0 if tx["verb"] in {"gather", "dig"} else 55.0, "adapter_feedback")
            self.log("transaction_rejected", action=tx["action"], feedback=current_feedback[-140:])
            self.tx = None
            self.work = None
            self.last_feedback = current_feedback
            return
        if self.action_success(tx, obs, inv, now):
            self.complete_tx(obs, inv, now)
            return
        if now - tx["started"] >= self.active_limit(tx["action"]):
            target = self.work.get("target") if self.work else tx["action"]
            self.suppress(target, now, 140.0 if tx["verb"] in {"gather", "dig"} else 60.0, "no_postcondition")
            self.log("transaction_timed_out", action=tx["action"], phase=self.phase)
            self.tx = None
            self.work = None

    def hard_emergency(self, obs, me, inv, blocks):
        if me.get("dead"):
            self.tx = None
            self.work = None
            self.table_hint = None
            self.furnace_hint = None
            self.place_wait_until = -1.0
            self.table_return_attempted = False
            self.furnace_started.clear()
            return "respawn", "normal respawn after death"
        age = num(obs.get("observation_age_s"), 0)
        if age > 3.0:
            return ("cancel" if has_task(obs) else "noop"), "stale observation"
        if me.get("is_in_lava") or me.get("is_in_fire") or me.get("on_fire"):
            if has_task(obs) or self.tx:
                return "cancel", "interrupt continuous fire/lava damage"
            return "controls forward,jump", "escape fire or lava"
        if me.get("is_in_water") and num(me.get("oxygen"), 300) < 6:
            if has_task(obs) or self.tx:
                return "cancel", "interrupt drowning risk"
            return "controls jump", "surface for oxygen"
        food = food_choice(inv)
        if food and (num(me.get("health"), 20) <= 6 or num(me.get("food"), 20) <= 5):
            if has_task(obs) or self.tx:
                return "cancel", "interrupt work for critical food"
            if name(me.get("held_item")) != food:
                return "equip %s hand" % food, "equip food before eating"
            return "eat " + food, "restore critical hunger or health"
        if num(me.get("fall_distance_estimate"), 0) > 4:
            if has_task(obs) or self.tx:
                return "cancel", "interrupt a dangerous fall"
            return "controls forward,jump", "escape a known fall state"
        return None, ""

    def hazard_rows(self, obs, blocks):
        me = obs.get("self") if isinstance(obs.get("self"), dict) else {}
        found = []
        if me.get("is_in_lava") or me.get("is_in_fire") or me.get("on_fire"):
            found.append({"kind": "lava_or_fire", "id": "self", "distance": 0.0, "range": "touching"})
        if me.get("is_in_water") and num(me.get("oxygen"), 300) < 6:
            found.append({"kind": "drowning", "id": "self", "distance": 0.0, "range": "touching"})
        for row in blocks:
            if row["_name"] in HAZARDS and row["_distance"] <= 4.5:
                found.append({"kind": row["_name"], "id": row["_id"], "distance": row["_distance"], "range": "near"})
        entities = obs.get("entities", obs.get("nearby_entities", []))
        if isinstance(entities, list):
            here = position_of_self(obs)
            for row in entities:
                if not isinstance(row, dict):
                    continue
                kind = name(row.get("name", row.get("type", "")))
                hostile = bool(row.get("hostile") or row.get("conditional_threat") or kind in HOSTILES)
                p = xyz(row)
                d = num(row.get("distance"), dist(here, p))
                if hostile and row.get("visible") and d <= 8.0:
                    found.append({"kind": kind or "hostile", "id": str(row.get("id", "")),
                                  "distance": d, "range": "touching" if d <= 3.2 else "near"})
        found.sort(key=lambda x: x["distance"])
        return found

    def threat_options(self, obs, hazards, stands, inv):
        actions = []
        me = obs.get("self") if isinstance(obs.get("self"), dict) else {}
        if has_task(obs) or self.tx:
            actions.append("cancel")
        if inv.get("shield"):
            actions.append("shield on")
        close = [h for h in hazards if h.get("id") != "self" and h.get("distance", 99) <= 3.2]
        weapon = next((w for w in WEAPONS if inv.get(w)), "")
        if close and weapon and close[0].get("id"):
            actions.append("equip %s hand" % weapon if name(me.get("held_item")) != weapon else "attack " + str(close[0].get("id")))
        nearest = next((h for h in hazards if h.get("id") != "self"), None)
        if nearest:
            threat_pos = None
            entities = obs.get("entities", obs.get("nearby_entities", []))
            if isinstance(entities, list):
                match = next((x for x in entities if str(x.get("id", "")) == str(nearest.get("id"))), None)
                threat_pos = xyz(match) if match else None
            choices = []
            for stand in stands:
                p = stand.get("_pos")
                if p is None or stand["_distance"] > 24 or stand["_distance"] < 1:
                    continue
                away = dist(p, threat_pos) if threat_pos is not None else 0
                choices.append((away, stand))
            choices.sort(key=lambda x: x[0], reverse=True)
            if choices:
                actions.append("navigate " + choices[0][1]["_id"])
        if "controls" in self.allowed:
            actions.append("controls back,jump")
        if not actions:
            actions.append("noop")
        return list(dict.fromkeys(a for a in actions if self.legal(a)))[:6]

    def poll_jev(self, obs, options, hazards):
        if self.jev_future is None:
            return None
        fut = self.jev_future
        now = self.now()
        if not fut.done():
            if now - self.jev_started >= 0.7:
                self.jev_committed = True
                return None
            return "__PENDING__"
        self.jev_future = None
        try:
            answers = fut.result()
            if self.jev_committed:
                self.jev_committed = False
                self.jev_cooldown_until = now + 1.0
                return None
            a = answers.get("response_a") if isinstance(answers, dict) else None
            b = answers.get("response_b") if isinstance(answers, dict) else None
            selected_a = getattr(a, "value", None)
            selected_b = getattr(b, "value", None)
            probs_a = getattr(a, "probabilities", {}) or {}
            probs_b = getattr(b, "probabilities", {}) or {}
            pa = num(probs_a.get(selected_a), 0.0) if isinstance(probs_a, dict) else 0.0
            pb = num(probs_b.get(selected_b), 0.0) if isinstance(probs_b, dict) else 0.0
            fresh = (self.jev_episode == obs.get("episode") and
                     (obs.get("seq") is None or self.jev_seq is None or
                      num(obs.get("seq")) - num(self.jev_seq) <= 8) and
                     self.jev_signature == (tuple(options), tuple((h["kind"], h["id"], h["range"]) for h in hazards)))
            if fresh and selected_a == selected_b and selected_a in options and min(pa, pb) >= 0.62:
                self.log("jev_threat_choice", action=selected_a, p_a=round(pa, 3), p_b=round(pb, 3))
                self.jev_cooldown_until = now + 1.5
                return selected_a
            self.log("jev_answer_ignored", fresh=fresh, selected=str(selected_a), p_a=round(pa, 3), p_b=round(pb, 3))
            self.jev_cooldown_until = now + 1.0
        except (BudgetExceeded, ModelUnavailable):
            raise
        except Exception as exc:
            self.log("jev_unusable", error=type(exc).__name__)
            self.jev_cooldown_until = now + 1.0
        return None

    def start_jev(self, obs, hazards, options):
        if not options or self.jev_future is not None or self.now() < self.jev_cooldown_until:
            return
        if self.ctx.budget.remaining().get("system1_calls", 0) <= 2:
            return
        me = obs.get("self") if isinstance(obs.get("self"), dict) else {}
        state = {
            "purpose": "Immediate hazard response only; ordinary work is not part of this choice.",
            "self": {
                "health_band": "critical" if num(me.get("health"), 20) < 7 else "low" if num(me.get("health"), 20) < 12 else "stable",
                "food_band": "low" if num(me.get("food"), 20) < 8 else "fed",
                "dimension": str(me.get("dimension", "overworld")),
                "on_ground": bool(me.get("on_ground")),
            },
            "current_phase": self.phase,
            "astra_focus": self.focus,
            "snapshot_freshness": "fresh" if num(obs.get("observation_age_s"), 0) <= 1.5 else "aging",
            "hazards": [{"kind": h["kind"], "range": h["range"], "observed_id": str(h["id"])} for h in hazards[:6]],
            "legal_responses": list(options),
            "rule": "Choose only an exact legal response. Do not invent IDs or coordinates. Prefer immediate escape/defense over mining.",
        }
        forward = list(options)
        reverse = list(reversed(options))
        questions = {
            "response_a": Choice("Which supplied response best reduces the immediate danger in this exact state?", {a: "Exact action: " + a for a in forward}),
            "response_b": Choice("Independently choose the safest supplied response. The order is reversed; use the explicit hazard facts, not option position.", {a: "Exact action: " + a for a in reverse}),
            "continue_work": Noul("Does an explicit hazard in `hazards` require interrupting ordinary mining or crafting now?", {
                "true": "A listed touching or near hazard requires interrupting ordinary work.",
                "false": "No listed hazard requires interrupting ordinary work.",
            }),
        }
        try:
            self.jev_future = self.ctx.spawn(lambda s=state, q=questions: self.ctx.decide(s, q, timeout_s=1.8))
            self.jev_started = self.now()
            self.jev_seq = obs.get("seq")
            self.jev_episode = obs.get("episode")
            self.jev_signature = (tuple(options), tuple((h["kind"], h["id"], h["range"]) for h in hazards))
            self.jev_actions = tuple(options)
            self.jev_committed = False
            self.log("jev_started", hazards=len(hazards), options=len(options))
        except (BudgetExceeded, ModelUnavailable):
            raise
        except Exception as exc:
            self.log("jev_spawn_failed", error=type(exc).__name__)

    def safety_action(self, obs, me, inv, blocks, stands, now):
        hazards = self.hazard_rows(obs, blocks)
        if not hazards:
            self.jev_signature = None
            self.jev_committed = True
            return None
        hard = any(h["kind"] in {"lava", "fire", "soul_fire", "lava_or_fire", "drowning"} for h in hazards)
        if hard:
            if has_task(obs) or self.tx:
                return "cancel", "hard hazard interruption"
            if any(h["kind"] == "drowning" for h in hazards):
                return "controls jump", "drowning escape"
            if any(h.get("id") == "self" for h in hazards):
                return "controls forward,jump", "leave fire/lava"
            return "controls back,jump", "avoid nearby fire/lava"
        options = self.threat_options(obs, hazards, stands, inv)
        choice = self.poll_jev(obs, options, hazards)
        if choice == "__PENDING__":
            # Do not let inference own survival. Use deterministic conservative
            # behavior while its sub-tick answer is pending.
            pass
        elif choice in options:
            return choice, "fresh Jev hazard response"
        self.start_jev(obs, hazards, options)
        # Deterministic fallback: cancel owned jobs; otherwise retreat or block.
        if has_task(obs) or self.tx:
            return "cancel", "interrupt work near visible hostile"
        if inv.get("shield"):
            return "shield on", "block while assessing visible hostile"
        if "controls back,jump" in options:
            return "controls back,jump", "conservative retreat fallback"
        return options[0], "conservative response fallback"

    def poll_astra(self):
        if self.astra_future is None or not self.astra_future.done():
            return
        fut = self.astra_future
        self.astra_future = None
        try:
            result = fut.result()
            text = str(getattr(result, "text", ""))
            start, end = text.find("{"), text.rfind("}")
            data = json.loads(text[start:end + 1]) if start >= 0 and end > start else {}
            focus = str(data.get("focus", "")).strip().lower()
            if focus in {"balanced", "food", "bed", "stone", "iron", "recover"}:
                self.focus = focus
                self.log("astra_focus_updated", focus=focus)
            self.astra_session = getattr(result, "session_id", self.astra_session)
        except (BudgetExceeded, ModelUnavailable):
            raise
        except Exception as exc:
            self.log("astra_output_ignored", error=type(exc).__name__)

    def maybe_start_astra(self, obs, me, inv, now):
        self.poll_astra()
        remaining = self.ctx.budget.remaining()
        if self.astra_future is not None or now < self.astra_next or self.astra_count >= 3:
            return
        if remaining.get("system2_calls", 0) <= 1 or remaining.get("wall_s", 0) < 170:
            return
        # Ask only at a quiet strategic checkpoint or after a long verified stall.
        if now < 50.0 and self.last_progress > 0:
            return
        if now - self.last_progress < 45.0 and not (now >= 55.0 and self.astra_count == 0):
            return
        goals = ["balanced", "food", "bed", "stone"]
        if any(inv.get(x) for x in STONE_PICKS):
            goals = ["balanced", "food", "bed", "iron", "recover"]
        observed_pos = xyz(me.get("position")) or (0.0, 64.0, 0.0)
        snapshot = {
            "objective": "survive and maximize verified progression toward Free the End",
            "phase": self.phase,
            "current_focus": self.focus,
            "dimension": str(me.get("dimension", "overworld")),
            "health_band": "critical" if num(me.get("health"), 20) < 7 else "low" if num(me.get("health"), 20) < 12 else "stable",
            "food_band": "low" if num(me.get("food"), 20) < 9 else "fed",
            "position_band": "surface" if observed_pos[1] > 50 else "underground",
            "inventory": {k: int(v) for k, v in sorted(inv.items()) if v and (k in FOOD_ITEMS or k in {"stick", "crafting_table", "cobblestone", "iron_ingot", "raw_iron", "iron_ore", "coal", "shield", "furnace"} or k.endswith(("_log", "_stem", "_planks", "_wool", "_bed", "_pickaxe")))},
            "verified_options": goals,
            "recent_stall_s": round(now - self.last_progress, 1),
            "rule": "Choose a focus label only; do not emit actions, coordinates, recipes, or claims about hidden terrain.",
        }
        prompt = (
            "You are a slow macro adviser for an ordinary vanilla Minecraft 1.21.1 Survival player. "
            "The controller owns exact recipes, candidate IDs, geometry, hazards, inventory counts and action execution. "
            "Choose one strategic focus from verified_options. Prefer food/bed only when an observed source exists; "
            "otherwise keep the next progression prerequisite. Never assume a cave, structure, ore, or seed. "
            "Reply as JSON only: {\"focus\":\"one exact supplied label\"}. State: "
            + json.dumps(snapshot, sort_keys=True, default=str)
        )
        session = self.astra_session
        try:
            self.astra_future = self.ctx.spawn(lambda p=prompt, s=session: self.ctx.think(p, session_id=s, timeout_s=90.0))
            self.astra_count += 1
            self.astra_next = now + 240.0
            self.log("astra_plan_requested", phase=self.phase, stall_s=round(now - self.last_progress, 1), allowed=goals)
        except (BudgetExceeded, ModelUnavailable):
            raise
        except Exception as exc:
            self.log("astra_spawn_failed", error=type(exc).__name__)

    def local_table_action(self, obs, blocks, here, inv, now):
        tables = self.table_rows(blocks, here)
        if tables:
            self.table_return_attempted = False
            return None
        if now < self.place_wait_until:
            return None
        if inv.get("crafting_table"):
            support = self.compile_place_support(blocks, here)
            me = obs.get("self") if isinstance(obs.get("self"), dict) else {}
            if support:
                if name(me.get("held_item")) != "crafting_table":
                    return self.add("equip crafting_table hand", "stage local workstation placement")
                return self.add("place crafting_table %s up" % support["_id"], "place a workstation at the current mining checkpoint")
            return None
        # If the just-placed station is still near, make one bounded return to
        # its observed support neighborhood after the candidate stream catches
        # up. A distant old station does not gate the local portable handoff.
        if self.table_hint is not None and here is not None and dist(here, self.table_hint) <= 10.0:
            hint_distance = dist(here, self.table_hint)
            if not self.table_return_attempted and hint_distance > 1.8:
                q = self.table_hint
                self.table_return_attempted = True
                return self.add("goto %.2f %.2f %.2f" % (q[0] + 1.0, q[1], q[2] + 1.0), "one bounded return to the newly placed local table")
            if not self.table_return_attempted:
                self.table_return_attempted = True
            return None
        # The inventory-grid recipe does not require a table; craft a station
        # only from actual planks once the old hint is clearly remote or absent.
        if plank_count(inv) >= 4:
            action = self.craft("crafting_table", 1, inv)
            return self.add(action, "craft a station in the inventory grid at the stone checkpoint") if action else None
        source = self.plank_item_for_log(inv)
        if source:
            action = self.craft(source, 4, inv)
            return self.add(action, "convert carried reserve logs into local station planks") if action else None
        return None

    def approach_table_hint(self, here, now):
        if self.table_hint is None or here is None or self.table_return_attempted:
            return None
        d = dist(here, self.table_hint)
        if d > 48.0 or now < self.place_wait_until:
            return None
        if d <= 1.8:
            self.table_return_attempted = True
            return None
        q = self.table_hint
        self.table_return_attempted = True
        self.place_wait_until = max(self.place_wait_until, now + 7.0)
        return self.add("goto %.2f %.2f %.2f" % (q[0] + 1.0, q[1], q[2] + 1.0), "one bounded return to the previously observed workstation")

    def compiler(self, obs, blocks, stands, inv, me, now):
        here = position_of_self(obs)
        # Death invalidates any remembered local station; inventory is audited
        # and the normal opening is rebuilt rather than assumed restored.
        if me.get("dead"):
            self.phase = "dead"
            return self.add("respawn", "ordinary respawn")
        if food_count(inv) == 0:
            phase = "opening"
        else:
            phase = "progress"

        logs = wood_count(inv)
        planks = plank_count(inv)
        tables = self.table_rows(blocks, here)
        station_exists = bool(inv.get("crafting_table") or tables or self.table_hint)
        if (not inv.get("crafting_table") and not tables and self.table_hint is not None and
                here is not None and dist(here, self.table_hint) <= 10.0 and
                self.table_return_attempted and now < self.place_wait_until):
            self.phase = "await_local_table"
            return self.add("noop", "allow the observed placed workstation to enter the nearby candidate view")
        # The first workstation costs four planks; after it exists, only the
        # pick/handles plus one portable second table must remain reserved.
        reserve_target = 9 if station_exists else 13
        reserve_needed = now < 95.0 and not any(inv.get(x) for x in WOODEN_PICKS) and logs * 4 + planks < reserve_target
        if not any(inv.get(x) for x in WOODEN_PICKS) and logs * 4 + planks >= 9:
            reserve_needed = False
        if not logs and planks < 4 and not inv.get("crafting_table") and not tables:
            target = self.closest_resource(blocks, suffixes=WOOD_ENDINGS, now=now)
            if target:
                self.set_work("wood", target, None, "gather first exposed logs and reserve")
                return None
            self.phase = "find_wood"
            return self.explore_action(stands, now)
        if reserve_needed:
            target = self.closest_resource(blocks, suffixes=WOOD_ENDINGS, now=now)
            if target:
                self.set_work("wood", target, None, "collect enough real wood for opening and local craft")
                self.phase = "reserve_wood"
                return None

        # Make the first table from real carried planks and place it only on a
        # currently observed adjacent support.  This is not a cached-coordinate
        # gate: later mining is allowed to continue independently.
        tables = self.table_rows(blocks, here)
        if not inv.get("crafting_table") and not tables and self.table_hint is None and not any(inv.get(x) for x in WOODEN_PICKS):
            source = self.plank_item_for_log(inv)
            if planks < 4 and source:
                action = self.craft(source, 4, inv)
                if action:
                    self.phase = "make_table"
                    return self.add(action, "craft feasible opening planks")
            if planks >= 4:
                action = self.craft("crafting_table", 1, inv)
                if action:
                    self.phase = "make_table"
                    return self.add(action, "craft the opening table")
            target = self.closest_resource(blocks, suffixes=WOOD_ENDINGS, now=now)
            if target:
                self.set_work("wood", target, None, "collect another log for the first table")
                return None
        if inv.get("crafting_table") and not tables:
            action = self.local_table_action(obs, blocks, here, inv, now)
            if action:
                self.phase = "place_table"
                return action
            self.phase = "place_table"
            return self.explore_action(stands, now)

        # A useful early tool is a real wooden pick. Do not require a table
        # candidate to remain in view if its inventory transaction already
        # succeeded; the 2x2 materials for the wooden pick are stationless only
        # for sticks, so table access is required for the pick itself.
        pick = next((x for x in WOODEN_PICKS if inv.get(x)), "")
        if not pick:
            if planks < 5 and logs:
                source = self.plank_item_for_log(inv)
                action = self.craft(source, 4, inv) if source else None
                if action:
                    self.phase = "wooden_tool"
                    return self.add(action, "make planks for sticks and wooden pick")
            if inv.get("stick", 0) < 2 and planks >= 2:
                action = self.craft("stick", 4, inv)
                if action:
                    self.phase = "wooden_tool"
                    return self.add(action, "craft feasible tool handles")
            if not tables:
                if inv.get("crafting_table"):
                    action = self.local_table_action(obs, blocks, here, inv, now)
                    if action:
                        self.phase = "wooden_tool"
                        return action
                # Reacquire only from an observed workstation or visible safe
                # stand; one attempt is cooled if navigation fails.
                remote = next((r for r in blocks if r["_name"] == "crafting_table"), None)
                if remote and remote["_distance"] <= 40:
                    return self.add("navigate " + remote["_id"], "approach a currently observed crafting table")
                hinted = self.approach_table_hint(here, now)
                if hinted:
                    return hinted
                return self.explore_action(stands, now)
            if tables[0]["_distance"] <= 4.5 and inv.get("stick", 0) >= 2 and planks >= 3:
                action = self.craft("wooden_pickaxe", 1, inv, tables[0])
                if action:
                    self.phase = "wooden_tool"
                    return self.add(action, "craft the first pick from observed inputs")
            # Missing logs are a real resource goal, not a recipe retry.
            target = self.closest_resource(blocks, suffixes=WOOD_ENDINGS, now=now)
            if target:
                self.set_work("wood", target, None, "collect real wood for the wooden pick")
                return None
            self.phase = "wooden_tool"
            return self.explore_action(stands, now)

        # Secure score-bearing food and a real bed opportunistically, but never
        # preempt a currently owned job.  Visible food sources are real targets.
        if food_count(inv) == 0 and (self.focus in {"balanced", "food", "recover"} or num(me.get("food"), 20) <= 12):
            food = self.closest_resource(blocks, names=FOOD_BLOCKS, now=now)
            if food:
                self.set_work("food", food, None, "collect a real observed food source")
                self.phase = "food"
                return None
        if (not any(k.endswith("_bed") and v for k, v in inv.items()) and not me.get("bed_position")
                and self.focus in {"balanced", "bed", "recover"}):
            bed = self.closest_resource(blocks, names={r["_name"] for r in blocks if r["_name"].endswith("_bed")}, now=now)
            if bed:
                self.set_work("bed", bed, None, "collect a currently observed bed")
                self.phase = "bed"
                return None
            wool = next((k for k, v in inv.items() if k.endswith("_wool") and v >= 3), "")
            if wool and planks >= 3 and tables and tables[0]["_distance"] <= 4.5:
                bed_name = wool[:-5] + "_bed"
                # Bed recipe is not part of the small explicit compiler above;
                # only issue it when the public feasible recipe hint says so.
                recipes = obs.get("feasible_recipes") or obs.get("recipes") or []
                feasible = any(isinstance(r, dict) and name(r.get("item", r.get("output"))) == bed_name for r in recipes) if isinstance(recipes, list) else False
                if feasible:
                    self.phase = "bed"
                    return self.add("craft %s 1 %s" % (bed_name, tables[0]["_id"]), "craft a bed from observed wool and feasible recipe hint")
        bed_item = next((k for k, v in inv.items() if k.endswith("_bed") and v), "")
        if bed_item:
            day_time = num(((obs.get("world") or {}).get("time_of_day")), 0) % 24000
            beds = [r for r in blocks if r["_name"] == bed_item]
            if beds and 12500 <= day_time <= 23000 and beds[0].get("in_reach"):
                return self.add("sleep " + beds[0]["_id"], "set a normal respawn point at night")

        # When three real cobble drops exist, perform the station handoff at the
        # current checkpoint. A carried reserve is made into a fresh local table
        # by the 2x2 recipe; no old spawn coordinate can block this transition.
        if not any(inv.get(x) for x in STONE_PICKS):
            if cobble_count(inv) >= 3:
                if inv.get("stick", 0) < 2:
                    if planks >= 2:
                        action = self.craft("stick", 4, inv)
                        if action:
                            self.phase = "stone_handles"
                            return self.add(action, "craft missing handles from real planks")
                    else:
                        target = self.closest_resource(blocks, suffixes=WOOD_ENDINGS, now=now)
                        if target:
                            self.set_work("wood", target, None, "collect real planks for stone-pick handles")
                            self.phase = "stone_handles"
                            return None
                        self.phase = "stone_handles"
                        return self.explore_action(stands, now)
                if not tables:
                    action = self.local_table_action(obs, blocks, here, inv, now)
                    if action:
                        self.phase = "local_stone_station"
                        return action
                    # Reuse the recorded table only as one bounded recovery
                    # attempt; it is never retried continuously.
                    if self.table_hint and here and dist(here, self.table_hint) <= 48 and not self.table_return_attempted and now >= self.bad.get("old_table_return", -1):
                        self.bad["old_table_return"] = now + 100
                        q = self.table_hint
                        self.table_return_attempted = True
                        return self.add("goto %.2f %.2f %.2f" % (q[0] + 1, q[1], q[2] + 1), "one bounded return to the observed original station")
                    self.phase = "stone_station_wait"
                    return self.explore_action(stands, now)
                if tables[0]["_distance"] <= 4.5:
                    action = self.craft("stone_pickaxe", 1, inv, tables[0])
                    if action:
                        self.phase = "stone_pick"
                        return self.add(action, "craft stone pick after verified cobble handoff")
            else:
                # First mine a currently exposed stone candidate with gather,
                # which chooses a correct tool and handles the ordinary pickup.
                target = self.closest_resource(blocks, names=STONE, now=now)
                if target:
                    tool = "wooden_pickaxe" if not any(inv.get(x) for x in STONE_PICKS) else "stone_pickaxe"
                    self.set_work("stone", target, tool, "gather exposed stone for cobblestone")
                    self.phase = "stone_gather"
                    return None
                # No visible seam: mine one adjacent, observed cell and verify
                # its disappearance before entering it.  This is bounded and
                # never targets unknown chunks or the support under the player.
                if not self.work:
                    step = self.step_target(obs, blocks)
                    if step:
                        self.begin_step(step, obs)
                        self.phase = "staircase"
                        return None
                if self.mine_steps >= 36 or (here and math.floor(here[1]) <= 44):
                    self.phase = "stone_frontier"
                    return self.explore_action(stands, now)
                self.phase = "stone_frontier"
                target = self.closest_resource(blocks, names=STONE, now=now)
                return self.explore_action(stands, now, target.get("_pos") if target else None)

        # Exposed ore and furnace progression are compiled only from real
        # inventory and currently listed blocks; no structure or coordinate is
        # inferred.  The route remains an honest midgame continuation.
        if any(inv.get(x) for x in STONE_PICKS):
            if inv.get("iron_ingot", 0) >= 1 and not inv.get("shield"):
                if planks < 6:
                    source = self.plank_item_for_log(inv)
                    if source:
                        action = self.craft(source, 4, inv)
                        if action:
                            self.phase = "shield_planks"
                            return self.add(action, "make the six real planks needed for a shield")
                    target = self.closest_resource(blocks, suffixes=WOOD_ENDINGS, now=now)
                    if target:
                        self.set_work("wood", target, None, "collect wood for shield planks")
                        self.phase = "shield_planks"
                        return None
                if planks >= 6 and tables and tables[0]["_distance"] <= 4.5:
                    action = self.craft("shield", 1, inv, tables[0])
                    if action:
                        self.phase = "shield"
                        return self.add(action, "craft a shield from real iron and planks")
            raw_iron = inv.get("raw_iron", 0) + inv.get("iron_ore", 0) + inv.get("deepslate_iron_ore", 0)
            if raw_iron < 1:
                ore = self.closest_resource(blocks, names=IRON_ORES, now=now, require_reach=True)
                if ore:
                    self.set_work("iron", ore, "stone_pickaxe", "collect exposed iron ore")
                    self.phase = "iron"
                    return None
            if not inv.get("coal") and not inv.get("charcoal"):
                coal = self.closest_resource(blocks, names=COAL_ORES, now=now, require_reach=True)
                if coal:
                    self.set_work("fuel", coal, "stone_pickaxe", "collect exposed furnace fuel")
                    self.phase = "fuel"
                    return None
            if raw_iron:
                furnaces = self.furnace_rows(blocks)
                if not furnaces and not inv.get("furnace") and cobble_count(inv) >= 8 and tables and tables[0]["_distance"] <= 4.5:
                    action = self.craft("furnace", 1, inv, tables[0])
                    if action:
                        self.phase = "furnace"
                        return self.add(action, "craft a real furnace from eight cobble")
                if inv.get("furnace") and not furnaces:
                    support = self.compile_place_support(blocks, here)
                    if support:
                        if name(me.get("held_item")) != "furnace":
                            return self.add("equip furnace hand", "equip the carried furnace")
                        return self.add("place furnace %s up" % support["_id"], "place a real furnace")
                if furnaces and furnaces[0]["_distance"] <= 4.5:
                    fid = furnaces[0]["_id"]
                    fuel = next((x for x in ("coal", "charcoal") if inv.get(x)), "")
                    if not fuel:
                        fuel = next((x for x, v in inv.items() if v and x.endswith("_planks")), "")
                    started = self.furnace_started.get(fid)
                    if started is None and fuel:
                        input_item = next((x for x in ("raw_iron", "iron_ore", "deepslate_iron_ore") if inv.get(x)), "raw_iron")
                        return self.add("furnace_start %s %s 1 %s 1" % (fid, input_item, fuel), "load one real iron input and fuel")
                    if started is not None and now - started >= 14:
                        return self.add("furnace_take " + fid, "retrieve smelted iron after real furnace time")
            # Dig only real adjacent loaded terrain; iron candidates hidden in a
            # solid interior are not assumed.
            iron_target = self.closest_resource(blocks, names=IRON_ORES | COAL_ORES, require_reach=True, now=now)
            if iron_target:
                self.set_work("ore", iron_target, "stone_pickaxe", "collect exposed ore or fuel")
                self.phase = "ore"
                return None

        # Hunger is handled before optional progress once it affects sprinting.
        if num(me.get("food"), 20) <= 12:
            food = food_choice(inv)
            if food:
                if name(me.get("held_item")) != food:
                    return self.add("equip %s hand" % food, "equip real food before eating")
                return self.add("eat " + food, "restore hunger")
        # If a visible passive animal is adjacent and food is absent, harvest it
        # with an existing weapon; do not follow it through unsafe terrain.
        if food_count(inv) == 0:
            entities = obs.get("entities", obs.get("nearby_entities", []))
            if isinstance(entities, list):
                animal = next((e for e in entities if isinstance(e, dict) and name(e.get("name", e.get("type"))) in {"sheep", "cow", "pig", "chicken"} and e.get("visible") and num(e.get("distance"), 99) <= 3.0), None)
                weapon = next((w for w in WEAPONS if inv.get(w)), "")
                if animal and weapon and now - self.last_attack > 1.0:
                    self.last_attack = now
                    if name(me.get("held_item")) != weapon:
                        return self.add("equip " + weapon + " hand", "prepare a weapon for an adjacent food animal")
                    return self.add("attack " + str(animal.get("id")), "harvest only an adjacent visible animal")
        # Repeatedly expand the loaded observation window only via safe listed
        # stands; no manual movement lease is allowed to run without a goal.
        self.phase = "explore"
        action = self.explore_action(stands, now)
        return action

    def run_episode(self, game):
        self.allowed = set(getattr(getattr(game, "action_spec", None), "names", ()) or ())
        self._last_obs = {}
        while not self.ctx.done:
            now = self.now()
            raw = self.ctx.observe()
            obs = raw if isinstance(raw, dict) else {}
            self._last_obs = obs
            me = obs.get("self") if isinstance(obs.get("self"), dict) else {}
            inv = inventory(obs)
            blocks, stands = self.candidates(obs)
            self.mark_progress(obs, inv, now)
            self.update_tx(obs, inv, now)

            urgent, why = self.hard_emergency(obs, me, inv, blocks)
            if urgent:
                self.post(urgent, obs, why)
                self.ctx.wait_for_tick()
                continue
            if num(obs.get("observation_age_s"), 0) > 3.0:
                self.post("cancel" if has_task(obs) else "noop", obs, "discard stale world state")
                self.ctx.wait_for_tick()
                continue

            threat_action = self.safety_action(obs, me, inv, blocks, stands, now)
            if threat_action:
                self.post(threat_action[0], obs, threat_action[1])
                self.ctx.wait_for_tick()
                continue

            self.maybe_start_astra(obs, me, inv, now)
            if has_task(obs):
                self.post("noop", obs, "continue the current bounded adaptor job")
                self.ctx.wait_for_tick()
                continue
            if self.tx is not None:
                self.post("noop", obs, "wait for a transaction postcondition")
                self.ctx.wait_for_tick()
                continue
            if self.work is not None:
                action = self.work_action(obs, blocks, stands, inv, now)
                if action:
                    self.post(action["action"], obs, action["purpose"])
                    self.ctx.wait_for_tick()
                    continue
                if self.work is not None:
                    self.post("noop", obs, "advance verified work stage")
                    self.ctx.wait_for_tick()
                    continue

            action = self.compiler(obs, blocks, stands, inv, me, now)
            # A work transaction may have been opened by the compiler; the next
            # loop posts only its first operation after the latest snapshot.
            if action:
                self.post(action["action"], obs, action["purpose"])
            else:
                self.post("noop", obs, "no currently supported observed transaction")
            self.ctx.wait_for_tick()


def build(ctx):
    return Controller(ctx)
