"""Candidate 0020: a transaction-based survival opener with a real mine.

The controller does not regard a vanished surface block as a resource plan.  It
makes the first descent as a small, observed staircase, then uses the same
postcondition machinery for a two-block branch.  All coordinates used for a
mine are derived from the current observation; no seed or map is assumed.
"""
from __future__ import annotations

import json
import math
import re
from typing import Any

from metaharness.core.contract import BudgetExceeded, ModelUnavailable
from metaharness.models.jev import Choice, Noul


WOOD_ENDINGS = ("_log", "_stem")
GROUND = {
    "grass_block", "dirt", "coarse_dirt", "rooted_dirt", "podzol", "mycelium",
    "sand", "red_sand", "gravel", "clay", "mud", "snow", "snow_block",
}
STONE = {"stone", "cobblestone", "deepslate", "cobbled_deepslate"}
IRON = {"iron_ore", "deepslate_iron_ore"}
COAL = {"coal_ore", "deepslate_coal_ore"}
FOOD_BLOCKS = {"hay_block", "melon", "sweet_berry_bush", "wheat", "carrots", "potatoes", "beetroots"}
FOODS = (
    "cooked_beef", "cooked_porkchop", "cooked_mutton", "cooked_chicken",
    "bread", "baked_potato", "carrot", "apple", "sweet_berries",
    "melon_slice", "beef", "porkchop", "mutton", "chicken", "rabbit",
    "cod", "salmon", "potato", "wheat",
)
PICKS = ("netherite_pickaxe", "diamond_pickaxe", "iron_pickaxe", "stone_pickaxe", "wooden_pickaxe")
STONE_PICK = {"netherite_pickaxe", "diamond_pickaxe", "iron_pickaxe", "stone_pickaxe"}
WEAPONS = (
    "netherite_sword", "diamond_sword", "iron_sword", "stone_sword",
    "wooden_sword", "iron_axe", "stone_axe", "wooden_axe",
)
HOSTILES = {
    "zombie", "skeleton", "creeper", "spider", "drowned", "husk", "stray",
    "witch", "phantom", "enderman", "slime", "magma_cube", "blaze", "ghast",
    "piglin", "piglin_brute", "zombified_piglin", "hoglin", "wither_skeleton",
    "ender_dragon", "shulker", "silverfish",
}
HAZARDS = {"lava", "fire", "soul_fire", "magma_block", "cactus", "powder_snow", "campfire", "soul_campfire"}
FAIL_WORDS = (
    "failed", "failure", "error", "unreachable", "timeout", "cannot", "rejected",
    "illegal", "not enough", "not in reach", "no path", "busy", "invalid",
    "unknown", "out of range",
)
ID_RE = re.compile(r"^[bs]:-?\d+,-?\d+,-?\d+$")


def n(value: Any, default: float = 0.0) -> float:
    try:
        value = float(value)
        return value if math.isfinite(value) else default
    except (TypeError, ValueError, OverflowError):
        return default


def item(value: Any) -> str:
    if isinstance(value, dict):
        value = value.get("name", value.get("item", value.get("type", value.get("id", ""))))
    if not isinstance(value, str):
        return ""
    value = value.removeprefix("minecraft:")
    return value if re.fullmatch(r"[a-z0-9_]+", value) else ""


def pnt(value: Any) -> tuple[float, float, float] | None:
    if isinstance(value, dict):
        if "position" in value:
            value = value.get("position")
        elif all(k in value for k in ("x", "y", "z")):
            value = [value.get("x"), value.get("y"), value.get("z")]
    if not isinstance(value, (list, tuple)) or len(value) < 3:
        return None
    q = tuple(n(x, math.inf) for x in value[:3])
    return q if all(math.isfinite(x) for x in q) else None


def id_pnt(ident: Any) -> tuple[float, float, float] | None:
    if not isinstance(ident, str) or not ID_RE.fullmatch(ident):
        return None
    try:
        q = tuple(float(x) for x in ident[2:].split(","))
        return q if len(q) == 3 else None
    except (TypeError, ValueError):
        return None


def d(a: Any, b: Any) -> float:
    a, b = pnt(a), pnt(b)
    if a is None or b is None:
        return math.inf
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def self_row(obs: dict) -> dict:
    return obs.get("self") if isinstance(obs.get("self"), dict) else {}


def self_pos(obs: dict) -> tuple[float, float, float] | None:
    me = self_row(obs)
    return pnt(me.get("position")) or pnt(me)


def _counts(source: Any) -> dict[str, int]:
    result: dict[str, int] = {}
    if isinstance(source, dict) and isinstance(source.get("items"), (dict, list)):
        source = source["items"]
    if isinstance(source, dict):
        pairs = source.items()
        for key, raw in pairs:
            name = item(key)
            raw_count = raw.get("count", 0) if isinstance(raw, dict) else raw
            count = max(0, int(n(raw_count)))
            if name and count:
                result[name] = result.get(name, 0) + count
    elif isinstance(source, list):
        for raw in source:
            if not isinstance(raw, dict):
                continue
            name = item(raw)
            count = max(0, int(n(raw.get("count", 0))))
            if name and count:
                result[name] = result.get(name, 0) + count
    return result


def inv_of(obs: dict) -> dict[str, int]:
    # The audit is authoritative.  The client view is only a labelled fallback
    # during its short join-time lag; neither cursor nor a container is counted.
    source = obs.get("inventory")
    if source in (None, {}, []):
        source = obs.get("inventory_authoritative")
    result = _counts(source)
    if not result:
        result = _counts(obs.get("inventory_client"))
    return result


def inv_sig(inv: dict[str, int]) -> tuple[tuple[str, int], ...]:
    return tuple(sorted((str(k), int(v)) for k, v in inv.items() if int(v) > 0))


def all_rows(obs: dict, prefix: str | None = None) -> list[dict]:
    here = self_pos(obs)
    result = []
    source = obs.get("candidates")
    if not isinstance(source, list):
        source = []
    for raw in source:
        if not isinstance(raw, dict):
            continue
        ident = raw.get("id")
        if not isinstance(ident, str) or not ID_RE.fullmatch(ident):
            continue
        if prefix and not ident.startswith(prefix + ":"):
            continue
        row = dict(raw)
        row["_id"] = ident
        row["_name"] = item(raw.get("name", raw.get("block", raw.get("type", ""))))
        row["_pos"] = pnt(raw.get("position")) or id_pnt(ident)
        row["_distance"] = n(raw.get("distance"), d(here, row["_pos"]))
        result.append(row)
    result.sort(key=lambda x: x["_distance"])
    return result


def entity_rows(obs: dict) -> list[dict]:
    here = self_pos(obs)
    source = obs.get("entities", obs.get("nearby_entities", []))
    if not isinstance(source, list):
        source = []
    result = []
    for raw in source:
        if not isinstance(raw, dict):
            continue
        row = dict(raw)
        row["_name"] = item(raw.get("name", raw.get("type", "")))
        row["_pos"] = pnt(raw.get("position"))
        row["_distance"] = n(raw.get("distance"), d(here, row["_pos"]))
        result.append(row)
    return result


def has_task(obs: dict) -> bool:
    value = obs.get("task")
    return value not in (None, "", {}, [])


def verb(action: str) -> str:
    return str(action or "").split(" ", 1)[0]


def food_count(inv: dict[str, int]) -> int:
    return sum(inv.get(x, 0) for x in FOODS)


def planks(inv: dict[str, int]) -> int:
    return sum(v for k, v in inv.items() if k.endswith("_planks"))


def logs(inv: dict[str, int]) -> int:
    return sum(v for k, v in inv.items() if k.endswith(WOOD_ENDINGS))


def best_food(inv: dict[str, int]) -> str:
    return next((x for x in FOODS if inv.get(x, 0)), "")


def current_block(obs: dict, ident: str) -> dict | None:
    return next((r for r in all_rows(obs, "b") if r["_id"] == ident), None)


def progress_sig(obs: dict) -> str:
    progress = obs.get("progress")
    adv = obs.get("advancements")
    try:
        return json.dumps([progress, adv], sort_keys=True, default=str)
    except Exception:
        return repr((progress, adv))


class Controller:
    """A finite milestone machine with explicit mine and recovery transactions."""

    def __init__(self, ctx: Any):
        self.ctx = ctx
        self.names: set[str] = set()
        self.tx: dict | None = None
        self.job: dict | None = None
        self.last_action = "noop"
        self.feedback_marker = ""
        self.blocked: dict[str, float] = {}
        self.visited: dict[str, float] = {}
        self.table_pos: tuple[float, float, float] | None = None
        self.table_wait = -1.0
        self.heading: tuple[int, int] = (1, 0)
        self.stair_steps = 0
        self.last_progress_at = 0.0
        self.last_inv = ()
        self.last_progress = ""
        self.dead_latched = False
        self.last_attack = -100.0
        self.hazard_cache: tuple[tuple, str, float] | None = None
        self.smelt_started: dict[str, float] = {}
        self._obs: dict = {}

    def log(self, event: str, **fields: Any) -> None:
        self.ctx.log(event, **fields)

    def now(self) -> float:
        return n(self.ctx.clock.now())

    def allowed(self, action: str) -> bool:
        return not self.names or verb(action) in self.names

    def blocked_now(self, key: str, now: float) -> bool:
        return now < self.blocked.get(key, -1.0)

    def suppress(self, key: str, now: float, seconds: float = 90.0, reason: str = "") -> None:
        if key and key not in {"noop", "cancel"}:
            self.blocked[key] = max(self.blocked.get(key, -1.0), now + seconds)
            self.log("suppressed", key=key, seconds=seconds, reason=reason)

    def mark_progress(self, obs: dict, now: float) -> None:
        inv = inv_sig(inv_of(obs))
        prog = progress_sig(obs)
        if inv != self.last_inv or prog != self.last_progress:
            self.last_inv, self.last_progress = inv, prog
            self.last_progress_at = now
            self.log("verified_change", inventory=True)
        for row in all_rows(obs, "b"):
            if row["_name"] == "crafting_table" and row["_pos"]:
                self.table_pos = row["_pos"]
                self.table_wait = -1.0

    def feedback_failed(self, value: Any) -> bool:
        if isinstance(value, dict):
            if value.get("ok") is False or value.get("success") is False:
                return True
            status = str(value.get("status", value.get("result", ""))).lower()
            if any(x in status for x in FAIL_WORDS):
                return True
            for key in ("message", "feedback", "reason", "error"):
                text = value.get(key)
                if isinstance(text, str) and any(x in text.lower() for x in FAIL_WORDS):
                    return True
            return False
        return isinstance(value, str) and any(x in value.lower() for x in FAIL_WORDS)

    def read_feedback(self, obs: dict, now: float) -> None:
        values = [obs.get(k) for k in ("action_feedback", "action_result", "last_result", "feedback")]
        try:
            marker = json.dumps(values, sort_keys=True, default=str)
        except Exception:
            marker = repr(values)
        if marker == self.feedback_marker:
            return
        self.feedback_marker = marker
        if self.tx and any(self.feedback_failed(x) for x in values):
            action = str(self.tx.get("action", ""))
            self.suppress(action, now, 150.0 if verb(action) in {"gather", "dig"} else 75.0, "adapter_feedback")
            if self.job and self.job.get("target"):
                self.suppress(str(self.job["target"]), now, 150.0, "job_feedback")
            self.log("transaction_rejected", action=action)
            self.tx = None
            if self.job and self.job.get("kind") in {"step", "tunnel", "gather"}:
                self.job = None

    def tx_limit(self, action: str) -> float:
        v = verb(action)
        return {
            "gather": 28.0, "navigate": 17.0, "goto": 17.0, "dig": 11.0,
            "craft": 7.0, "place": 8.0, "equip": 6.0, "look_at": 3.0,
            "furnace_start": 8.0, "furnace_take": 8.0, "eat": 6.0,
        }.get(v, 7.0)

    def update_tx(self, obs: dict, now: float) -> None:
        tx = self.tx
        if not tx:
            return
        action = str(tx.get("action", ""))
        v = verb(action)
        inv = inv_of(obs)
        before = tx.get("before_inv", ())
        changed = inv_sig(inv) != before
        before_pos = tx.get("before_pos")
        moved = d(before_pos, self_pos(obs)) >= 0.65
        done = False
        dig_vanished = False
        if v == "equip":
            wanted = item(action.split()[1]) if len(action.split()) > 1 else ""
            done = item(self_row(obs).get("held_item")) == wanted
        elif v in {"craft", "place", "furnace_start", "furnace_take", "eat"}:
            done = changed
            if v == "place":
                parts = action.split()
                if len(parts) >= 3 and changed:
                    support = id_pnt(parts[2])
                    if support:
                        self.table_pos = (support[0], support[1] + 1.0, support[2]) if parts[1] == "crafting_table" else self.table_pos
                        if parts[1] == "crafting_table":
                            self.table_wait = now + 8.0
                            self.log("table_consumed", position=self.table_pos)
        elif v in {"goto", "navigate"}:
            done = moved
        elif v == "look_at":
            done = now - n(tx.get("started"), now) >= 0.45
        elif v == "dig":
            target = str(tx.get("target", ""))
            row = current_block(obs, target) if target else None
            vanished = row is None or row.get("_name") in {"air", "water", "lava"}
            changes = obs.get("block_changes", [])
            if isinstance(changes, list):
                for change in changes:
                    if not isinstance(change, dict):
                        continue
                    cp = pnt(change.get("position"))
                    tp = id_pnt(target)
                    if tp and cp and all(abs(cp[i] - tp[i]) < 0.2 for i in range(3)) and item(change.get("name")) == "air":
                        vanished = True
                        break
            done = vanished or changed
            dig_vanished = vanished
        elif v == "gather":
            target = str(tx.get("target", ""))
            done = changed or (target and current_block(obs, target) is None and not has_task(obs))
        if done:
            # A direct dig can change inventory before the candidate stream
            # reflects the new air cell.  Keep the bound in a short verification
            # wait instead of issuing the same dig again; repeated direct digs
            # were the main failure mode of the surface controller.
            if v == "dig" and self.job and self.job.get("kind") in {"step", "tunnel"}:
                self.job["stage"] = "wait_dig"
                self.job["dig_done_at"] = now
                self.job["dig_vanished"] = dig_vanished
            self.log("transaction_done", action=action)
            self.tx = None
            return
        age = now - n(tx.get("started"), now)
        if age >= self.tx_limit(action) and not has_task(obs):
            self.suppress(action, now, 150.0 if v in {"gather", "dig"} else 75.0, "postcondition_timeout")
            if self.job and self.job.get("target"):
                self.suppress(str(self.job["target"]), now, 150.0, "target_timeout")
            self.log("transaction_timeout", action=action)
            self.tx = None
            if self.job and self.job.get("kind") in {"step", "tunnel", "gather"}:
                self.job = None

    def post(self, action: str, obs: dict, now: float, why: str = "") -> None:
        action = str(action or "noop").strip() or "noop"
        if not self.allowed(action) and action != "noop":
            action = "noop"
        v = verb(action)
        if action == self.last_action and v not in {"noop", "controls", "attack", "shield"}:
            action = "noop"
            v = "noop"
        if action == "cancel":
            self.tx = None
            self.job = None
        try:
            self.ctx.act(action)
        except (BudgetExceeded, ModelUnavailable):
            raise
        except Exception as exc:
            self.log("act_rejected", action=action, error=type(exc).__name__)
            action = "noop"
            v = "noop"
            try:
                self.ctx.act("noop")
            except (BudgetExceeded, ModelUnavailable):
                raise
            except Exception:
                pass
        self.last_action = action
        if action != "noop":
            self.log("posted", action=action, why=why)
        if action == "noop" or v in {"controls", "attack", "shield", "wait", "cancel"}:
            return
        self.tx = {
            "action": action,
            "started": now,
            "before_inv": inv_sig(inv_of(obs)),
            "before_pos": self_pos(obs),
            "target": self._target_from_action(action),
        }

    @staticmethod
    def _target_from_action(action: str) -> str:
        parts = action.split()
        if not parts:
            return ""
        if parts[0] in {"gather", "dig"} and len(parts) >= 2:
            return parts[1] if parts[1].startswith("b:") else ""
        return ""

    def cancel_first(self, obs: dict, now: float, why: str) -> bool:
        if has_task(obs) or self.tx or self.job:
            if has_task(obs):
                self.post("cancel" if self.allowed("cancel") else "noop", obs, now, why)
            else:
                self.tx = None
                self.job = None
            return True
        return False

    def support(self, obs: dict, now: float) -> dict | None:
        pos = self_pos(obs)
        if pos is None:
            return None
        fy = math.floor(pos[1]) - 1
        fx, fz = math.floor(pos[0]), math.floor(pos[2])
        choices = []
        for row in all_rows(obs, "b"):
            q = row["_pos"]
            if q is None or row["_name"] not in GROUND | STONE:
                continue
            if row.get("visible") is False and not row.get("in_reach"):
                continue
            if math.floor(q[1]) != fy or (math.floor(q[0]) == fx and math.floor(q[2]) == fz):
                continue
            key = "place crafting_table %s up" % row["_id"]
            if self.blocked_now(key, now):
                continue
            choices.append(row)
        choices.sort(key=lambda r: (not bool(r.get("in_reach")), r["_distance"]))
        return choices[0] if choices else None

    def near_table(self, obs: dict) -> dict | None:
        rows = [r for r in all_rows(obs, "b") if r["_name"] == "crafting_table" and r["_distance"] <= 4.6]
        rows.sort(key=lambda r: r["_distance"])
        return rows[0] if rows else None

    def target_ok(self, row: dict, now: float, names: set[str] | None = None, suffixes: tuple[str, ...] = ()) -> bool:
        name = row.get("_name", "")
        if names is not None and name not in names:
            return False
        if suffixes and not any(name.endswith(s) for s in suffixes):
            return False
        if not row.get("_id") or self.blocked_now(row["_id"], now):
            return False
        if row.get("harvestable") is False or name in HAZARDS | {"air", "water", "lava", "bubble_column"}:
            return False
        return bool(row.get("visible") or row.get("in_reach"))

    def choose_resource(self, obs: dict, now: float, names: set[str] | None = None, suffixes: tuple[str, ...] = ()) -> dict | None:
        choices = [r for r in all_rows(obs, "b") if self.target_ok(r, now, names, suffixes)]
        choices.sort(key=lambda r: (not bool(r.get("in_reach")), not bool(r.get("visible")), r["_distance"]))
        return choices[0] if choices else None

    def choose_surface_step(self, obs: dict, now: float) -> dict | None:
        pos = self_pos(obs)
        if pos is None:
            return None
        fy = math.floor(pos[1]) - 1
        fx, fz = math.floor(pos[0]), math.floor(pos[2])
        choices = []
        for row in all_rows(obs, "b"):
            q = row["_pos"]
            if q is None or row["_name"] not in (GROUND | STONE):
                continue
            if not self.target_ok(row, now) or math.floor(q[1]) != fy:
                continue
            dx, dz = math.floor(q[0]) - fx, math.floor(q[2]) - fz
            if abs(dx) + abs(dz) != 1:
                continue
            # A known solid lower cell makes a one-block descent safe.  If the
            # adaptor has not supplied collision data, the exposed adjacent
            # floor is still a legal conservative probe.
            below = self.collision(obs, (math.floor(q[0]), math.floor(q[1]) - 1, math.floor(q[2])))
            if below == "passable":
                continue
            choices.append((0 if below == "solid" else 1, row["_distance"], row))
        choices.sort(key=lambda x: (x[0], x[1]))
        return choices[0][2] if choices else None

    def collision(self, obs: dict, absolute: tuple[int, int, int]) -> str | None:
        pos = self_pos(obs)
        world = obs.get("world") if isinstance(obs.get("world"), dict) else {}
        values = world.get("local_collision", obs.get("local_collision", []))
        if pos is None or not isinstance(values, list):
            return None
        base = (math.floor(pos[0]), math.floor(pos[1]), math.floor(pos[2]))
        delta = (absolute[0] - base[0], absolute[1] - base[1], absolute[2] - base[2])
        for raw in values:
            if isinstance(raw, (list, tuple)) and len(raw) >= 4:
                try:
                    if tuple(int(raw[i]) for i in range(3)) == delta:
                        return str(raw[3])
                except (TypeError, ValueError):
                    pass
        return None

    def choose_tunnel_block(self, obs: dict, now: float) -> dict | None:
        pos = self_pos(obs)
        if pos is None:
            return None
        fx, fy, fz = math.floor(pos[0]), math.floor(pos[1]), math.floor(pos[2])
        dx, dz = self.heading
        # Clear the roof first: a two-high tunnel cannot be entered while the
        # head cell is solid.  Then clear the foot cell and walk into it.
        wanted = {
            (fx + dx, fy + 1, fz + dz),
            (fx + dx, fy, fz + dz),
        }
        candidates = []
        for row in all_rows(obs, "b"):
            q = row["_pos"]
            if q is None or row["_name"] in HAZARDS:
                continue
            cell = (math.floor(q[0]), math.floor(q[1]), math.floor(q[2]))
            if cell not in wanted or not self.target_ok(row, now):
                continue
            if row["_name"] in {"air", "water", "lava"}:
                continue
            # head before foot; the foot block is explicitly not the current
            # support block because it is one cell ahead.
            rank = 0 if cell[1] == fy + 1 else 1
            candidates.append((rank, row["_distance"], row))
        candidates.sort(key=lambda x: (x[0], x[1]))
        return candidates[0][2] if candidates else None

    def safe_stand(self, obs: dict, now: float, around: tuple[float, float, float] | None = None) -> dict | None:
        here = self_pos(obs)
        choices = []
        for row in all_rows(obs, "s"):
            q = row["_pos"]
            if q is None or row["_distance"] < 1.5 or row["_distance"] > 45:
                continue
            if self.blocked_now("goto " + row["_id"], now) or self.blocked_now("navigate " + row["_id"], now):
                continue
            if around is not None and d(q, around) > 6.5:
                continue
            last = self.visited.get(row["_id"], -1000.0)
            cell = (math.floor(q[0] / 5), math.floor(q[2] / 5))
            choices.append((0 if last < now - 45 else 1, 0 if cell not in {(math.floor((here or q)[0] / 5), math.floor((here or q)[2] / 5))} else 1, -row["_distance"], row))
        choices.sort(key=lambda x: (x[0], x[1], x[2]))
        return choices[0][3] if choices else None

    def explore(self, obs: dict, now: float) -> str:
        stand = self.safe_stand(obs, now)
        if stand:
            self.visited[stand["_id"]] = now
            q = stand["_pos"]
            # goto is deliberately used for a concrete safe feet position;
            # navigate candidates are retained as a fallback after a rejection.
            return "goto %.2f %.2f %.2f" % q
        stand = self.safe_stand(obs, now)
        if stand:
            return "navigate " + stand["_id"]
        return "wait 0.5"

    def begin_gather(self, row: dict, kind: str, now: float, count: int = 1) -> None:
        self.job = {
            "kind": "gather", "target": row["_id"], "name": row["_name"],
            "started": now, "before": inv_sig(inv_of(self._obs)), "count": count,
            "purpose": kind,
        }
        self.log("gather_begin", target=row["_id"], block=row["_name"], purpose=kind)

    def begin_step(self, row: dict, now: float) -> None:
        q = row["_pos"]
        pos = self_pos(self._obs)
        if q is None or pos is None:
            return
        dx = math.floor(q[0]) - math.floor(pos[0])
        dz = math.floor(q[2]) - math.floor(pos[2])
        if abs(dx) + abs(dz) == 1:
            self.heading = (max(-1, min(1, dx)), max(-1, min(1, dz)))
        self.job = {
            "kind": "step", "target": row["_id"], "position": q,
            "stage": "tool", "started": now, "before": inv_sig(inv_of(self._obs)),
        }
        self.stair_steps += 1
        self.log("stair_step_begin", target=row["_id"], block=row["_name"], step=self.stair_steps)

    def begin_tunnel(self, row: dict, now: float) -> None:
        pos = self_pos(self._obs)
        role = "foot"
        if pos is not None and row.get("_pos") is not None:
            role = "head" if math.floor(row["_pos"][1]) == math.floor(pos[1]) + 1 else "foot"
        self.job = {
            "kind": "tunnel", "target": row["_id"], "position": row["_pos"],
            "role": role, "stage": "tool", "started": now,
            "before": inv_sig(inv_of(self._obs)),
        }
        self.log("branch_begin", target=row["_id"], block=row["_name"], heading=self.heading, role=role)

    def job_action(self, obs: dict, now: float) -> str | None:
        job = self.job
        if not job:
            return None
        kind = job.get("kind")
        inv = inv_of(obs)
        if now - n(job.get("started"), now) > (38.0 if kind == "gather" else 45.0):
            self.suppress(str(job.get("target", "")), now, 150.0, "job_deadline")
            self.job = None
            return None
        if kind == "gather":
            target = str(job.get("target", ""))
            if inv_sig(inv) != job.get("before"):
                self.log("gather_confirmed", target=target)
                self.job = None
                return None
            row = current_block(obs, target)
            if row is None:
                self.suppress(target, now, 150.0, "target_missing")
                self.job = None
                return None
            if not self.tx:
                return "gather %s %d" % (target, int(job.get("count", 1)))
            return None

        target = str(job.get("target", ""))
        row = current_block(obs, target)
        stage = str(job.get("stage", ""))
        pos = job.get("position")
        if kind == "step":
            if stage == "tool":
                pick = next((x for x in PICKS if inv.get(x)), "")
                if not pick:
                    self.job = None
                    return None
                if item(self_row(obs).get("held_item")) != pick:
                    job["stage"] = "aim"
                    return "equip %s hand" % pick
                job["stage"] = "aim"
                return None
            if stage == "aim":
                if row is None:
                    self.job = None
                    return None
                job["stage"] = "dig"
                q = row["_pos"] or pos
                return "look_at %.2f %.2f %.2f" % q if q else None
            if stage == "dig":
                if row is None:
                    job["stage"] = "foot_select" if kind == "tunnel" and job.get("role") == "head" else "enter"
                    return None
                if not self.tx:
                    job["stage"] = "wait_dig"
                    q = row["_pos"] or pos
                    return "dig %d %d %d" % tuple(int(round(x)) for x in q) if q else None
                return None
            if stage == "wait_dig":
                if job.get("dig_vanished") or row is None or row.get("_name") in {"air", "water", "lava"}:
                    job["stage"] = "foot_select" if kind == "tunnel" and job.get("role") == "head" else "enter"
                elif now - n(job.get("dig_done_at"), now) > 2.5:
                    self.suppress(target, now, 150.0, "dig_postcondition_not_visible")
                    self.job = None
                return None
            if stage == "foot_select":
                base = self_pos(obs)
                foot = None
                if base is not None:
                    fx, fy, fz = math.floor(base[0]), math.floor(base[1]), math.floor(base[2])
                    dx, dz = self.heading
                    for candidate in all_rows(obs, "b"):
                        q = candidate.get("_pos")
                        if q is None:
                            continue
                        if (math.floor(q[0]), math.floor(q[1]), math.floor(q[2])) != (fx + dx, fy, fz + dz):
                            continue
                        if self.target_ok(candidate, now):
                            foot = candidate
                            break
                if foot is None:
                    self.job = None
                    return None
                job["target"] = foot["_id"]
                job["position"] = foot["_pos"]
                job["role"] = "foot"
                job["stage"] = "aim"
                self.log("branch_foot_selected", target=foot["_id"], block=foot["_name"])
                return None
            if stage == "enter":
                if pos is None:
                    self.job = None
                    return None
                job["stage"] = "verify"
                # The center of the removed floor cell is a one-block lower
                # feet position; goto can use the freshly exposed stair cell.
                return "goto %.2f %.2f %.2f" % (pos[0] + 0.5, pos[1], pos[2] + 0.5)
            if stage == "verify":
                here = self_pos(obs)
                if here and pos and d(here, (pos[0] + 0.5, pos[1], pos[2] + 0.5)) < 2.0:
                    self.log("stair_step_confirmed", target=target, position=pos)
                    self.job = None
                    return None
                if not self.tx:
                    self.suppress(target, now, 120.0, "stair_target_not_reached")
                    self.job = None
                return None
            return None

        if kind == "tunnel":
            if stage == "tool":
                pick = next((x for x in PICKS if inv.get(x)), "")
                if not pick:
                    self.job = None
                    return None
                if item(self_row(obs).get("held_item")) != pick:
                    job["stage"] = "aim"
                    return "equip %s hand" % pick
                job["stage"] = "aim"
                return None
            if stage == "aim":
                if row is None:
                    self.job = None
                    return None
                job["stage"] = "dig"
                q = row["_pos"] or pos
                return "look_at %.2f %.2f %.2f" % q if q else None
            if stage == "dig":
                if row is None:
                    job["stage"] = "foot_select" if kind == "tunnel" and job.get("role") == "head" else "enter"
                    return None
                if not self.tx:
                    job["stage"] = "wait_dig"
                    q = row["_pos"] or pos
                    return "dig %d %d %d" % tuple(int(round(x)) for x in q) if q else None
                return None
            if stage == "wait_dig":
                if job.get("dig_vanished") or row is None or row.get("_name") in {"air", "water", "lava"}:
                    job["stage"] = "foot_select" if kind == "tunnel" and job.get("role") == "head" else "enter"
                elif now - n(job.get("dig_done_at"), now) > 2.5:
                    self.suppress(target, now, 150.0, "dig_postcondition_not_visible")
                    self.job = None
                return None
            if stage == "foot_select":
                base = self_pos(obs)
                foot = None
                if base is not None:
                    fx, fy, fz = math.floor(base[0]), math.floor(base[1]), math.floor(base[2])
                    dx, dz = self.heading
                    for candidate in all_rows(obs, "b"):
                        q = candidate.get("_pos")
                        if q is None:
                            continue
                        if (math.floor(q[0]), math.floor(q[1]), math.floor(q[2])) != (fx + dx, fy, fz + dz):
                            continue
                        if self.target_ok(candidate, now):
                            foot = candidate
                            break
                if foot is None:
                    self.job = None
                    return None
                job["target"] = foot["_id"]
                job["position"] = foot["_pos"]
                job["role"] = "foot"
                job["stage"] = "aim"
                self.log("branch_foot_selected", target=foot["_id"], block=foot["_name"])
                return None
            if stage == "enter":
                if pos is None:
                    self.job = None
                    return None
                job["stage"] = "verify"
                return "goto %.2f %.2f %.2f" % (pos[0] + 0.5, pos[1], pos[2] + 0.5)
            if stage == "verify":
                here = self_pos(obs)
                if here and pos and d(here, (pos[0] + 0.5, pos[1], pos[2] + 0.5)) < 2.0:
                    self.log("branch_entered", target=target)
                    self.job = None
                    return None
                if not self.tx:
                    self.suppress(target, now, 120.0, "branch_target_not_reached")
                    self.job = None
                return None
        return None

    def threat_menu(self, obs: dict) -> tuple[list[dict], list[str]]:
        me = self_row(obs)
        threats = []
        for row in entity_rows(obs):
            hostile = bool(row.get("hostile") or row.get("conditional_threat") or row["_name"] in HOSTILES)
            if hostile and row.get("visible") and row["_distance"] <= 8.0:
                threats.append({"kind": row["_name"], "id": str(row.get("id", "")), "range": "touching" if row["_distance"] <= 3.2 else "near", "distance": row["_distance"]})
        if me.get("is_in_lava") or me.get("is_in_fire") or me.get("on_fire"):
            threats.append({"kind": "lava_or_fire", "id": "self", "range": "touching", "distance": 0.0})
        if me.get("is_in_water") and n(me.get("oxygen"), 20) < 6:
            threats.append({"kind": "drowning", "id": "self", "range": "touching", "distance": 0.0})
        inv = inv_of(obs)
        options = []
        food = best_food(inv)
        if food and (n(me.get("health"), 20) <= 8 or n(me.get("food"), 20) <= 5):
            options += [("equip %s hand" % food) if item(me.get("held_item")) != food else "eat " + food]
        close = [x for x in threats if x.get("id") != "self" and x.get("range") == "touching"]
        weapon = next((x for x in WEAPONS if inv.get(x)), "")
        if close and weapon:
            if item(me.get("held_item")) != weapon:
                options.append("equip %s hand" % weapon)
            options += ["attack " + x["id"] for x in close if x.get("id")]
        if inv.get("shield"):
            options.append("shield on")
        options += ["controls back,jump", "cancel" if has_task(obs) else "noop"]
        answer = []
        for x in options:
            if self.allowed(x) and x not in answer:
                answer.append(x)
        return threats, answer[:8]

    def tactical(self, obs: dict, now: float) -> str | None:
        threats, options = self.threat_menu(obs)
        if not threats:
            self.hazard_cache = None
            return None
        # Hard hazards never wait for inference.
        if any(x["kind"] == "lava_or_fire" for x in threats):
            return "controls forward,jump" if self.allowed("controls") else "cancel"
        if any(x["kind"] == "drowning" for x in threats):
            return "controls jump" if self.allowed("controls") else "cancel"
        if has_task(obs):
            return "cancel" if self.allowed("cancel") else "noop"
        if not options:
            return "noop"
        signature = tuple((x["kind"], x["id"], x["range"]) for x in threats)
        if self.hazard_cache and self.hazard_cache[0] == signature and now - self.hazard_cache[2] < 2.5:
            return self.hazard_cache[1] if self.hazard_cache[1] in options else options[0]
        state = {
            "purpose": "Immediate survival only; choose one supplied action.",
            "self": {"health_band": "critical" if n(self_row(obs).get("health"), 20) < 7 else "low" if n(self_row(obs).get("health"), 20) < 12 else "stable", "dimension": self_row(obs).get("dimension", ""), "in_lava": False},
            "hazards": [{"kind": x["kind"], "range": x["range"]} for x in threats[:6]],
            "legal_actions": options,
            "rule": "Never invent an id. Prefer retreat, shield, or food over mining. Do not attack lava or drowning.",
        }
        questions = {
            "response": Choice("Select the safest exact response from legal_actions.", {x: "Execute this supplied response: " + x for x in options}),
            "continue": Noul("Is it safe to continue the current work without a defensive response?", {"true": "No immediate threat requires defense.", "false": "A visible hostile requires defense, retreat, or interruption."}),
        }
        try:
            result = self.ctx.decide(state, questions, timeout_s=1.2)
            ans = result.get("response") if isinstance(result, dict) else None
            value = getattr(ans, "value", None)
            probs = getattr(ans, "probabilities", {}) or {}
            probability = n(probs.get(value), 0.0) if isinstance(probs, dict) else 0.0
            if value in options and probability >= 0.48:
                self.hazard_cache = (signature, value, now)
                self.log("jev_tactical", action=value, probability=round(probability, 3))
                return value
        except (BudgetExceeded, ModelUnavailable):
            raise
        except (TimeoutError, RuntimeError, TypeError, ValueError, AttributeError, KeyError) as exc:
            self.log("jev_fallback", error=type(exc).__name__)
        self.hazard_cache = (signature, options[0], now)
        return options[0]

    def safe_override(self, obs: dict, now: float) -> str | None:
        me = self_row(obs)
        inv = inv_of(obs)
        if me.get("dead"):
            return "respawn"
        if n(obs.get("observation_age_s"), 0) > 3.0:
            return "cancel" if has_task(obs) else "noop"
        if n(me.get("health"), 20) <= 7 or n(me.get("food"), 20) <= 5:
            food = best_food(inv)
            if food:
                if item(me.get("held_item")) != food:
                    return "equip %s hand" % food
                return "eat " + food
        if me.get("is_in_lava") or me.get("is_in_fire") or me.get("on_fire"):
            return "controls forward,jump"
        if me.get("is_in_water") and n(me.get("oxygen"), 20) < 6:
            return "controls jump"
        return None

    def phase(self, obs: dict) -> str:
        inv = inv_of(obs)
        table = self.near_table(obs) or (self.table_pos is not None)
        if logs(inv) == 0 and planks(inv) == 0 and not inv.get("crafting_table") and not table:
            return "wood"
        if not table and not inv.get("crafting_table"):
            return "table"
        if not any(inv.get(x) for x in PICKS):
            return "wood_tools"
        if not any(inv.get(x) for x in STONE_PICK):
            return "stone"
        if inv.get("raw_iron", 0) + inv.get("iron_ore", 0) + inv.get("deepslate_iron_ore", 0) < 3:
            return "iron"
        if not inv.get("iron_ingot") and not any(r["_name"] == "furnace" for r in all_rows(obs, "b")) and not inv.get("furnace"):
            return "furnace"
        if not inv.get("iron_ingot"):
            return "smelt"
        if not inv.get("shield"):
            return "shield"
        return "after_iron"

    def craft(self, name: str, count: int, table: dict | None = None) -> str:
        result = "craft %s %d" % (name, max(1, int(count)))
        if table:
            result += " " + table["_id"]
        return result

    def table_route(self, obs: dict, now: float) -> str | None:
        table = self.near_table(obs)
        if table:
            return None
        # A crafted table is still an inventory item until this explicit
        # transaction places it.  This branch must run before workstation
        # travel; otherwise the wooden-tool phase can wait forever with a
        # carried table and sticks.
        inv = inv_of(obs)
        if inv.get("crafting_table"):
            support = self.support(obs, now)
            if support:
                if item(self_row(obs).get("held_item")) != "crafting_table":
                    return "equip crafting_table hand"
                return "place crafting_table %s up" % support["_id"]
            return self.explore(obs, now)
        tables = [r for r in all_rows(obs, "b") if r["_name"] == "crafting_table"]
        if tables:
            tables.sort(key=lambda r: r["_distance"])
            q = tables[0]["_pos"]
            stand = self.safe_stand(obs, now, q)
            if stand:
                self.visited[stand["_id"]] = now
                return "goto %.2f %.2f %.2f" % stand["_pos"]
            return "goto %.2f %.2f %.2f" % (q[0] + 1, q[1], q[2] + 1) if q else None
        if self.table_pos and now >= self.table_wait:
            q = self.table_pos
            return "goto %.2f %.2f %.2f" % (q[0] + 1, q[1], q[2] + 1)
        return "noop"

    def compile(self, obs: dict, now: float) -> str | None:
        inv = inv_of(obs)
        phase = self.phase(obs)
        table = self.near_table(obs)
        # Once the basic pick exists, harvest a real visible food block when
        # one is available.  This is opportunistic: an absent food candidate
        # never diverts the mine or invents an animal/structure.
        if phase not in {"wood", "table"} and food_count(inv) < 3:
            food_row = self.choose_resource(obs, now, FOOD_BLOCKS)
            if food_row:
                self.begin_gather(food_row, "visible food reserve", now, 1)
                return None
        if phase == "wood":
            if logs(inv) < 3:
                target = self.choose_resource(obs, now, suffixes=WOOD_ENDINGS)
                if target:
                    self.begin_gather(target, "wood for table, tools, and shield", now, 3)
                    return None
                return self.explore(obs, now)
            log = next((k for k, v in inv.items() if v and k.endswith(WOOD_ENDINGS)), "")
            if log:
                return self.craft(log.replace("_log", "_planks").replace("_stem", "_planks"), min(inv[log] * 4, 16))
        if phase == "table":
            if planks(inv) < 4:
                log = next((k for k, v in inv.items() if v and k.endswith(WOOD_ENDINGS)), "")
                if log:
                    return self.craft(log.replace("_log", "_planks").replace("_stem", "_planks"), min(inv[log] * 4, 16))
                target = self.choose_resource(obs, now, suffixes=WOOD_ENDINGS)
                if target:
                    self.begin_gather(target, "wood for workstation", now, 2)
                    return None
                return self.explore(obs, now)
            if not inv.get("crafting_table"):
                return self.craft("crafting_table", 1)
            support = self.support(obs, now)
            if support:
                if item(self_row(obs).get("held_item")) != "crafting_table":
                    return "equip crafting_table hand"
                return "place crafting_table %s up" % support["_id"]
            return self.explore(obs, now)
        if phase == "wood_tools":
            if planks(inv) < 3:
                log = next((k for k, v in inv.items() if v and k.endswith(WOOD_ENDINGS)), "")
                if log:
                    return self.craft(log.replace("_log", "_planks").replace("_stem", "_planks"), min(inv[log] * 4, 8))
                target = self.choose_resource(obs, now, suffixes=WOOD_ENDINGS)
                if target:
                    self.begin_gather(target, "extra wood", now, 1)
                    return None
                return self.explore(obs, now)
            if inv.get("stick", 0) < 2:
                return self.craft("stick", 4)
            if not table:
                return self.table_route(obs, now) or self.explore(obs, now)
            return self.craft("wooden_pickaxe", 1, table)
        if phase == "stone":
            cobble = inv.get("cobblestone", 0) + inv.get("cobbled_deepslate", 0)
            if cobble < 3:
                row = self.choose_resource(obs, now, STONE)
                if row:
                    self.begin_gather(row, "cobblestone for stone pick", now, min(3, 3 - cobble))
                    return None
                row = self.choose_surface_step(obs, now)
                if row:
                    self.begin_step(row, now)
                    return None
                row = self.choose_tunnel_block(obs, now)
                if row:
                    self.begin_tunnel(row, now)
                    return None
                return self.explore(obs, now)
            if inv.get("stick", 0) < 2:
                return self.craft("stick", 4)
            if not table:
                return self.table_route(obs, now) or self.explore(obs, now)
            return self.craft("stone_pickaxe", 1, table)
        if phase == "iron":
            raw = inv.get("raw_iron", 0) + inv.get("iron_ore", 0) + inv.get("deepslate_iron_ore", 0)
            if raw < 3:
                row = self.choose_resource(obs, now, IRON)
                if row:
                    self.begin_gather(row, "iron for shield", now, min(3, 3 - raw))
                    return None
                row = self.choose_resource(obs, now, COAL)
                if row and not inv.get("coal"):
                    self.begin_gather(row, "fuel while mining", now, 1)
                    return None
                row = self.choose_surface_step(obs, now)
                if row:
                    self.begin_step(row, now)
                    return None
                row = self.choose_tunnel_block(obs, now)
                if row:
                    self.begin_tunnel(row, now)
                    return None
                # When a loaded branch runs out of exposed wall, walk to a new
                # safe stand instead of digging a guessed coordinate.
                return self.explore(obs, now)
        if phase == "furnace":
            if not inv.get("furnace") and not any(r["_name"] == "furnace" for r in all_rows(obs, "b")):
                table = self.near_table(obs)
                if inv.get("cobblestone", 0) + inv.get("cobbled_deepslate", 0) >= 8 and table:
                    return self.craft("furnace", 1, table)
                row = self.choose_resource(obs, now, STONE)
                if row:
                    self.begin_gather(row, "furnace stone", now, 1)
                    return None
                row = self.choose_surface_step(obs, now)
                if row:
                    self.begin_step(row, now)
                    return None
                row = self.choose_tunnel_block(obs, now)
                if row:
                    self.begin_tunnel(row, now)
                    return None
                return self.explore(obs, now)
            if inv.get("furnace"):
                support = self.support(obs, now)
                if support:
                    if item(self_row(obs).get("held_item")) != "furnace":
                        return "equip furnace hand"
                    return "place furnace %s up" % support["_id"]
                return self.explore(obs, now)
        if phase == "smelt":
            furnaces = [r for r in all_rows(obs, "b") if r["_name"] == "furnace"]
            furnaces.sort(key=lambda r: r["_distance"])
            if not furnaces:
                return self.table_route(obs, now) or self.explore(obs, now)
            furnace = furnaces[0]
            if furnace["_distance"] > 4.2:
                stand = self.safe_stand(obs, now, furnace["_pos"])
                if stand:
                    return "goto %.2f %.2f %.2f" % stand["_pos"]
                return "goto %.2f %.2f %.2f" % furnace["_pos"]
            fid = furnace["_id"]
            raw = inv.get("raw_iron", 0)
            fuel = next((x for x in ("coal", "charcoal") if inv.get(x)), "")
            if not fuel:
                fuel = next((x for x, v in inv.items() if x.endswith("_planks") and v), "")
            if raw and fuel and fid not in self.smelt_started:
                self.smelt_started[fid] = now
                return "furnace_start %s raw_iron %d %s %d" % (fid, min(3, raw), fuel, min(3, inv[fuel]))
            if fid in self.smelt_started and now - self.smelt_started[fid] >= 30.0:
                return "furnace_take " + fid
            return "wait 0.5"
        if phase == "shield":
            if planks(inv) < 5:
                log = next((k for k, v in inv.items() if v and k.endswith(WOOD_ENDINGS)), "")
                if log:
                    return self.craft(log.replace("_log", "_planks").replace("_stem", "_planks"), min(inv[log] * 4, 8))
                target = self.choose_resource(obs, now, suffixes=WOOD_ENDINGS)
                if target:
                    self.begin_gather(target, "shield planks", now, 1)
                    return None
                return self.explore(obs, now)
            if not table:
                return self.table_route(obs, now) or self.explore(obs, now)
            return self.craft("shield", 1, table)
        # Keep food topped up and continue the observed branch after the first
        # iron milestone.  This is also a harmless fallback if the objective is
        # replaced with a resource/advancement target.
        if n(self_row(obs).get("food"), 20) <= 12 and best_food(inv):
            food = best_food(inv)
            return ("equip %s hand" % food) if item(self_row(obs).get("held_item")) != food else "eat " + food
        row = self.choose_resource(obs, now, IRON)
        if row:
            self.begin_gather(row, "additional iron", now, 1)
            return None
        return self.explore(obs, now)

    def run_episode(self, game: Any) -> None:
        spec = getattr(game, "action_spec", None)
        self.names = set(getattr(spec, "names", ()) or ())
        self.last_progress_at = self.now()
        while not self.ctx.done:
            now = self.now()
            raw = self.ctx.observe()
            obs = raw if isinstance(raw, dict) else {}
            self._obs = obs
            if not obs:
                self.post("noop", obs, now, "no_observation")
                self.ctx.wait_for_tick()
                continue
            self.read_feedback(obs, now)
            self.mark_progress(obs, now)
            self.update_tx(obs, now)
            me = self_row(obs)
            if me.get("dead"):
                self.tx = None
                self.job = None
                if not self.dead_latched:
                    self.dead_latched = True
                    self.post("respawn", obs, now, "ordinary_respawn")
                else:
                    self.post("noop", obs, now, "await_respawn")
                self.ctx.wait_for_tick()
                continue
            self.dead_latched = False
            urgent = self.safe_override(obs, now)
            if urgent:
                if has_task(obs) and urgent not in {"noop", "controls forward,jump", "controls jump"}:
                    self.post("cancel", obs, now, "safety_interrupt")
                else:
                    self.post(urgent, obs, now, "hard_safety")
                self.ctx.wait_for_tick()
                continue
            tactical = self.tactical(obs, now)
            if tactical:
                self.post(tactical, obs, now, "threat_policy")
                self.ctx.wait_for_tick()
                continue
            if has_task(obs):
                self.post("noop", obs, now, "continue_job")
                self.ctx.wait_for_tick()
                continue
            if self.tx:
                self.post("noop", obs, now, "await_postcondition")
                self.ctx.wait_for_tick()
                continue
            if self.job:
                action = self.job_action(obs, now)
                if action:
                    self.post(action, obs, now, "mine_transaction")
                else:
                    self.post("noop", obs, now, "advance_transaction")
                self.ctx.wait_for_tick()
                continue
            action = self.compile(obs, now)
            if action:
                self.post(action, obs, now, "milestone_policy")
            else:
                self.post("noop", obs, now, "transaction_created")
            self.ctx.wait_for_tick()


def build(ctx: Any) -> Controller:
    return Controller(ctx)
