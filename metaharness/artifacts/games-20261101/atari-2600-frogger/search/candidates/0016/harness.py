"""0016: evidence-conditioned hazard overlays, not model-owned joystick routes.

Only public observations are read. The named bonus labels are hypotheses: home
alert phases and board-located bonus sprites can be obstacles. Code tracks their
geometry; Jev interprets compact evidence and selects a response protocol; Astra
reviews which semantic conditions should trigger that protocol.
"""
from __future__ import annotations

import json
import math
import time
from collections import deque
from concurrent.futures import CancelledError

from metaharness.core.contract import BudgetExceeded, EpisodeOver, ModelUnavailable
from metaharness.models.jev import Choice, Noul

COPIES = ((0,), (0, 16), (0, 32), (0, 16, 32), (0, 64), (0,), (0, 32, 64), (0,))
WIDTHS = (8, 8, 8, 8, 8, 16, 8, 32)
MOVES = (("UP", 1, 0), ("LEFT", 0, -8), ("RIGHT", 0, 8), ("NOOP", 0, 0), ("DOWN", -1, 0))


def number(v, default=0.0):
    try:
        v = float(v)
        return v if math.isfinite(v) else default
    except (ValueError, TypeError, OverflowError):
        return default


def integer(v, default=0):
    return int(number(v, default))


def unpack(v):
    if isinstance(v, str):
        try:
            v = json.loads(v)
        except (ValueError, TypeError):
            return {}
    return v if isinstance(v, dict) else {}


def fragment(text):
    if not isinstance(text, str):
        return {}
    try:
        return unpack(text[text.index("{"):text.rindex("}") + 1])
    except ValueError:
        return {}


def row(o):
    r = integer(o.get("frog", {}).get("row"), 255)
    return -1 if r == 255 else r


def epoch(o):
    return tuple(integer(o.get(k), d) for k, d in (("deaths", 0), ("frogs_home", 0), ("level", 1)))


def delta(a, b):
    return (number(a) - number(b) + 80) % 160 - 80


def ram(o, name, address, default=0):
    fields = o.get("ram_fields", {})
    if name in fields:
        return integer(fields[name], default)
    raw = o.get("raw_ram", [])
    i = address - 0x80
    return integer(raw[i], default) if isinstance(raw, list) and 0 <= i < len(raw) else default


def alerts(o):
    flags = ram(o, "collision_e3", 0xE3)
    target = ram(o, "collision_row_e5", 0xE5)
    phase = "raised" if flags & 128 else "warning" if flags & 64 else "fly" if flags & 32 else "absent"
    slot = str(target - 1) if 1 <= target <= 5 else None
    return phase, slot


def intruder(o):
    flags = ram(o, "bonus_flags_c7", 0xC7)
    r = ram(o, "fly_row_ae", 0xAE, 255)
    if flags & 128 and 0 <= r <= 10:
        return {"row": r, "x": float(ram(o, "fly_x_b2", 0xB2)), "flags": flags}
    return None


def merge(intervals):
    out = []
    for a, b in sorted(intervals):
        if out and a <= out[-1][1] + .001:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return out


def agreement(ans, first, second, threshold=.72):
    if not isinstance(ans, dict):
        return None, 0.0
    a, b = ans.get(first), ans.get(second)
    value = getattr(a, "value", None)
    if not isinstance(value, str) or value != getattr(b, "value", None):
        return None, 0.0
    pa, pb = getattr(a, "probabilities", None), getattr(b, "probabilities", None)
    if not isinstance(pa, dict) or not isinstance(pb, dict):
        return None, 0.0
    p = min(number(pa.get(value)), number(pb.get(value)))
    return (value if p >= threshold else None), p


class Field:
    """Time-layered carrier support with a separate moving-obstacle overlay."""
    def __init__(self, h, o, depth, relevant=None, overlay_intruder=True):
        self.h, self.o = h, o
        self.layers, self.drift = [], {}
        self.strict_bank = epoch(o)[2] >= 4 or h.bank_alarm
        self.homes = {str(z.get("slot")): z for z in h.open_homes(o)}
        self.phase, self.alert_bay = alerts(o)
        self.block_warning = h.policy["home_phase"] == "full_cycle"
        if h.semantic and h.semantic.get("signature") == h.signature(o):
            self.block_warning = self.block_warning or h.semantic.get("home") == "CAUTION"
        self.threat = intruder(o) if overlay_intruder else None
        self.overlay_intruder = overlay_intruder
        items_by_row = {}
        for lane in o.get("road", []) + o.get("river", []):
            r = integer(lane.get("row"))
            if relevant is not None and r not in relevant:
                continue
            items = []
            for z in lane.get("objects", []):
                n = integer(z.get("nusiz")) & 7
                shape = integer(z.get("shape_ptr"))
                sign = -1 if z.get("direction", lane.get("direction")) == "left" else 1
                v = h.velocity.get(str(z.get("id")), sign / max(1, integer(lane.get("activity_mask"), 3) + 1))
                width = 24 if n == 7 and shape in (163, 170) else WIDTHS[n]
                safe = not z.get("hazard") and z.get("kind") in ("log", "turtle")
                items.append((number(z.get("x")), v, COPIES[n], width, safe))
            items_by_row[r] = items
            self.drift[r] = sum(z[1] for z in items) / max(1, len(items)) if r >= 6 else 0.0
        for d in range(depth + 1):
            dt, layer = d * h.step, {}
            for r, items in items_by_row.items():
                if r < 5:
                    values = [20.0] * 161
                    for x, v, copies, width, safe in items:
                        for off in copies:
                            a, b = x + off + v * (dt + 1), x + off + v * (dt + h.step + 3)
                            self.paint(values, min(a, b) - 10, max(a, b) + width + 2)
                else:
                    intervals = []
                    for x, v, copies, width, safe in items:
                        if safe:
                            for off in copies:
                                a = x + off + v * (dt + 1)
                                for wrap in (-320, -160, 0, 160, 320):
                                    intervals.append((a + wrap, a + width + wrap))
                    values = [-1.0] * 161
                    for a, b in merge(intervals):
                        lo, hi = a - 2, b - 6
                        for px in range(max(0, math.ceil(lo)), min(160, math.floor(hi)) + 1):
                            values[px] = max(values[px], min(px - lo, hi - px))
                layer[r] = values
            # Bonus semantics are not merged into support. A snake on a turtle
            # remains lethal even where the turtle offers a generous margin.
            if self.threat:
                r, x = self.threat["row"], self.threat["x"]
                if relevant is None or r in relevant:
                    values = layer.setdefault(r, [20.0] * 161 if r == 5 else [-1.0] * 161)
                    v = 0.0 if self.threat["flags"] & 64 else h.intruder_v
                    a, b = x + v * dt, x + v * (dt + h.step + 3)
                    uncertainty = min(10.0, 2.0 + dt * .025) + (3.0 if h.policy["intruder_response"] == "detour" else 0.0)
                    self.paint(values, min(a, b) - 9 - uncertainty, max(a, b) + 19 + uncertainty)
            # In observed late levels AE=128 is the off-board phase. The
            # same bonus-channel obstacle reappears at the left of row8.
            # Reserve only its birth corridor, not the entire row or carrier.
            if overlay_intruder and self.threat is None and epoch(o)[2] >= 5 and ram(o, "fly_row_ae", 0xAE, 255) == 128:
                if relevant is None or 8 in relevant:
                    values = layer.get(8)
                    if values is not None:
                        self.paint(values, -10, 24)
            self.layers.append(layer)

    @staticmethod
    def paint(values, lo, hi):
        for wrap in (-320, -160, 0, 160, 320):
            start, end = max(0, math.floor(lo + wrap)), min(160, math.ceil(hi + wrap))
            if end >= start:
                values[start:end + 1] = [-1.0] * (end - start + 1)

    def margin(self, r, x, d=0):
        if not 4 <= x <= 148 or not -1 <= r <= 10:
            return -100.0
        layer = self.layers[min(d, len(self.layers) - 1)]
        if r == -1:
            return 20.0
        if r == 5 and r not in layer:
            return 20.0
        values = layer.get(r)
        return values[int(round(x))] if values else -100.0

    def blocked(self, slot):
        return slot == self.alert_bay and (self.phase == "raised" or (self.phase == "warning" and self.block_warning))

    def entry(self, x, slot):
        home = self.homes.get(str(slot))
        if home is None or self.blocked(str(slot)):
            return -100.0
        center = number(home.get("x_center"), 80)
        return min(x - center + 4, center - x)

    def advance(self, r, x, move, d, slot=None, bank_steps=0):
        act, dr, dx = move
        nr, nx = r + dr, x + dx
        if nr < -1 or nr > 11 or not 4 <= nx <= 148:
            return None
        if nr == 11:
            if act != "UP" or slot is None:
                return None
            m = self.entry(nx, slot)
            # Home contact is not an ordinary six-frame river dwell. The
            # live guard verifies observed carry/settling; do not require a
            # fictional future six-frame support interval after leaving it.
            if m < 0:
                return None
            return nr, nx, m, 0
        count = bank_steps + 1 if nr == 5 else 0
        if self.strict_bank and nr == 5 and r >= 5:
            return None
        if not self.strict_bank and count > 3:
            return None
        source = self.margin(r, x, d)
        edge_escape = (6 <= r <= 10 and act in ("LEFT", "RIGHT")
                       and bool(self.o.get("frog", {}).get("moving_with_platform"))
                       and ((x > 148 and act == "LEFT") or (x < 4 and act == "RIGHT")))
        if source < 0 and not edge_escape:
            return None
        m = self.margin(nr, nx, d)
        if m < 0:
            return None
        endx = nx + self.drift.get(nr, 0.0) * self.h.step
        if nr >= 6:
            endm = self.margin(nr, endx, d + 1)
            if endm < 0:
                return None
            m = min(m, endm)
        return nr, endx, m, count


class Harness:
    def __init__(self, ctx):
        self.ctx = ctx
        self.step = 6
        self.prev = None
        self.current_epoch = None
        self.history, self.velocity = {}, {}
        self.intruder_history = deque(maxlen=80)
        self.intruder_v = 0.0
        self.intruder_move_tick = None
        self.rearm_at = 0
        self.target = None
        self.target_owner = "code_upstream"
        self.policy = {"home_phase": "full_cycle", "response": "bypass", "intruder_response": "escape"}
        self.version = 0
        self.semantic = None
        self.bank_alarm = False
        self.ready = 3
        self.press = None
        self.press_tick = -100
        self.origin = None
        self.top_entered = None
        self.last_lateral = -100
        self.last_action = "NOOP"
        self.last_plan = None
        self.last_log = -100
        self.last_progress = 0
        self.routes = []
        self.search_ms = 0.0
        self.ring = deque(maxlen=12)
        self.failures = deque(maxlen=6)
        self.revision = 0
        self.seen_anomalies = set()
        self.s1 = self.s2 = None
        self.s1_meta = self.s2_meta = None
        self.next_s1 = 0
        self.next_s2 = 240
        self.session = None
        self.latencies = deque(maxlen=30)
        self.hazard_signature = None
        self.hazard_since = 0

    def log(self, event, **fields):
        self.ctx.log(event, **fields)

    def open_homes(self, o):
        return [z for z in o.get("homes", []) if not z.get("occupied") and z.get("available", True)
                and z.get("crocodile") not in (True, "present", "dangerous")]

    def signature(self, o):
        p, b = alerts(o)
        q = intruder(o)
        return p, b, q["row"] if q else None

    def home_blocked(self, o, slot=None):
        p, bay = alerts(o)
        target = self.target if slot is None else str(slot)
        return bay == target and bay is not None and (p == "raised" or
                (p == "warning" and self.policy["home_phase"] == "full_cycle"))

    def snapshot(self, o):
        r = row(o)
        return {"tick": o.get("tick"), "epoch": epoch(o), "frog": o.get("frog"),
                "ram_fields": o.get("ram_fields"), "intruder": intruder(o), "alert": alerts(o),
                "action": self.last_action, "target": self.target,
                "objects": [l for l in o.get("river", []) if integer(l.get("row")) in (r, r + 1)]}

    def update(self, o):
        t, ep, rr = integer(o.get("tick")), epoch(o), row(o)
        old = self.prev
        if old and ep[0] > epoch(old)[0]:
            p, bay = alerts(old)
            q = intruder(old)
            oldx = number(old.get("frog", {}).get("x"))
            witness = {"tick": t, "previous": self.snapshot(old), "current": self.snapshot(o),
                       "plan": self.last_plan, "frames": list(self.ring)[-5:],
                       "matching_home_alert": row(old) == 10 and p == "raised" and bay == self.target,
                       "near_board_intruder": bool(q and row(old) == q["row"] and -10 <= delta(oldx, q["x"]) <= 20),
                       "bank": row(old) == 5}
            self.failures.append(witness)
            self.revision += 1
            self.bank_alarm = self.bank_alarm or witness["bank"]
            self.log("semantic_failure", **witness)
            self.next_s2 = min(self.next_s2, t + 60)
        if self.current_epoch != ep:
            if self.current_epoch is not None:
                self.log("crossing_outcome", tick=t, before=self.current_epoch, after=ep,
                         target=self.target, target_owner=self.target_owner, last_plan=self.last_plan)
            if self.current_epoch and self.current_epoch[2] != ep[2]:
                self.history.clear()
                self.velocity.clear()
                self.intruder_history.clear()
                self.intruder_v = 0.0
            self.current_epoch = ep
            self.target = None
            self.semantic = None
            self.press, self.origin = None, None
            self.ready = t + 3
            self.rearm_at = t + 2
            self.top_entered = None
            self.next_s1 = t
        if old and rr != row(old):
            self.last_progress = t
            self.top_entered = t if rr == 10 else None
        for lane in o.get("road", []) + o.get("river", []):
            for z in lane.get("objects", []):
                ident, x = str(z.get("id")), number(z.get("x"))
                history = self.history.setdefault(ident, deque(maxlen=60))
                if not history or history[-1][0] != t:
                    history.append((t, x))
                older = next((a for a in history if 24 <= t - a[0] <= 56), None)
                sign = -1 if z.get("direction", lane.get("direction")) == "left" else 1
                v = sign / max(1, integer(lane.get("activity_mask"), 3) + 1)
                if older:
                    measured = delta(x, older[1]) / max(1, t - older[0])
                    if measured * sign > 0 and abs(measured) <= 2:
                        v = sign * min((1/16, 1/8, 1/4, 1/2, 1, 2), key=lambda k: abs(k - abs(measured)))
                self.velocity[ident] = v
        q = intruder(o)
        previous_q = intruder(old) if old else None
        if q:
            changed = not previous_q or previous_q["row"] != q["row"] or (previous_q["flags"] & 64) != (q["flags"] & 64)
            if changed:
                self.intruder_history.clear()
                self.intruder_move_tick = None
                self.intruder_v = 0.0 if q["flags"] & 64 else .5
            if q["flags"] & 64:
                # Observed C7=192 is the stationary x108 interval. Do not
                # extrapolate the preceding motion through this phase change.
                self.intruder_v = 0.0
                self.intruder_move_tick = None
            elif previous_q and not changed:
                moved = delta(q["x"], previous_q["x"])
                if abs(moved) > .1 and abs(moved) <= 8:
                    gap = t - self.intruder_move_tick if self.intruder_move_tick is not None else 0
                    if 1 <= gap <= 8:
                        self.intruder_v = max(-2.0, min(2.0, moved / gap))
                    self.intruder_move_tick = t
            self.intruder_history.append((t, q["x"]))
        else:
            self.intruder_history.clear()
            self.intruder_move_tick = None
            self.intruder_v = 0.0
        signature = self.signature(o)
        if signature != self.hazard_signature:
            self.hazard_signature = signature
            self.hazard_since = t
            self.semantic = None
            self.next_s1 = min(self.next_s1, t)
            self.log("hazard_overlay_changed", tick=t, signature=signature, intruder=q)
        if signature[0] in ("raised", "warning"):
            self.seen_anomalies.add("home_alert")
        if q:
            self.seen_anomalies.add("board_intruder")
        self.choose_target(o)
        self.prev = o
        self.ring.append({"tick": t, "frog": o.get("frog"), "alert": alerts(o), "intruder": q,
                          "action": self.last_action})

    def choose_target(self, o):
        homes = self.open_homes(o)
        if not homes:
            self.target = None
            return
        p, bay = alerts(o)
        blocked = bay if bay is not None and self.home_blocked(o, bay) else None
        available = [z for z in homes if str(z.get("slot")) != blocked]
        top = next((l for l in o.get("river", []) if integer(l.get("row")) == 10), {})
        sign = -1 if top.get("direction") == "left" else 1
        slots = {str(z.get("slot")) for z in homes}
        need = self.target not in slots
        semantic_response = (self.semantic or {}).get("response")
        bypass = semantic_response == "BYPASS" or (not semantic_response and self.policy["response"] == "bypass" and integer(o.get("tick")) - self.hazard_since >= 90)
        if self.target == blocked and available and (row(o) < 5 or (bypass and row(o) <= 10)):
            need = True
        if need:
            candidates = available or homes
            xx = number(o.get("frog", {}).get("x"), 80)
            if row(o) >= 8:
                home = min(candidates, key=lambda z: abs(number(z.get("x_center")) - xx))
            else:
                home = min(candidates, key=lambda z: sign * number(z.get("x_center")))
            before = self.target
            self.target = str(home.get("slot"))
            self.target_owner = "jev_bypass" if semantic_response == "BYPASS" else "code_hazard_route"
            if before is not None and before != self.target:
                self.log("hazard_route_changed", tick=o.get("tick"), before=before, after=self.target, owner=self.target_owner)

    def target_x(self, o):
        home = next((z for z in self.open_homes(o) if str(z.get("slot")) == self.target), None)
        return number(home.get("x_center"), 80) - 2 if home else 78.0

    def search(self, o):
        began = time.perf_counter()
        rr, xx = row(o), number(o.get("frog", {}).get("x"), 80)
        if not -1 <= rr <= 10:
            return []
        endpoint, depth = (7, 22) if rr < 5 else (11, 22)
        relevant = set(range(8)) if endpoint == 7 else set(range(4, 11))
        field = Field(self, o, depth, relevant)
        target = self.target_x(o)
        beam = [(rr, xx, (), 20.0, 0.0, 0)]
        completed, partial = {}, {}
        for d in range(depth):
            expanded = []
            for r, x, path, bottleneck, cost, bank_count in beam:
                for move in MOVES:
                    a, dr, dx = move
                    if r < 5 and r + dr > endpoint:
                        continue
                    if rr >= 6 and r + dr < 5:
                        continue
                    z = field.advance(r, x, move, d, self.target, bank_count)
                    if z is None:
                        continue
                    nr, nx, m, banks = z
                    nm = min(bottleneck, m)
                    nc = cost + 1 + (.15 if dx else 0) + (1 if a == "DOWN" else 0)
                    if nr == 5:
                        nc += 4 if field.strict_bank else 1
                    seq = path + (a,)
                    dist = abs(nx - target)
                    if nr == endpoint:
                        value = 400 - nc * 2 + min(nm, 5) * .6 - (dist * .30 if endpoint == 7 else 0)
                        candidate = {"action": seq[0], "path": seq, "value": value, "margin": round(nm, 2),
                                     "complete": True, "endpoint": endpoint, "endx": round(nx, 2), "steps": len(seq)}
                        if seq[0] not in completed or value > completed[seq[0]]["value"]:
                            completed[seq[0]] = candidate
                        continue
                    attraction = .075 if nr < 4 else .18 if nr < 8 else .25
                    value = nr * 20 - nc * .6 - dist * attraction + min(nm, 5) * .3
                    if nr > rr:
                        candidate = {"action": seq[0], "path": seq, "value": value - 400, "margin": round(nm, 2),
                                     "complete": False, "endpoint": nr, "endx": round(nx, 2), "steps": len(seq)}
                        if seq[0] not in partial or candidate["value"] > partial[seq[0]]["value"]:
                            partial[seq[0]] = candidate
                    expanded.append((value, (nr, nx, seq, nm, nc, banks)))
            if not expanded:
                break
            expanded.sort(key=lambda z: z[0], reverse=True)
            beam, seen = [], set()
            for _, state in expanded:
                key = state[0], round(state[1] / 2), state[5]
                if key in seen:
                    continue
                seen.add(key)
                beam.append(state)
                if len(beam) >= 36:
                    break
            if completed and d > min(z["steps"] for z in completed.values()) + 3:
                break
        routes = list(completed.values()) or list(partial.values())
        if not routes:
            for move in MOVES:
                z = field.advance(rr, xx, move, 0, self.target)
                if z:
                    nr, nx, m, _ = z
                    value = nr * 10 + min(m, 4) - abs(nx - target) * .03 - (move[0] != "NOOP")
                    routes.append({"action": move[0], "path": (move[0],), "value": value, "margin": m,
                                   "complete": False, "endpoint": nr, "endx": nx, "steps": 1})
        routes.sort(key=lambda z: z["value"], reverse=True)
        self.search_ms = (time.perf_counter() - began) * 1000
        return routes

    def guard(self, o, action, field):
        rr, xx = row(o), number(o.get("frog", {}).get("x"), 80)
        move = next((z for z in MOVES if z[0] == action), MOVES[3])
        if field.advance(rr, xx, move, 0, self.target) is None:
            return False
        if rr == 10 and action == "UP":
            if not o.get("frog", {}).get("moving_with_platform"):
                return False
            t = integer(o.get("tick"))
            if t - self.last_lateral < 6 or (self.top_entered is not None and t - self.top_entered < 6):
                return False
            if self.home_blocked(o):
                return False
        return True

    def pick(self, o, routes):
        field = Field(self, o, 1)
        valid = [z for z in routes if self.guard(o, z["action"], field)]
        q = intruder(o)
        near = bool(q and q["row"] == row(o) and -22 <= delta(o.get("frog", {}).get("x"), q["x"]) <= 30)
        if near and (self.semantic or {}).get("response") == "ESCAPE":
            # A fresh semantic escape selects a safe row exit even if the
            # long-horizon target score would prefer a lateral detour.
            for action in ("UP", "DOWN"):
                if self.guard(o, action, field):
                    selected = next((z for z in valid if z["action"] == action),
                                    {"action": action, "fallback": True})
                    if not valid or valid[0]["action"] != action:
                        self.log("jev_changes_escape", tick=o.get("tick"), before=valid[0]["action"] if valid else "NOOP", after=action)
                    return action, dict(selected, semantic_escape=True)
        if valid:
            return valid[0]["action"], valid[0]
        # No model result is needed for survival. Never substitute an untested
        # wait for an unsafe first action at the bank or beside a board intruder.
        q = intruder(o)
        urgent = row(o) == 5 or bool(q and q["row"] == row(o) and -22 <= delta(o.get("frog", {}).get("x"), q["x"]) <= 30)
        order = ("UP", "DOWN", "LEFT", "RIGHT", "NOOP") if urgent else ("NOOP", "UP", "LEFT", "RIGHT", "DOWN")
        action = next((a for a in order if self.guard(o, a, field)), "NOOP")
        return action, {"action": action, "fallback": True, "alert": alerts(o), "intruder": q}

    def model_state(self, o):
        p, bay = alerts(o)
        q = intruder(o)
        xx = number(o.get("frog", {}).get("x"))
        homes = self.open_homes(o)
        return {
            "objective": "Preserve frogs and fill unoccupied home bays in Atari Frogger.",
            "role": "Classify uncertain object meaning and select a durable hazard response, never an action or coordinate.",
            "rules": "A fly bonus belongs in a home bay, not on a traversable river row. Later levels have snakes. Safe carrier ink does not make an overlapping snake safe. Unknown crocodile is not absent crocodile.",
            "evidence": "Archive observations: 13/13 immediate home deaths matched raised home alert and bay; 12/12 row-8 deaths had an active board bonus sprite at collision distance. These are conditional hazards, not unsafe entire rows.",
            "policy": dict(self.policy),
            "facts": {"home_phase": p, "alert_matches_target": bay == self.target and bay is not None,
                      "other_open_home": any(str(z.get("slot")) != self.target for z in homes),
                      "board_sprite_active": q is not None, "sprite_on_frog_row": bool(q and q["row"] == row(o)),
                      "sprite_collision_near": bool(q and q["row"] == row(o) and -22 <= delta(xx, q["x"]) <= 30),
                      "at_dock": row(o) == 10, "river_committed": row(o) >= 6,
                      "timer_urgent": number(o.get("timer_s"), 30) < 8,
                      "new_counterexample": bool(self.failures), "observed_bank_contradiction": self.bank_alarm},
            "bridge": "Raised matching home alerts are always excluded in code. Active board sprites always receive spatial obstacles. BYPASS may change target, STAGE preserves target and waits on certified support, ESCAPE prioritizes leaving a threatened row. Responses must be rechecked on live facts."}

    def request_jev(self, o):
        t = integer(o.get("tick"))
        if self.s1 is not None or t < self.next_s1 or getattr(self.ctx, "system1", True) is None:
            return
        if self.ctx.budget.remaining().get("system1_calls", 0) < 3:
            return
        state = self.model_state(o)
        home = {"HOSTILE": "Raised phase is active at the current target; prohibit entry.",
                "CAUTION": "Warning phase matches the target and policy.home_phase=full_cycle; stage through this warning.",
                "CLEAR": "No raised or warning phase matches the current target. This is not a claim about all hazards.",
                "UNKNOWN": "The facts do not establish which condition applies."}
        obj = {"ROW_HAZARD": "An active board-located sprite is a snake-like obstacle, not a collectible fly.",
               "ABSENT": "No active board sprite is present.", "UNKNOWN": "Insufficient evidence about this sprite."}
        response = {"ESCAPE": "A board sprite is collision-near on the frog's row; leave its spatial obstacle.",
                    "BYPASS": "A home alert matches target and another home is open; route to another open bay.",
                    "STAGE": "A home alert matches target but there is no alternative, or policy requests staging; maintain supported waiting.",
                    "PROCEED": "Neither a matching home alert nor a near board sprite requires a protocol change.",
                    "UNKNOWN": "Retain code's verified fallback."}
        hi = "Using facts, classify only the current target's home alert. Matching raised means HOSTILE; matching warning means CAUTION only under policy.home_phase=full_cycle; otherwise CLEAR. Do not infer danger from unrelated bays."
        oi = "Using facts.board_sprite_active and the supplied observational evidence, classify the board sprite. Active means ROW_HAZARD, absent means ABSENT. Do not trust the decoder's fly name over the location evidence."
        ri = "Choose a hazard response from facts and policy. Near board threat takes ESCAPE priority. Matching raised alert (or warning with policy.home_phase=full_cycle) takes BYPASS if another home is open and policy.response=bypass, otherwise STAGE. With neither threat PROCEED; UNKNOWN if unsupported."
        questions = {"home": Choice(hi, home), "home_check": Choice(hi, dict(reversed(list(home.items())))),
                     "object": Choice(oi, obj), "object_check": Choice(oi, dict(reversed(list(obj.items())))),
                     "response": Choice(ri, response), "response_check": Choice(ri, dict(reversed(list(response.items())))),
                     "review": Noul("Do facts explicitly report a new counterexample or observed bank contradiction requiring the slow policy reviewer?")}
        self.s1_meta = {"tick": t, "epoch": epoch(o), "version": self.version, "signature": self.signature(o), "target": self.target}
        def call():
            try:
                began = self.ctx.clock.now()
                answer = self.ctx.decide(state, questions, timeout_s=3.0)
                return answer, self.ctx.clock.now() - began
            except (BudgetExceeded, EpisodeOver, ModelUnavailable):
                raise
            except (TimeoutError, RuntimeError, TypeError, ValueError):
                return None, 0.0
        self.s1 = self.ctx.spawn(call)
        self.next_s1 = t + (36 if intruder(o) or alerts(o)[0] in ("warning", "raised") else 150)

    def request_astra(self, o):
        t = integer(o.get("tick"))
        if self.s2 is not None or t < self.next_s2 or not self.seen_anomalies or getattr(self.ctx, "system2", True) is None:
            return
        if self.ctx.budget.remaining().get("system2_calls", 0) < 2:
            return
        summary = {"policy": self.policy, "current": self.model_state(o), "failures": list(self.failures)[-2:],
                   "anomalies": sorted(self.seen_anomalies), "level": o.get("level"), "score": o.get("score")}
        prompt = (
            "Review Frogger's semantic hazard protocol, not actions. The adaptor calls E3/E5 collision flags and C7/AE/B2 a fly; "
            "they are not a guarantee of harmlessness. Archived public observations establish 13/13 immediate home failures at "
            "E3 bit7 with E5=bay+1, and 12/12 river-row8 failures within -8..16 pixels of active C7 bit7 / AE=row8 / B2 sprite. "
            "We therefore exclude a matching raised home and overlay board sprites as moving obstacles without changing carrier widths. "
            "Code owns exact geometry and arithmetic. Jev receives named booleans for matching alert, warning/raised phase, "
            "board-sprite presence, near collision, alternatives and urgency; it answers reversed Choices for semantic home/object "
            "roles and BYPASS/STAGE/ESCAPE/PROCEED. Require both winning probabilities >=.72 and matching live signature, epoch, "
            "version, age<=75 ticks. There is no generated code and no action from prose. Your enums compile into its next state "
            "and its code-owned target/overlay protocol. Choose full_cycle for precaution during warning, raised_only only if "
            "evidence suggests warning waits cause stalls; bypass an alerted bay when alternatives exist, stage when preserving "
            "the upstream itinerary is better. Never globally enlarge carriers or trust bonus labels. Return ONLY JSON with "
            '"home_phase":"full_cycle|raised_only", "response":"bypass|stage", "intruder_response":"escape|detour". '
            "Keep a safe default if evidence is inconclusive. Evidence=" + json.dumps(summary, separators=(",", ":")))
        self.s2_meta = {"tick": t, "revision": self.revision}
        session = self.session
        def call():
            try:
                return self.ctx.think(prompt, session_id=session, timeout_s=120.0)
            except (BudgetExceeded, EpisodeOver, ModelUnavailable):
                raise
            except (TimeoutError, RuntimeError, TypeError):
                return None
        self.s2 = self.ctx.spawn(call)
        self.next_s2 = t + 4200

    def poll(self, o):
        t = integer(o.get("tick"))
        if self.s1 is not None and self.s1.done():
            future, self.s1 = self.s1, None
            try:
                answers, latency = future.result()
            except (BudgetExceeded, EpisodeOver, ModelUnavailable):
                raise
            except (TimeoutError, RuntimeError, TypeError, ValueError, CancelledError):
                answers, latency = None, 0.0
            if isinstance(answers, dict):
                self.latencies.append(latency)
                meta = self.s1_meta or {}
                fresh = epoch(o) == meta.get("epoch") and self.version == meta.get("version") and self.signature(o) == meta.get("signature") and self.target == meta.get("target") and t - meta.get("tick", t) <= 75
                home, hp = agreement(answers, "home", "home_check")
                obj, op = agreement(answers, "object", "object_check")
                response, rp = agreement(answers, "response", "response_check")
                if fresh:
                    facts = self.model_state(o)["facts"]
                    if response == "ESCAPE" and not facts["sprite_collision_near"]:
                        response = None
                    if response in ("BYPASS", "STAGE") and not (facts["alert_matches_target"] and facts["home_phase"] in ("warning", "raised")):
                        response = None
                    self.semantic = {"home": home, "object": obj, "response": response,
                                     "signature": self.signature(o), "until": t + 90}
                    before = self.target
                    self.choose_target(o)
                    self.log("jev_semantic_overlay", tick=t, home=home, object=obj, response=response,
                             probabilities=[hp, op, rp], latency_s=latency, target_changed=before != self.target)
                else:
                    self.log("jev_semantic_discarded", tick=t, latency_s=latency)
        if self.semantic and t > self.semantic.get("until", -1):
            self.semantic = None
        if self.s2 is not None and self.s2.done():
            future, self.s2 = self.s2, None
            try:
                result = future.result()
            except (BudgetExceeded, EpisodeOver, ModelUnavailable):
                raise
            except (TimeoutError, RuntimeError, TypeError, CancelledError):
                result = None
            if result is not None:
                self.session = getattr(result, "session_id", None)
                proposed = fragment(getattr(result, "text", ""))
                allowed = {"home_phase": ("full_cycle", "raised_only"), "response": ("bypass", "stage"),
                           "intruder_response": ("escape", "detour")}
                changes = {k: v for k, v in proposed.items() if k in allowed and isinstance(v, str) and v in allowed[k]}
                if changes and integer((self.s2_meta or {}).get("revision")) == self.revision:
                    self.policy.update(changes)
                    self.version += 1
                    self.semantic = None
                    self.next_s1 = t
                    self.log("astra_semantics_compiled", tick=t, policy=dict(self.policy), version=self.version)
                else:
                    self.log("astra_semantics_rejected", tick=t)

    def run_episode(self, game):
        del game
        while not self.ctx.done:
            o = unpack(self.ctx.observe())
            if not o:
                self.ctx.act("NOOP")
                self.ctx.wait_for_tick()
                continue
            self.update(o)
            self.poll(o)
            t, rr = integer(o.get("tick")), row(o)
            active = integer(o.get("frog", {}).get("state")) == 128 and -1 <= rr <= 10
            action = "NOOP"
            busy = self.press is not None
            if busy:
                moved = self.origin is not None and (rr != self.origin[0] or abs(delta(o.get("frog", {}).get("x"), self.origin[1])) >= 7)
                if moved or not active or t - self.press_tick >= 4:
                    self.log("pulse_release", tick=t, action=self.press, moved=moved)
                    self.press, self.origin = None, None
                    self.ready = max(self.ready, t + 2)
                    self.rearm_at = t + 2
                else:
                    # A newly raised home alert can invalidate a still-pending
                    # press. Cancel it rather than waiting for the next search.
                    action = "NOOP" if rr == 10 and self.press == "UP" and self.home_blocked(o) else self.press
                    q = intruder(o)
                    dr = 1 if self.press == "UP" else -1 if self.press == "DOWN" else 0
                    if q and q["row"] in (rr, rr + dr) and not self.guard(o, self.press, Field(self, o, 1)):
                        action = "NOOP"
                        self.press, self.origin = None, None
                        self.rearm_at = t + 2
                        self.ready = min(self.ready, t + 2)
                        self.log("pending_hop_cancelled", tick=t, intruder=q)
            emergency = False
            q = intruder(o)
            if not busy and active and q and q["row"] == rr and -22 <= delta(o.get("frog", {}).get("x"), q["x"]) <= 30 and t >= self.rearm_at:
                # Newly appearing/stopping hazards may invalidate a six-frame
                # dwell. A neutral-rearmed urgent exit must not wait for its
                # original route timer or for a fresh model answer.
                field = Field(self, o, 1)
                escape = next((a for a in ("UP", "DOWN", "LEFT", "RIGHT") if self.guard(o, a, field)), None)
                if escape is None:
                    # If the overlay occupies the only one-hop landing, use
                    # an unambiguous inward lateral escape while retaining
                    # the carrier/road checks. This is preferable to waiting
                    # at x<4/x>148 for a moving obstacle to hit.
                    plain = Field(self, o, 1, overlay_intruder=False)
                    x = number(o.get("frog", {}).get("x"), 80)
                    preferred = "LEFT" if x > 148 else "RIGHT" if x < 4 else ("RIGHT" if delta(q["x"], x) < 0 else "LEFT")
                    escape = next((a for a in (preferred, "LEFT" if preferred == "RIGHT" else "RIGHT")
                                   if self.guard(o, a, plain)), None)
                if escape:
                    action = escape
                    self.press, self.press_tick = action, t
                    self.origin = rr, number(o.get("frog", {}).get("x"))
                    self.ready = t + self.step
                    if action in ("LEFT", "RIGHT"):
                        self.last_lateral = t
                    self.last_plan = {"tick": t, "origin": list(self.origin), "action": action, "target": self.target,
                                      "intruder": q, "emergency": True}
                    self.log("overlay_emergency_escape", tick=t, plan=self.last_plan)
                    emergency = True
            if not busy and not emergency and active and t >= self.ready:
                self.routes = self.search(o)
                latest = unpack(self.ctx.observe())
                if latest and epoch(latest) == epoch(o) and row(latest) == rr:
                    action, selected = self.pick(latest, self.routes)
                    t = integer(latest.get("tick"), t)
                    self.last_plan = {"tick": t, "origin": [rr, latest.get("frog", {}).get("x")],
                                      "target": self.target, "alert": alerts(latest), "intruder": intruder(latest), **selected}
                    if action != "NOOP" or t - self.last_log >= 60:
                        self.log("overlay_action", tick=t, plan=self.last_plan, search_ms=round(self.search_ms, 2))
                        self.last_log = t
                    self.ready = t + (self.step if action != "NOOP" else 3)
                    if action != "NOOP":
                        self.press, self.press_tick = action, t
                        self.origin = rr, number(latest.get("frog", {}).get("x"))
                        if action in ("LEFT", "RIGHT"):
                            self.last_lateral = t
                self.request_jev(o)
            self.request_astra(o)
            self.last_action = action
            self.ctx.act(action)
            self.ctx.wait_for_tick()


def build(ctx):
    return Harness(ctx)
