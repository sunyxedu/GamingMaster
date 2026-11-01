"""0024: latched carrier-edge recovery over the route portfolio.

Based on 0023's public-observation search and hazard layers. This revision adds
rider-motion sampling and a checked, hysteretic recovery latch to prevent a
later route choice from undoing a carrier-edge correction.
"""
from __future__ import annotations

import json
import math
import time
from collections import deque

from metaharness.core.contract import BudgetExceeded, EpisodeOver, ModelUnavailable
from metaharness.models.jev import Choice, Noul

COPIES = ((0,), (0, 16), (0, 32), (0, 16, 32), (0, 64), (0,), (0, 32, 64), (0,))
WIDTHS = (8, 8, 8, 8, 8, 16, 8, 32)
MOVES = (("UP", 1, 0), ("LEFT", 0, -8), ("RIGHT", 0, 8), ("NOOP", 0, 0), ("DOWN", -1, 0))


def number(v, default=0.0):
    try:
        f = float(v)
        return f if math.isfinite(f) else default
    except (ValueError, TypeError, OverflowError):
        return default


def integer(v, default=0):
    return int(number(v, default))


def unpack(v):
    if isinstance(v, str):
        try:
            v = json.loads(v)
        except (TypeError, ValueError):
            return {}
    return v if isinstance(v, dict) else {}


def fragment(v):
    if not isinstance(v, str):
        return {}
    try:
        return unpack(v[v.index("{"):v.rindex("}") + 1])
    except ValueError:
        return {}


def row(o):
    r = integer(o.get("frog", {}).get("row"), 255)
    return -1 if r == 255 else r


def epoch(o):
    return tuple(integer(o.get(k), d) for k, d in (("deaths", 0), ("frogs_home", 0), ("level", 1)))


def merge(intervals):
    out = []
    for a, b in sorted(intervals, key=lambda p: (p[0], p[1])):
        if out and a <= out[-1][1] + .001:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return out


def agreed(answers, a, b, threshold):
    aa, bb = answers.get(a), answers.get(b)
    v = getattr(aa, "value", None)
    if not isinstance(v, str) or v != getattr(bb, "value", None):
        return None, 0.0
    pa, pb = getattr(aa, "probabilities", None), getattr(bb, "probabilities", None)
    if not isinstance(pa, dict) or not isinstance(pb, dict):
        return None, 0.0
    p = min(number(pa.get(v)), number(pb.get(v)))
    return (v if p >= threshold else None), p


def t_elapsed(o, since):
    return integer(o.get("tick"))-since


def home_alert(o, slot):
    """Conservative, observation-established home danger signature.

    E3 low bits are occupancy. Across the parent traces every final-UP failure
    had its high bit set AND E5 equal to the entered slot plus one. Do not use
    an unrelated home's alert as a global ban, nor assert other states safe.
    """
    rf = o.get("ram_fields", {})
    return bool(integer(rf.get("collision_e3")) & 128) and integer(rf.get("collision_row_e5")) == integer(slot, -2)+1


def auxiliary(o):
    """A candidate on a playable row is an actor, not a home-fly reward."""
    rf = o.get("ram_fields", {})
    z = o.get("fly", {})
    zr = z.get("row")
    rr = integer(zr if zr is not None else rf.get("fly_row_ae"), 255)
    active = bool(z.get("present")) or bool(integer(rf.get("bonus_flags_c7")) & 128)
    if active and 0 <= rr <= 10:
        zx = z.get("x")
        return rr, number(zx if zx is not None else rf.get("fly_x_b2"))
    return None


class Field:
    """Time layers in frog-left coordinates, including the entire release dwell."""
    def __init__(self, h, o, depth, relevant=None):
        self.h, self.o = h, o
        self.strict_bank = integer(o.get("level"), 1) >= 4 or h.bank_alarm
        self.rows, self.drift, self.objects = [], {}, {}
        self.homes = [z for z in o.get("homes", []) if not z.get("occupied") and z.get("available", True)
                      and z.get("crocodile") not in (True, "present", "dangerous")]
        self.home_centers = {str(z.get("slot")): number(z.get("x_center"), 80) for z in self.homes}
        self.alerted = {slot for slot in self.home_centers if home_alert(o, slot)}
        actor = auxiliary(o)
        for lane in o.get("road", []) + o.get("river", []):
            rr = integer(lane.get("row"))
            if relevant is not None and rr not in relevant:
                continue
            items = []
            for z in lane.get("objects", []):
                n = integer(z.get("nusiz")) & 7
                shape = integer(z.get("shape_ptr"))
                sign = -1 if z.get("direction", lane.get("direction")) == "left" else 1
                v = h.vel.get(str(z.get("id")), sign / max(1, integer(lane.get("activity_mask"), 3) + 1))
                w = 24 if n == 7 and shape in (163, 170) else WIDTHS[n]
                safe = not z.get("hazard") and z.get("kind") in ("log", "turtle")
                items.append((number(z.get("x")), v, COPIES[n], w, safe, shape, n, str(z.get("id"))))
            self.objects[rr] = items
            self.drift[rr] = sum(z[1] for z in items) / max(1, len(items)) if rr >= 6 else 0.0
        # The median is omitted from decoded road/river lists. Public RAM
        # positions 0..10 and 11..21 match those lists; the second median
        # track at index16 explains the archived bank-contact witnesses.
        ram = o.get("raw_ram", [])
        if self.strict_bank and len(ram) >= 22 and (relevant is None or 5 in relevant):
            self.objects[5] = [(number(ram[16]), h.bank_velocity, (0,), 16, False, 0, 0, "bank_auxiliary")]
        for d in range(depth + 1):
            now = d * h.step
            layer = {}
            for rr, items in self.objects.items():
                if rr <= 5:
                    values = [20.0] * 161
                    for x, v, copies, width, safe, shape, n, ident in items:
                        for off in copies:
                            a, b = x + off + v*(now+1), x + off + v*(now+h.step+2)
                            lo, hi = min(a, b)-9.5, max(a, b)+width+1.5
                            for wrap in (-320, -160, 0, 160, 320):
                                start, end = max(0, math.floor(lo+wrap)), min(160, math.ceil(hi+wrap))
                                if end >= start:
                                    values[start:end+1] = [-1.0]*(end-start+1)
                else:
                    ints = []
                    for x, v, copies, width, safe, shape, n, ident in items:
                        if safe:
                            for off in copies:
                                a = x + off + v*(now+1)
                                for wrap in (-320, -160, 0, 160, 320):
                                    ints.append((a+wrap, a+width+wrap))
                    values = [-1.0] * 161
                    for a, b in merge(ints):
                        lo, hi = a-2.0, b-6.0
                        for px in range(max(0, math.ceil(lo)), min(160, math.floor(hi))+1):
                            values[px] = max(values[px], min(px-lo, hi-px))
                if actor is not None and actor[0] == rr:
                    # It advances and can pause: carrier support alone is not
                    # safety. Its observed disappearance is NOT a wraparound.
                    av = h.aux_velocity
                    a = actor[1] + av*(now+1)
                    b = actor[1] + av*(now+h.step+3)
                    lo = max(0, math.floor(min(a, b)-11))
                    hi = min(160, math.ceil(max(a, b)+19))
                    if hi >= lo:
                        values[lo:hi+1] = [-1.0]*(hi-lo+1)
                elif actor is None and h.aux_spawn is not None and h.aux_spawn[0] == rr:
                    # An absent actor can reappear between a plan and its hop.
                    # Reserve only its observed spawn mouth, not the whole row.
                    sx = h.aux_spawn[1]
                    lo, hi = max(0, math.floor(sx-11)), min(160, math.ceil(sx+23))
                    if hi >= lo:
                        values[lo:hi+1] = [-1.0]*(hi-lo+1)
                layer[rr] = values
            self.rows.append(layer)

    def margin(self, rr, xx, layer=0):
        if not 4 <= xx <= 148:
            return -100.0
        if rr == -1 or (rr == 5 and 5 not in self.rows[layer]):
            return 20.0
        if rr < 0 or rr > 10:
            return -100.0
        a = self.rows[layer].get(rr)
        return a[int(round(xx))] if a else -100.0

    def entry(self, x, slot):
        center = self.home_centers.get(str(slot))
        if center is None or str(slot) in self.alerted:
            return -100.0
        # Dock at the interior; unlike a new river landing, verify the source too.
        low, high = self.h.dock_window
        return min(x-center-low, center+high-x)

    def advance(self, rr, xx, move, d, slot=None, bank_steps=0):
        act, dr, dx = move
        nr, nx = rr+dr, xx+dx
        if nr < -1 or nr > 11 or not 4 <= nx <= 148:
            return None
        if nr == 11:
            if act != "UP" or slot is None:
                return None
            m = self.entry(nx, slot)
            if m < 0:
                return None
            return nr, nx, m, 0
        # The bank is a transit link, NOT an endpoint / unlimited sanctuary.
        count = bank_steps+1 if nr == 5 else 0
        if self.strict_bank:
            if nr == 5 and (rr == 5 or rr >= 6):
                return None
        elif count > 3:
            return None
        m = self.margin(nr, nx, d)
        if m < 0:
            return None
        endx = nx + self.drift.get(nr, 0.0)*self.h.step
        if nr >= 6:
            endm = self.margin(nr, endx, d+1)
            if endm < 0:
                return None
            m = min(m, endm)
        return nr, endx, m, count


class Harness:
    def __init__(self, ctx):
        self.ctx = ctx
        self.step = 6
        self.prev = None
        self.hist, self.vel = {}, {}
        self.current_epoch = None
        self.target = None
        self.target_owner = "code_upstream"
        self.protocol = "TRANSIT"
        self.policy = {"itinerary": "upstream", "launch": "transit", "dock": "settle"}
        self.version = 0
        self.dock_window = (-4, 0)
        self.bank_alarm = False
        self.bank_entered = None
        self.top_entered = None
        self.last_lateral = -100
        self.last_progress = 0
        self.ready = 3
        self.press = None
        self.press_tick = -100
        self.origin = None
        self.last_action = "NOOP"
        self.last_plan = None
        self.recent = deque(maxlen=8)
        self.ring = deque(maxlen=16)
        self.entry_failures = {}
        self.failed_carriers = {}
        self.evidence_revision = 0
        self.diagnosis = None
        self.s1 = None
        self.s1_meta = None
        self.next_s1 = 0
        self.s2 = None
        self.s2_meta = None
        self.next_s2 = 1200
        self.session = None
        self.last_search_ms = 0.0
        self.last_log = -100
        self.routes = []
        self.aux_velocity = .5
        self.aux_running_speed = .5
        self.aux_spawn = None
        self.aux_was_present = False
        self.aux_history = deque(maxlen=48)
        self.aux_epoch = None
        self.route_goal = None
        self.route_goal_until = 0
        self.bank_velocity = .25
        self.bank_history = deque(maxlen=60)
        self.last_hazard_signature = None
        # Direct carried-frog motion is learned only across neutral frames on
        # one river row. A recovery latch prevents the route selector from
        # immediately undoing an edge correction.
        self.rider_history = {}
        self.edge_latch = None

    def log(self, event, **fields):
        self.ctx.log(event, **fields)

    def open_homes(self, o):
        return [h for h in o.get("homes", []) if not h.get("occupied") and h.get("available", True)
                and h.get("crocodile") not in (True, "present", "dangerous")]

    def carrier(self, o):
        rr, xx = row(o), number(o.get("frog", {}).get("x"))
        lane = next((l for l in o.get("river", []) if integer(l.get("row")) == rr), {})
        found = []
        for z in lane.get("objects", []):
            n = integer(z.get("nusiz")) & 7
            for off in COPIES[n]:
                relative = (xx-number(z.get("x"))-off+80) % 160-80
                if -4 <= relative <= WIDTHS[n]:
                    found.append({"id": str(z.get("id")), "shape": integer(z.get("shape_ptr")),
                                  "offset": round(relative, 1), "kind": z.get("kind"), "hazard": bool(z.get("hazard"))})
        return found

    def snapshot(self, o):
        return {"tick": o.get("tick"), "frog": o.get("frog"), "timer": o.get("timer_s"),
                "epoch": epoch(o), "homes": o.get("homes"), "contact": self.carrier(o),
                "ram_fields": o.get("ram_fields"), "last_action": self.last_action,
                "auxiliary_track": auxiliary(o), "auxiliary_velocity": self.aux_velocity,
                "known_spawn": self.aux_spawn,
                "bank_track": (o.get("raw_ram") or [None]*17)[16] if len(o.get("raw_ram", [])) > 16 else None}

    def update(self, o):
        t, ep, rr = integer(o.get("tick")), epoch(o), row(o)
        if self.prev:
            old = self.prev
            oldrow = row(old)
            oldfrog, newfrog = old.get("frog", {}), o.get("frog", {})
            dt = max(1, t-integer(old.get("tick"), t-1))
            if (rr == oldrow and 6 <= rr <= 10 and dt <= 8
                    and oldfrog.get("moving_with_platform") and newfrog.get("moving_with_platform")
                    and self.last_action == "NOOP"):
                dx = (number(newfrog.get("x"))-number(oldfrog.get("x"))+80) % 160-80
                v = dx/dt
                if abs(v) <= 2.0:
                    q = self.rider_history.setdefault(rr, deque(maxlen=9))
                    q.append(v)
            if rr != oldrow or ep != epoch(old):
                if ep != epoch(old):
                    self.rider_history.clear()
                if self.edge_latch is not None:
                    self.log("carrier_edge_latch_cleared", tick=t, reason="row_or_epoch_change",
                             previous=self.edge_latch, row=rr, epoch=ep)
                self.edge_latch = None
        if self.prev and ep[0] > epoch(self.prev)[0]:
            oldrow = row(self.prev)
            plan = self.last_plan or {}
            age = t-integer(plan.get("tick"), t-99)
            entry = oldrow == 10 and self.last_action == "UP" and age < 6
            bank = oldrow == 5
            slot = str(self.target) if self.target is not None else "unknown"
            contacts = self.carrier(self.prev)
            evidence = {"tick": t, "entry_failure": entry, "bank_failure": bank,
                        "carrier_failure": 6 <= oldrow <= 10 and not entry,
                        "repeat_entry": entry and self.entry_failures.get((ep[2], slot), 0) > 0,
                        "previous": self.snapshot(self.prev), "current": self.snapshot(o),
                        "plan": plan, "last_frames": list(self.ring)[-6:],
                        "home_alert_matches": home_alert(self.prev, slot),
                        "auxiliary": auxiliary(self.prev)}
            if entry:
                key = (ep[2], slot)
                self.entry_failures[key] = self.entry_failures.get(key, 0)+1
                for c in contacts:
                    self.failed_carriers[(ep[2], slot, c["id"])] = t
            if bank:
                self.bank_alarm = True
            self.recent.append(evidence)
            self.evidence_revision += 1
            self.log("gate_failure", **evidence)
        if self.current_epoch != ep:
            if self.current_epoch is not None:
                self.log("crossing_outcome", tick=t, before=self.current_epoch, after=ep, target=self.target,
                         target_owner=self.target_owner, last_plan=self.last_plan)
            self.current_epoch = ep
            self.target = None
            self.route_goal = None
            self.press, self.origin = None, None
            self.ready = t+3
            self.top_entered, self.bank_entered = None, None
            self.next_s1 = t
        if self.prev and rr != row(self.prev):
            self.last_progress = t
            self.bank_entered = t if rr == 5 else None
            self.top_entered = t if rr == 10 else None
        if self.target is None:
            homes = self.open_homes(o)
            top = next((l for l in o.get("river", []) if integer(l.get("row")) == 10), {})
            sign = -1 if top.get("direction") == "left" else 1
            if homes:
                upstream = min(homes, key=lambda h: sign*number(h.get("x_center")))
                self.target = str(upstream.get("slot"))
                self.target_owner = "code_upstream"
        ram = o.get("raw_ram", [])
        if len(ram) >= 22:
            bx = number(ram[16])
            q = self.bank_history
            if not q or q[-1][0] != t:
                q.append((t, bx))
            old = next((v for v in q if 12 <= t-v[0] <= 40), None)
            if old:
                v = ((bx-old[1]+80) % 160-80) / max(1, t-old[0])
                if 0 < v <= 2:
                    self.bank_velocity = min((.125, .25, .5, 1.), key=lambda q: abs(q-v))
        actor = auxiliary(o)
        signature = (next((str(h.get("slot")) for h in self.open_homes(o) if home_alert(o, h.get("slot"))), None),
                     actor[0] if actor is not None else None, ep[2] >= 4 or self.bank_alarm)
        if signature != self.last_hazard_signature:
            self.log("hazard_layer_changed", tick=t, alerted_bay=signature[0], lane_actor=actor,
                     bank_actor_modeled=signature[2])
            self.last_hazard_signature = signature
        if actor is not None and not self.aux_was_present:
            if actor[1] <= 4:
                self.aux_spawn = (actor[0], actor[1])
            self.log("auxiliary_appearance", tick=t, actor=actor, spawn_mouth=self.aux_spawn)
        self.aux_was_present = actor is not None
        if actor is None:
            self.aux_history.clear()
            self.aux_epoch = None
        else:
            ar, ax = actor
            tag = (ep[2], ar)
            if tag != self.aux_epoch:
                self.aux_history.clear()
                self.aux_epoch = tag
                self.aux_velocity = self.aux_running_speed
            if self.aux_history and abs(ax-self.aux_history[-1][1]) > 20:
                self.aux_history.clear()
            if not self.aux_history or t != self.aux_history[-1][0]:
                self.aux_history.append((t, ax))
            old = next((v for v in self.aux_history if 8 <= t-v[0] <= 32), None)
            if old:
                v = (ax-old[1]) / max(1, t-old[0])
                if .05 <= abs(v) <= 2:
                    self.aux_velocity = min((-1., -.5, -.25, .25, .5, 1.), key=lambda q: abs(q-v))
                    self.aux_running_speed = self.aux_velocity
                elif v == 0:
                    self.aux_velocity = 0.0
        for lane in o.get("road", []) + o.get("river", []):
            for z in lane.get("objects", []):
                ident, xx = str(z.get("id")), number(z.get("x"))
                q = self.hist.setdefault(ident, deque(maxlen=60))
                if not q or t != q[-1][0]:
                    q.append((t, xx))
                old = next((v for v in q if 24 <= t-v[0] <= 56), None)
                sign = -1 if z.get("direction", lane.get("direction")) == "left" else 1
                v = sign/max(1, integer(lane.get("activity_mask"), 3)+1)
                if old:
                    observed = ((xx-old[1]+80) % 160-80)/max(1, t-old[0])
                    if observed*sign > 0 and abs(observed) <= 2:
                        v = sign*min((1/16, 1/8, 1/4, 1/2, 1, 2), key=lambda k: abs(k-abs(observed)))
                self.vel[ident] = v
        self.prev = o
        self.ring.append(self.snapshot(o))

    def target_x(self, o):
        home = next((h for h in self.open_homes(o) if str(h.get("slot")) == str(self.target)), None)
        return number(home.get("x_center"), 80) if home else 80.0

    def search(self, o):
        start = time.perf_counter()
        rr, xx = row(o), number(o.get("frog", {}).get("x"), 80)
        if not -1 <= rr <= 10:
            return []
        # Cross-region launch planning: road routes terminate on row 7, not bank 5.
        endpoint = 7 if rr < 5 else 11
        depth = 22 if endpoint == 7 else 22
        relevant = set(range(8)) if endpoint == 7 else set(range(4, 11))
        field = Field(self, o, depth, relevant)
        bank_cost = 4.0 if field.strict_bank else 1.0
        target = self.target_x(o)-2
        centers = [(str(z.get("slot")), number(z.get("x_center"))) for z in field.homes]
        portfolio = endpoint == 11 and (str(self.target) in field.alerted
                                        or t_elapsed(o, self.last_progress) > 240
                                        or auxiliary(o) is not None
                                        or (self.aux_spawn is not None and rr <= 8)
                                        or number(o.get("timer_s"), 30) < 8)
        available_centers = [(sid, cx) for sid, cx in centers if sid not in field.alerted]
        beam = [(rr, xx, (), 20.0, 0.0, 0)]
        completed, partial = {}, {}
        for d in range(depth):
            expanded = []
            for r, x, path, bottleneck, cost, bank_count in beam:
                for move in MOVES:
                    a, dr, dx = move
                    if r < 5 and r+dr > endpoint:
                        continue
                    if rr >= 6 and r+dr < 5:
                        continue
                    slot = self.target
                    if r == 10 and a == "UP" and portfolio:
                        choices = [(sid, field.entry(x, sid)) for sid, cx in centers]
                        feasible = [(sid, m) for sid, m in choices if m >= 0]
                        if feasible:
                            slot = max(feasible, key=lambda z: z[1])[0]
                    z = field.advance(r, x, move, d, slot, bank_count)
                    if z is None:
                        continue
                    nr, nx, margin, banks = z
                    nm = min(bottleneck, margin)
                    nc = cost + 1 + (0.15 if a in ("LEFT", "RIGHT") else 0) + (1.0 if a == "DOWN" else 0)
                    if nr == 5:
                        nc += bank_cost
                    seq = path+(a,)
                    dist = abs(nx-target)
                    if nr == endpoint:
                        # A road route must arrive on a useful river corridor, not
                        # simply jump onto the nearest carrier at the wrong edge.
                        value = 400-nc*2+min(nm, 5)*.6 - (dist*.30 if endpoint == 7 else 0)
                        if endpoint == 11:
                            value += 5 if slot == self.target else 0
                            value += 4 if slot == self.route_goal else 0
                        key = (slot, seq[0]) if endpoint == 11 else seq[0]
                        item = {"action": seq[0], "path": seq, "value": value, "margin": round(nm, 2),
                                "goal": slot, "complete": True, "endpoint": endpoint, "endx": round(nx, 2), "steps": len(seq)}
                        if key not in completed or value > completed[key]["value"]:
                            completed[key] = item
                        continue
                    # Alignment matters BEFORE the bank. It is expensive or lethal there.
                    if portfolio and available_centers:
                        dist = min(abs(nx-cx+2) + (0 if sid == self.target else 12) for sid, cx in available_centers)
                    attraction = .075 if nr < 4 else .18 if nr < 8 else .25
                    value = nr*20 - nc*.6 - dist*attraction + min(nm, 5)*.3
                    if nr > rr:
                        key = seq[0]
                        item = {"action": key, "path": seq, "value": value-400, "margin": round(nm, 2),
                                "complete": False, "endpoint": nr, "endx": round(nx, 2), "steps": len(seq)}
                        if key not in partial or item["value"] > partial[key]["value"]:
                            partial[key] = item
                    expanded.append((value, (nr, nx, seq, nm, nc, banks)))
            if not expanded:
                break
            expanded.sort(key=lambda z: z[0], reverse=True)
            beam, seen = [], set()
            for _, state in expanded:
                key = (state[0], round(state[1]/2), state[5])
                if key in seen:
                    continue
                seen.add(key)
                beam.append(state)
                if len(beam) >= 36:
                    break
            if completed and d > min(z["steps"] for z in completed.values())+3:
                break
        routes = list(completed.values()) or list(partial.values())
        if not routes:
            for move in MOVES:
                z = field.advance(rr, xx, move, 0, self.target)
                if z:
                    nr, nx, m, _ = z
                    value = nr*10+min(m, 4)-abs(nx-target)*.03-(0 if move[0] == "NOOP" else 1)
                    routes.append({"action": move[0], "path": (move[0],), "value": value,
                                   "margin": m, "complete": False, "endpoint": nr, "endx": nx, "steps": 1})
        routes.sort(key=lambda z: z["value"], reverse=True)
        self.last_search_ms = (time.perf_counter()-start)*1000
        return routes

    def guard(self, o, action, field=None):
        f = field if field is not None else Field(self, o, 1)
        rr, xx = row(o), number(o.get("frog", {}).get("x"), 80)
        move = next((m for m in MOVES if m[0] == action), MOVES[3])
        z = f.advance(rr, xx, move, 0, self.target)
        if z is None:
            return False
        if rr == 10 and action == "UP":
            t = integer(o.get("tick"))
            # A dock is a distinct protocol. Confirm support and let lateral input
            # settle before the irreversible press; do not reinterpret it as a log.
            if t-self.last_lateral < 6:
                return False
            if self.top_entered is not None and t-self.top_entered < 6:
                return False
            if self.diagnosis == "CHANGE_APPROACH":
                contacts = self.carrier(o)
                banned = any((epoch(o)[2], str(self.target), c["id"]) in self.failed_carriers for c in contacts)
                if banned and t-self.last_progress < 180:
                    return False
            if self.policy["dock"] == "different_carrier" and self.entry_failures.get((epoch(o)[2], str(self.target)), 0):
                if t-self.last_progress < 180:
                    return False
        return True

    def model_state(self, o):
        homes = self.open_homes(o)
        top = next((l for l in o.get("river", []) if integer(l.get("row")) == 10), {})
        sign = -1 if top.get("direction") == "left" else 1
        ordered = sorted(homes, key=lambda z: sign*number(z.get("x_center")))
        frontier = str(ordered[0].get("slot")) if ordered else None
        xx = number(o.get("frog", {}).get("x"), 80)
        nearest = str(min(homes, key=lambda z: abs(number(z.get("x_center"))-xx)).get("slot")) if homes else None
        facts = [{"bay": str(h.get("slot")), "upstream_frontier": str(h.get("slot")) == frontier,
                  "nearest_now": str(h.get("slot")) == nearest, "current_commitment": str(h.get("slot")) == self.target,
                  "failed_entry_this_level": self.entry_failures.get((epoch(o)[2], str(h.get("slot"))), 0) > 0,
                  "matched_home_alert": home_alert(o, h.get("slot")),
                  "complete_route_now": any(r.get("goal") == str(h.get("slot")) and r.get("complete") for r in self.routes)}
                 for h in ordered]
        ev = self.recent[-1] if self.recent else {}
        return {"game": "Frogger: fill different open bays while preserving lives.",
                "phase": "road_launch" if row(o) < 5 else "bank_transit" if row(o) == 5 else "dock" if row(o) == 10 else "river",
                "policy": dict(self.policy), "homes": facts,
                "facts": {"higher_level_bank_threat": epoch(o)[2] >= 4 or self.bank_alarm,
                          "complete_launch_exists": any(r["complete"] for r in self.routes),
                          "auxiliary_on_playable_row": auxiliary(o) is not None,
                          "auxiliary_on_frog_row": auxiliary(o) is not None and auxiliary(o)[0] == row(o),
                          "auxiliary_spawn_reserved": self.aux_spawn is not None and auxiliary(o) is None,
                          "target_has_matched_alert": home_alert(o, self.target),
                          "urgent_timer": number(o.get("timer_s"), 30) < 8,
                          "entry_failed": bool(ev.get("entry_failure")), "bank_failed": bool(ev.get("bank_failure")),
                          "repeat_entry": bool(ev.get("repeat_entry")), "carrier_failed": bool(ev.get("carrier_failure"))},
                "rules": "The top carrier flow makes upstream bays harder to revisit. Select an itinerary early, not a joystick. The median is a transit link, not a safe long wait. A failed final UP does NOT establish that all carriers are water. Unknown crocodile status is not an absent crocodile. Code rechecks geometry and docking every frame."}

    def request_jev(self, o):
        t = integer(o.get("tick"))
        if self.s1 is not None or t < self.next_s1 or getattr(self.ctx, "system1", True) is None:
            return
        if self.ctx.budget.remaining().get("system1_calls", 0) < 2:
            return
        state = self.model_state(o)
        homes = state["homes"]
        if not homes:
            return
        itinerary = {h["bay"]: {"what": "Commit this open bay for the whole crossing.", "facts": h} for h in homes}
        itinerary["UNKNOWN"] = "Keep code's current commitment; evidence is inadequate."
        instruction = ("Choose one bay from homes. This is a WHOLE-CROSSING itinerary made before the river, not a nearest-frame action. "
                       "Under policy.itinerary=upstream prefer upstream_frontier; under nearest prefer nearest_now. "
                       "Reject a matched_home_alert bay. Once in river retain current_commitment unless it has a matched_home_alert or failed_entry_this_level; "
                       "then prefer an unalerted complete_route_now alternative. "
                       "Do not claim an unknown crocodile is absent. UNKNOWN if undecided.")
        gates = {"TRANSIT": "Plan through the bank, align before it, and prefer a complete launch into the river.",
                 "STAGE": "Favor an available wider launch rather than partial forward progress; do not wait on the bank.",
                 "RECOVER": "A failed entry needs a different supported approach, not another identical final UP.",
                 "SCREEN": "Treat an on-lane auxiliary candidate as an independent moving hazard; prefer a complete low-exposure crossing.",
                 "UNKNOWN": "No justified protocol change."}
        gate_instruction = ("Select a crossing protocol using facts, not coordinates. Prefer SCREEN when auxiliary_on_playable_row is true; "
                            "SCREEN also when auxiliary_spawn_reserved; RECOVER when repeat_entry is true or target_has_matched_alert; "
                            "TRANSIT when higher_level_bank_threat or bank_failed; STAGE when carrier_failed and not urgent_timer; otherwise TRANSIT. "
                            "Each describes code's future control protocol, not a delayed action.")
        questions = {"itinerary": Choice(instruction, itinerary),
                     "itinerary_check": Choice(instruction, dict(reversed(list(itinerary.items())))),
                     "gate": Choice(gate_instruction, gates),
                     "gate_check": Choice(gate_instruction, dict(reversed(list(gates.items())))),
                     "review": Noul("Do facts explicitly show repeated entry failure or a bank failure requiring Astra to revise the crossing protocol?")}
        self.s1_meta = {"tick": t, "epoch": epoch(o), "version": self.version, "revision": self.evidence_revision,
                        "road": row(o) < 5, "state": state}
        def call():
            try:
                before = self.ctx.clock.now()
                ans = self.ctx.decide(state, questions, timeout_s=3.0)
                return ans, self.ctx.clock.now()-before
            except (BudgetExceeded, EpisodeOver, ModelUnavailable):
                raise
            except (TimeoutError, RuntimeError, TypeError, ValueError):
                return None, 0.0
        self.s1 = self.ctx.spawn(call)
        self.next_s1 = t+90

    def request_astra(self, o):
        t = integer(o.get("tick"))
        if self.s2 is not None or t < self.next_s2 or not self.recent or getattr(self.ctx, "system2", True) is None:
            return
        if self.ctx.budget.remaining().get("system2_calls", 0) < 2:
            return
        summary = {"policy": self.policy, "failures": list(self.recent)[-2:], "score": o.get("score"),
                   "level": o.get("level"), "search_ms": round(self.last_search_ms, 2), "protocol": self.protocol}
        prompt = ("You review Frogger crossing PROTOCOLS asynchronously, never frame actions. Code uses separated sprite copies, "
                  "body-aware collision intervals and neutral-rearmed six-frame hops. Unlike the prior controller, road search "
                  "ends on river row7, not median5; at higher levels median cannot be a lateral/NOOP waypoint. "
                  "Jev gets compact named facts and makes two order-reversed Choices each for whole-crossing bay itinerary "
                  "and TRANSIT/STAGE/RECOVER/SCREEN gate policy. Both choices require >=0.70 winning probability, not confidence; "
                  "unknown or missing probabilities preserve code fallback. Itinerary changes only before river entry. "
                  "A repeated final-UP failure may be latent support, home hazard or sampling, so RECOVER changes the approach "
                  "rather than expanding every obstacle. A public auxiliary fly candidate on a playable lane is separately "
                  "screened as a moving hazard, based on archived contact deaths; its observed spawn mouth is reserved "
                  "while absent and its stationary phase is tracked, not assumed to wrap. E3 high bit plus E5=bay+1 is a matched "
                  "home-alert veto; unknown is not safe. Complete unalerted alternatives can override a blocked itinerary. "
                  "The second median position track in public RAM is also screened at higher levels, as archived bank "
                  "deaths align with that moving track. These local hard exclusions cannot be disabled by model policy. "
                  "No invented RAM semantics. Choose only enums, return ONLY JSON: "
                  "{\"itinerary\":\"upstream|nearest\",\"launch\":\"transit|wide\",\"dock\":\"settle|different_carrier\"}. "
                  "Upstream-first avoids leaving an up-current last bay; wide may delay launch, transit minimizes bank exposure; "
                  "different_carrier avoids a specifically failed approach. Do not generate code or questions. Evidence="
                  + json.dumps(summary, separators=(",", ":")))
        self.s2_meta = {"revision": self.evidence_revision, "tick": t}
        session = self.session
        def call():
            try:
                return self.ctx.think(prompt, session_id=session, timeout_s=120.0)
            except (BudgetExceeded, EpisodeOver, ModelUnavailable):
                raise
            except (TimeoutError, RuntimeError, TypeError):
                return None
        self.s2 = self.ctx.spawn(call)
        self.next_s2 = t+3600

    def poll(self, o):
        t = integer(o.get("tick"))
        if self.s1 is not None and self.s1.done():
            f, self.s1 = self.s1, None
            try:
                answers, latency = f.result()
            except (BudgetExceeded, EpisodeOver, ModelUnavailable):
                raise
            except (TimeoutError, RuntimeError, TypeError, ValueError):
                answers, latency = None, 0.0
            if isinstance(answers, dict):
                meta = self.s1_meta or {}
                fresh = epoch(o) == meta.get("epoch") and self.version == meta.get("version") and t-meta.get("tick", t) <= 90
                chosen, p = agreed(answers, "itinerary", "itinerary_check", .70)
                if fresh and meta.get("road") and row(o) < 5 and chosen in {str(z.get("slot")) for z in self.open_homes(o)}:
                    before = self.target
                    self.target, self.target_owner = chosen, "jev_itinerary"
                    self.log("jev_commits_crossing", tick=t, before=before, after=chosen, probability=p, latency_s=latency)
                if fresh and row(o) >= 5 and chosen is not None and not home_alert(o, chosen):
                    feasible = [r for r in self.routes if r.get("complete") and r.get("goal") == chosen]
                    if feasible:
                        self.route_goal, self.route_goal_until = chosen, t+180
                        self.log("jev_portfolio_goal", tick=t, goal=chosen, probability=p)
                mode, prob = agreed(answers, "gate", "gate_check", .70)
                if fresh and mode in ("TRANSIT", "STAGE", "RECOVER", "SCREEN"):
                    before = self.protocol
                    self.protocol = mode
                    if mode == "RECOVER" and meta.get("revision") == self.evidence_revision:
                        self.diagnosis = "CHANGE_APPROACH"
                    self.log("jev_gate_protocol", tick=t, before=before, after=mode, probability=prob, latency_s=latency)
                if number(getattr(answers.get("review"), "value", 0)) >= .85:
                    self.request_astra(o)
        if self.s2 is not None and self.s2.done():
            f, self.s2 = self.s2, None
            try:
                result = f.result()
            except (BudgetExceeded, EpisodeOver, ModelUnavailable):
                raise
            except (TimeoutError, RuntimeError, TypeError):
                result = None
            if result is not None:
                self.session = getattr(result, "session_id", None)
                proposed = fragment(getattr(result, "text", ""))
                allowed = {"itinerary": ("upstream", "nearest"), "launch": ("transit", "wide"), "dock": ("settle", "different_carrier")}
                changes = {k: v for k, v in proposed.items() if k in allowed and isinstance(v, str) and v in allowed[k]}
                if changes:
                    self.policy.update(changes)
                    self.version += 1
                    self.next_s1 = t
                    if self.policy["dock"] == "different_carrier":
                        self.diagnosis = "CHANGE_APPROACH"
                    self.log("astra_protocol_compiled", tick=t, policy=dict(self.policy), version=self.version)
                else:
                    self.log("astra_protocol_rejected", tick=t)

    def rider_velocity(self, rr, fallback):
        values = list(self.rider_history.get(rr, ()))
        if len(values) < 3:
            return fallback, False
        values.sort()
        return values[len(values)//2], True

    def latched_edge_recovery(self, o, action, chosen):
        """Keep edge recovery durable across replans; never invent support."""
        rr = row(o)
        frog = o.get("frog", {})
        if rr < 6 or rr > 10 or not frog.get("moving_with_platform"):
            return action, chosen
        xx = number(frog.get("x"), 80)
        field = Field(self, o, 4)
        fallback = field.drift.get(rr, 0.0)
        drift, measured = self.rider_velocity(rr, fallback)
        horizon, layer = self.step*3, 3
        neutral_x = xx + drift*horizon
        move = next((m for m in MOVES if m[0] == action), MOVES[3])
        z = field.advance(rr, xx, move, 0, self.target)
        action_projection = neutral_x
        if z is not None and 6 <= z[0] <= 10:
            dest_drift = drift if z[0] == rr else field.drift.get(z[0], 0.0)
            action_projection = z[1] + dest_drift*horizon

        ep = epoch(o)
        latch = self.edge_latch
        if latch is not None and (latch.get("row") != rr or latch.get("epoch") != ep):
            self.edge_latch = latch = None
        if latch is not None:
            returned = neutral_x <= 132.0 if latch["side"] == "right" else neutral_x >= 28.0
            if returned:
                self.log("carrier_edge_latch_cleared", tick=o.get("tick"), row=rr,
                         side=latch["side"], neutral_projection=round(neutral_x, 2),
                         measured_rider_drift=measured)
                self.edge_latch = latch = None

        if latch is None:
            side = "right" if max(neutral_x, action_projection) > 142.0 else (
                "left" if min(neutral_x, action_projection) < 18.0 else None)
            if side is None:
                return action, chosen
            latch = {"side": side, "direction": "LEFT" if side == "right" else "RIGHT",
                     "row": rr, "epoch": ep, "since": integer(o.get("tick"))}
            self.edge_latch = latch
            self.log("carrier_edge_latch_started", tick=o.get("tick"), row=rr, side=side,
                     correction=latch["direction"], frog_x=round(xx, 2),
                     neutral_projection=round(neutral_x, 2),
                     action_projection=round(action_projection, 2), rider_drift=round(drift, 3),
                     measured=measured)

        inward = latch["direction"]
        outward = "RIGHT" if inward == "LEFT" else "LEFT"

        def verified(candidate):
            mv = next((m for m in MOVES if m[0] == candidate), MOVES[3])
            advanced = field.advance(rr, xx, mv, 0, self.target)
            if advanced is None:
                return None
            nr, nx, margin, _ = advanced
            if 6 <= nr <= 10:
                nd = drift if nr == rr else field.drift.get(nr, 0.0)
                projected = nx + nd*horizon
                future_margin = field.margin(nr, projected, layer)
                # The warning band is deliberately stricter than support: the
                # recovery may need one supported hop while still near the edge.
                # Reject only loss of support, not merely a projection above 142.
                if future_margin < 0:
                    return None
                return (nr, nx, min(margin, future_margin), projected)
            return (nr, nx, margin, nx)

        # Preserve a safe vertical itinerary or an already-inward correction.
        current_safe = verified(action)
        if current_safe is not None and action not in (outward, "NOOP"):
            return action, chosen

        # Recovery is latched until the neutral drift projection is safely back
        # inside the board. Prefer an inward lateral hop, then supported progress.
        order = (inward, "UP", "DOWN", "NOOP", outward)
        selected = None
        for candidate in order:
            if candidate == outward:
                continue
            proof = verified(candidate)
            if proof is not None:
                selected = (candidate, proof)
                break
        if selected is None:
            # There is no proven escape. Keep the route selector's already
            # checked action rather than replacing it with an unverified NOOP.
            self.log("carrier_edge_latch_unresolved", tick=o.get("tick"), row=rr,
                     side=latch["side"], action=action, fallback=action,
                     frog_x=round(xx, 2), neutral_projection=round(neutral_x, 2),
                     rider_drift=round(drift, 3), measured=measured)
            return action, dict(chosen or {}, action=action, edge_latch_unresolved=True)

        replacement, proof = selected
        self.log("carrier_edge_latch_override", tick=o.get("tick"), row=rr,
                 side=latch["side"], previous_action=action, replacement=replacement,
                 frog_x=round(xx, 2), neutral_projection=round(neutral_x, 2),
                 projected_landing=round(proof[3], 2), future_margin=round(proof[2], 2),
                 rider_drift=round(drift, 3), measured=measured,
                 latch_age=integer(o.get("tick"))-integer(latch.get("since")))
        return replacement, dict(chosen or {}, action=replacement, edge_latched=True,
                                 edge_latch_side=latch["side"], projected_landing=round(proof[3], 2),
                                 edge_future_margin=round(proof[2], 2))

    def pick(self, o, routes):
        if not routes:
            return "NOOP", None
        chosen = routes[0]
        if self.protocol == "STAGE" or self.policy["launch"] == "wide":
            near = [z for z in routes if z["complete"] == chosen["complete"] and z["value"] >= chosen["value"]-5]
            if near:
                chosen = max(near, key=lambda z: (z["margin"], z["value"]))
        if self.protocol == "SCREEN":
            near = [z for z in routes if z["complete"] == chosen["complete"] and z["value"] >= chosen["value"]-4]
            if near:
                chosen = min(near, key=lambda z: (sum(a in ("NOOP", "LEFT", "RIGHT") for a in z["path"]), -z["value"]))
        if chosen["action"] != routes[0]["action"]:
            self.log("semantic_protocol_changes_action", tick=o.get("tick"),
                     baseline=routes[0]["action"], selected=chosen["action"],
                     jev_protocol=self.protocol, astra_launch=self.policy["launch"], version=self.version)
        if chosen.get("complete") and chosen.get("endpoint") == 11 and chosen.get("goal") is not None:
            new_target = chosen["goal"]
            if new_target != self.target:
                self.log("portfolio_changes_itinerary", tick=o.get("tick"), before=self.target, after=new_target,
                         matched_alert=home_alert(o, self.target), owner="jev" if new_target == self.route_goal else "code")
                self.target, self.target_owner = new_target, "complete_portfolio"
        action = chosen["action"]
        live_field = Field(self, o, 1)
        if not self.guard(o, action, live_field):
            # Preserve an executable escape, not a blanket NOOP on an unsafe bank.
            valid = [z for z in routes if self.guard(o, z["action"], live_field)]
            if valid:
                chosen = valid[0]
                action = chosen["action"]
            else:
                ordered = ("UP", "DOWN", "LEFT", "RIGHT", "NOOP") if row(o) == 5 else ("NOOP", "UP", "LEFT", "RIGHT", "DOWN")
                action = next((a for a in ordered if self.guard(o, a, live_field)), "NOOP")
                chosen = dict(chosen, action=action, fallback=True)

        # A one-step NOOP can look safe while the carrier keeps moving on every
        # real-time frame. Before accepting that wait, test the rider's existing
        # row-drift forecast through three decision intervals. Only override it
        # for the demonstrated failure modes: same-row auxiliary contact or an
        # approaching screen edge. Any replacement must retain support now and
        # through the same long horizon; otherwise keep the parent's fallback.
        rr = row(o)
        rider = o.get("frog", {})
        if action == "NOOP" and rr >= 6 and rr <= 10 and rider.get("moving_with_platform"):
            long_field = Field(self, o, 4)
            xx = number(rider.get("x"), 80)
            drift = long_field.drift.get(rr, 0.0)
            look_frames = self.step * 3
            future_x = xx + drift * look_frames
            future_margin = long_field.margin(rr, future_x, 3)
            actor = auxiliary(o)
            actor_close = False
            if actor is not None and actor[0] == rr:
                actor_future_x = actor[1] + self.aux_velocity * look_frames
                actor_close = abs(future_x - actor_future_x) <= 28.0
            # Start an edge correction while an 18-frame lateral recovery is
            # still possible, rather than waiting for the projected point to
            # leave the playable interval. Obstacle-triggered corrections still
            # require the modeled auxiliary actor to be near the projection.
            edge_approach = future_x < 18.0 or future_x > 142.0
            danger = edge_approach or (future_margin < 0 and actor_close)
            if danger:
                exits = []
                for move in MOVES:
                    name = move[0]
                    if name == "NOOP":
                        continue
                    advanced = long_field.advance(rr, xx, move, 0, self.target)
                    if advanced is None:
                        continue
                    nr, nx, margin, _ = advanced
                    long_margin = 20.0
                    if 6 <= nr <= 10:
                        projected = nx + long_field.drift.get(nr, 0.0) * look_frames
                        long_margin = long_field.margin(nr, projected, 3)
                        if long_margin < 0:
                            continue
                    # Safety is a hard filter; forward progress breaks ties only
                    # after both present and longer-horizon support are certified.
                    value = long_margin * 2.0 + (nr - rr) * 5.0 - abs(nx - xx) * 0.1
                    exits.append((value, name, nr, nx, long_margin))
                if exits:
                    exits.sort(reverse=True)
                    _, replacement, nr, nx, exit_margin = exits[0]
                    self.log("carrier_wait_lookahead_veto", tick=o.get("tick"), row=rr,
                             frog_x=round(xx, 2), projected_x=round(future_x, 2),
                             projected_margin=round(future_margin, 2), actor=actor,
                             replacement=replacement, exit_row=nr, exit_x=round(nx, 2),
                             exit_margin=round(exit_margin, 2))
                    action = replacement
                    chosen = dict(chosen, action=action, lookahead_escape=True,
                                  projected_x=round(future_x, 2), exit_margin=round(exit_margin, 2))
                else:
                    self.log("carrier_wait_lookahead_unresolved", tick=o.get("tick"), row=rr,
                             frog_x=round(xx, 2), projected_x=round(future_x, 2),
                             projected_margin=round(future_margin, 2), actor=actor)
        action, chosen = self.latched_edge_recovery(o, action, chosen)
        return action, chosen

    def run_episode(self, game):
        del game
        while not self.ctx.done:
            o = unpack(self.ctx.observe())
            if not o:
                self.ctx.act("NOOP")
                self.ctx.wait_for_tick()
                continue
            self.update(o)
            if integer(o.get("tick")) > self.route_goal_until:
                self.route_goal = None
            self.poll(o)
            t, rr = integer(o.get("tick")), row(o)
            active = integer(o.get("frog", {}).get("state")) == 128 and -1 <= rr <= 10
            action = "NOOP"
            busy = self.press is not None
            if busy:
                moved = self.origin is not None and (rr != self.origin[0] or abs(number(o.get("frog", {}).get("x"))-self.origin[1]) >= 7)
                if moved or not active or t-self.press_tick >= 4:
                    self.log("pulse_release", tick=t, action=self.press, moved=moved)
                    self.press, self.origin = None, None
                    self.ready = max(self.ready, t+2)
                else:
                    action = self.press
            if not busy and active and t >= self.ready:
                self.routes = self.search(o)
                latest = unpack(self.ctx.observe())
                if latest and epoch(latest) == epoch(o) and row(latest) == rr:
                    action, selected = self.pick(latest, self.routes)
                    t = integer(latest.get("tick"), t)
                    if selected:
                        self.last_plan = {"tick": t, "origin": (rr, latest.get("frog", {}).get("x")),
                                          "target": self.target, "protocol": self.protocol, **selected}
                        if action != "NOOP" or t-self.last_log >= 60:
                            self.log("gate_action", tick=t, plan=self.last_plan, search_ms=round(self.last_search_ms, 2))
                            self.last_log = t
                    # Search consumes real frames. Schedule the NEXT search
                    # early by the observed computation lag, so the intended
                    # six-frame hop interval does not become seven/eight frames.
                    search_lag = max(0, min(2, t-integer(o.get("tick"), t)))
                    self.ready = t+(max(4, self.step-search_lag) if action != "NOOP" else 3)
                    if action != "NOOP":
                        self.press, self.press_tick = action, t
                        self.origin = (rr, number(latest.get("frog", {}).get("x")))
                        if action in ("LEFT", "RIGHT"):
                            self.last_lateral = t
                self.request_jev(o)
            self.request_astra(o)
            self.last_action = action
            self.ctx.act(action)
            self.ctx.wait_for_tick()


def build(ctx):
    return Harness(ctx)
