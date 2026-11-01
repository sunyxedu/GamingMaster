"""0020: transit routes with a measured carrier-drift escape watchdog.

The controller retains a time-expanded Frogger route portfolio and separately
tracks the frog's realized motion while it is carried. A short-horizon reserve
check overrides waiting only for a projected edge or a nearby lane actor.
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


def wrap_delta(a, b):
    """Shortest displacement in the cartridge's 160-pixel world."""
    return (number(a) - number(b) + 80.0) % 160.0 - 80.0


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
    def __init__(self, h, o, depth, relevant=None, actor_overlay=True):
        self.h, self.o = h, o
        self.actor_overlay = actor_overlay
        self.strict_bank = integer(o.get("level"), 1) >= 4 or h.bank_alarm
        self.rows, self.drift, self.objects = [], {}, {}
        self.homes = [z for z in o.get("homes", []) if not z.get("occupied") and z.get("available", True)
                      and z.get("crocodile") not in (True, "present", "dangerous")]
        self.home_centers = {str(z.get("slot")): number(z.get("x_center"), 80) for z in self.homes}
        self.alerted = {slot for slot in self.home_centers if home_alert(o, slot)}
        actor = auxiliary(o) if actor_overlay else None
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
                if actor_overlay and actor is not None and actor[0] == rr:
                    # It advances and can pause: carrier support alone is not
                    # safety. Its observed disappearance is NOT a wraparound.
                    av = h.aux_velocity
                    a = actor[1] + av*(now+1)
                    b = actor[1] + av*(now+h.step+3)
                    lo = max(0, math.floor(min(a, b)-11))
                    hi = min(160, math.ceil(max(a, b)+19))
                    if hi >= lo:
                        values[lo:hi+1] = [-1.0]*(hi-lo+1)
                elif actor_overlay and actor is None and h.aux_spawn is not None and h.aux_spawn[0] == rr:
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
        self.policy = {"itinerary": "upstream", "launch": "transit", "dock": "settle",
                       "hold": "short", "escape": "inward"}
        self.drift_advice = "KEEP"
        self.drift_advice_until = -1
        self.drift_advice_signature = None
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
        # Motion of the frog itself is the only drift quantity used by the
        # watchdog.  It is reset at each life and quantized conservatively.
        self.frog_motion = deque(maxlen=24)
        self.frog_drift = 0.0
        self.frog_drift_confidence = 0
        self.watchdog_until = -1
        self.watchdog_reason = None
        self.watchdog_signature = None
        self.drift_revision = 0
        self.last_watch_log = -100

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
                "known_spawn": self.aux_spawn, "frog_drift": round(self.frog_drift, 3),
                "frog_drift_confidence": self.frog_drift_confidence,
                "bank_track": (o.get("raw_ram") or [None]*17)[16] if len(o.get("raw_ram", [])) > 16 else None}

    def update_frog_motion(self, o):
        """Estimate the frog's realized carrier drift, excluding hop impulses."""
        old = self.prev
        t, rr = integer(o.get("tick")), row(o)
        f = o.get("frog", {}) if isinstance(o.get("frog"), dict) else {}
        moving = bool(f.get("moving_with_platform")) and integer(f.get("state")) == 128
        if not old or epoch(old) != epoch(o):
            self.frog_motion.clear()
            self.frog_drift = 0.0
            self.frog_drift_confidence = 0
            return
        pf = old.get("frog", {}) if isinstance(old.get("frog"), dict) else {}
        same = rr == row(old) == rr and moving and bool(pf.get("moving_with_platform")) \
               and integer(pf.get("state")) == 128
        if not same:
            if rr != row(old):
                self.frog_motion.clear()
                self.frog_drift = 0.0
                self.frog_drift_confidence = 0
            return
        dx = wrap_delta(f.get("x"), pf.get("x"))
        # A normal hop changes x by about eight pixels; carrier drift is much
        # slower.  Rejecting that impulse is important after a lateral escape.
        if abs(dx) > 2.25:
            self.frog_motion.clear()
            self.frog_drift = 0.0
            self.frog_drift_confidence = 0
            return
        if abs(dx) >= .05:
            self.frog_motion.append((t, dx))
        samples = [v for tick, v in self.frog_motion if t - tick <= 18 and abs(v) >= .05]
        if samples:
            samples.sort()
            self.frog_drift = samples[len(samples)//2]
            self.frog_drift = min((-1.0, -.5, -.25, -.125, 0.0, .125, .25, .5, 1.0),
                                  key=lambda v: abs(v-self.frog_drift))
            self.frog_drift_confidence = min(8, len(samples))
        elif t - (self.frog_motion[-1][0] if self.frog_motion else t) > 18:
            self.frog_drift = 0.0
            self.frog_drift_confidence = 0

    def drift_assessment(self, o):
        """Return a short-horizon carried-frog reserve assessment."""
        rr, xx = row(o), number(o.get("frog", {}).get("x"), 80)
        f = o.get("frog", {}) if isinstance(o.get("frog"), dict) else {}
        if not (6 <= rr <= 10 and f.get("moving_with_platform") and integer(f.get("state")) == 128):
            return {"status": "CLEAR", "reason": "not_carried", "row": rr, "x_band": "unknown",
                    "drift_band": "unknown", "edge": False, "actor": False,
                    "future_support": "unknown", "horizon": 0}
        t = integer(o.get("tick"))
        # Use a longer horizon for a passive wait than for the four-frame
        # pulse itself.  The slow reviewer can only choose between these two
        # bounded policies; it cannot remove the edge guard.
        horizon = 10 if self.policy.get("hold") == "short" else 14
        drift = self.frog_drift if self.frog_drift_confidence else 0.0
        q = auxiliary(o)
        av = self.aux_velocity if q is not None else 0.0
        actor_near = False
        actor_min = 999.0
        if q is not None and q[0] == rr:
            for k in range(0, horizon + 1, 2):
                actor_min = min(actor_min, abs(wrap_delta(q[1] + av*k, xx + drift*k)))
            actor_near = actor_min <= 25.0
        projected = xx + drift*horizon
        edge = projected < 16.0 or projected > 144.0 or (xx < 9 and drift <= 0) or (xx > 151 and drift >= 0)
        # A carrier can look safe now but lose the frog during the hold.  Use
        # the actor-free field for support, since actor contact is reported
        # separately and must not hide the underlying landing certificate.
        support = "stable"
        if self.frog_drift_confidence:
            try:
                plain = Field(self, o, 3, relevant={rr}, actor_overlay=False)
                checks = [plain.margin(rr, xx, 0), plain.margin(rr, xx + drift*5, 1),
                          plain.margin(rr, projected, 2)]
                if min(checks) < 0:
                    support = "lost"
            except (IndexError, TypeError, ValueError):
                support = "uncertain"
        elif xx < 18 or xx > 142:
            support = "uncertain"
        future = support
        status = "ESCAPE" if (edge or actor_near) else \
                 "WATCH" if (self.frog_drift_confidence and (abs(drift) >= .125 or xx < 24 or xx > 136 or support != "stable")) else "CLEAR"
        xb = "left_edge" if xx < 24 else "right_edge" if xx > 136 else "interior"
        db = "left" if drift < -.05 else "right" if drift > .05 else "stationary"
        return {"status": status, "reason": "actor_contact" if actor_near else "edge_drift" if edge else future,
                "row": rr, "x": round(xx, 1), "x_band": xb, "drift_band": db,
                "drift": round(drift, 3), "confidence": self.frog_drift_confidence,
                "projected_x": round(projected, 1), "actor": actor_near,
                "actor_min_band": "contact" if actor_min <= 25 else "near" if actor_min < 40 else "far",
                "future_support": future, "edge": edge, "horizon": horizon}

    def watchdog_action(self, o, assessment):
        """Convert a carried-frog reserve warning into a verified escape hop.

        The normal Field deliberately overlays the auxiliary actor and can
        therefore make every option look blocked.  An emergency is different:
        it is allowed to use carrier geometry without the actor overlay, then
        selects a one-hop exit away from the watched contact.  The actor is
        never ignored for a *new* route; this is only the immediate escape
        bridge when waiting is the demonstrated lethal choice.
        """
        if assessment.get("status") != "ESCAPE":
            return None, None
        rr = integer(assessment.get("row"), row(o))
        xx = number(o.get("frog", {}).get("x"), 80)
        q = auxiliary(o)
        preferred = None
        if assessment.get("edge"):
            drift = number(assessment.get("drift"))
            preferred = "LEFT" if drift > 0 or xx > 136 else "RIGHT"
        elif q is not None and q[0] == rr:
            preferred = "LEFT" if wrap_delta(q[1], xx) > 0 else "RIGHT"
        order = []
        advice = (self.drift_advice if integer(o.get("tick")) <= self.drift_advice_until
                  and self.drift_advice_signature == self.drift_signature(o) else "KEEP")
        if assessment.get("edge"):
            # Edge protection is invariant: steer against drift before any
            # model-selected row exit.
            order.append(preferred)
            order.extend(["UP", "DOWN"])
        elif advice == "FORWARD" or (advice == "KEEP" and self.policy.get("escape") == "forward_first"):
            order.extend(["UP", "DOWN"])
            if preferred:
                order.append(preferred)
        elif advice == "BACK":
            order.append("DOWN")
            if preferred:
                order.append(preferred)
            order.append("UP")
        elif advice == "INWARD":
            if preferred:
                order.append(preferred)
            order.extend(["UP", "DOWN"])
        else:
            if preferred:
                order.append(preferred)
            order.extend(["UP", "DOWN"])
        order.append("RIGHT" if preferred == "LEFT" else "LEFT")
        seen = set()
        order = [a for a in order if not (a in seen or seen.add(a))]
        # During a rescue we certify the actual one-frame landing, not the
        # planner's six-frame dwell. A lateral pulse can move the frog away
        # from a hazard before the next observation, but an off-platform
        # landing is still rejected by carrier geometry.
        normal_step = self.step
        self.step = 1
        try:
            quick = Field(self, o, 5, relevant=set(range(4, 11)), actor_overlay=False)
            for action in order:
                if self.guard(o, action, quick):
                    return action, {"action": action, "reason": assessment.get("reason"),
                                    "assessment": assessment, "actor": q, "emergency": True}
            # The six-frame route guard can reject an otherwise supported
            # inward hop because its average lane velocity is noisier than the
            # observed frog drift. Certify this escape against the actual row
            # support for the hop/release interval instead of waiting at an
            # edge: both its landing and four-frame carried position must be
            # supported by safe carrier geometry.
            safe_contact = any(not c.get("hazard") and c.get("kind") in ("log", "turtle")
                               for c in self.carrier(o))
            sides = [preferred] if preferred in ("LEFT", "RIGHT") else []
            sides += [a for a in ("LEFT", "RIGHT") if a not in sides]
            if safe_contact and (assessment.get("edge") or assessment.get("actor")):
                drift = number(assessment.get("drift"))
                for action in sides:
                    dx = -8 if action == "LEFT" else 8
                    land = xx + dx
                    future = land + drift*4
                    if not (4 <= land <= 148 and 4 <= future <= 148):
                        continue
                    if quick.margin(rr, land, 0) < 0 or quick.margin(rr, future, 4) < 0:
                        continue
                    if q is not None and q[0] == rr:
                        before_gap = abs(wrap_delta(q[1], land))
                        after_gap = abs(wrap_delta(q[1] + self.aux_velocity*4, future))
                        if min(before_gap, after_gap) <= 18:
                            continue
                    return action, {"action": action, "reason": assessment.get("reason"),
                                    "assessment": assessment, "actor": q,
                                    "verified_as": "supported_inward_drift_hop", "emergency": True}
        finally:
            self.step = normal_step
        # No checked exit is available in this frame. Fall through to the
        # normal route portfolio rather than leasing NOOP as an emergency.
        return None, None

    def drift_signature(self, o):
        a = self.drift_assessment(o)
        return (epoch(o), a.get("row"), a.get("status"), a.get("x_band"),
                a.get("drift_band"), a.get("actor_min_band"), a.get("future_support"))

    def update(self, o):
        t, ep, rr = integer(o.get("tick")), epoch(o), row(o)
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
                        "auxiliary": auxiliary(self.prev),
                        "drift_witness": self.drift_assessment(self.prev)}
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
            if evidence["carrier_failure"] and evidence["previous"].get("frog", {}).get("moving_with_platform"):
                self.next_s2 = min(self.next_s2, t + 60)
        # Preserve the just-ended life’s drift estimate for its failure witness;
        # only then reset/advance the motion filter for the current observation.
        self.update_frog_motion(o)
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
            self.watchdog_until = -1
            self.watchdog_reason = None
            self.watchdog_signature = None
            self.drift_advice = "KEEP"
            self.drift_advice_until = -1
            self.drift_advice_signature = None
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
        drift = self.drift_assessment(o)
        return {"game": "Frogger: fill different open bays while preserving lives.",
                "phase": "road_launch" if row(o) < 5 else "bank_transit" if row(o) == 5 else "dock" if row(o) == 10 else "river",
                "policy": dict(self.policy), "homes": facts,
                "drift": {k: drift.get(k) for k in ("status", "x_band", "drift_band", "actor_min_band", "future_support", "edge", "horizon")},
                "facts": {"higher_level_bank_threat": epoch(o)[2] >= 4 or self.bank_alarm,
                          "complete_launch_exists": any(r["complete"] for r in self.routes),
                          "auxiliary_on_playable_row": auxiliary(o) is not None,
                          "auxiliary_on_frog_row": auxiliary(o) is not None and auxiliary(o)[0] == row(o),
                          "auxiliary_spawn_reserved": self.aux_spawn is not None and auxiliary(o) is None,
                          "target_has_matched_alert": home_alert(o, self.target),
                          "urgent_timer": number(o.get("timer_s"), 30) < 8,
                          "entry_failed": bool(ev.get("entry_failure")), "bank_failed": bool(ev.get("bank_failure")),
                          "repeat_entry": bool(ev.get("repeat_entry")), "carrier_failed": bool(ev.get("carrier_failure")),
                          "drift_watch": drift.get("status") in ("WATCH", "ESCAPE"),
                          "drift_escape_required": drift.get("status") == "ESCAPE"},
                "rules": "The top carrier flow makes upstream bays harder to revisit. Select an itinerary early, not a joystick. The median is a transit link, not a safe long wait. A failed final UP does NOT establish that all carriers are water. Unknown crocodile status is not an absent crocodile. Code rechecks geometry and docking every frame. A carried frog must leave an edge or active actor before a passive wait."}

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
        reserves = {"CLEAR": "The carried frog has ample reserve for the short passive horizon; no emergency protocol is needed.",
                    "WATCH": "A short passive horizon is acceptable, but recheck drift promptly and avoid a long neutral dwell.",
                    "ESCAPE": "A projected screen-edge loss or same-row actor contact requires a verified one-hop escape now.",
                    "WATCH": "Support forecast is uncertain or drift is meaningful; shorten the observation cadence.",
                    "UNKNOWN": "Evidence is insufficient; preserve the code fallback and do not authorize a risky wait."}
        reserve_instruction = ("Classify only drift. `drift.status` and `facts.drift_escape_required` are code-derived. "
                               "ESCAPE is appropriate only when a projected screen-edge loss or same-row actor contact is explicit; "
                               "WATCH means support or drift is uncertain and the next wait should be short. Do not calculate distances or invent actions.")
        protocols = {"KEEP": "Keep the code-selected action if its live guard passes.",
                     "INWARD": "Prefer a code-generated lateral hop toward the interior when it is live-verified.",
                     "FORWARD": "Prefer a verified UP exit from the watched carrier row.",
                     "BACK": "Prefer a verified DOWN exit when UP is not supported." ,
                     "UNKNOWN": "Use the conservative code fallback."}
        protocol_instruction = ("Choose among named executable protocols, never a raw joystick string. If facts.drift_escape_required, "
                                "do not choose KEEP; prefer INWARD, then FORWARD, then BACK according to the named situation. "
                                "With CLEAR choose KEEP. UNKNOWN when facts do not support a protocol.")
        questions = {"itinerary": Choice(instruction, itinerary),
                     "itinerary_check": Choice(instruction, dict(reversed(list(itinerary.items())))),
                     "gate": Choice(gate_instruction, gates),
                     "gate_check": Choice(gate_instruction, dict(reversed(list(gates.items())))),
                     "reserve": Choice(reserve_instruction, reserves),
                     "reserve_check": Choice(reserve_instruction, dict(reversed(list(reserves.items())))),
                     "protocol": Choice(protocol_instruction, protocols),
                     "protocol_check": Choice(protocol_instruction, dict(reversed(list(protocols.items())))),
                     "review": Noul("Do facts explicitly show repeated entry, bank, or drift failure requiring Astra to revise the crossing protocol?")}
        self.s1_meta = {"tick": t, "epoch": epoch(o), "version": self.version, "revision": self.evidence_revision,
                        "road": row(o) < 5, "state": state, "reserve_signature": self.drift_signature(o)}
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
        drift_status = self.drift_assessment(o).get("status")
        interval = 24 if drift_status in ("WATCH", "ESCAPE") else 90
        self.next_s1 = t+interval

    def request_astra(self, o):
        t = integer(o.get("tick"))
        if self.s2 is not None or t < self.next_s2 or not self.recent or getattr(self.ctx, "system2", True) is None:
            return
        if self.ctx.budget.remaining().get("system2_calls", 0) < 2:
            return
        summary = {"policy": self.policy, "failures": list(self.recent)[-2:], "score": o.get("score"),
                   "level": o.get("level"), "search_ms": round(self.last_search_ms, 2), "protocol": self.protocol,
                   "drift": self.drift_assessment(o)}
        prompt = ("You review Frogger crossing and drift-reserve PROTOCOLS asynchronously, never frame actions. Code uses separated sprite copies, "
                  "body-aware collision intervals, neutral-rearmed six-frame hops, and a measured frog-carrier drift watchdog. "
                  "The watchdog forecasts only a short passive hold and uses a hard live guard for edge, support and auxiliary contact; "
                  "it is not a global carrier pad. Jev gets compact named facts and makes order-reversed Choices for itinerary, gate, "
                  "drift reserve and KEEP/INWARD/FORWARD/BACK protocol. Both choices require >=0.70 winning probability, not confidence; "
                  "unknown or missing probabilities preserve code fallback. Local edge/contact exclusions cannot be disabled. "
                  "A repeated final-UP failure may be latent support, home hazard or sampling; change approach rather than enlarging every obstacle. "
                  "Return ONLY JSON with bounded enums: {\"itinerary\":\"upstream|nearest\",\"launch\":\"transit|wide\","
                  "\"dock\":\"settle|different_carrier\",\"hold\":\"short|normal\",\"escape\":\"inward|forward_first\"}. "
                  "Do not generate code, questions, coordinates or actions. Prefer hold=short and escape=inward when drift evidence is inconclusive. Evidence="
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
                fresh = (epoch(o) == meta.get("epoch") and self.version == meta.get("version")
                         and t-meta.get("tick", t) <= 70
                         and self.drift_signature(o) == meta.get("reserve_signature"))
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
                reserve, reserve_p = agreed(answers, "reserve", "reserve_check", .70)
                protocol, protocol_p = agreed(answers, "protocol", "protocol_check", .70)
                if fresh:
                    assessment = self.drift_assessment(o)
                    # A typed answer may shorten the lease, but cannot soften a
                    # code-derived ESCAPE or make an unsafe protocol executable.
                    if reserve in ("CLEAR", "WATCH", "ESCAPE") and reserve == assessment.get("status"):
                        self.drift_advice = protocol if protocol in ("KEEP", "INWARD", "FORWARD", "BACK") else "KEEP"
                        self.drift_advice_until = t + (24 if reserve == "ESCAPE" else 60)
                        self.drift_advice_signature = self.drift_signature(o)
                        self.log("jev_drift_protocol", tick=t, reserve=reserve, protocol=self.drift_advice,
                                 probabilities=[reserve_p, protocol_p], latency_s=latency)
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
                allowed = {"itinerary": ("upstream", "nearest"), "launch": ("transit", "wide"), "dock": ("settle", "different_carrier"),
                           "hold": ("short", "normal"), "escape": ("inward", "forward_first")}
                changes = {k: v for k, v in proposed.items() if k in allowed and isinstance(v, str) and v in allowed[k]}
                if changes and integer((self.s2_meta or {}).get("revision")) == self.evidence_revision:
                    self.policy.update(changes)
                    self.version += 1
                    self.next_s1 = t
                    if self.policy["dock"] == "different_carrier":
                        self.diagnosis = "CHANGE_APPROACH"
                    self.log("astra_protocol_compiled", tick=t, policy=dict(self.policy), version=self.version)
                else:
                    self.log("astra_protocol_rejected", tick=t, revision=self.evidence_revision,
                             source_revision=(self.s2_meta or {}).get("revision"))

    def pick(self, o, routes):
        t = integer(o.get("tick"))
        assessment = self.drift_assessment(o)
        signature = (epoch(o), assessment.get("row"), assessment.get("x_band"),
                     assessment.get("drift_band"), assessment.get("actor_min_band"),
                     assessment.get("future_support"))
        if assessment.get("status") == "ESCAPE":
            action, selected = self.watchdog_action(o, assessment)
            if action is not None:
                self.watchdog_until = t + 6
                self.watchdog_reason = assessment.get("reason")
                self.watchdog_signature = signature
                self.drift_revision += 1
                self.log("drift_emergency", tick=t, action=action, assessment=assessment,
                         selected=selected, target=self.target)
                self.press = action if action != "NOOP" else None
                self.press_tick = t
                self.origin = (row(o), number(o.get("frog", {}).get("x"))) if action != "NOOP" else None
                self.ready = t + 4
                if action in ("LEFT", "RIGHT"):
                    self.last_lateral = t
                self.last_plan = {"tick": t, "origin": self.origin, "target": self.target,
                                  "action": action, "emergency": True, "assessment": assessment}
                if t - self.last_watch_log >= 5:
                    self.last_watch_log = t
                return action, selected
            self.log("drift_watchdog_block", tick=t, assessment=assessment)
        else:
            if assessment.get("status") == "WATCH" and t - self.last_watch_log >= 60:
                self.log("drift_watch", tick=t, assessment=assessment)
                self.last_watch_log = t
            if self.watchdog_until >= t and self.watchdog_signature:
                # Release a no-longer-needed watchdog reservation immediately
                # when the frog has moved to another row or away from danger.
                if self.watchdog_signature[0] != epoch(o) or self.watchdog_signature[1] != row(o):
                    self.watchdog_until = -1
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
