"""0022: viability search with a latched vertical carrier escape.

The controller plans across the road/river, but treats a dangerous carrier as
an exit problem rather than repeatedly steering sideways on the same carrier.
"""
from __future__ import annotations

import json
import math
from collections import deque

from metaharness.core.contract import BudgetExceeded, EpisodeOver, ModelUnavailable
from metaharness.models.jev import Choice, Noul

# Atari object copy layouts, indexed by TIA NUSIZ low bits.
COPIES = ((0,), (0, 16), (0, 32), (0, 16, 32), (0, 64), (0,), (0, 32, 64), (0,))
WIDTHS = (8, 8, 8, 8, 8, 16, 8, 32)
MOVES = (("UP", 1, 0), ("LEFT", 0, -8), ("RIGHT", 0, 8), ("DOWN", -1, 0), ("NOOP", 0, 0))


def num(value, default=0.0):
    try:
        x = float(value)
        return x if math.isfinite(x) else default
    except (TypeError, ValueError, OverflowError):
        return default


def integer(value, default=0):
    return int(num(value, default))


def mapping(value):
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except (TypeError, ValueError):
            return {}
    return value if isinstance(value, dict) else {}


def fragment(value):
    if not isinstance(value, str):
        return {}
    try:
        a, b = value.index("{"), value.rindex("}") + 1
        out = json.loads(value[a:b])
        return out if isinstance(out, dict) else {}
    except (ValueError, TypeError, json.JSONDecodeError):
        return {}


def row_of(o):
    r = integer(mapping(o.get("frog")).get("row"), 255)
    return -1 if r == 255 else r


def epoch_of(o):
    return tuple(integer(o.get(k), d) for k, d in (("deaths", 0), ("frogs_home", 0), ("level", 1)))


def wrap_delta(a, b):
    return (num(a) - num(b) + 80.0) % 160.0 - 80.0


def join_spans(spans):
    out = []
    for lo, hi in sorted(spans):
        if out and lo <= out[-1][1] + 0.001:
            out[-1][1] = max(out[-1][1], hi)
        else:
            out.append([lo, hi])
    return out


def actor_at(o):
    """Public bonus candidate, considered a possible actor only on a board row."""
    f = mapping(o.get("fly"))
    ram = mapping(o.get("ram_fields"))
    rr = integer(f.get("row") if f.get("row") is not None else ram.get("fly_row_ae"), 255)
    active = bool(f.get("present")) or bool(integer(ram.get("bonus_flags_c7")) & 128)
    if active and 0 <= rr <= 10:
        xx = f.get("x") if f.get("x") is not None else ram.get("fly_x_b2")
        return rr, num(xx)
    return None


def alerted(o, bay):
    ram = mapping(o.get("ram_fields"))
    return (bool(integer(ram.get("collision_e3")) & 128)
            and integer(ram.get("collision_row_e5"), -2) == integer(bay, -3) + 1)


class Forecast:
    """Per-hop collision/support raster over a compact moving-object model."""
    def __init__(self, brain, o, depth, hop=5, overlay=True, rows=None):
        self.brain, self.o, self.hop = brain, o, hop
        self.strict_bank = integer(o.get("level"), 1) >= 4 or brain.bank_alarm
        self.tracks = {}
        self.drift = {}
        self.homes = [h for h in o.get("homes", []) if not h.get("occupied") and h.get("available", True)]
        self.centers = {str(h.get("slot")): num(h.get("x_center"), 80) for h in self.homes}
        self.warned = {k for k in self.centers if alerted(o, k)}
        actor = actor_at(o) if overlay else None
        for lane in list(o.get("road", [])) + list(o.get("river", [])):
            rr = integer(lane.get("row"), -9)
            if rows is not None and rr not in rows:
                continue
            vals = []
            for z in lane.get("objects", []):
                n = integer(z.get("nusiz")) & 7
                direction = z.get("direction", lane.get("direction"))
                sign = -1 if direction == "left" else 1
                ident = str(z.get("id", ""))
                speed = brain.speed.get(ident, sign / max(1, integer(lane.get("activity_mask"), 3) + 1))
                shape = integer(z.get("shape_ptr"))
                width = 24 if n == 7 and shape in (163, 170) else WIDTHS[n]
                safe = z.get("kind") in ("log", "turtle") and not z.get("hazard")
                vals.append((num(z.get("x")), speed, COPIES[n], width, safe, ident))
            self.tracks[rr] = vals
            if rr >= 6:
                self.drift[rr] = sum(v[1] for v in vals) / max(1, len(vals))
        raw = o.get("raw_ram", [])
        if self.strict_bank and isinstance(raw, list) and len(raw) > 16 and (rows is None or 5 in rows):
            if 5 not in self.tracks:
                self.tracks[5] = [(num(raw[16]), brain.bank_velocity, (0,), 16, False, "median_track")]

        self.cells = []
        for d in range(depth + 2):
            elapsed = d * hop
            raster = {}
            for rr, sprites in self.tracks.items():
                if rr <= 5:
                    safe = [18.0] * 161
                    for x, v, copies, width, _, _ in sprites:
                        lo_t, hi_t = x + v * (elapsed + 1), x + v * (elapsed + hop + 2)
                        for off in copies:
                            self._mark(safe, min(lo_t, hi_t) + off - 9.5,
                                       max(lo_t, hi_t) + off + width + 1.5)
                else:
                    safe = [-1.0] * 161
                    spans = []
                    for x, v, copies, width, usable, _ in sprites:
                        if usable:
                            at = x + v * (elapsed + 1)
                            for off in copies:
                                for wrap in (-320, -160, 0, 160, 320):
                                    spans.append((at + off + wrap, at + off + width + wrap))
                    for lo, hi in join_spans(spans):
                        left, right = lo - 2, hi - 6
                        for px in range(max(0, math.ceil(left)), min(160, math.floor(right)) + 1):
                            safe[px] = max(safe[px], min(px - left, right - px))
                if overlay and actor is not None and actor[0] == rr:
                    av = brain.actor_velocity
                    a = actor[1] + av * (elapsed + 1)
                    b = actor[1] + av * (elapsed + hop + 3)
                    self._block(safe, min(a, b) - 11, max(a, b) + 19)
                elif overlay and actor is None and brain.actor_spawn and brain.actor_spawn[0] == rr:
                    sx = brain.actor_spawn[1]
                    self._block(safe, sx - 11, sx + 23)
                raster[rr] = safe
            self.cells.append(raster)

    @staticmethod
    def _mark(cells, low, high):
        for shift in (-320, -160, 0, 160, 320):
            a, b = max(0, math.floor(low + shift)), min(160, math.ceil(high + shift))
            if a <= b:
                cells[a:b + 1] = [-1.0] * (b - a + 1)

    @staticmethod
    def _block(cells, low, high):
        a, b = max(0, math.floor(low)), min(160, math.ceil(high))
        if a <= b:
            cells[a:b + 1] = [-1.0] * (b - a + 1)

    def margin(self, rr, x, d=0):
        if not math.isfinite(x) or x < 4 or x > 148 or d >= len(self.cells):
            return -100.0
        if rr == -1 or (rr == 5 and 5 not in self.cells[d]):
            return 18.0
        if rr < 0 or rr > 10:
            return -100.0
        cells = self.cells[d].get(rr)
        return cells[int(max(0, min(160, round(x))))] if cells else -100.0

    def home_margin(self, x, bay):
        center = self.centers.get(str(bay))
        if center is None or str(bay) in self.warned:
            return -100.0
        return min(x - center + 4, center - x)

    def transition(self, rr, x, move, d, bay=None, bank_visits=0):
        action, dr, dx = move
        nr, nx = rr + dr, x + dx
        if not -1 <= nr <= 11:
            return None
        if nr == 11:
            if action != "UP" or bay is None:
                return None
            m = self.home_margin(nx, bay)
            return (nr, nx, m, 0) if m >= 0 else None
        if nx < 4 or nx > 148:
            return None
        visits = bank_visits + 1 if nr == 5 else 0
        if self.strict_bank and nr == 5 and (rr == 5 or rr >= 6):
            return None
        if not self.strict_bank and visits > 3:
            return None
        m = self.margin(nr, nx, d)
        if m < 0:
            return None
        if nr >= 6:
            drifted = nx + self.drift.get(nr, 0.0) * self.hop
            tail = self.margin(nr, drifted, d + 1)
            if tail < 0:
                return None
            m = min(m, tail)
            nx = drifted
        return nr, nx, m, visits


class Frogger:
    def __init__(self, ctx):
        self.ctx = ctx
        self.prev = None
        self.speed = {}
        self.pos = {}
        self.actor_velocity = .5
        self.actor_cruise = .5
        self.actor_points = deque(maxlen=40)
        self.actor_spawn = None
        self.actor_present = False
        self.bank_velocity = .25
        self.bank_points = deque(maxlen=48)
        self.bank_alarm = False
        self.frog_points = deque(maxlen=18)
        self.frog_drift = 0.0
        self.drift_samples = 0
        self.epoch = None
        self.target = None
        self.target_mode = "UPSTREAM"
        self.escape_bias = "UP"
        self.policy = {"itinerary": "upstream", "escape": "vertical"}
        self.revision = 0
        self.failures = deque(maxlen=6)
        self.s1 = None
        self.s1_meta = None
        self.s1_due = 0
        self.s2 = None
        self.s2_meta = None
        self.s2_due = 1800
        self.session = None
        self.press = None
        self.press_at = -100
        self.origin = None
        self.ready_at = 3
        self.last_action = "NOOP"
        self.last_plan = None
        self.portfolio = []
        self.last_search = 0.0
        self.last_log = -100
        self.last_row = None
        self.progress_at = 0
        self.escape_latch = None
        self.escape_latch_until = -1
        self.escape_count = 0

    def log(self, name, **data):
        self.ctx.log(name, **data)

    def homes(self, o):
        return [h for h in o.get("homes", []) if not h.get("occupied") and h.get("available", True)]

    def choose_initial_target(self, o):
        homes = self.homes(o)
        if not homes:
            return None
        top = next((z for z in o.get("river", []) if integer(z.get("row")) == 10), {})
        sign = -1 if top.get("direction") == "left" else 1
        if self.target_mode == "NEAREST":
            x = num(mapping(o.get("frog")).get("x"), 80)
            return str(min(homes, key=lambda h: abs(num(h.get("x_center"), 80) - x)).get("slot"))
        return str(min(homes, key=lambda h: sign * num(h.get("x_center"), 80)).get("slot"))

    def track_motion(self, o):
        t = integer(o.get("tick"))
        for lane in list(o.get("road", [])) + list(o.get("river", [])):
            for obj in lane.get("objects", []):
                ident, x = str(obj.get("id", "")), num(obj.get("x"))
                q = self.pos.setdefault(ident, deque(maxlen=40))
                if not q or q[-1][0] != t:
                    q.append((t, x))
                old = next((p for p in q if 16 <= t - p[0] <= 50), None)
                sign = -1 if obj.get("direction", lane.get("direction")) == "left" else 1
                v = sign / max(1, integer(lane.get("activity_mask"), 3) + 1)
                if old:
                    observed = wrap_delta(x, old[1]) / max(1, t - old[0])
                    if observed * sign > 0 and abs(observed) <= 2:
                        candidates = (1/16, 1/8, 1/4, 1/2, 1, 2)
                        v = sign * min(candidates, key=lambda c: abs(c - abs(observed)))
                self.speed[ident] = v
        actor = actor_at(o)
        if actor is None:
            self.actor_points.clear()
            self.actor_present = False
        else:
            rr, x = actor
            if not self.actor_present and x <= 4:
                self.actor_spawn = (rr, x)
            self.actor_present = True
            if self.actor_points and abs(x - self.actor_points[-1][1]) > 20:
                self.actor_points.clear()
            if not self.actor_points or self.actor_points[-1][0] != t:
                self.actor_points.append((t, x))
            old = next((p for p in self.actor_points if 8 <= t - p[0] <= 30), None)
            if old:
                v = (x - old[1]) / max(1, t - old[0])
                if .05 <= abs(v) <= 2:
                    self.actor_velocity = min((-.1, -.25, -.5, -1, .1, .25, .5, 1), key=lambda c: abs(c-v))
                    self.actor_cruise = self.actor_velocity
                elif abs(v) < .025:
                    self.actor_velocity = 0.0
        raw = o.get("raw_ram", [])
        if isinstance(raw, list) and len(raw) > 16 and t % 4 == 0:
            self.bank_points.append((t, num(raw[16])))
            old = next((p for p in self.bank_points if 12 <= t-p[0] <= 40), None)
            if old:
                v = wrap_delta(raw[16], old[1]) / max(1, t-old[0])
                if 0 < v <= 2:
                    self.bank_velocity = min((.125, .25, .5, 1), key=lambda c: abs(c-v))
        frog = mapping(o.get("frog"))
        if self.prev and epoch_of(self.prev) == epoch_of(o) and row_of(self.prev) == row_of(o) and 6 <= row_of(o) <= 10:
            oldfrog = mapping(self.prev.get("frog"))
            if frog.get("moving_with_platform") and oldfrog.get("moving_with_platform"):
                dx = wrap_delta(frog.get("x"), oldfrog.get("x"))
                if abs(dx) <= 2.25 and abs(dx) >= .05:
                    self.frog_points.append((t, dx))
                    vals = sorted(x[1] for x in self.frog_points if t-x[0] <= 18)
                    if vals:
                        self.frog_drift = vals[len(vals)//2]
                        self.drift_samples = len(vals)
                elif abs(dx) > 2.25:
                    self.frog_points.clear()
                    self.drift_samples = 0
        else:
            if self.prev and row_of(self.prev) != row_of(o):
                self.frog_points.clear()
                self.drift_samples = 0
                self.frog_drift = 0.0
        self.prev = o

    def observe(self, o):
        ep = epoch_of(o)
        t, rr = integer(o.get("tick")), row_of(o)
        prior = self.prev
        # track_motion retains the old observation long enough for failure diagnosis.
        if prior is not None and ep[0] > epoch_of(prior)[0]:
            oldrow = row_of(prior)
            entry = oldrow == 10 and self.last_action == "UP" and t - integer((self.last_plan or {}).get("tick"), t - 99) <= 6
            witness = {"tick": t, "row": oldrow, "frog": mapping(prior.get("frog")),
                       "last_action": self.last_action, "entry": entry,
                       "actor": actor_at(prior), "plan": self.last_plan,
                       "timer": prior.get("timer_s"), "reserve": prior.get("reserve_frogs")}
            self.failures.append(witness)
            self.log("life_loss_witness", **witness)
            if oldrow == 5:
                self.bank_alarm = True
            self.revision += 1
            self.escape_latch = None
        if self.epoch != ep:
            self.epoch = ep
            self.target = None
            self.press = None
            self.origin = None
            self.ready_at = t + 3
            self.escape_latch = None
            self.frog_points.clear()
            self.drift_samples = 0
            self.frog_drift = 0.0
        if prior is not None and row_of(prior) != rr:
            self.progress_at = t
            # Preserve a vertical escape commitment across a row transition;
            # the next live support check decides whether it remains executable.
        if self.target is None:
            self.target = self.choose_initial_target(o)
        self.track_motion(o)

    def make_field(self, o, depth=18, hop=5, rows=None, overlay=True):
        return Forecast(self, o, depth, hop=hop, rows=rows, overlay=overlay)

    def route_search(self, o):
        """Bounded dynamic-programming search; retain diverse first-hop plans."""
        import time
        started = time.perf_counter()
        rr, x = row_of(o), num(mapping(o.get("frog")).get("x"), 80)
        if not -1 <= rr <= 10:
            return []
        road = rr < 5
        endpoint = 7 if road else 11
        horizon = 21
        relevant = set(range(8)) if road else set(range(4, 11))
        field = self.make_field(o, horizon, 5, relevant, True)
        homes = [(str(h.get("slot")), num(h.get("x_center"), 80)) for h in field.homes
                 if str(h.get("slot")) not in field.warned]
        if self.target is None or self.target not in dict(homes):
            self.target = self.choose_initial_target(o)
        # While crossing the road, favor the central approach corridor rather
        # than dragging the frog laterally toward a distant home bay. River
        # search takes over bay alignment after the median transition.
        goal_x = (80.0 if road else field.centers.get(str(self.target), 80)) - (0 if road else 2)
        beam = [(rr, x, (), 18.0, 0, 0.0)]  # row, x, path, clearance, bank, effort
        finals, partial = {}, {}
        for depth in range(horizon):
            expanded = []
            for r, px, path, clear, banks, effort in beam:
                for move in MOVES:
                    act, dr, dx = move
                    nr = r + dr
                    if road and nr > endpoint:
                        continue
                    if not road and nr < 5:
                        continue
                    bay = self.target
                    # When a target is locally alerted, use only a complete open alternative.
                    if not road and r == 10 and act == "UP" and str(bay) in field.warned:
                        bay = None
                    moved = field.transition(r, px, move, depth, bay, banks)
                    if moved is None:
                        if nr == 11 and not road:
                            for bid, _ in homes:
                                candidate = field.transition(r, px, move, depth, bid, banks)
                                if candidate is not None:
                                    moved = candidate
                                    bay = bid
                                    break
                        if moved is None:
                            continue
                    nr, nx, margin, new_banks = moved
                    new_clear = min(clear, margin)
                    new_effort = effort + 1 + (0.18 if act in ("LEFT", "RIGHT") else 0) + (0.9 if act == "DOWN" else 0)
                    if nr == 5:
                        new_effort += 3.5 if field.strict_bank else .8
                    seq = path + (act,)
                    if nr == endpoint:
                        dist = abs(nx - goal_x) if road else 0
                        value = 440 - 2.1 * new_effort - (.28 * dist if road else 0) + min(new_clear, 6) * .45
                        if not road:
                            value += 5 if str(bay) == str(self.target) else 0
                        key = (str(bay), seq[0]) if not road else seq[0]
                        rec = {"action": seq[0], "path": seq, "goal": bay if not road else None,
                               "complete": True, "endpoint": nr, "endx": round(nx, 2),
                               "margin": round(new_clear, 2), "value": value, "steps": len(seq)}
                        if key not in finals or value > finals[key]["value"]:
                            finals[key] = rec
                        continue
                    # A forward-biased heuristic guides partial paths; it does not certify safety.
                    attraction = .08 if nr < 4 else .16 if nr < 8 else .22
                    near = abs(nx - goal_x)
                    if not road and homes:
                        near = min(abs(nx - cx) + (0 if bid == self.target else 14) for bid, cx in homes)
                    utility = nr * 18 - new_effort * .7 - near * attraction + min(new_clear, 6) * .3
                    if nr > rr:
                        rec = {"action": seq[0], "path": seq, "goal": None, "complete": False,
                               "endpoint": nr, "endx": round(nx, 2), "margin": round(new_clear, 2),
                               "value": utility - 350, "steps": len(seq)}
                        if seq[0] not in partial or rec["value"] > partial[seq[0]]["value"]:
                            partial[seq[0]] = rec
                    expanded.append((utility, (nr, nx, seq, new_clear, new_banks, new_effort)))
            if not expanded:
                break
            expanded.sort(key=lambda item: item[0], reverse=True)
            kept, seen = [], set()
            for _, node in expanded:
                key = (node[0], round(node[1] / 3), node[4], node[2][0])
                if key in seen:
                    continue
                seen.add(key)
                kept.append(node)
                if len(kept) >= 38:
                    break
            beam = kept
            if finals and depth >= min(v["steps"] for v in finals.values()) + 3:
                break
        choices = list(finals.values()) or list(partial.values())
        if not choices:
            for move in MOVES:
                q = field.transition(rr, x, move, 0, self.target, 0)
                if q is not None:
                    nr, nx, m, _ = q
                    choices.append({"action": move[0], "path": (move[0],), "goal": None,
                                    "complete": False, "endpoint": nr, "endx": nx,
                                    "margin": m, "value": nr * 10 + min(m, 4) - (move[0] == "NOOP")})
        choices.sort(key=lambda z: z["value"], reverse=True)
        self.last_search = (time.perf_counter() - started) * 1000
        return choices

    def rider_threat(self, o):
        """Risk over a short hold, separate from route clearance ranking."""
        rr = row_of(o)
        frog = mapping(o.get("frog"))
        x = num(frog.get("x"), 80)
        if not (6 <= rr <= 10 and frog.get("moving_with_platform") and integer(frog.get("state")) == 128):
            return None
        drift = self.frog_drift if self.drift_samples else 0.0
        horizon = 9
        projected = x + drift * horizon
        edge = projected < 12 or projected > 148 or (x < 8 and drift <= 0) or (x > 152 and drift >= 0)
        actor = actor_at(o)
        actor_min = 999.0
        if actor is not None and actor[0] == rr:
            for k in range(0, horizon + 1, 2):
                actor_min = min(actor_min, abs(wrap_delta(actor[1] + self.actor_velocity * k, x + drift * k)))
        contact = actor_min <= 36.0
        return {"row": rr, "x": x, "drift": drift, "projected": projected,
                "edge": edge, "actor": contact, "actor_min": actor_min,
                "threat": bool(edge or contact)}

    def emergency(self, o):
        """Exit the threatened row; do not counter-steer repeatedly on its carrier."""
        risk = self.rider_threat(o)
        if not risk or not risk["threat"]:
            self.escape_latch = None
            return None, risk
        rr, x = risk["row"], risk["x"]
        # Evaluate exits over the actual short hop/release interval. Vertical
        # movement changes the carrier row and therefore breaks persistent drift.
        quick = self.make_field(o, 5, 4, set(range(4, 11)), overlay=True)
        t = integer(o.get("tick"))
        if self.escape_latch and t <= self.escape_latch_until:
            latched = next((m for m in MOVES if m[0] == self.escape_latch), None)
            if latched is not None:
                if rr == 10 and latched[0] == "UP":
                    valid_latch = any(quick.transition(rr, x, latched, 0, bid, 0) is not None
                                      for bid in (str(self.target),) + tuple(str(h.get("slot")) for h in self.homes(o)))
                else:
                    valid_latch = quick.transition(rr, x, latched, 0, self.target, 0) is not None
                if valid_latch:
                    self.log("carrier_exit_latched", tick=t, action=self.escape_latch, risk=risk)
                    return self.escape_latch, risk
            self.escape_latch = None
        candidates = []
        if rr < 10:
            q = quick.transition(rr, x, MOVES[0], 0, self.target, 0)
            if q is not None:
                future_x = q[1] + quick.drift.get(rr + 1, 0.0) * 8
                interior = max(-40.0, min(future_x - 12.0, 148.0 - future_x))
                candidates.append((115 + min(30.0, interior) + (12 if self.escape_bias == "UP" else 0), "UP", q[2]))
        if rr > 6:
            q = quick.transition(rr, x, MOVES[3], 0, self.target, 0)
            if q is not None:
                future_x = q[1] + quick.drift.get(rr - 1, 0.0) * 8
                interior = max(-40.0, min(future_x - 12.0, 148.0 - future_x))
                candidates.append((95 + min(30.0, interior) + (12 if self.escape_bias == "DOWN" else 0), "DOWN", q[2]))
        # At the top row, UP is a home entry only if it passes the live bay guard.
        if rr == 10:
            for bid in (str(self.target),) + tuple(str(h.get("slot")) for h in self.homes(o)):
                q = quick.transition(rr, x, MOVES[0], 0, bid, 0)
                if q is not None:
                    candidates.append((150 if bid == str(self.target) else 120, "UP", q[2]))
        # Lateral corrections are an explicitly weaker last resort. They can
        # change position for one hop but cannot cancel a moving carrier's drift.
        lateral = ("LEFT", "RIGHT")
        if risk["edge"]:
            lateral = ("LEFT", "RIGHT") if risk["projected"] > x else ("RIGHT", "LEFT")
        elif actor_at(o) is not None and actor_at(o)[0] == rr:
            actor = actor_at(o)[1]
            lateral = ("LEFT", "RIGHT") if wrap_delta(actor, x) > 0 else ("RIGHT", "LEFT")
        for rank, act in enumerate(lateral):
            move = next(m for m in MOVES if m[0] == act)
            q = quick.transition(rr, x, move, 0, self.target, 0)
            if q is not None:
                bias = 12 if self.escape_bias == "INWARD" and rank == 0 else 0
                candidates.append((35 - rank + bias, act, q[2]))
        if self.escape_bias == "INWARD" and risk["actor"] and not risk["edge"]:
            for rank, act in enumerate(lateral):
                move = next(m for m in MOVES if m[0] == act)
                q = quick.transition(rr, x, move, 0, self.target, 0)
                if q is not None:
                    candidates.append((165 - rank, act, q[2]))
        if not candidates:
            self.log("escape_no_verified_exit", tick=o.get("tick"), risk=risk)
            return None, risk
        candidates.sort(key=lambda z: (z[0], z[2]), reverse=True)
        _, action, margin = candidates[0]
        self.escape_latch = action
        self.escape_latch_until = integer(o.get("tick")) + 18
        self.escape_count += 1
        self.log("carrier_row_exit", tick=o.get("tick"), action=action, risk=risk,
                 margin=round(margin, 2), count=self.escape_count)
        return action, risk

    def pick(self, o, routes):
        t, rr = integer(o.get("tick")), row_of(o)
        action, risk = self.emergency(o)
        if action is not None:
            selected = {"action": action, "complete": False, "escape": True, "risk": risk}
            return action, selected
        if not routes:
            return "NOOP", None
        chosen = routes[0]
        # The median is a legitimate observation point in standard modes. If
        # no complete river/home path currently exists, hold position and let
        # the carrier phase change instead of walking laterally along the bank.
        if rr == 5 and not any(z.get("complete") for z in routes):
            bank_field = self.make_field(o, 2, 1, set(range(4, 11)), overlay=True)
            x_now = num(mapping(o.get("frog")).get("x"), 80)
            if bank_field.transition(rr, x_now, MOVES[-1], 0, self.target, 0) is not None:
                return "NOOP", {"action": "NOOP", "complete": False, "reason": "wait_for_river_window"}
        # Road planning can otherwise alternate lateral first-hops while a long
        # complete itinerary continually re-optimizes. If an immediate forward
        # landing is certified now, take that monotone progress instead.
        if rr < 5:
            probe = self.make_field(o, 2, 1, set(range(8)), overlay=True)
            x_now = num(mapping(o.get("frog")).get("x"), 80)
            safe_up = probe.transition(rr, x_now, MOVES[0], 0, self.target, 0)
            if safe_up is not None:
                chosen = {"action": "UP", "path": ("UP",), "goal": None,
                          "complete": False, "endpoint": rr + 1,
                          "endx": round(x_now, 2), "margin": round(safe_up[2], 2),
                          "value": chosen.get("value", 0) + 2, "steps": 1,
                          "reason": "safe_forward_hop"}
        # If an itinerary has become invalid, only a complete route can replace it.
        if rr >= 5:
            available = {str(h.get("slot")) for h in self.homes(o)}
            if self.target not in available or alerted(o, self.target):
                alternatives = [z for z in routes if z.get("complete") and z.get("goal") is not None]
                if alternatives:
                    chosen = alternatives[0]
        # Keep model and route changes subordinate to current feasibility.
        if chosen.get("complete") and chosen.get("endpoint") == 11 and chosen.get("goal") is not None:
            self.target = str(chosen["goal"])
        action = chosen.get("action", "NOOP")
        live_rows = set(range(8)) if rr < 5 else set(range(4, 11))
        live = self.make_field(o, 2, 1, live_rows, overlay=True)
        move = next((m for m in MOVES if m[0] == action), MOVES[-1])
        if live.transition(rr, num(mapping(o.get("frog")).get("x"), 80), move, 0, self.target, 0) is None:
            valid = []
            for r in routes:
                m = next((z for z in MOVES if z[0] == r.get("action")), MOVES[-1])
                if live.transition(rr, num(mapping(o.get("frog")).get("x"), 80), m, 0, self.target, 0) is not None:
                    valid.append(r)
            if valid:
                chosen, action = valid[0], valid[0]["action"]
            else:
                action = "NOOP"
                chosen = dict(chosen, action=action, live_fallback=True)
        return action, chosen

    def risk_signature(self, o):
        risk = self.rider_threat(o)
        if risk is None:
            return (row_of(o), "not_carried")
        return (risk["row"], bool(risk["threat"]), bool(risk["edge"]), bool(risk["actor"]),
                "left" if risk["x"] < 24 else "right" if risk["x"] > 136 else "center")

    def state_for_jev(self, o):
        rr, x = row_of(o), num(mapping(o.get("frog")).get("x"), 80)
        actor = actor_at(o)
        risk = self.rider_threat(o)
        open_homes = self.homes(o)
        options = []
        for h in open_homes:
            slot = str(h.get("slot"))
            options.append({"bay": slot, "upstream": slot == str(self.choose_initial_target(o)),
                            "committed": slot == str(self.target), "matched_alert": alerted(o, slot),
                            "complete_route": any(z.get("complete") and str(z.get("goal")) == slot for z in self.portfolio)})
        return {"game": "Atari Frogger; cross traffic and river, enter an open bay, preserve frogs.",
                "policy": dict(self.policy),
                "epoch": list(epoch_of(o)), "phase": "road" if rr < 5 else "median" if rr == 5 else "river" if rr < 10 else "top_row",
                "row": rr, "x_band": "left" if x < 32 else "right" if x > 128 else "center",
                "timer_band": "urgent" if num(o.get("timer_s"), 30) < 6 else "normal",
                "homes": options, "committed_bay": self.target,
                "actor": None if actor is None else {"same_row": actor[0] == rr, "near_band": "near" if risk and risk.get("actor") else "far"},
                "carrier_risk": None if risk is None else {"threat": risk["threat"], "edge": risk["edge"], "actor": risk["actor"]},
                "risk_signature": self.risk_signature(o),
                "recent_loss_class": "carrier" if self.failures and 6 <= integer(self.failures[-1].get("row"), -1) <= 10 else "other" if self.failures else "none",
                "rules": "Choose a whole-crossing bay preference, not a raw joystick. Unknown home crocodile state is not proof of safety. A threatened carried frog should leave that row vertically if code confirms support; lateral movement is a last resort. Code checks every action against the latest object geometry."}

    def request_jev(self, o):
        t = integer(o.get("tick"))
        if self.s1 is not None or t < self.s1_due or getattr(self.ctx, "system1", True) is None:
            return
        if self.ctx.budget.remaining().get("system1_calls", 0) < 2:
            return
        state = self.state_for_jev(o)
        if not state["homes"]:
            return
        opts = {h["bay"]: {"what": "Use this open bay as the whole-crossing itinerary.", "facts": h} for h in state["homes"]}
        opts["KEEP"] = "Retain the current code commitment; do not change the frame action."
        route_instruction = ("Choose the open bay for the whole crossing, not the next joystick action. Prefer the upstream bay in `homes` while on road. "
                            "In river retain `committed_bay` if it remains open and unalerted; otherwise choose a listed complete_route bay. "
                            "Never infer safety from unknown crocodile data. Choose KEEP if evidence is inadequate.")
        esc = {"UP": "Choose a code-verified forward row exit when a carrier threat is present.",
               "DOWN": "Choose a code-verified backward row exit when forward is not supported.",
               "INWARD": "Use a code-verified lateral correction only if no vertical exit is safe.",
               "KEEP": "No immediate row exit is justified by the named risk facts."}
        escape_instruction = ("Classify the immediate carrier response only. If `carrier_risk.threat` is true, prefer UP, then DOWN, then INWARD, but code must verify support. "
                              "If there is no threat, choose KEEP. Never invent an action or override code geometry.")
        questions = {"bay": Choice(route_instruction, opts),
                     "bay_check": Choice(route_instruction, dict(reversed(list(opts.items())))),
                     "escape": Choice(escape_instruction, esc),
                     "escape_check": Choice(escape_instruction, dict(reversed(list(esc.items())))),
                     "review": Noul("Do the named recent loss or carrier-risk facts warrant an Astra review of the crossing protocol?")}
        self.s1_meta = {"tick": t, "epoch": epoch_of(o), "risk": state["carrier_risk"],
                        "risk_signature": state["risk_signature"], "road": row_of(o) < 5}

        def call():
            try:
                start = self.ctx.clock.now()
                ans = self.ctx.decide(state, questions, timeout_s=3.0)
                return ans, self.ctx.clock.now() - start
            except (BudgetExceeded, EpisodeOver, ModelUnavailable):
                raise
            except (TimeoutError, RuntimeError, TypeError, ValueError):
                return None, 0.0
        self.s1 = self.ctx.spawn(call)
        self.s1_due = t + (24 if self.rider_threat(o) and self.rider_threat(o).get("threat") else 90)

    def request_astra(self, o):
        t = integer(o.get("tick"))
        if self.s2 is not None or t < self.s2_due or not self.failures or getattr(self.ctx, "system2", True) is None:
            return
        if self.ctx.budget.remaining().get("system2_calls", 0) < 2:
            return
        evidence = {"policy": self.policy, "losses": list(self.failures)[-2:], "level": o.get("level"),
                    "score": o.get("score"), "target": self.target, "route_ms": round(self.last_search, 2)}
        prompt = ("Review this Frogger controller's protocol, never produce a frame action. Its route search is code-owned. "
                  "The new safety hypothesis is that repeated lateral counter-steering on a moving carrier can oscillate; prefer a verified vertical row exit for an imminent carrier hazard, with lateral movement only as a last resort. "
                  "Choose bounded JSON only: {\"itinerary\":\"upstream|nearest\",\"escape\":\"vertical|interior\"}. Do not disable geometry checks or invent RAM meanings. Evidence="
                  + json.dumps(evidence, separators=(",", ":")))
        self.s2_meta = {"revision": self.revision, "tick": t}
        session = self.session

        def call():
            try:
                return self.ctx.think(prompt, session_id=session, timeout_s=100.0)
            except (BudgetExceeded, EpisodeOver, ModelUnavailable):
                raise
            except (TimeoutError, RuntimeError, TypeError, ValueError):
                return None
        self.s2 = self.ctx.spawn(call)
        self.s2_due = t + 3600

    @staticmethod
    def choice_pair(answers, first, second, threshold=.70):
        a, b = answers.get(first), answers.get(second)
        v = getattr(a, "value", None)
        if not isinstance(v, str) or v != getattr(b, "value", None):
            return None, 0.0
        p, q = getattr(a, "probabilities", None), getattr(b, "probabilities", None)
        if not isinstance(p, dict) or not isinstance(q, dict):
            return None, 0.0
        chance = min(num(p.get(v)), num(q.get(v)))
        return (v if chance >= threshold else None), chance

    def poll(self, o):
        t = integer(o.get("tick"))
        if self.s1 is not None and self.s1.done():
            f, self.s1 = self.s1, None
            try:
                result = f.result()
            except (BudgetExceeded, EpisodeOver, ModelUnavailable):
                raise
            except (TimeoutError, RuntimeError, TypeError, ValueError):
                result = None
            if result and isinstance(result, tuple) and isinstance(result[0], dict):
                answers, latency = result
                meta = self.s1_meta or {}
                fresh = (epoch_of(o) == meta.get("epoch") and t - meta.get("tick", t) <= 75
                         and tuple(self.risk_signature(o)) == tuple(meta.get("risk_signature", ())))
                bay, bp = self.choice_pair(answers, "bay", "bay_check")
                valid = {str(h.get("slot")) for h in self.homes(o)}
                if fresh and meta.get("road") and row_of(o) < 5 and bay in valid:
                    self.target_mode = "NEAREST" if bay != str(self.choose_initial_target(o)) else "UPSTREAM"
                    self.target = bay
                    self.log("jev_itinerary", tick=t, bay=bay, probability=bp, latency_s=latency)
                esc, ep = self.choice_pair(answers, "escape", "escape_check", .75)
                if fresh and esc in ("UP", "DOWN", "INWARD", "KEEP"):
                    self.escape_bias = esc
                    self.log("jev_escape_preference", tick=t, preference=esc, probability=ep)
                if fresh and num(getattr(answers.get("review"), "value", 0)) >= .85:
                    self.request_astra(o)
        if self.s2 is not None and self.s2.done():
            f, self.s2 = self.s2, None
            try:
                result = f.result()
            except (BudgetExceeded, EpisodeOver, ModelUnavailable):
                raise
            except (TimeoutError, RuntimeError, TypeError, ValueError):
                result = None
            if result is not None:
                self.session = getattr(result, "session_id", None)
                parsed = fragment(getattr(result, "text", ""))
                if integer((self.s2_meta or {}).get("revision"), -1) == self.revision:
                    if parsed.get("itinerary") in ("upstream", "nearest") and parsed.get("escape") in ("vertical", "interior"):
                        self.policy.update(parsed)
                        self.target_mode = "NEAREST" if parsed["itinerary"] == "nearest" else "UPSTREAM"
                        self.escape_bias = "INWARD" if parsed["escape"] == "interior" else "UP"
                        self.revision += 1
                        self.s1_due = t
                        self.log("astra_policy_compiled", tick=t, policy=self.policy)
                    else:
                        self.log("astra_policy_rejected", tick=t)

    def run_episode(self, game):
        del game
        while not self.ctx.done:
            o = mapping(self.ctx.observe())
            if not o:
                self.ctx.act("NOOP")
                self.ctx.wait_for_tick()
                continue
            self.observe(o)
            self.poll(o)
            t, rr = integer(o.get("tick")), row_of(o)
            frog = mapping(o.get("frog"))
            living = integer(frog.get("state")) == 128 and -1 <= rr <= 10
            action = "NOOP"
            if self.press is not None:
                moved = self.origin is not None and (rr != self.origin[0] or abs(num(frog.get("x")) - self.origin[1]) >= 7)
                if moved or not living or t - self.press_at >= 4:
                    self.log("pulse_release", tick=t, action=self.press, moved=moved)
                    self.press, self.origin = None, None
                    self.ready_at = max(self.ready_at, t + 2)
                else:
                    action = self.press
            if self.press is None and living and t >= self.ready_at:
                self.portfolio = self.route_search(o)
                latest = mapping(self.ctx.observe())
                if latest and epoch_of(latest) == epoch_of(o) and row_of(latest) == rr:
                    action, selected = self.pick(latest, self.portfolio)
                    t = integer(latest.get("tick"), t)
                    self.last_plan = {"tick": t, "row": rr, "x": mapping(latest.get("frog")).get("x"),
                                      "target": self.target, **(selected or {"action": action})}
                    if action != "NOOP" or t - self.last_log >= 60:
                        self.log("viability_decision", tick=t, plan=self.last_plan, search_ms=round(self.last_search, 2))
                        self.last_log = t
                    lag = max(0, min(2, t - integer(o.get("tick"), t)))
                    # Route-search layers advance in five-frame quanta. Match passive wait
                    # cadence to that model; shorter receding NOOPs chase a forecast gap that
                    # has not yet arrived and can starve forward movement.
                    self.ready_at = t + (max(4, 5 - lag) if action != "NOOP" else 5)
                    if action != "NOOP":
                        self.press, self.press_at = action, t
                        self.origin = (rr, num(mapping(latest.get("frog")).get("x"), 80))
                self.request_jev(o)
            self.request_astra(o)
            self.last_action = action
            self.ctx.act(action)
            self.ctx.wait_for_tick()


def build(ctx):
    return Frogger(ctx)
