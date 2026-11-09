"""Frogger controller: conservative forecast search with model-selected intent.

The game-specific arithmetic stays local; models choose a short-lived control
posture over independently generated, executable route alternatives.
"""
from __future__ import annotations

import json
import math
import time
from collections import deque

from metaharness.models.jev import Choice, Noul
from metaharness.core.contract import BudgetExceeded, EpisodeOver, ModelUnavailable

MOVES = (("UP", 1, 0), ("LEFT", 0, -8), ("RIGHT", 0, 8), ("NOOP", 0, 0), ("DOWN", -1, 0))
COPIES = {0: (0,), 1: (0, 16), 2: (0, 32), 3: (0, 16, 32),
          4: (0, 64), 5: (0,), 6: (0, 32, 64), 7: (0,)}
WIDTH = {0: 8, 1: 8, 2: 8, 3: 8, 4: 8, 5: 16, 6: 8, 7: 32}
SPEEDS = (1/16, 1/8, 1/4, 1/2, 1.0, 2.0)


def number(v, default=0.0):
    try:
        x = float(v)
        return x if math.isfinite(x) else default
    except (TypeError, ValueError, OverflowError):
        return default


def integer(v, default=0):
    return int(number(v, default))


def decode(value):
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except (ValueError, TypeError):
            return {}
    return value if isinstance(value, dict) else {}


def row_of(o):
    value = integer(o.get("frog", {}).get("row"), 255)
    return -1 if value == 255 else value


def episode_key(o):
    return (integer(o.get("deaths")), integer(o.get("frogs_home")), integer(o.get("level"), 1))


def wrap_delta(new, old):
    return (new - old + 80.0) % 160.0 - 80.0


def joined(intervals):
    result = []
    for lo, hi in sorted(intervals):
        if result and lo <= result[-1][1] + 0.01:
            result[-1][1] = max(result[-1][1], hi)
        else:
            result.append([lo, hi])
    return result


def home_alarm(o, slot):
    fields = o.get("ram_fields", {})
    # This is a matched signature only; unknown home-crocodile status remains unknown.
    return bool(integer(fields.get("collision_e3")) & 128) and integer(fields.get("collision_row_e5"), -99) == integer(slot, -2) + 1


def available_homes(o):
    return [h for h in o.get("homes", [])
            if not h.get("occupied") and h.get("available", True)
            and h.get("crocodile") not in (True, "present", "dangerous")
            and not home_alarm(o, h.get("slot"))]


class Forecast:
    """Short time layers in frog-left coordinates, recomputed from each RAM view."""
    def __init__(self, owner, obs, depth):
        self.owner, self.obs, self.depth = owner, obs, depth
        self.drift = {}
        self.layers = [dict() for _ in range(depth + 1)]
        actors = self._actors(obs)
        raw_lanes = list(obs.get("road", [])) + list(obs.get("river", []))
        row_items = {}
        for lane in raw_lanes:
            rr = integer(lane.get("row"), -99)
            items = []
            for obj in lane.get("objects", []):
                n = integer(obj.get("nusiz")) & 7
                width = integer(obj.get("width_px_estimate"), WIDTH[n])
                if n == 7 and integer(obj.get("shape_ptr")) in (163, 170):
                    width = 24
                width = max(8, min(40, width))
                ident = str(obj.get("id", (rr, obj.get("index"))))
                velocity = owner.object_speed(ident, obj, lane, obs)
                safe = (not obj.get("hazard") and obj.get("kind") in ("log", "turtle"))
                items.append((number(obj.get("x")), velocity, COPIES.get(n, (0,)), width, safe, ident))
            row_items[rr] = items
            if rr >= 6:
                solids = [z[1] for z in items if z[4]]
                self.drift[rr] = sum(solids) / len(solids) if solids else 0.0
            else:
                self.drift[rr] = 0.0
        self.row_items = row_items
        raw = obs.get("raw_ram", [])
        if (integer(obs.get("level"), 1) >= 4 or owner.bank_alarm) and isinstance(raw, list) and len(raw) > 16:
            bank_x = number(raw[16])
            row_items[5] = [(bank_x, owner.bank_speed(obs, bank_x), (0,), 16, False, "median_track")]
            self.drift[5] = 0.0
        self.actor_tracks = actors
        for layer_index in range(depth + 1):
            elapsed = layer_index * owner.hop_frames
            if 5 in row_items:
                self.layers[layer_index][5] = self._road_safe(5, row_items[5], elapsed)
            for rr, items in row_items.items():
                if rr <= 4:
                    self.layers[layer_index][rr] = self._road_safe(rr, items, elapsed)
                elif rr >= 6:
                    self.layers[layer_index][rr] = self._river_margin(rr, items, elapsed)
            for rr, ax, av in actors:
                base = self.layers[layer_index].get(rr)
                if base is None:
                    continue
                ahead = ax + av * elapsed
                after = ax + av * (elapsed + owner.hop_frames)
                lo = min(ahead, after) - 11.0
                hi = max(ahead, after) + 19.0
                start, end = max(0, int(math.floor(lo))), min(160, int(math.ceil(hi)))
                if end >= start:
                    for x in range(start, end + 1):
                        base[x] = -1.0

    @staticmethod
    def _position(x, v, ticks):
        return (x + v * ticks) % 160.0

    def _road_safe(self, rr, items, elapsed):
        safe = [18.0] * 161
        # Reserve the full swept vehicle interval during the next one-hop dwell.
        for x, velocity, copies, width, _, _ in items:
            p0 = x + velocity * (elapsed + 1)
            p1 = x + velocity * (elapsed + self.owner.hop_frames + 2)
            # An object wrapping at an edge occupies copies on the opposite edge.
            for offset in copies:
                a, b = p0 + offset, p1 + offset
                low, high = min(a, b) - 8.0 - self.owner.road_pad, max(a, b) + width + self.owner.road_pad
                for wrap in (-320, -160, 0, 160, 320):
                    first = max(0, int(math.floor(low + wrap)))
                    last = min(160, int(math.ceil(high + wrap)))
                    for px in range(first, last + 1):
                        safe[px] = -1.0
        return safe

    def _river_margin(self, rr, items, elapsed):
        intervals = []
        for x, velocity, copies, width, can_ride, _ in items:
            if not can_ride:
                continue
            p = self._position(x, velocity, elapsed + 1)
            for offset in copies:
                start = p + offset
                for wrap in (-320, -160, 0, 160, 320):
                    intervals.append((start + wrap, start + width + wrap))
        output = [-1.0] * 161
        for lo_obj, hi_obj in joined(intervals):
            # Frog body overlaps the carrier; leave a modest inward buffer.
            lo, hi = lo_obj + self.owner.river_pad - 4.0, hi_obj - self.owner.river_pad - 4.0
            first, last = max(0, int(math.ceil(lo))), min(160, int(math.floor(hi)))
            for px in range(first, last + 1):
                output[px] = max(output[px], min(px - lo, hi - px))
        return output

    def margin(self, rr, x, layer=0):
        if x < 4 or x > 148:
            return -100.0
        if rr == -1:
            return 18.0
        if rr == 5 and rr not in self.layers[min(max(layer, 0), self.depth)]:
            return 18.0
        if rr < 0 or rr > 10:
            return -100.0
        values = self.layers[min(max(layer, 0), self.depth)].get(rr)
        if values is None:
            return -100.0
        return values[max(0, min(160, int(round(x))))]

    def move(self, rr, x, action, layer, slot=None, bank_count=0):
        lookup = next((m for m in MOVES if m[0] == action), MOVES[3])
        nr, nx = rr + lookup[1], x + lookup[2]
        if not 4 <= nx <= 148 or not -1 <= nr <= 11:
            return None
        if nr == 11:
            if action != "UP" or slot is None:
                return None
            h = next((z for z in available_homes(self.obs) if str(z.get("slot")) == str(slot)), None)
            if h is None:
                return None
            center = number(h.get("x_center"), 80)
            clearance = min(nx - (center - 4.0), center - nx)
            return (nr, nx, clearance, 0)
        count = bank_count + 1 if nr == 5 else 0
        # At higher levels do not let the frog use the median as a lateral parking lot.
        if integer(self.obs.get("level"), 1) >= 4 and nr == 5 and rr >= 5:
            return None
        if count > 3:
            return None
        clearance = self.margin(nr, nx, layer)
        if clearance < 0:
            return None
        end_x = nx + self.drift.get(nr, 0.0) * self.owner.hop_frames if nr >= 6 else nx
        if nr >= 6:
            trailing = self.margin(nr, end_x, min(layer + 1, self.depth))
            if trailing < 0:
                return None
            clearance = min(clearance, trailing)
        return nr, end_x, clearance, count

    def _actors(self, obs):
        fly = obs.get("fly", {})
        fields = obs.get("ram_fields", {})
        row = integer(fly.get("row") if fly.get("row") is not None else fields.get("fly_row_ae"), 255)
        active = bool(fly.get("present")) or bool(integer(fields.get("bonus_flags_c7")) & 128)
        if not active or not 0 <= row <= 10:
            return []
        x = number(fly.get("x") if fly.get("x") is not None else fields.get("fly_x_b2"), 80)
        velocity = self.owner.actor_speed(obs, row, x)
        return [(row, x, velocity)]


class Harness:
    def __init__(self, ctx):
        self.ctx = ctx
        self.hop_frames = 6
        self.road_pad = 1.5
        self.river_pad = 2.0
        self.previous = None
        self.position_history = {}
        self.actor_history = deque(maxlen=32)
        self.bank_history = deque(maxlen=64)
        self.bank_velocity = .25
        self.bank_alarm = False
        self.velocities = {}
        self.actor_velocity = .5
        self.failures = deque(maxlen=12)
        self.current_epoch = None
        self.crossing_target = None
        self.intent = "BALANCED"
        self.intent_until = -1
        self.intent_meta = None
        self.jev_future = None
        self.jev_meta = None
        self.next_jev = 0
        self.astra_future = None
        self.astra_meta = None
        self.next_astra = 0
        self.session = None
        self.pressed = None
        self.press_tick = -100
        self.origin = None
        self.ready_tick = 3
        self.routes = []
        self.last_route = None
        self.last_log_tick = -100
        self.last_lateral_tick = -100
        self.total_deaths_seen = 0
        self.current_policy = "BALANCED"

    def log(self, name, **fields):
        self.ctx.log(name, **fields)

    def object_speed(self, ident, obj, lane, obs):
        t, x = integer(obs.get("tick")), number(obj.get("x"))
        history = self.position_history.setdefault(ident, deque(maxlen=64))
        if not history or history[-1][0] != t:
            history.append((t, x))
        direction = obj.get("direction", lane.get("direction"))
        sign = -1.0 if direction == "left" else 1.0
        estimate = sign / max(1, integer(lane.get("activity_mask"), 3) + 1)
        old = next((q for q in history if 20 <= t - q[0] <= 56), None)
        if old:
            measured = wrap_delta(x, old[1]) / max(1, t - old[0])
            if measured * sign > 0 and abs(measured) <= 2:
                estimate = sign * min(SPEEDS, key=lambda s: abs(s - abs(measured)))
        if number(obj.get("velocity_px_per_tick")) and not old:
            reported = number(obj.get("velocity_px_per_tick"))
            if abs(reported) <= 2 and reported * sign >= 0:
                estimate = reported
        self.velocities[ident] = estimate
        return estimate

    def bank_speed(self, obs, x):
        tick = integer(obs.get("tick"))
        if not self.bank_history or self.bank_history[-1][0] != tick:
            self.bank_history.append((tick, x))
        old = next((item for item in self.bank_history if 12 <= tick - item[0] <= 40), None)
        if old:
            measured = wrap_delta(x, old[1]) / max(1, tick - old[0])
            if .05 <= measured <= 2.0:
                self.bank_velocity = min((.125, .25, .5, 1.0), key=lambda q: abs(q - measured))
        return self.bank_velocity

    def actor_speed(self, obs, rr, x):
        t = integer(obs.get("tick"))
        if not self.actor_history or self.actor_history[-1][0] != t:
            self.actor_history.append((t, rr, x))
        old = next((q for q in self.actor_history if q[1] == rr and 8 <= t - q[0] <= 32), None)
        if old:
            v = (x - old[2]) / max(1, t - old[0])
            if 0.05 <= abs(v) <= 2:
                self.actor_velocity = min((-1.0, -.5, -.25, .25, .5, 1.0), key=lambda a: abs(a-v))
            elif v == 0:
                self.actor_velocity = 0.0
        return self.actor_velocity

    def reset_epoch(self, obs):
        key = episode_key(obs)
        t = integer(obs.get("tick"))
        if key != self.current_epoch:
            if self.current_epoch is not None:
                self.log("crossing_epoch", tick=t, before=self.current_epoch, after=key,
                         target=self.crossing_target, prior_route=self.last_route)
            self.current_epoch = key
            self.crossing_target = None
            self.intent = "BALANCED"
            self.intent_until = -1
            self.pressed = None
            self.origin = None
            self.ready_tick = t + 3
            self.next_jev = t
            self.actor_history.clear()
        # Pick an initial target following the current oncoming top-carrier flow.
        if self.crossing_target is None:
            homes = available_homes(obs)
            top = next((lane for lane in obs.get("river", []) if integer(lane.get("row")) == 10), {})
            sign = -1 if top.get("direction") == "left" else 1
            if homes:
                choice = min(homes, key=lambda h: sign * number(h.get("x_center"), 80))
                self.crossing_target = str(choice.get("slot"))
        self.previous = obs

    def update(self, obs):
        t, r = integer(obs.get("tick")), row_of(obs)
        old = self.previous
        if old is not None and integer(obs.get("deaths")) > integer(old.get("deaths")):
            self.bank_alarm = self.bank_alarm or row_of(old) == 5
            oldrow = row_of(old)
            evidence = {"tick": t, "old_row": oldrow, "old_frog": old.get("frog"),
                        "new_frog": obs.get("frog"), "old_score": old.get("score"),
                        "level": old.get("level"), "last_action": self.last_route.get("action") if self.last_route else None,
                        "last_route": self.last_route, "target": self.crossing_target,
                        "old_carrier": self.carrier_summary(old)}
            self.failures.append(evidence)
            self.log("life_loss_witness", **evidence)
            # Defer policy changes to a diagnosis over repeated evidence, not one noisy hit.
            self.ready_tick = t + 3
        if old is not None and (r != row_of(old) or obs.get("score") != old.get("score")):
            self.log("frog_progress", tick=t, before=old.get("frog"), after=obs.get("frog"),
                     score=obs.get("score"), homes=obs.get("frogs_home"))
        if self.intent_until < t:
            self.intent = "BALANCED"
        self.reset_epoch(obs)
        self.previous = obs

    def carrier_summary(self, obs):
        rr, x = row_of(obs), number(obs.get("frog", {}).get("x"), 80)
        lane = next((l for l in obs.get("river", []) if integer(l.get("row")) == rr), {})
        nearby = []
        for z in lane.get("objects", []):
            if z.get("kind") in ("log", "turtle", "diving_turtle"):
                nearby.append({"kind": z.get("kind"), "x": z.get("x"), "hazard": bool(z.get("hazard")),
                               "nusiz": z.get("nusiz"), "id": z.get("id")})
        return {"row": rr, "x": x, "moving_with_platform": bool(obs.get("frog", {}).get("moving_with_platform")),
                "objects": nearby}

    def target_x(self, obs):
        match = next((h for h in available_homes(obs) if str(h.get("slot")) == str(self.crossing_target)), None)
        return number(match.get("x_center"), 80) if match else 80.0

    def solve(self, obs):
        started = time.perf_counter()
        start_row, start_x = row_of(obs), number(obs.get("frog", {}).get("x"), 80)
        if not -1 <= start_row <= 10:
            return [], None
        launch = start_row < 5
        endpoint = 7 if launch else 11
        depth = 21
        world = Forecast(self, obs, depth + 1)
        homes = available_homes(obs)
        if not homes:
            return [], world
        open_centers = [(str(h.get("slot")), number(h.get("x_center"), 80)) for h in homes]
        wanted = self.target_x(obs)
        if launch:
            # Search an itinerary through the first safe carrier rows, not a long median wait.
            preferred_center = wanted
        else:
            preferred_center = wanted
        # Node fields: row, frog-left x, first move, path, minimum clearance, cost, median visits.
        fringe = [(start_row, start_x, None, (), 18.0, 0.0, 0)]
        completed = {}
        partial = {}
        for layer in range(depth):
            candidates = []
            for rr, xx, first, path, bottleneck, cost, bank_visits in fringe:
                for action, dr, dx in MOVES:
                    nr = rr + dr
                    if launch and nr > endpoint:
                        continue
                    if not launch and start_row >= 5 and nr < 5:
                        continue
                    if nr < -1 or nr > endpoint:
                        continue
                    if launch and nr == -1 and rr == -1 and action == "DOWN":
                        continue
                    if nr == 11:
                        if action != "UP" or rr != 10:
                            continue
                        for slot, center in open_centers:
                            moved = world.move(rr, xx, action, layer, slot, bank_visits)
                            if moved is None:
                                continue
                            _, endx, clearance, _ = moved
                            nm = min(bottleneck, clearance)
                            seq = path + (action,)
                            first_action = first or action
                            length_cost = cost + 1
                            home_match = 5.0 if slot == self.crossing_target else 0.0
                            value = 520.0 - 2.0 * length_cost + min(nm, 6.0) * .7 + home_match
                            route = {"label": "HOME_" + slot, "action": first_action, "path": seq,
                                     "value": value, "margin": round(nm, 2), "goal": slot,
                                     "complete": True, "endpoint": 11, "endx": round(endx, 2), "steps": len(seq)}
                            key = (slot, first_action)
                            if key not in completed or route["value"] > completed[key]["value"]:
                                completed[key] = route
                        continue
                    moved = world.move(rr, xx, action, layer, None, bank_visits)
                    if moved is None:
                        continue
                    nr, nx, clearance, banks = moved
                    new_margin = min(bottleneck, clearance)
                    seq = path + (action,)
                    first_action = first or action
                    lateral_cost = .18 if action in ("LEFT", "RIGHT") else 0.0
                    reverse_cost = .9 if action == "DOWN" else 0.0
                    new_cost = cost + 1.0 + lateral_cost + reverse_cost
                    if nr == endpoint:
                        if endpoint == 7:
                            approach = abs(nx - preferred_center)
                            value = 520.0 - new_cost * 2.0 + min(new_margin, 6.0) * .7 - approach * .20
                            candidate = {"label": "LAUNCH_" + first_action, "action": first_action,
                                         "path": seq, "value": value, "margin": round(new_margin, 2),
                                         "goal": self.crossing_target, "complete": True, "endpoint": 7,
                                         "endx": round(nx, 2), "steps": len(seq)}
                            key = first_action
                            if key not in completed or value > completed[key]["value"]:
                                completed[key] = candidate
                        else:
                            # A home endpoint is generated only by the explicit UP case above.
                            continue
                    ahead_x = nx + world.drift.get(nr, 0.0) * self.hop_frames if nr >= 6 else nx
                    closeness = min(abs(ahead_x - center) for _, center in open_centers)
                    score = nr * 24.0 - new_cost * .55 - closeness * (.21 if nr >= 8 else .035)
                    score += min(new_margin, 6.0) * .25 - banks * 3.0
                    if nr > start_row:
                        route = {"label": "PROGRESS_" + first_action, "action": first_action,
                                 "path": seq, "value": score - 450.0, "margin": round(new_margin, 2),
                                 "goal": self.crossing_target, "complete": False,
                                 "endpoint": nr, "endx": round(ahead_x, 2), "steps": len(seq)}
                        if first_action not in partial or route["value"] > partial[first_action]["value"]:
                            partial[first_action] = route
                    candidates.append((score, (nr, ahead_x, first_action, seq, new_margin, new_cost, banks)))
            if not candidates:
                break
            candidates.sort(key=lambda item: item[0], reverse=True)
            next_fringe, seen = [], set()
            for _, node in candidates:
                key = (node[0], int(round(node[1] / 3.0)), node[2], node[6])
                if key in seen:
                    continue
                seen.add(key)
                next_fringe.append(node)
                if len(next_fringe) >= 40:
                    break
            fringe = next_fringe
            if completed and layer > min(v["steps"] for v in completed.values()) + 4:
                break
        routes = list(completed.values()) or list(partial.values())
        if not routes:
            for action, _, _ in MOVES:
                result = world.move(start_row, start_x, action, 0, None, 0)
                if result:
                    nr, nx, margin, _ = result
                    routes.append({"label": "SAFE_" + action, "action": action, "path": (action,),
                                   "value": nr * 10.0 + min(margin, 4.0) - (1 if action != "NOOP" else 0),
                                   "margin": margin, "goal": self.crossing_target,
                                   "complete": False, "endpoint": nr, "endx": nx, "steps": 1})
        routes.sort(key=lambda z: z["value"], reverse=True)
        self.routes = routes
        self.last_search_ms = (time.perf_counter() - started) * 1000
        return routes, world

    def route_state(self, obs, routes, world):
        rr, x = row_of(obs), number(obs.get("frog", {}).get("x"), 80)
        predicted = x
        if 6 <= rr <= 10 and world is not None:
            predicted += world.drift.get(rr, 0.0) * 18.0
        edge = "near" if predicted < 22 or predicted > 138 else "interior"
        if predicted < 10 or predicted > 150:
            edge = "critical"
        lane = next((z for z in obs.get("river", []) if integer(z.get("row")) == rr), {})
        carrier_kinds = [z.get("kind") for z in lane.get("objects", []) if z.get("kind") in ("log", "turtle", "diving_turtle")]
        on_diver = "diving_turtle" in carrier_kinds and any(z.get("hazard") for z in lane.get("objects", []) if z.get("kind") == "diving_turtle")
        options = []
        for z in routes[:6]:
            action = z.get("action", "NOOP")
            options.append({"route": z.get("label"), "first_hop": action,
                            "destination": "home" if z.get("endpoint") == 11 else "river" if z.get("endpoint", 0) >= 6 else "road_or_median",
                            "search_endpoint_reached": bool(z.get("complete")),
                            "margin": "wide" if number(z.get("margin")) >= 3 else "thin",
                            "action_effect": {"UP": "advance", "DOWN": "retreat", "LEFT": "shift_left",
                                              "RIGHT": "shift_right", "NOOP": "wait"}.get(action, "wait"),
                            "committed_bay": z.get("goal")})
        return {"game": "Frogger; legal actions are five labels; code rechecks all moves before posting.",
                "observation": {"tick": integer(obs.get("tick")), "zone": obs.get("frog", {}).get("zone"),
                                "row": rr, "platform_carry": bool(obs.get("frog", {}).get("moving_with_platform")),
                                "current_x_bucket": "left" if x < 50 else "center" if x < 110 else "right",
                                "projected_18_tick_edge": edge, "diving_carrier_on_row": bool(on_diver),
                                "timer_bucket": "urgent" if number(obs.get("timer_s"), 30) < 8 else "normal",
                                "level": integer(obs.get("level"), 1), "open_bay_count": len(available_homes(obs)),
                                "target_bay": self.crossing_target,
                                "actor_on_frog_row": any(z.get("present") and integer(z.get("row"), 255) == rr for z in [obs.get("fly", {})]),
                                "current_model_intent": self.intent},
                "candidate_routes": options,
                "rules": "Choose a planning posture, not a joystick. Every posted hop is independently collision/support checked from the newest observation. Unknown home-crocodile state is not safe evidence."}

    def request_jev(self, obs, routes, world):
        t = integer(obs.get("tick"))
        if self.jev_future is not None or t < self.next_jev or not routes:
            return
        if getattr(self.ctx, "system1", True) is None or self.ctx.budget.remaining().get("system1_calls", 0) < 1:
            return
        state = self.route_state(obs, routes, world)
        choices = {
            "PROTECT": "Favor routes with the broadest forecast support and low exposure. Never select an unsafe route.",
            "ADVANCE": "Favor fewer hops and forward/home progress while retaining hard safety checks.",
            "ESCAPE": "If carrying exposure is evident, favor a checked route that leaves the current row. Never jump without a certified landing.",
            "ALIGN": "Favor routes that align with the committed open bay; do not change bays without a meaningful route advantage.",
            "UNSURE": "Evidence does not justify a different posture; keep the deterministic balanced plan."}
        instruction = ("Select the best CONTROL POSTURE for the current Frogger snapshot and candidate_routes. "
                       "PROTECT is for thin support, projected edge, diving or actor exposure; "
                       "ADVANCE favors a short route reaching its local endpoint (row 7 on a road launch, or an open home from the river); "
                       "ESCAPE only when the current ride is becoming unsafe; ALIGN when reaching the home approach is the remaining task. "
                       "Options describe how code should rank its freshly recomputed certified routes, not a delayed action. UNSURE if unclear.")
        question_set = {"posture": Choice(instruction, choices),
                        "posture_reverse": Choice(instruction, dict(reversed(list(choices.items())))),
                        "threat": Noul("Does this observation explicitly suggest near-term lethal exposure to the active frog from a car, unsupported water, diving carrier, actor, or screen edge?")}
        self.jev_meta = {"tick": t, "epoch": episode_key(obs), "phase": "road" if row_of(obs) < 5 else "river"}

        def ask():
            try:
                started = self.ctx.clock.now()
                answers = self.ctx.decide(state, question_set, timeout_s=3.0)
                return answers, max(0.0, self.ctx.clock.now() - started)
            except (BudgetExceeded, EpisodeOver, ModelUnavailable):
                raise
            except (TimeoutError, RuntimeError, TypeError, ValueError):
                return None, 0.0
        self.jev_future = self.ctx.spawn(ask)
        self.next_jev = t + 54

    def request_astra(self, obs):
        t = integer(obs.get("tick"))
        if self.astra_future is not None or t < self.next_astra or len(self.failures) < 2:
            return
        if getattr(self.ctx, "system2", True) is None or self.ctx.budget.remaining().get("system2_calls", 0) < 1:
            return
        recent = list(self.failures)[-3:]
        summary = {"score": obs.get("score"), "level": obs.get("level"), "target": self.crossing_target,
                   "failures": recent, "current_policy": {"risk": self.current_policy,
                   "road_pad": self.road_pad, "river_pad": self.river_pad}}
        prompt = ("You are the slow Frogger policy reviewer, not the frame controller. Code estimates object motion, "
                  "searches separated sprite copies and verifies every hop; Jev chooses a transient posture among PROTECT, "
                  "ADVANCE, ESCAPE and ALIGN. Diagnose repeated witnessed losses and choose only a conservative reusable policy. "
                  "Return one JSON object, no prose, exactly these optional keys: risk ('balanced'|'conservative'|'progress'), "
                  "road_margin ('standard'|'wide'), river_margin ('standard'|'wide'). Do not infer a crocodile is absent. "
                  "Change a setting only when the supplied failure evidence supports it; otherwise return {}. Evidence="
                  + json.dumps(summary, separators=(",", ":")))
        self.astra_meta = {"tick": t, "epoch": episode_key(obs), "failure_count": len(self.failures)}
        session = self.session

        def advise():
            try:
                return self.ctx.think(prompt, session_id=session, timeout_s=90.0)
            except (BudgetExceeded, EpisodeOver, ModelUnavailable):
                raise
            except (TimeoutError, RuntimeError, TypeError, ValueError):
                return None
        self.astra_future = self.ctx.spawn(advise)
        self.next_astra = t + 1800

    @staticmethod
    def _probability(answer, key):
        probs = getattr(answer, "probabilities", None)
        return number(probs.get(key), 0.0) if isinstance(probs, dict) else 0.0

    def poll_models(self, obs):
        t = integer(obs.get("tick"))
        if self.jev_future is not None and self.jev_future.done():
            future, self.jev_future = self.jev_future, None
            try:
                result, latency = future.result()
            except (BudgetExceeded, EpisodeOver, ModelUnavailable):
                raise
            except Exception:
                result, latency = None, 0.0
            if isinstance(result, dict):
                a, b = result.get("posture"), result.get("posture_reverse")
                value = getattr(a, "value", None)
                p = min(self._probability(a, value), self._probability(b, value)) if isinstance(value, str) else 0.0
                meta = self.jev_meta or {}
                same = (episode_key(obs) == meta.get("epoch") and
                        ("road" if row_of(obs) < 5 else "river") == meta.get("phase"))
                age = t - integer(meta.get("tick"), t)
                if (same and 0 <= age <= 72 and value == getattr(b, "value", None)
                        and value in ("PROTECT", "ADVANCE", "ESCAPE", "ALIGN", "UNSURE") and p >= .60):
                    before = self.intent
                    self.intent = "BALANCED" if value == "UNSURE" else value
                    self.intent_until = t + 72
                    self.log("jev_posture_applied", tick=t, before=before, after=self.intent,
                             probability=p, age=age, latency_s=latency)
                else:
                    self.log("jev_posture_rejected", tick=t, value=value, age=age, same_epoch=same, probability=p)
                threat = getattr(result.get("threat"), "value", 0)
                if number(threat) >= .85 and len(self.failures) >= 2:
                    self.request_astra(obs)
        if self.astra_future is not None and self.astra_future.done():
            future, self.astra_future = self.astra_future, None
            try:
                result = future.result()
            except (BudgetExceeded, EpisodeOver, ModelUnavailable):
                raise
            except Exception:
                result = None
            if result is not None:
                self.session = getattr(result, "session_id", None)
                text = getattr(result, "text", "")
                proposal = {}
                if isinstance(text, str):
                    try:
                        left, right = text.index("{"), text.rindex("}") + 1
                        proposal = decode(text[left:right])
                    except (ValueError, TypeError):
                        proposal = {}
                allowed = {"risk": ("balanced", "conservative", "progress"),
                           "road_margin": ("standard", "wide"), "river_margin": ("standard", "wide")}
                changes = {k: v for k, v in proposal.items() if k in allowed and v in allowed[k]}
                meta = self.astra_meta or {}
                if changes and episode_key(obs) == meta.get("epoch"):
                    before = {"risk": self.current_policy, "road_pad": self.road_pad, "river_pad": self.river_pad}
                    if "risk" in changes:
                        self.current_policy = changes["risk"].upper()
                    self.road_pad = 2.5 if changes.get("road_margin") == "wide" else (1.5 if changes.get("road_margin") == "standard" else self.road_pad)
                    self.river_pad = 3.0 if changes.get("river_margin") == "wide" else (2.0 if changes.get("river_margin") == "standard" else self.river_pad)
                    self.log("astra_policy_compiled", tick=t, before=before, changes=changes,
                             policy={"risk": self.current_policy, "road_pad": self.road_pad, "river_pad": self.river_pad})
                else:
                    self.log("astra_policy_rejected", tick=t)

    def select_route(self, routes, obs):
        if not routes:
            return "NOOP", None
        t, rr = integer(obs.get("tick")), row_of(obs)
        posture = self.intent if t <= self.intent_until else "BALANCED"
        risk = self.current_policy
        def adjusted(route):
            value = number(route.get("value"))
            margin = number(route.get("margin"))
            path = route.get("path", ())
            action = route.get("action", "NOOP")
            if posture == "PROTECT" or risk == "CONSERVATIVE":
                value += min(margin, 6.0) * (2.0 if posture == "PROTECT" else 1.5)
                value -= 1.0 * sum(a == "NOOP" for a in path) if rr >= 6 else 0.0
            elif posture == "ADVANCE" or risk == "PROGRESS":
                value -= len(path) * 0.45
                value += 2.0 if route.get("complete") else 0.0
            elif posture == "ESCAPE":
                if rr >= 6 and action in ("UP", "DOWN"):
                    value += 10.0
                if action == "NOOP":
                    value -= 7.0
            elif posture == "ALIGN":
                if route.get("goal") == self.crossing_target:
                    value += 9.0
                if action in ("LEFT", "RIGHT") and rr >= 8:
                    value += 1.0
            return value
        ordered = sorted(routes, key=adjusted, reverse=True)
        best = ordered[0]
        # A semantic posture changes rank only among paths the local forecast accepted.
        action = best.get("action", "NOOP")
        self.last_route = dict(best, posture=posture, effective_value=round(adjusted(best), 2))
        if action in ("LEFT", "RIGHT"):
            self.last_lateral_tick = t
        return action, best

    def actuation(self, obs, routes, world):
        t = integer(obs.get("tick"))
        rr = row_of(obs)
        active = integer(obs.get("frog", {}).get("state")) == 128 and -1 <= rr <= 10
        action = "NOOP"
        if self.pressed is not None:
            moved = self.origin is not None and (rr != self.origin[0] or abs(number(obs.get("frog", {}).get("x")) - self.origin[1]) >= 7)
            if moved or not active or t - self.press_tick >= 4:
                self.log("neutral_rearm", tick=t, pressed=self.pressed, moved=moved)
                self.pressed = None
                self.origin = None
                self.ready_tick = max(self.ready_tick, t + 2)
            else:
                return self.pressed
        if active and t >= self.ready_tick:
            action, route = self.select_route(routes, obs)
            if route is not None:
                self.last_route = {"tick": t, "origin": (rr, number(obs.get("frog", {}).get("x"))), **self.last_route}
                if action != "NOOP" or t - self.last_log_tick >= 60:
                    self.log("route_posted", tick=t, action=action, route=self.last_route,
                             top_routes=routes[:4], intent=self.intent)
                    self.last_log_tick = t
            # Search executes between frames; compensate for its measured lag.
            lag = 0
            if self.previous is not None:
                lag = max(0, min(2, t - integer(self.previous.get("tick"), t)))
            self.ready_tick = t + (max(4, self.hop_frames - lag) if action != "NOOP" else 3)
            if action != "NOOP":
                self.pressed, self.press_tick = action, t
                self.origin = (rr, number(obs.get("frog", {}).get("x")))
        return action

    def run_episode(self, game):
        del game
        while not self.ctx.done:
            obs = decode(self.ctx.observe())
            if not obs:
                self.ctx.act("NOOP")
                self.ctx.wait_for_tick()
                continue
            self.update(obs)
            self.poll_models(obs)
            t = integer(obs.get("tick"))
            ready_to_plan = (integer(obs.get("frog", {}).get("state")) == 128
                             and self.pressed is None and t >= self.ready_tick)
            if ready_to_plan:
                routes, world = self.solve(obs)
                self.request_jev(obs, routes, world)
            else:
                routes, world = [], None
            self.request_astra(obs)
            action = self.actuation(obs, routes, world)
            self.ctx.act(action)
            self.ctx.wait_for_tick()


def build(ctx):
    return Harness(ctx)
