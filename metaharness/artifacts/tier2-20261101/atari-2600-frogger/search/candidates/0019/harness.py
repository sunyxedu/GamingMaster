"""0019: a commitment-first transit planner with interruptible hazard avoidance.

The controller combines a time-layered route portfolio with a small reactive
escape layer.  Route intentions are sticky; close actor contacts can interrupt
an intention only through a currently supported move.
"""
from __future__ import annotations

import json
import math
import time
from collections import deque

from metaharness.core.contract import BudgetExceeded, EpisodeOver, ModelUnavailable
from metaharness.models.jev import Choice, Noul

# One frog hop is eight world pixels. These offsets describe repeated TIA copies.
COPIES = ((0,), (0, 16), (0, 32), (0, 16, 32), (0, 64), (0,), (0, 32, 64), (0,))
BODY = (8, 8, 8, 8, 8, 16, 8, 32)
MOVES = (("UP", 1, 0), ("LEFT", 0, -8), ("RIGHT", 0, 8), ("NOOP", 0, 0), ("DOWN", -1, 0))


def num(value, fallback=0.0):
    try:
        result = float(value)
        return result if math.isfinite(result) else fallback
    except (TypeError, ValueError, OverflowError):
        return fallback


def integer(value, fallback=0):
    return int(num(value, fallback))


def obj(value):
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except (TypeError, ValueError):
            return {}
    return value if isinstance(value, dict) else {}


def json_fragment(value):
    if not isinstance(value, str):
        return {}
    try:
        left, right = value.index("{"), value.rindex("}") + 1
        parsed = json.loads(value[left:right])
        return parsed if isinstance(parsed, dict) else {}
    except (ValueError, TypeError, json.JSONDecodeError):
        return {}


def frog_row(observation):
    value = integer(observation.get("frog", {}).get("row"), 255)
    return -1 if value == 255 else value


def crossing_epoch(observation):
    return tuple(integer(observation.get(key), default) for key, default in
                 (("deaths", 0), ("frogs_home", 0), ("level", 1)))


def lane_actor(observation):
    """Return the observed bonus candidate only while it occupies a playable row."""
    ram = observation.get("ram_fields", {})
    fly = observation.get("fly", {})
    rr = integer(fly.get("row") if fly.get("row") is not None else ram.get("fly_row_ae"), 255)
    active = bool(fly.get("present")) or bool(integer(ram.get("bonus_flags_c7")) & 128)
    if not active or rr < 0 or rr > 10:
        return None
    xx = fly.get("x") if fly.get("x") is not None else ram.get("fly_x_b2")
    return rr, num(xx)


def bay_alert(observation, bay):
    # Observed correlation only: require the high flag and its matching target.
    ram = observation.get("ram_fields", {})
    return bool(integer(ram.get("collision_e3")) & 128) and integer(ram.get("collision_row_e5")) == integer(bay, -2) + 1


def combined_probabilities(answers, first, second, minimum):
    one, two = answers.get(first), answers.get(second)
    selected = getattr(one, "value", None)
    if not isinstance(selected, str) or selected != getattr(two, "value", None):
        return None, 0.0
    p1, p2 = getattr(one, "probabilities", None), getattr(two, "probabilities", None)
    if not isinstance(p1, dict) or not isinstance(p2, dict):
        return None, 0.0
    probability = min(num(p1.get(selected)), num(p2.get(selected)))
    return (selected, probability) if probability >= minimum else (None, probability)


def _merge(spans):
    result = []
    for low, high in sorted(spans):
        if result and low <= result[-1][1]:
            result[-1][1] = max(result[-1][1], high)
        else:
            result.append([low, high])
    return result


class CollisionForecast:
    """Rasterized landing/support forecasts at the controller's hop interval."""

    def __init__(self, controller, observation, horizon):
        self.owner = controller
        self.observation = observation
        self.hard_bank = integer(observation.get("level"), 1) >= 4 or controller.bank_alarm
        self.depth = horizon
        self.clearance = []
        self.platform_flow = {}
        self.tracks = {}
        self.open_bays = [bay for bay in observation.get("homes", [])
                          if not bay.get("occupied") and bay.get("available", True)
                          and bay.get("crocodile") not in (True, "present", "dangerous")]
        self.bay_centers = {str(bay.get("slot")): num(bay.get("x_center"), 80)
                            for bay in self.open_bays}
        self.warned_bays = {slot for slot in self.bay_centers if bay_alert(observation, slot)}
        for lane in observation.get("road", []) + observation.get("river", []):
            rr = integer(lane.get("row"))
            items = []
            for sprite in lane.get("objects", []):
                shape = integer(sprite.get("shape_ptr"))
                size_code = integer(sprite.get("nusiz")) & 7
                direction = sprite.get("direction", lane.get("direction"))
                sign = -1 if direction == "left" else 1
                key = str(sprite.get("id"))
                velocity = controller.velocity.get(key, sign / max(1, integer(lane.get("activity_mask"), 3) + 1))
                width = 24 if size_code == 7 and shape in (163, 170) else BODY[size_code]
                usable = sprite.get("kind") in ("log", "turtle") and not sprite.get("hazard")
                items.append((num(sprite.get("x")), velocity, COPIES[size_code], width, usable, key))
            self.tracks[rr] = items
            if rr >= 6:
                self.platform_flow[rr] = sum(item[1] for item in items) / max(1, len(items))

        raw = observation.get("raw_ram", [])
        if self.hard_bank and len(raw) >= 17 and 5 not in self.tracks:
            self.tracks[5] = [(num(raw[16]), controller.bank_velocity, (0,), 16, False, "median-secondary")]

        actor = lane_actor(observation)
        for layer_index in range(horizon + 2):
            elapsed = layer_index * controller.hop_frames
            row_cells = {}
            for rr, sprites in self.tracks.items():
                if rr < 5:
                    cells = [18.0] * 161
                    for x, speed, copies, width, _, _ in sprites:
                        first = x + speed * (elapsed + 1)
                        last = x + speed * (elapsed + controller.hop_frames + 2)
                        for offset in copies:
                            lo = min(first, last) + offset - 9.5
                            hi = max(first, last) + offset + width + 1.5
                            self._stamp(cells, lo, hi, -1.0)
                elif rr == 5:
                    cells = [18.0] * 161
                    if self.hard_bank and sprites and sprites[0][5] == "median-secondary":
                        x, speed, copies, width, _, _ = sprites[0]
                        self._stamp(cells, x + speed * (elapsed + 1) - 10,
                                    x + speed * (elapsed + controller.hop_frames + 2) + width + 10, -1.0)
                else:
                    cells = [-1.0] * 161
                    supported = []
                    for x, speed, copies, width, usable, _ in sprites:
                        if not usable:
                            continue
                        at = x + speed * (elapsed + 1)
                        for offset in copies:
                            for wrap in (-320, -160, 0, 160, 320):
                                supported.append((at + offset + wrap, at + offset + width + wrap))
                    for lo, hi in _merge(supported):
                        left, right = lo - 2.0, hi - 6.0
                        for px in range(max(0, math.ceil(left)), min(160, math.floor(right)) + 1):
                            cells[px] = max(cells[px], min(px - left, right - px))
                row_cells[rr] = cells

            if actor is not None and actor[0] in row_cells:
                speed = controller.actor_velocity
                left = actor[1] + speed * (elapsed + 1)
                right = actor[1] + speed * (elapsed + controller.hop_frames + 3)
                self._stamp(row_cells[actor[0]], min(left, right) - 11,
                            max(left, right) + 19, -1.0)
            elif actor is None and controller.actor_spawn and controller.actor_spawn[0] in row_cells:
                spawn_row, spawn_x = controller.actor_spawn
                self._stamp(row_cells[spawn_row], spawn_x - 11, spawn_x + 23, -1.0)
            self.clearance.append(row_cells)

    @staticmethod
    def _stamp(cells, low, high, value):
        for offset in (-320, -160, 0, 160, 320):
            start = max(0, math.floor(low + offset))
            stop = min(160, math.ceil(high + offset))
            if start <= stop:
                cells[start:stop + 1] = [value] * (stop - start + 1)

    def at(self, rr, x, time_layer=0):
        if x < 4 or x > 148 or time_layer >= len(self.clearance):
            return -100.0
        if rr in (-1, 5) and rr not in self.clearance[time_layer]:
            return 18.0
        if not 0 <= rr <= 10:
            return -100.0
        cells = self.clearance[time_layer].get(rr)
        return cells[max(0, min(160, int(round(x))))] if cells else -100.0

    def dock_margin(self, x, bay):
        center = self.bay_centers.get(str(bay))
        if center is None or str(bay) in self.warned_bays:
            return -100.0
        low, high = self.owner.dock_band
        return min(x - center - low, center + high - x)

    def step(self, rr, x, move, layer, bay=None, bank_visits=0):
        action, dr, dx = move
        nr, nx = rr + dr, x + dx
        if nr < -1 or nr > 11 or nx < 4 or nx > 148:
            return None
        if nr == 11:
            if action != "UP" or bay is None:
                return None
            margin = self.dock_margin(nx, bay)
            return (nr, nx, margin, 0) if margin >= 0 else None
        if self.hard_bank and nr == 5 and (rr == 5 or rr >= 6):
            return None
        visits = bank_visits + 1 if nr == 5 else 0
        if not self.hard_bank and visits > 3:
            return None
        margin = self.at(nr, nx, layer)
        if margin < 0:
            return None
        drifted = nx + self.platform_flow.get(nr, 0.0) * self.owner.hop_frames if nr >= 6 else nx
        if nr >= 6:
            end_margin = self.at(nr, drifted, layer + 1)
            if end_margin < 0:
                return None
            margin = min(margin, end_margin)
        return nr, drifted, margin, visits


class FrogController:
    def __init__(self, ctx):
        self.ctx = ctx
        self.hop_frames = 6
        self.dock_band = (-4.0, 0.0)
        self.velocity = {}
        self.positions = {}
        self.last_observation = None
        self.current_epoch = None
        self.home_goal = None
        self.goal_source = "upstream"
        self.protocol = "TRANSIT"
        self.astra_policy = {"itinerary": "upstream", "launch": "transit", "dock": "settle"}
        self.policy_revision = 0
        self.bank_alarm = False
        self.bank_velocity = .25
        self.bank_positions = deque(maxlen=64)
        self.actor_velocity = .5
        self.actor_cruise = .5
        self.actor_spawn = None
        self.actor_present = False
        self.actor_positions = deque(maxlen=48)
        self.actor_row_epoch = None
        self.last_target_missing = None
        self.pending_goal = None
        self.pending_since = None
        self.last_progress = 0
        self.last_lateral = -100
        self.last_top_arrival = None
        self.press = None
        self.press_tick = -100
        self.press_origin = None
        self.next_motor_tick = 3
        self.last_posted = "NOOP"
        self.plan_record = None
        self.route_portfolio = []
        self.target_failures = {}
        self.failed_carrier = {}
        self.failures = deque(maxlen=8)
        self.frames = deque(maxlen=12)
        self.s1_future = None
        self.s1_context = None
        self.s1_due = 0
        self.s2_future = None
        self.s2_context = None
        self.s2_due = 1200
        self.astra_session = None
        self.s1_generation = 0
        self.last_search_ms = 0.0
        self.last_action_log = -100
        self.evasion_total = 0

    def log(self, name, **details):
        self.ctx.log(name, **details)

    @staticmethod
    def available_bays(observation):
        return [bay for bay in observation.get("homes", [])
                if not bay.get("occupied") and bay.get("available", True)
                and bay.get("crocodile") not in (True, "present", "dangerous")]

    def frontier(self, observation):
        bays = self.available_bays(observation)
        if not bays:
            return None
        top = next((lane for lane in observation.get("river", []) if integer(lane.get("row")) == 10), {})
        direction = -1 if top.get("direction") == "left" else 1
        return min(bays, key=lambda bay: direction * num(bay.get("x_center")))

    def contacts(self, observation):
        rr, x = frog_row(observation), num(observation.get("frog", {}).get("x"))
        lane = next((entry for entry in observation.get("river", []) if integer(entry.get("row")) == rr), {})
        results = []
        for sprite in lane.get("objects", []):
            code = integer(sprite.get("nusiz")) & 7
            for offset in COPIES[code]:
                delta = (x - num(sprite.get("x")) - offset + 80) % 160 - 80
                if -4 <= delta <= BODY[code]:
                    results.append({"id": str(sprite.get("id")), "shape": integer(sprite.get("shape_ptr")),
                                    "offset": round(delta, 1), "kind": sprite.get("kind"),
                                    "hazard": bool(sprite.get("hazard"))})
        return results

    def snapshot(self, observation):
        ram = observation.get("raw_ram", [])
        return {"tick": observation.get("tick"), "frog": observation.get("frog"),
                "timer": observation.get("timer_s"), "epoch": crossing_epoch(observation),
                "homes": observation.get("homes"), "contact": self.contacts(observation),
                "ram_fields": observation.get("ram_fields"), "last_action": self.last_posted,
                "actor": lane_actor(observation), "actor_velocity": self.actor_velocity,
                "actor_spawn": self.actor_spawn, "median_second": ram[16] if len(ram) > 16 else None}

    def _learn_track_speeds(self, observation):
        tick = integer(observation.get("tick"))
        for lane in observation.get("road", []) + observation.get("river", []):
            for sprite in lane.get("objects", []):
                ident, x = str(sprite.get("id")), num(sprite.get("x"))
                history = self.positions.setdefault(ident, deque(maxlen=64))
                if not history or history[-1][0] != tick:
                    history.append((tick, x))
                old = next((point for point in history if 24 <= tick - point[0] <= 56), None)
                direction = sprite.get("direction", lane.get("direction"))
                sign = -1 if direction == "left" else 1
                speed = sign / max(1, integer(lane.get("activity_mask"), 3) + 1)
                if old is not None:
                    measured = ((x - old[1] + 80) % 160 - 80) / max(1, tick - old[0])
                    if measured * sign > 0 and abs(measured) <= 2:
                        candidates = (1/16, 1/8, 1/4, 1/2, 1, 2)
                        speed = sign * min(candidates, key=lambda value: abs(value - abs(measured)))
                self.velocity[ident] = speed

    def _learn_actor(self, observation):
        tick, actor = integer(observation.get("tick")), lane_actor(observation)
        if actor is None:
            self.actor_positions.clear()
            self.actor_row_epoch = None
            self.actor_present = False
            return
        rr, x = actor
        if not self.actor_present and x <= 4:
            self.actor_spawn = (rr, x)
        self.actor_present = True
        identity = (integer(observation.get("level"), 1), rr)
        if identity != self.actor_row_epoch:
            self.actor_positions.clear()
            self.actor_row_epoch = identity
            self.actor_velocity = self.actor_cruise
        if self.actor_positions and abs(x - self.actor_positions[-1][1]) > 20:
            self.actor_positions.clear()
        if not self.actor_positions or self.actor_positions[-1][0] != tick:
            self.actor_positions.append((tick, x))
        old = next((point for point in self.actor_positions if 8 <= tick - point[0] <= 32), None)
        if old is not None:
            speed = (x - old[1]) / max(1, tick - old[0])
            if .05 <= abs(speed) <= 2:
                options = (-1.0, -.5, -.25, .25, .5, 1.0)
                self.actor_velocity = min(options, key=lambda value: abs(value - speed))
                self.actor_cruise = self.actor_velocity
            elif speed == 0:
                self.actor_velocity = 0.0

    def observe_update(self, observation):
        tick, epoch, rr = integer(observation.get("tick")), crossing_epoch(observation), frog_row(observation)
        previous = self.last_observation
        if previous is not None and epoch[0] > crossing_epoch(previous)[0]:
            lost_row = frog_row(previous)
            previous_plan = self.plan_record or {}
            plan_age = tick - integer(previous_plan.get("tick"), tick - 99)
            entry_loss = lost_row == 10 and self.last_posted == "UP" and plan_age < 6
            bank_loss = lost_row == 5
            goal = str(self.home_goal) if self.home_goal is not None else "unknown"
            report = {"tick": tick, "entry_failure": entry_loss, "bank_failure": bank_loss,
                      "carrier_failure": 6 <= lost_row <= 10 and not entry_loss,
                      "repeat_entry": entry_loss and self.target_failures.get((epoch[2], goal), 0) > 0,
                      "previous": self.snapshot(previous), "current": self.snapshot(observation),
                      "plan": previous_plan, "recent_frames": list(self.frames)[-6:],
                      "matched_home_alert": bay_alert(previous, goal), "actor": lane_actor(previous)}
            if entry_loss:
                key = (epoch[2], goal)
                self.target_failures[key] = self.target_failures.get(key, 0) + 1
                for contact in self.contacts(previous):
                    self.failed_carrier[(epoch[2], goal, contact["id"])] = tick
            if bank_loss:
                self.bank_alarm = True
            self.failures.append(report)
            self.log("crossing_loss_witness", **report)

        if epoch != self.current_epoch:
            if self.current_epoch is not None:
                self.log("crossing_epoch_changed", from_epoch=self.current_epoch, to_epoch=epoch,
                         old_goal=self.home_goal, last_plan=self.plan_record)
            self.current_epoch = epoch
            self.home_goal = None
            self.pending_goal = None
            self.last_target_missing = None
            self.press = None
            self.press_origin = None
            self.next_motor_tick = tick + 3
            self.last_top_arrival = None
            self.s1_due = tick
        if previous is not None and rr != frog_row(previous):
            self.last_progress = tick
            self.last_top_arrival = tick if rr == 10 else None

        if self.home_goal is None:
            first = self.frontier(observation)
            if first:
                self.home_goal = str(first.get("slot"))
                self.goal_source = "code_upstream"

        raw = observation.get("raw_ram", [])
        if len(raw) >= 17 and (not self.bank_positions or self.bank_positions[-1][0] != tick):
            self.bank_positions.append((tick, num(raw[16])))
            old = next((point for point in self.bank_positions if 12 <= tick - point[0] <= 40), None)
            if old is not None:
                measured = ((num(raw[16]) - old[1] + 80) % 160 - 80) / max(1, tick - old[0])
                if 0 < measured <= 2:
                    self.bank_velocity = min((.125, .25, .5, 1.0), key=lambda value: abs(value - measured))

        self._learn_track_speeds(observation)
        self._learn_actor(observation)
        self.last_observation = observation
        self.frames.append(self.snapshot(observation))

    def target_x(self, observation):
        home = next((bay for bay in self.available_bays(observation)
                     if str(bay.get("slot")) == str(self.home_goal)), None)
        return num(home.get("x_center"), 80) if home else 80.0

    def route_search(self, observation):
        started = time.perf_counter()
        row, x = frog_row(observation), num(observation.get("frog", {}).get("x"), 80)
        if not -1 <= row <= 10:
            return []
        road_phase = row < 5
        destination_row = 7 if road_phase else 11
        horizon = 22
        forecast = CollisionForecast(self, observation, horizon)
        target_x = self.target_x(observation) - 2
        center_table = [(str(bay.get("slot")), num(bay.get("x_center"))) for bay in forecast.open_bays
                        if str(bay.get("slot")) not in forecast.warned_bays]
        portfolio = (not road_phase and (self.home_goal is None
                     or str(self.home_goal) in forecast.warned_bays
                     or str(self.home_goal) not in forecast.bay_centers
                     or tick_age(observation, self.last_progress) > 240
                     or lane_actor(observation) is not None
                     or (self.actor_spawn is not None and row <= 8)
                     or num(observation.get("timer_s"), 30) < 8))
        candidates = [(row, x, (), 18.0, 0.0, 0)]
        completed, partial = {}, {}
        for depth in range(horizon):
            next_frontier = []
            for r, px, path, weakest, cost, bank_count in candidates:
                for move in MOVES:
                    action, dr, dx = move
                    next_row = r + dr
                    if road_phase and next_row > destination_row:
                        continue
                    if not road_phase and next_row < 5:
                        continue
                    bay_id = self.home_goal
                    if not road_phase and r == 10 and action == "UP" and portfolio:
                        feasible = [(slot, forecast.dock_margin(px, slot)) for slot, _ in center_table]
                        feasible = [item for item in feasible if item[1] >= 0]
                        if feasible:
                            bay_id = max(feasible, key=lambda item: item[1])[0]
                    moved = forecast.step(r, px, move, depth, bay_id, bank_count)
                    if moved is None:
                        continue
                    nr, nx, clearance, new_bank = moved
                    min_clearance = min(weakest, clearance)
                    effort = cost + 1 + (.15 if action in ("LEFT", "RIGHT") else 0) + (1.0 if action == "DOWN" else 0)
                    if nr == 5:
                        effort += 4 if forecast.hard_bank else 1
                    sequence = path + (action,)
                    distance = abs(nx - target_x)
                    if nr == destination_row:
                        value = 400 - effort * 2 + min(min_clearance, 5) * .6
                        if road_phase:
                            value -= distance * .30
                            goal = None
                        else:
                            goal = bay_id
                            value += 6 if str(goal) == str(self.home_goal) else 0
                            value += 4 if str(goal) == str(self.pending_goal) else 0
                        key = (str(goal), sequence[0]) if not road_phase else sequence[0]
                        route = {"action": sequence[0], "path": sequence, "value": value,
                                 "margin": round(min_clearance, 2), "goal": goal,
                                 "complete": True, "endpoint": nr, "endx": round(nx, 2), "steps": len(sequence)}
                        if key not in completed or route["value"] > completed[key]["value"]:
                            completed[key] = route
                        continue
                    pull = abs(nx - target_x)
                    if portfolio and center_table:
                        pull = min(abs(nx - cx + 2) + (0 if sid == self.home_goal else 12)
                                   for sid, cx in center_table)
                    weight = .075 if nr < 4 else .18 if nr < 8 else .25
                    utility = nr * 20 - effort * .6 - pull * weight + min(min_clearance, 5) * .3
                    if nr > row:
                        record = {"action": sequence[0], "path": sequence, "value": utility - 400,
                                  "margin": round(min_clearance, 2), "goal": None,
                                  "complete": False, "endpoint": nr, "endx": round(nx, 2), "steps": len(sequence)}
                        if sequence[0] not in partial or record["value"] > partial[sequence[0]]["value"]:
                            partial[sequence[0]] = record
                    next_frontier.append((utility, (nr, nx, sequence, min_clearance, effort, new_bank)))
            if not next_frontier:
                break
            next_frontier.sort(key=lambda item: item[0], reverse=True)
            survivors, occupied = [], set()
            for _, node in next_frontier:
                key = (node[0], round(node[1] / 2), node[5], node[2][0])
                if key in occupied:
                    continue
                occupied.add(key)
                survivors.append(node)
                if len(survivors) >= 42:
                    break
            candidates = survivors
            if completed and depth > min(route["steps"] for route in completed.values()) + 3:
                break

        results = list(completed.values()) or list(partial.values())
        if not results:
            for move in MOVES:
                feasible = forecast.step(row, x, move, 0, self.home_goal, 0)
                if feasible is not None:
                    nr, nx, clearance, _ = feasible
                    score = nr * 10 + min(clearance, 4) - abs(nx - target_x) * .03 - (move[0] != "NOOP")
                    results.append({"action": move[0], "path": (move[0],), "value": score,
                                    "margin": clearance, "goal": None, "complete": False,
                                    "endpoint": nr, "endx": nx, "steps": 1})
        results.sort(key=lambda route: route["value"], reverse=True)
        self.last_search_ms = (time.perf_counter() - started) * 1000
        return results

    def landing_guard(self, observation, action, forecast=None):
        forecast = forecast or CollisionForecast(self, observation, 1)
        move = next((move for move in MOVES if move[0] == action), MOVES[3])
        result = forecast.step(frog_row(observation), num(observation.get("frog", {}).get("x"), 80), move, 0,
                               self.home_goal, 0)
        if result is None:
            return False
        if frog_row(observation) == 10 and action == "UP":
            tick = integer(observation.get("tick"))
            if tick - self.last_lateral < 6:
                return False
            if self.last_top_arrival is not None and tick - self.last_top_arrival < 6:
                return False
            if self.astra_policy["dock"] == "different_carrier" and self.target_failures.get(
                    (crossing_epoch(observation)[2], str(self.home_goal)), 0) and tick - self.last_progress < 180:
                return False
        return True

    def _escape_from_actor(self, observation, forecast):
        actor, rr = lane_actor(observation), frog_row(observation)
        if actor is None or actor[0] != rr or not 6 <= rr <= 10:
            return None
        tick, x = integer(observation.get("tick")), num(observation.get("frog", {}).get("x"), 80)
        relative = (actor[1] - x + 80) % 160 - 80
        gap = abs(relative)
        frog_speed = num(forecast.platform_flow.get(rr))
        closes_at = (frog_speed - self.actor_velocity) if relative > 0 else (self.actor_velocity - frog_speed)
        if gap > 20 and not (closes_at > .03 and gap <= 12 + closes_at * 24):
            return None
        escapes = []
        future_actor = actor[1] + self.actor_velocity * self.hop_frames
        for move in MOVES:
            action, dr, _ = move
            if action == "NOOP":
                continue
            result = forecast.step(rr, x, move, 0, self.home_goal, 0)
            if result is None:
                continue
            new_row, new_x, margin, _ = result
            if new_row == rr:
                new_gap = abs((future_actor - new_x + 80) % 160 - 80)
                if new_gap < 18 or new_gap < gap + 2:
                    continue
                direction_away = (action == "LEFT" and relative > 0) or (action == "RIGHT" and relative < 0)
                ranking = (70 if direction_away else 35) + min(new_gap, 40) * .3 + min(margin, 8) * .2
            else:
                ranking = (110 if action == "UP" else 55) + min(margin, 8) * .2
            escapes.append((ranking, action, margin))
        if not escapes:
            return None
        _, action, margin = max(escapes)
        self.evasion_total += 1
        self.log("actor_intercept_escape", tick=tick, frog=(rr, round(x, 1)), actor=actor,
                 actor_speed=self.actor_velocity, action=action, safe_margin=round(margin, 2),
                 escape_count=self.evasion_total)
        return action

    def choose_action(self, observation, routes):
        if not routes:
            return "NOOP", None
        tick, rr = integer(observation.get("tick")), frog_row(observation)
        pool = list(routes)
        permitted_retarget = True
        open_slots = {str(bay.get("slot")) for bay in self.available_bays(observation)}
        if 6 <= rr <= 10 and self.home_goal is not None:
            goal_valid = str(self.home_goal) in open_slots and not bay_alert(observation, self.home_goal)
            incumbent = [route for route in routes if route.get("complete") and route.get("endpoint") == 11
                         and str(route.get("goal")) == str(self.home_goal)]
            if goal_valid and incumbent:
                pool = incumbent
                self.last_target_missing = None
                self.pending_goal = None
                self.pending_since = None
            elif goal_valid:
                if self.last_target_missing is None:
                    self.last_target_missing = tick
                alternatives = [route for route in routes if route.get("complete") and route.get("endpoint") == 11
                               and route.get("goal") is not None]
                if alternatives:
                    viable = {str(route.get("goal")) for route in alternatives}
                    if str(self.pending_goal) not in viable:
                        self.pending_goal = str(alternatives[0].get("goal"))
                        self.pending_since = tick
                    pool = [route for route in alternatives if str(route.get("goal")) == str(self.pending_goal)] or alternatives
                    permitted_retarget = (num(observation.get("timer_s"), 30) <= 4
                                          or tick - self.last_target_missing >= 18)
                else:
                    permitted_retarget = False
            else:
                self.last_target_missing = tick
                self.pending_goal = None
                self.pending_since = None

        selected = pool[0] if pool else routes[0]
        if self.protocol == "STAGE" or self.astra_policy["launch"] == "wide":
            close = [route for route in pool if route["complete"] == selected["complete"]
                     and route["value"] >= selected["value"] - 5]
            if close:
                selected = max(close, key=lambda route: (route["margin"], route["value"]))
        if self.protocol == "SCREEN":
            close = [route for route in pool if route["complete"] == selected["complete"]
                     and route["value"] >= selected["value"] - 4]
            if close:
                selected = min(close, key=lambda route: (sum(step in ("NOOP", "LEFT", "RIGHT") for step in route["path"]),
                                                         -route["value"]))

        if permitted_retarget and selected.get("complete") and selected.get("endpoint") == 11 and selected.get("goal") is not None:
            revised_goal = str(selected["goal"])
            if revised_goal != str(self.home_goal):
                old_goal = self.home_goal
                self.home_goal, self.goal_source = revised_goal, "portfolio"
                self.last_target_missing = None
                self.pending_goal = None
                self.pending_since = None
                self.log("durable_goal_replaced", tick=tick, old=old_goal, new=revised_goal,
                         reason="invalid" if not (old_goal is not None and str(old_goal) in open_slots) else "no_complete_incumbent")

        forecast = CollisionForecast(self, observation, 1)
        action = selected["action"]
        if not self.landing_guard(observation, action, forecast):
            legal = [route for route in pool if self.landing_guard(observation, route["action"], forecast)]
            if legal:
                selected, action = legal[0], legal[0]["action"]
            else:
                sequence = ("UP", "DOWN", "LEFT", "RIGHT", "NOOP") if rr == 5 else ("NOOP", "UP", "LEFT", "RIGHT", "DOWN")
                action = next((candidate for candidate in sequence if self.landing_guard(observation, candidate, forecast)), "NOOP")
                selected = dict(selected, action=action, guarded_fallback=True)
        dodge = self._escape_from_actor(observation, forecast)
        if dodge is not None and dodge != action:
            old_action = action
            action = dodge
            selected = dict(selected, action=action, actor_escape=True)
            self.log("actor_escape_overrides_route", tick=tick, planned=old_action, action=action,
                     durable_goal=self.home_goal, protocol=self.protocol)
        return action, selected

    def _jev_state(self, observation):
        available = self.available_bays(observation)
        top = next((lane for lane in observation.get("river", []) if integer(lane.get("row")) == 10), {})
        sign = -1 if top.get("direction") == "left" else 1
        ordered = sorted(available, key=lambda bay: sign * num(bay.get("x_center")))
        frontier_id = str(ordered[0].get("slot")) if ordered else None
        x = num(observation.get("frog", {}).get("x"), 80)
        nearest = str(min(available, key=lambda bay: abs(num(bay.get("x_center")) - x)).get("slot")) if available else None
        bay_facts = [{"bay": str(bay.get("slot")), "upstream_frontier": str(bay.get("slot")) == frontier_id,
                      "nearest_now": str(bay.get("slot")) == nearest, "current_commitment": str(bay.get("slot")) == str(self.home_goal),
                      "failed_entry_this_level": self.target_failures.get((crossing_epoch(observation)[2], str(bay.get("slot"))), 0) > 0,
                      "matched_alert": bay_alert(observation, bay.get("slot")),
                      "complete_route_now": any(route.get("goal") == str(bay.get("slot")) and route.get("complete")
                                                 for route in self.route_portfolio)} for bay in ordered]
        failure = self.failures[-1] if self.failures else {}
        actor = lane_actor(observation)
        rr = frog_row(observation)
        return {"game": "Frogger; bring frogs to open home bays while preserving lives.",
                "phase": "road_launch" if rr < 5 else "median_transit" if rr == 5 else "river" if rr < 10 else "dock",
                "policy": dict(self.astra_policy), "candidate_bays": bay_facts,
                "facts": {"route_to_row7_exists": any(route.get("complete") and route.get("endpoint") == 7
                                                       for route in self.route_portfolio),
                          "strict_median": integer(observation.get("level"), 1) >= 4 or self.bank_alarm,
                          "actor_on_playable_row": actor is not None,
                          "actor_on_frog_row": actor is not None and actor[0] == rr,
                          "actor_spawn_reserved": actor is None and self.actor_spawn is not None,
                          "current_target_alerted": bay_alert(observation, self.home_goal),
                          "urgent_timer": num(observation.get("timer_s"), 30) < 8,
                          "entry_failed": bool(failure.get("entry_failure")),
                          "bank_failed": bool(failure.get("bank_failure")),
                          "repeat_entry": bool(failure.get("repeat_entry")),
                          "carrier_failed": bool(failure.get("carrier_failure"))},
                "rules": "Choose a whole-crossing bay, not a joystick direction. The river flow makes upstream bays harder to revisit. The median is a transit link at harder levels. A matched home alert is a local veto; unknown crocodile status is not evidence of safety. Code validates every hop against live geometry and can escape an approaching same-row actor only on a supported move."}

    def _request_jev(self, observation):
        tick = integer(observation.get("tick"))
        if (self.s1_future is not None or tick < self.s1_due or getattr(self.ctx, "system1", True) is None
                or self.ctx.budget.remaining().get("system1_calls", 0) < 2):
            return
        state = self._jev_state(observation)
        if not state["candidate_bays"]:
            return
        itinerary_options = {bay["bay"]: {"what": "Commit to this open home bay for the crossing.", "evidence": bay}
                            for bay in state["candidate_bays"]}
        itinerary_options["UNKNOWN"] = "Keep the code-selected commitment; the evidence is insufficient."
        routes = {"TRANSIT": "Move through the median with a complete river route.",
                  "STAGE": "Prefer wider near-equal launch routes without treating the median as safe.",
                  "RECOVER": "After repeated entry failure change approach, not the clearance constants.",
                  "SCREEN": "Minimize exposure to an auxiliary actor or its reserved spawn region.",
                  "UNKNOWN": "Keep the current protocol."}
        choose_bay = ("Choose a durable whole-crossing home from candidate_bays. If still on road, prefer upstream_frontier. "
                      "In river retain current_commitment unless alerted or there is no complete route now; then choose a complete unalerted route. "
                      "Never infer an unknown crocodile is absent. UNKNOWN if unsure.")
        choose_protocol = ("Select a route-control protocol from facts. Prefer SCREEN if actor_on_playable_row or actor_spawn_reserved; "
                           "RECOVER on repeat_entry or current_target_alerted; TRANSIT on strict_median or bank_failed; "
                           "STAGE for a carrier_failed witness when timer is not urgent; otherwise TRANSIT. This is policy, not a frame action.")
        questions = {"itinerary": Choice(choose_bay, itinerary_options),
                     "itinerary_check": Choice(choose_bay, dict(reversed(list(itinerary_options.items())))),
                     "protocol": Choice(choose_protocol, routes),
                     "protocol_check": Choice(choose_protocol, dict(reversed(list(routes.items())))),
                     "review": Noul("Do facts show a repeated entry or median contradiction that warrants Astra revising route policy?")}
        self.s1_generation += 1
        meta = {"tick": tick, "epoch": crossing_epoch(observation), "revision": self.policy_revision,
                "generation": self.s1_generation, "road_phase": frog_row(observation) < 5}
        self.s1_context = meta

        def request():
            try:
                begin = self.ctx.clock.now()
                answer = self.ctx.decide(state, questions, timeout_s=3.0)
                return answer, self.ctx.clock.now() - begin
            except (BudgetExceeded, EpisodeOver, ModelUnavailable):
                raise
            except (TimeoutError, RuntimeError, TypeError, ValueError):
                return None, 0.0
        self.s1_future = self.ctx.spawn(request)
        self.s1_due = tick + 90

    def _request_astra(self, observation):
        tick = integer(observation.get("tick"))
        if (self.s2_future is not None or tick < self.s2_due or not self.failures
                or getattr(self.ctx, "system2", True) is None
                or self.ctx.budget.remaining().get("system2_calls", 0) < 2):
            return
        evidence = {"current_policy": self.astra_policy, "recent_failures": list(self.failures)[-2:],
                    "score": observation.get("score"), "level": observation.get("level"),
                    "protocol": self.protocol, "forecast_ms": round(self.last_search_ms, 2)}
        prompt = ("You review Frogger route protocol asynchronously, never frame actions. The controller searches time-layered "
                  "body-aware routes through median transit into row7; it keeps a feasible whole-crossing bay commitment, and only "
                  "switches after its incumbent has no complete route or is invalid. An independent public auxiliary candidate is "
                  "screened as a possible moving actor; immediate escape is code-owned and requires a supported move. A matched E3/E5 "
                  "alert vetoes only its bay. Unknown crocodile state is not safe. Jev receives compact named state and reversed Choices; "
                  "responses require agreement and >=0.70 probability. Choose only policy enums; do not invent geometry or code. "
                  "Return ONLY JSON: {\"itinerary\":\"upstream|nearest\",\"launch\":\"transit|wide\",\"dock\":\"settle|different_carrier\"}. Evidence="
                  + json.dumps(evidence, separators=(",", ":")))
        meta = {"tick": tick, "failure_count": len(self.failures)}
        self.s2_context = meta
        session = self.astra_session

        def request():
            try:
                return self.ctx.think(prompt, session_id=session, timeout_s=120.0)
            except (BudgetExceeded, EpisodeOver, ModelUnavailable):
                raise
            except (TimeoutError, RuntimeError, TypeError, ValueError):
                return None
        self.s2_future = self.ctx.spawn(request)
        self.s2_due = tick + 3600

    def _collect_models(self, observation):
        tick = integer(observation.get("tick"))
        if self.s1_future is not None and self.s1_future.done():
            future, self.s1_future = self.s1_future, None
            try:
                result = future.result()
            except (BudgetExceeded, EpisodeOver, ModelUnavailable):
                raise
            except (TimeoutError, RuntimeError, TypeError, ValueError):
                result = None
            if result and isinstance(result, tuple):
                answers, latency = result
                answers = answers if isinstance(answers, dict) else {}
                meta = self.s1_context or {}
                current = crossing_epoch(observation) == meta.get("epoch") and self.policy_revision == meta.get("revision")
                fresh = current and tick - meta.get("tick", tick) <= 90
                chosen, probability = combined_probabilities(answers or {}, "itinerary", "itinerary_check", .70)
                if fresh and meta.get("road_phase") and frog_row(observation) < 5:
                    valid = {str(bay.get("slot")) for bay in self.available_bays(observation)}
                    if chosen in valid:
                        previous = self.home_goal
                        self.home_goal, self.goal_source = chosen, "jev_commitment"
                        self.log("jev_durable_home_choice", tick=tick, old=previous, new=chosen,
                                 probability=probability, latency_s=latency)
                elif fresh and chosen is not None and not bay_alert(observation, chosen):
                    feasible = [route for route in self.route_portfolio if route.get("complete")
                                and str(route.get("goal")) == chosen]
                    if feasible:
                        self.pending_goal = chosen
                        self.pending_since = tick
                        self.log("jev_route_preference", tick=tick, goal=chosen, probability=probability)
                protocol, chance = combined_probabilities(answers or {}, "protocol", "protocol_check", .70)
                if fresh and protocol in ("TRANSIT", "STAGE", "RECOVER", "SCREEN"):
                    before = self.protocol
                    self.protocol = protocol
                    self.log("jev_protocol", tick=tick, old=before, new=protocol,
                             probability=chance, latency_s=latency)
                review = num(getattr((answers or {}).get("review"), "value", 0))
                if review >= .85:
                    self._request_astra(observation)
        if self.s2_future is not None and self.s2_future.done():
            future, self.s2_future = self.s2_future, None
            try:
                answer = future.result()
            except (BudgetExceeded, EpisodeOver, ModelUnavailable):
                raise
            except (TimeoutError, RuntimeError, TypeError, ValueError):
                answer = None
            if answer is not None:
                self.astra_session = getattr(answer, "session_id", None)
                policy = json_fragment(getattr(answer, "text", ""))
                allowed = {"itinerary": ("upstream", "nearest"), "launch": ("transit", "wide"),
                           "dock": ("settle", "different_carrier")}
                changes = {key: value for key, value in policy.items()
                           if key in allowed and isinstance(value, str) and value in allowed[key]}
                if changes:
                    self.astra_policy.update(changes)
                    self.policy_revision += 1
                    self.s1_due = tick
                    self.log("astra_policy_compiled", tick=tick, policy=dict(self.astra_policy),
                             revision=self.policy_revision)
                else:
                    self.log("astra_policy_ignored", tick=tick)

    def _issue(self, action, route, observation):
        tick = integer(observation.get("tick"))
        rr = frog_row(observation)
        if route:
            self.plan_record = {"tick": tick, "origin": (rr, observation.get("frog", {}).get("x")),
                                "goal": self.home_goal, "protocol": self.protocol, **route}
            if action != "NOOP" or tick - self.last_action_log >= 60:
                self.log("route_decision", tick=tick, route=self.plan_record,
                         search_ms=round(self.last_search_ms, 2))
                self.last_action_log = tick
        if action != "NOOP":
            self.press = action
            self.press_tick = tick
            self.press_origin = (rr, num(observation.get("frog", {}).get("x")))
            if action in ("LEFT", "RIGHT"):
                self.last_lateral = tick
        return action

    def run_episode(self, game):
        del game
        while not self.ctx.done:
            observation = obj(self.ctx.observe())
            if not observation:
                self.ctx.act("NOOP")
                self.ctx.wait_for_tick()
                continue
            self.observe_update(observation)
            self._collect_models(observation)
            tick, rr = integer(observation.get("tick")), frog_row(observation)
            living = integer(observation.get("frog", {}).get("state")) == 128 and -1 <= rr <= 10
            action = "NOOP"
            if self.press is not None:
                position = num(observation.get("frog", {}).get("x"))
                changed = self.press_origin is not None and (
                    rr != self.press_origin[0] or abs(position - self.press_origin[1]) >= 7)
                if changed or not living or tick - self.press_tick >= 4:
                    self.log("direction_released", tick=tick, pressed=self.press, moved=changed)
                    self.press = None
                    self.press_origin = None
                    self.next_motor_tick = max(self.next_motor_tick, tick + 2)
                else:
                    action = self.press
            if self.press is None and living and tick >= self.next_motor_tick:
                self.route_portfolio = self.route_search(observation)
                latest = obj(self.ctx.observe())
                if latest and crossing_epoch(latest) == crossing_epoch(observation) and frog_row(latest) == rr:
                    action, chosen = self.choose_action(latest, self.route_portfolio)
                    tick = integer(latest.get("tick"), tick)
                    action = self._issue(action, chosen, latest)
                    lag = max(0, min(2, tick - integer(observation.get("tick"), tick)))
                    self.next_motor_tick = tick + (max(4, self.hop_frames - lag) if action != "NOOP" else 3)
                self._request_jev(observation)
            self._request_astra(observation)
            self.last_posted = action
            self.ctx.act(action)
            self.ctx.wait_for_tick()


def tick_age(observation, tick):
    return integer(observation.get("tick")) - tick


def build(ctx):
    return FrogController(ctx)
