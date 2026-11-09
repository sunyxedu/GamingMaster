"""Frogger controller with failure-conditioned phase diversity.

The route finder is closed loop: it forecasts the currently visible traffic and
carrier copies, but it never commits to a multi-hop script.  A death is also
an observation about timing.  In particular, a repeated final-bay failure is
not retried at the same start phase; the next frog gets an episode-local,
bounded sidewalk delay (and, when possible, a different open bay).  This is
intended to break deterministic bad traffic/home-crocodile phase replays
without globally inflating every collision margin.
"""
from __future__ import annotations

import json
import math
from collections import defaultdict, deque

from metaharness.core.contract import BudgetExceeded, EpisodeOver, ModelUnavailable
from metaharness.models.jev import Choice, Noul


# NUSIZ values describe either separated copies or one stretched copy.  The
# adaptor exposes the same low three bits, so a collision table must not fill
# the gaps between copies.
COPY_OFFSETS = (
    (0,), (0, 16), (0, 32), (0, 16, 32),
    (0, 64), (0,), (0, 32, 64), (0,),
)
COPY_WIDTHS = (8, 8, 8, 8, 8, 16, 8, 32)
MOVES = (
    ("UP", 1, 0), ("LEFT", 0, -8), ("RIGHT", 0, 8),
    ("NOOP", 0, 0), ("DOWN", -1, 0),
)
LEGAL = {"UP", "LEFT", "RIGHT", "NOOP", "DOWN"}


def number(value, default=0.0):
    try:
        result = float(value)
        return result if math.isfinite(result) else default
    except (TypeError, ValueError):
        return default


def obj(value):
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except (TypeError, ValueError):
            return {}
    return value if isinstance(value, dict) else {}


def integer(value, default=0):
    return int(number(value, default))


def frog_row(observation):
    value = integer(obj(observation).get("frog", {}).get("row"), 255)
    return -1 if value == 255 else value


def active_frog(observation):
    frog = obj(observation).get("frog", {})
    return integer(frog.get("state"), 0) == 128 and -1 <= frog_row(observation) <= 10


def epoch(observation):
    observation = obj(observation)
    return (
        integer(observation.get("deaths"), 0),
        integer(observation.get("frogs_home"), 0),
        integer(observation.get("level"), 1),
    )


def phase_name(observation):
    row = frog_row(observation)
    if row < 0:
        return "SIDEWALK"
    if row < 5:
        return "ROAD"
    if row == 5:
        return "MEDIAN"
    if row <= 10:
        return "RIVER"
    return "HOME"


def merge_intervals(intervals):
    normalized = []
    for interval in intervals:
        try:
            a, b = float(interval[0]), float(interval[1])
        except (TypeError, ValueError, IndexError):
            continue
        if math.isfinite(a) and math.isfinite(b):
            normalized.append((min(a, b), max(a, b)))
    merged = []
    for a, b in sorted(normalized, key=lambda pair: (pair[0], pair[1])):
        if merged and a <= merged[-1][1] + 0.01:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    return merged


def probability(answer, option):
    probabilities = getattr(answer, "probabilities", {})
    if not isinstance(probabilities, dict):
        return 0.0
    return number(probabilities.get(option), 0.0)


def extract_json(text):
    if not isinstance(text, str):
        return {}
    left, right = text.find("{"), text.rfind("}")
    if left < 0 or right < left:
        return {}
    try:
        value = json.loads(text[left:right + 1])
    except (TypeError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


class Forecast:
    """Build short time-expanded safety tables from the public object tracks."""

    def __init__(self, controller, observation, horizon):
        self.controller = controller
        self.observation = observation
        self.horizon = horizon
        self.step = controller.forecast_step
        self.lanes = {}
        self.drift = {}
        self.tables = []
        self._collect_lanes()
        self._build_tables()

    def _collect_lanes(self):
        observation = self.observation
        for lane in list(observation.get("road", [])) + list(observation.get("river", [])):
            row = integer(lane.get("row"), -99)
            if row < 0:
                continue
            records = []
            velocities = []
            for item in lane.get("objects", []):
                if not isinstance(item, dict):
                    continue
                nusiz = integer(item.get("nusiz"), 0) & 7
                ident = str(item.get("id", (row, item.get("index", len(records)))))
                velocity = self.controller.velocities.get(ident)
                if velocity is None:
                    sign = -1.0 if item.get("direction", lane.get("direction")) == "left" else 1.0
                    velocity = sign / (integer(lane.get("activity_mask"), 3) + 1)
                velocity = number(velocity, 0.0)
                shape = integer(item.get("shape_ptr"), -1)
                width = COPY_WIDTHS[nusiz]
                # The two stretched forms have a narrower reliably usable end
                # in the public calibration traces.  Unknown forms retain the
                # ordinary conservative width instead of being relabelled.
                if nusiz == 7 and shape in (163, 170):
                    width = 24
                safe = (
                    row >= 6 and not bool(item.get("hazard"))
                    and item.get("kind") in ("log", "turtle")
                )
                records.append({
                    "x": number(item.get("x"), 0.0),
                    "velocity": velocity,
                    "offsets": COPY_OFFSETS[nusiz],
                    "width": width,
                    "safe": safe,
                    "shape": shape,
                    "nusiz": nusiz,
                })
                velocities.append(velocity)
            self.lanes[row] = records
            self.drift[row] = sum(velocities) / len(velocities) if velocities else 0.0

    def _pixel(self, layer, row, x, default=-1.0):
        values = self.tables[min(max(0, int(layer)), len(self.tables) - 1)].get(row)
        if values is None:
            return default
        pixel = max(0, min(160, int(round(x))))
        return values[pixel]

    def _build_tables(self):
        # Include one extra layer because a river landing is checked after a
        # short carrier-drift dwell, not only at the instant of the hop.
        for layer in range(self.horizon + 2):
            elapsed = layer * self.step
            rows = {}
            for row, records in self.lanes.items():
                if row < 5:
                    safe = [20.0] * 161
                    for record in records:
                        for offset in record["offsets"]:
                            x0 = record["x"] + offset + record["velocity"] * (elapsed + 1)
                            x1 = record["x"] + offset + record["velocity"] * (elapsed + self.step + 2)
                            lo, hi = min(x0, x1), max(x0, x1) + record["width"]
                            for wrap in (-320, -160, 0, 160, 320):
                                first = max(0, math.floor(lo + wrap - 8 - self.controller.road_pad))
                                last = min(160, math.ceil(hi + wrap + self.controller.road_pad))
                                for pixel in range(first, last + 1):
                                    safe[pixel] = -1.0
                    rows[row] = safe
                    continue

                supports = []
                for record in records:
                    if not record["safe"]:
                        continue
                    for offset in record["offsets"]:
                        base = record["x"] + offset + record["velocity"] * (elapsed + 1)
                        for wrap in (-320, -160, 0, 160, 320):
                            supports.append((base + wrap, base + record["width"] + wrap, record))
                values = [-1.0] * 161
                for left, right in merge_intervals(supports):
                    # Frog x is its left anchor.  Requiring its centre to stay
                    # inside the carrier leaves both a collision and drift pad.
                    low = left + self.controller.river_pad - 4
                    high = right - self.controller.river_pad - 4
                    first, last = max(0, math.ceil(low)), min(160, math.floor(high))
                    for pixel in range(first, last + 1):
                        values[pixel] = max(values[pixel], min(pixel - low, high - pixel))

                # A local ban is associated with a copy/form, not an absolute
                # screen x.  It follows the carrier as the phase advances.
                for record in self.lanes.get(row, []):
                    bans = self.controller.contact_bans.get((row, record["shape"], record["nusiz"]), ())
                    if not bans or not record["safe"]:
                        continue
                    for offset in record["offsets"]:
                        base = record["x"] + offset + record["velocity"] * (elapsed + 1)
                        for low, high in bans:
                            for wrap in (-320, -160, 0, 160, 320):
                                first = max(0, math.floor(base + wrap + low))
                                last = min(160, math.ceil(base + wrap + high))
                                for pixel in range(first, last + 1):
                                    values[pixel] = -1.0
                rows[row] = values
            self.tables.append(rows)

    def home_centres(self):
        answer = []
        for home in self.observation.get("homes", []):
            if not isinstance(home, dict):
                continue
            if home.get("occupied") or not home.get("available", True):
                continue
            slot = str(home.get("slot", len(answer)))
            # Unknown crocodile status is deliberately retained as unknown;
            # a prior failed entry can temporarily quarantine the bay.
            answer.append((slot, number(home.get("x_center"), 80.0)))
        return answer

    def home_margin(self, x, slot, centre):
        low, high = self.controller.entry_window(slot)
        return min(x - (centre + low), centre + high - x)

    def one_hop_safe(self, observation, action):
        row = frog_row(observation)
        x = number(observation.get("frog", {}).get("x"), 80.0)
        move = next((move for move in MOVES if move[0] == action), MOVES[3])
        new_row, new_x = row + move[1], x + move[2]
        if not -1 <= new_row <= 11 or not 4 <= new_x <= 148:
            return False
        if new_row == 5 and (new_x < 8 or new_x > 136):
            return False
        if new_row == 5 and action in ("LEFT", "RIGHT") and self.controller.median_lateral_unsafe:
            return False
        if row == 5 and action == "NOOP" and self.controller.median_wait_unsafe:
            return False
        if row == 5 and action == "NOOP" and self.controller.bank_transit_until > integer(observation.get("tick"), 0):
            return False
        if new_row in (-1, 5):
            return True
        if new_row == 11:
            centres = self.home_centres()
            if not centres:
                return False
            slot, centre = min(centres, key=lambda pair: abs(new_x - pair[1]))
            return self.home_margin(new_x, slot, centre) >= 0
        if new_row not in self.lanes:
            return False
        value = self._pixel(0, new_row, new_x, -1.0)
        if value < 0:
            return False
        if new_row >= 6:
            drifted = new_x + self.drift.get(new_row, 0.0) * self.step
            if not 4 <= drifted <= 148 or self._pixel(1, new_row, drifted, -1.0) < 0:
                return False
        return True


class RouteFinder:
    """Beam-search routes, returning only the first action to the motor."""

    def __init__(self, controller):
        self.controller = controller
        self.last_ms = 0.0

    def find(self, observation):
        import time
        started = time.perf_counter()
        row0 = frog_row(observation)
        x0 = number(observation.get("frog", {}).get("x"), 80.0)
        if not -1 <= row0 <= 10:
            return []
        river = row0 >= 5
        horizon = 24 if river else 16
        forecast = Forecast(self.controller, observation, horizon)
        homes = forecast.home_centres()
        tick = integer(observation.get("tick"), 0)
        # A failed bay is avoided only when there is another open bay.  If it
        # is the sole open bay, temporal diversity is safer than giving up.
        usable_homes = [pair for pair in homes if not self.controller.bay_blocked(pair[0], tick)]
        if usable_homes:
            homes = usable_homes
        if not homes:
            return []

        endpoint = 11 if river else 5
        # score, row, x, first action, path, bottleneck, cost
        beam = [(0.0, row0, x0, None, (), 20.0, 0.0)]
        complete = {}
        best_partial = None
        best_partial_value = -1e9
        for layer in range(horizon):
            next_states = []
            for _score, current_row, current_x, first, path, margin, cost in beam:
                for action, delta_row, delta_x in MOVES:
                    new_row = current_row + delta_row
                    new_x = current_x + delta_x
                    if new_row < -1 or new_row > endpoint:
                        continue
                    if river and new_row < 5:
                        continue
                    if not 4 <= new_x <= 148:
                        continue
                    if new_row == 5 and (new_x < 8 or new_x > 136):
                        continue
                    if new_row == 5 and action in ("LEFT", "RIGHT") and self.controller.median_lateral_unsafe:
                        continue
                    if new_row == 5 and action == "NOOP" and self.controller.median_wait_unsafe:
                        continue
                    if new_row == 5 and action == "NOOP" and self.controller.bank_transit_until > tick:
                        continue

                    if new_row in (-1, 5):
                        margin_here = 20.0
                        drifted_x = new_x
                    elif new_row == 11:
                        candidates = [(slot, centre) for slot, centre in homes]
                        slot, centre = min(candidates, key=lambda pair: abs(new_x - pair[1]))
                        margin_here = forecast.home_margin(new_x, slot, centre)
                        drifted_x = new_x
                        if margin_here < 0:
                            continue
                    else:
                        values = forecast.tables[min(layer, len(forecast.tables) - 1)].get(new_row, [])
                        if not values:
                            continue
                        margin_here = values[max(0, min(160, int(round(new_x))))]
                        if margin_here < 0:
                            continue
                        drifted_x = new_x
                        if new_row >= 6:
                            drifted_x += forecast.drift.get(new_row, 0.0) * forecast.step
                            if not 4 <= drifted_x <= 148:
                                continue
                            future = forecast.tables[min(layer + 1, len(forecast.tables) - 1)].get(new_row, [])
                            if not future or future[max(0, min(160, int(round(drifted_x))))] < 0:
                                continue
                            margin_here = min(margin_here, future[max(0, min(160, int(round(drifted_x))))])

                    new_margin = min(margin, margin_here)
                    new_cost = cost + 1.0
                    if action in ("LEFT", "RIGHT"):
                        new_cost += 0.10
                    elif action == "DOWN":
                        new_cost += 0.75
                    elif action == "NOOP":
                        new_cost += 0.25
                    new_path = path + (action,)
                    first_action = first or action

                    if new_row == endpoint:
                        if river:
                            slot, centre = min(homes, key=lambda pair: abs(new_x - pair[1]))
                            label = "HOME_" + slot
                            goal_distance = abs(new_x - centre)
                        else:
                            label = "MEDIAN_LEFT" if new_x < 56 else "MEDIAN_RIGHT" if new_x > 104 else "MEDIAN_CENTER"
                            goal_distance = abs(new_x - 80)
                        value = 245.0 - 2.1 * new_cost + min(new_margin, 7.0) * 0.9
                        value -= goal_distance * (0.04 if river else 0.005)
                        if self.controller.goal == label:
                            value += 8.0
                        if label == self.controller.last_failed_target and len(homes) > 1:
                            value -= 14.0
                        previous = complete.get(label)
                        candidate = {
                            "label": label, "action": first_action, "path": new_path,
                            "value": value, "margin": round(new_margin, 2),
                            "x": round(new_x, 1), "complete": True,
                            "steps": len(new_path),
                        }
                        if previous is None or candidate["value"] > previous["value"]:
                            complete[label] = candidate
                        continue

                    # Keep a high-row partial route as a recovery action, but
                    # make any complete route dominate it by construction.
                    partial_value = new_row * 20.0 - new_cost * 0.55
                    partial_value -= min(abs(drifted_x - centre) for _, centre in homes) * (0.14 if new_row >= 8 else 0.01)
                    partial_value += min(new_margin, 5.0) * 0.25
                    if new_row > row0 and partial_value > best_partial_value:
                        best_partial_value = partial_value
                        best_partial = {
                            "label": "PROGRESS", "action": first_action, "path": new_path,
                            "value": partial_value - 420.0, "margin": round(new_margin, 2),
                            "x": round(drifted_x, 1), "complete": False,
                            "steps": len(new_path),
                        }
                    next_states.append((partial_value, new_row, drifted_x, first_action, new_path, new_margin, new_cost))

            if not next_states:
                break
            next_states.sort(key=lambda state: state[0], reverse=True)
            beam = []
            seen = set()
            for state in next_states:
                key = (state[1], round(state[2] / 2), state[3])
                if key in seen:
                    continue
                seen.add(key)
                beam.append(state)
                if len(beam) >= 48:
                    break
            if complete and layer >= min(item["steps"] for item in complete.values()) + 3:
                break

        routes = sorted(complete.values(), key=lambda route: route["value"], reverse=True)
        if not routes and best_partial is not None:
            routes = [best_partial]
        if not routes:
            choices = []
            for action, delta_row, delta_x in MOVES:
                if not forecast.one_hop_safe(observation, action):
                    continue
                new_row, new_x = row0 + delta_row, x0 + delta_x
                value = new_row * 12.0 - (1.0 if action != "NOOP" else 0.0)
                choices.append({
                    "label": "SURVIVE", "action": action, "path": (action,),
                    "value": value, "margin": 0.0, "x": new_x,
                    "complete": False, "steps": 1,
                })
            routes = sorted(choices, key=lambda route: route["value"], reverse=True)
        self.last_ms = (time.perf_counter() - started) * 1000.0
        return routes


class Controller:
    def __init__(self, ctx):
        self.ctx = ctx
        self.previous = None
        self.histories = {}
        self.velocities = {}
        self.forecast_step = 6
        self.road_pad = 1.5
        self.river_pad = 2.0

        self.goal = None
        self.goal_until = 0
        self.last_failed_target = None
        self.failure_kind = None
        self.failure_count = 0
        self.same_target_failures = defaultdict(int)
        self.retry_wait_until = 0
        self.retry_serial = 0
        self.bank_transit_until = 0
        self.median_lateral_unsafe = False
        self.median_wait_unsafe = False
        self.bay_blacklist = {}
        self.entry_windows = {}
        self.contact_bans = defaultdict(list)
        self.failure_records = deque(maxlen=10)
        self.successful_entries = deque(maxlen=12)
        self.revision = 0

        self.pressed = None
        self.press_tick = -100
        self.press_origin = None
        self.ready_tick = 3
        self.last_action = "NOOP"
        self.last_plan = None
        self.last_log_tick = -100
        self.routes = []

        self.policy = {
            "route": "nearest",
            "risk": "normal",
            "tempo": "adaptive",
        }
        self.route_finder = RouteFinder(self)

        self.jev_future = None
        self.jev_meta = None
        self.next_jev = 0
        self.astra_future = None
        self.astra_meta = None
        self.next_astra = 0
        self.session_id = None

    def log(self, name, **fields):
        self.ctx.log(name, **fields)

    def compact(self, observation):
        observation = obj(observation)
        return {
            "tick": observation.get("tick"),
            "phase": phase_name(observation),
            "frog": observation.get("frog"),
            "score": observation.get("score"),
            "deaths": observation.get("deaths"),
            "homes": observation.get("frogs_home"),
            "timer": observation.get("timer_s"),
            "lanes": [
                {
                    "row": lane.get("row"),
                    "objects": [
                        {key: item.get(key) for key in ("id", "x", "nusiz", "kind", "hazard", "shape_ptr")}
                        for item in lane.get("objects", []) if isinstance(item, dict)
                    ],
                }
                for lane in list(observation.get("road", [])) + list(observation.get("river", []))
            ],
        }

    def entry_window(self, slot):
        # Start with the empirically successful center-minus-five..plus-one
        # window.  A failed entry narrows only that bay on the next retries.
        return self.entry_windows.get(str(slot), (-5.0, 1.0))

    def bay_blocked(self, slot, tick):
        return integer(self.bay_blacklist.get(str(slot), -1), -1) > tick

    def update_velocities(self, observation):
        tick = integer(observation.get("tick"), 0)
        for lane in list(observation.get("road", [])) + list(observation.get("river", [])):
            row = integer(lane.get("row"), -99)
            for index, item in enumerate(lane.get("objects", [])):
                if not isinstance(item, dict):
                    continue
                ident = str(item.get("id", (row, index)))
                history = self.histories.setdefault(ident, deque(maxlen=48))
                x = number(item.get("x"), 0.0)
                if not history or history[-1][0] != tick:
                    history.append((tick, x))
                old = next((entry for entry in history if 12 <= tick - entry[0] <= 36), None)
                direction = item.get("direction", lane.get("direction"))
                sign = -1.0 if direction == "left" else 1.0
                fallback = sign / (integer(lane.get("activity_mask"), 3) + 1)
                velocity = fallback
                if old is not None:
                    elapsed = max(1, tick - old[0])
                    displacement = (x - old[1] + 80.0) % 160.0 - 80.0
                    measured = displacement / elapsed
                    if abs(measured) <= 2.0 and measured * sign >= -0.05:
                        velocity = measured
                self.velocities[ident] = velocity

    def classify_failure(self, old_observation, new_observation):
        old_row = frog_row(old_observation)
        action = self.last_action if self.last_action in LEGAL else "NOOP"
        planned = self.last_plan or {}
        target = planned.get("label") if isinstance(planned, dict) else None
        if old_row == 10 and action == "UP":
            kind = "HOME_ENTRY"
        elif old_row == 5:
            kind = "MEDIAN"
        elif old_row >= 6:
            kind = "RIVER"
        else:
            kind = "ROAD"
        return kind, target

    def learn_contact(self, old_observation, kind):
        """Localize a carrier contradiction, never a final-entry failure."""
        if kind != "RIVER":
            return []
        old_row = frog_row(old_observation)
        if old_row < 6 or old_row > 10:
            return []
        frog_x = number(old_observation.get("frog", {}).get("x"), 80.0)
        examples = []
        lanes = [lane for lane in old_observation.get("river", []) if integer(lane.get("row"), -1) == old_row]
        for lane in lanes:
            for item in lane.get("objects", []):
                if not isinstance(item, dict) or item.get("kind") not in ("log", "turtle"):
                    continue
                nusiz = integer(item.get("nusiz"), 0) & 7
                shape = integer(item.get("shape_ptr"), -1)
                for offset in COPY_OFFSETS[nusiz]:
                    relative = (frog_x - number(item.get("x"), 0.0) - offset + 80.0) % 160.0 - 80.0
                    if -5 <= relative <= COPY_WIDTHS[nusiz] + 2:
                        band = (
                            max(-8.0, math.floor(relative) - 3.0),
                            min(float(COPY_WIDTHS[nusiz] + 8), math.ceil(relative) + 3.0),
                        )
                        key = (old_row, shape, nusiz)
                        self.contact_bans[key] = merge_intervals(self.contact_bans[key] + [band])[-8:]
                        examples.append({"row": old_row, "shape": shape, "nusiz": nusiz, "offset": relative})
        return examples

    def handle_epoch_change(self, observation):
        if self.previous is None:
            return
        old_epoch, new_epoch = epoch(self.previous), epoch(observation)
        if old_epoch == new_epoch:
            return
        tick = integer(observation.get("tick"), 0)
        if new_epoch[0] > old_epoch[0]:
            kind, target = self.classify_failure(self.previous, observation)
            self.failure_count += 1
            if target and str(target).startswith("HOME_"):
                self.same_target_failures[str(target)] += 1
                self.last_failed_target = str(target)
            else:
                self.last_failed_target = target
            if target and str(target).startswith("HOME_"):
                repeat = self.same_target_failures.get(str(target), 1)
            else:
                # Different-looking partial/median labels can still be the
                # same phase problem.  Increase the next spawn offset by the
                # total failure count, rather than resetting at each label.
                repeat = self.failure_count
            base = 30 if kind == "HOME_ENTRY" else 24
            delay = min(144, base * min(1 + repeat, 4))
            if self.policy.get("tempo") == "shift_large":
                delay = min(180, int(delay * 1.5))
            elif self.policy.get("tempo") == "shift_small":
                delay = min(160, int(delay * 1.2))
            self.retry_wait_until = tick + delay
            self.retry_serial += 1
            if kind == "MEDIAN":
                self.bank_transit_until = tick + 240
                if self.last_action in ("LEFT", "RIGHT"):
                    self.median_lateral_unsafe = True
                elif self.last_action == "NOOP":
                    self.median_wait_unsafe = True
            if target and str(target).startswith("HOME_"):
                slot = str(target).split("_", 1)[1]
                # Do not keep sending frogs into the same open bay if another
                # bay exists; the block expires, so it is not a permanent claim
                # that an unknown crocodile is present.
                self.bay_blacklist[slot] = tick + max(180, delay * 3)
                low, high = self.entry_window(slot)
                self.entry_windows[slot] = (max(-4.0, low + 1.0), min(0.0, high))
            examples = self.learn_contact(self.previous, kind)
            record = {
                "tick": tick, "kind": kind, "target": target,
                "before": old_epoch, "after": new_epoch,
                "delay_frames": delay, "retry_serial": self.retry_serial,
                "contact_examples": examples,
                "previous": self.compact(self.previous),
            }
            self.failure_kind = kind
            self.failure_records.append(record)
            self.goal = None
            self.pressed = None
            self.ready_tick = tick + 4
            self.revision += 1
            self.log("phase_retry_scheduled", **record)
        else:
            # A successful home invalidates stale bay blocks and provides an
            # entry sample without assuming that one successful x is universal.
            if new_epoch[1] > old_epoch[1]:
                plan = self.last_plan or {}
                if plan.get("label", "").startswith("HOME_"):
                    slot = plan["label"].split("_", 1)[1]
                    x = number(self.previous.get("frog", {}).get("x"), 80.0)
                    centre = next((number(h.get("x_center"), 80.0) for h in self.previous.get("homes", [])
                                   if str(h.get("slot")) == slot), None)
                    if centre is not None:
                        self.successful_entries.append({"slot": slot, "offset": x - centre, "tick": tick})
                self.bay_blacklist.clear()
            self.goal = None
            self.pressed = None
            self.ready_tick = tick + 4
            self.revision += 1
            self.log("progress_epoch", tick=tick, before=old_epoch, after=new_epoch,
                     last_plan=self.last_plan)
        self.previous = observation

    def observe_update(self, observation):
        observation = obj(observation)
        self.handle_epoch_change(observation)
        self.update_velocities(observation)
        if self.previous is not None:
            if frog_row(observation) != frog_row(self.previous) or observation.get("score") != self.previous.get("score"):
                self.log("progress", tick=observation.get("tick"), frog=observation.get("frog"),
                         score=observation.get("score"), homes=observation.get("frogs_home"),
                         deaths=observation.get("deaths"))
        self.previous = observation

    def compact_routes(self, routes):
        values = []
        for route in routes[:5]:
            values.append({
                "label": route.get("label"),
                "kind": "home" if str(route.get("label", "")).startswith("HOME_") else "median_or_progress",
                "verified": bool(route.get("complete")),
                "travel": "short" if integer(route.get("steps"), 99) <= 12 else "long",
                "clearance": "comfortable" if number(route.get("margin"), 0) >= 2 else "tight",
                "current_goal": route.get("label") == self.goal,
            })
        return values

    def request_jev(self, observation, routes):
        if self.jev_future is not None or not routes:
            return
        if getattr(self.ctx, "system1", True) is None:
            return
        if self.ctx.budget.remaining().get("system1_calls", 0) < 2:
            return
        tick = integer(observation.get("tick"), 0)
        candidates = [route for route in routes if route.get("complete")][:5]
        if not candidates:
            return
        options = {
            route["label"]: {
                "what": "A code-verified durable route goal, not an immediate joystick command.",
                "avoid": "Do not invent a direction or override the next-frame geometry check.",
            }
            for route in candidates
        }
        options["UNKNOWN"] = "No listed goal is supported by the current forecast."
        instruction = (
            "Choose a durable Frogger route goal from `candidates`. Prefer a current goal, "
            "then a comfortable route; a tight route is acceptable only when it is the only "
            "verified crossing. These options are generated from the latest object tracks. "
            "Do not calculate, choose an immediate action, or infer that an unknown home "
            "crocodile is absent. UNKNOWN if the facts do not support a goal."
        )
        retry_options = {
            "KEEP_PHASE": "No recent replay failure is established, or the current phase is adequate.",
            "SHIFT_PHASE": "A recent same-target failure makes a bounded start delay useful before retrying.",
            "USE_ALTERNATE": "Another currently open bay or verified route should be preferred over the failed target.",
            "UNKNOWN": "The supplied evidence does not support a retry preference.",
        }
        retry_instruction = (
            "Judge only the explicit `retry` facts. Choose SHIFT_PHASE after a recent repeated "
            "home-entry or river timing failure; choose USE_ALTERNATE only when an alternate "
            "open bay is listed. This changes a future route preference, never the current "
            "direction. KEEP_PHASE when no failure evidence is present."
        )
        state = {
            "game": "Atari 2600 Frogger; preserve frogs and fill open homes.",
            "tick_age": 0,
            "phase": phase_name(observation),
            "frog": {"row": frog_row(observation), "x": number(observation.get("frog", {}).get("x"), 80)},
            "timer": "urgent" if number(observation.get("timer_s"), 30) < 8 else "normal",
            "candidates": self.compact_routes(candidates),
            "current_goal": self.goal,
            "retry": {
                "recent_failure": self.failure_kind,
                "same_target_failures": self.same_target_failures.get(self.last_failed_target, 0),
                "phase_shift_already_scheduled": self.retry_wait_until > tick,
                "alternate_open_bay": any(
                    route.get("label") != self.last_failed_target and str(route.get("label", "")).startswith("HOME_")
                    for route in candidates
                ),
            },
            "rules": "Code owns arithmetic, object copies, collision margins, legal actions, and freshness.",
        }
        questions = {
            "goal": Choice(instruction, options),
            "goal_reverse": Choice(instruction, dict(reversed(list(options.items())))),
            "retry": Choice(retry_instruction, retry_options),
            "retry_reverse": Choice(retry_instruction, dict(reversed(list(retry_options.items())))),
            "replay_evidence": Noul(
                "Does `retry.recent_failure` explicitly establish a repeated timing or home-entry failure? "
                "True means yes; false means no or unknown."
            ),
        }
        self.jev_meta = {
            "tick": tick, "epoch": epoch(observation), "phase": phase_name(observation),
            "revision": self.revision, "labels": set(options),
        }
        next_jev = tick + (60 if self.failure_kind else 90)
        self.next_jev = next_jev

        def call():
            try:
                answer = self.ctx.decide(state, questions, timeout_s=3.0)
                return answer
            except (BudgetExceeded, EpisodeOver, ModelUnavailable):
                raise
            except (TimeoutError, RuntimeError, TypeError):
                return None

        self.jev_future = self.ctx.spawn(call)

    def request_astra(self, observation):
        if self.astra_future is not None:
            return
        if getattr(self.ctx, "system2", True) is None:
            return
        if self.ctx.budget.remaining().get("system2_calls", 0) < 2:
            return
        tick = integer(observation.get("tick"), 0)
        summary = {
            "policy": self.policy,
            "level": observation.get("level"),
            "score": observation.get("score"),
            "recent_failures": list(self.failure_records)[-3:],
            "successful_entries": list(self.successful_entries)[-4:],
            "open_homes": [
                {"slot": h.get("slot"), "occupied": h.get("occupied"), "available": h.get("available"),
                 "crocodile": h.get("crocodile")}
                for h in observation.get("homes", []) if isinstance(h, dict)
            ],
        }
        prompt = (
            "You are the slow strategy and experiment adviser for an Atari 2600 Frogger "
            "controller. Do not give a joystick action or code. The executor rebuilds a "
            "time-expanded route from RAM object tracks on every hop, models separated "
            "NUSIZ copies, releases direction presses, and rechecks the live observation. "
            "The new experiment treats repeated deaths as temporal evidence: after a "
            "home-entry/river failure it waits on the starting sidewalk for a bounded "
            "phase shift instead of replaying the same route. It may quarantine a failed "
            "bay only when another open bay exists. Home crocodile status is unknown, not "
            "safe. Jev selects only a durable goal and a retry preference from verified "
            "candidates. Return ONLY a JSON object with optional enum fields: "
            "route=nearest|left|right|wide, risk=normal|cautious, "
            "tempo=adaptive|shift_small|shift_large. Choose the smallest change supported "
            "by repeated failure evidence; do not turn a single failure into a global margin "
            "inflation. Current summary: " + json.dumps(summary, separators=(",", ":"))
        )
        self.astra_meta = {"level": observation.get("level"), "tick": tick}
        self.next_astra = tick + 3000
        session = self.session_id

        def call():
            try:
                return self.ctx.think(prompt, session_id=session, timeout_s=120.0)
            except (BudgetExceeded, EpisodeOver, ModelUnavailable):
                raise
            except (TimeoutError, RuntimeError):
                return None

        self.astra_future = self.ctx.spawn(call)

    def poll_models(self, observation):
        tick = integer(observation.get("tick"), 0)
        if self.jev_future is not None and self.jev_future.done():
            future, self.jev_future = self.jev_future, None
            try:
                answers = future.result()
            except (BudgetExceeded, EpisodeOver):
                answers = None
            if isinstance(answers, dict) and self.jev_meta is not None:
                meta = self.jev_meta
                fresh = (
                    epoch(observation) == meta["epoch"]
                    and phase_name(observation) == meta["phase"]
                    and self.revision == meta["revision"]
                    and tick - meta["tick"] <= 120
                )
                first = answers.get("goal")
                second = answers.get("goal_reverse")
                selected = getattr(first, "value", None)
                goal_ok = (
                    fresh and isinstance(selected, str) and selected == getattr(second, "value", None)
                    and selected in meta["labels"] and min(probability(first, selected), probability(second, selected)) >= 0.62
                )
                if goal_ok and selected != "UNKNOWN":
                    self.goal = selected
                    self.goal_until = tick + 180
                    self.log("jev_goal_adopted", tick=tick, goal=selected,
                             probability=min(probability(first, selected), probability(second, selected)))
                else:
                    self.log("jev_goal_rejected", tick=tick, fresh=fresh, selected=selected)

                retry = answers.get("retry")
                retry_other = answers.get("retry_reverse")
                retry_value = getattr(retry, "value", None)
                retry_ok = (
                    fresh and retry_value == getattr(retry_other, "value", None)
                    and retry_value in ("SHIFT_PHASE", "USE_ALTERNATE", "KEEP_PHASE")
                    and min(probability(retry, retry_value), probability(retry_other, retry_value)) >= 0.62
                )
                if retry_ok and retry_value == "SHIFT_PHASE" and frog_row(observation) < 0:
                    self.retry_wait_until = max(self.retry_wait_until, tick + 18)
                    self.log("jev_phase_shift_accepted", tick=tick, until=self.retry_wait_until)
                elif retry_ok and retry_value == "USE_ALTERNATE":
                    self.log("jev_alternate_preference", tick=tick)

        if self.astra_future is not None and self.astra_future.done():
            future, self.astra_future = self.astra_future, None
            try:
                result = future.result()
            except (BudgetExceeded, EpisodeOver):
                result = None
            if result is not None:
                self.session_id = getattr(result, "session_id", None)
                proposed = extract_json(getattr(result, "text", ""))
                allowed = {
                    "route": ("nearest", "left", "right", "wide"),
                    "risk": ("normal", "cautious"),
                    "tempo": ("adaptive", "shift_small", "shift_large"),
                }
                changes = {
                    key: value for key, value in proposed.items()
                    if key in allowed and isinstance(value, str) and value in allowed[key]
                }
                if changes and self.astra_meta and self.astra_meta.get("level") == observation.get("level"):
                    self.policy.update(changes)
                    self.road_pad = 2.25 if self.policy["risk"] == "cautious" else 1.5
                    self.river_pad = 2.5 if self.policy["risk"] == "cautious" else 2.0
                    self.revision += 1
                    self.goal = None
                    self.log("astra_policy_applied", tick=tick, policy=dict(self.policy))
                else:
                    self.log("astra_output_rejected", tick=tick)

    def route_choice(self, routes, tick):
        if not routes:
            return None
        best = routes[0]
        if self.goal and tick <= self.goal_until:
            same = next((route for route in routes if route.get("label") == self.goal), None)
            if same is not None and same.get("value", -1e9) >= best.get("value", -1e9) - 10:
                best = same
        # Astra's route preference only breaks near ties; it never overrides a
        # complete route with a partial or unverified route.
        if self.policy.get("route") in ("left", "right"):
            prefix = "MEDIAN_" + self.policy["route"].upper()
            alternatives = [route for route in routes if route.get("label") == prefix]
            if alternatives and alternatives[0].get("value", -1e9) >= best.get("value", -1e9) - 8:
                best = alternatives[0]
        elif self.policy.get("route") == "wide":
            wide = max(routes[:5], key=lambda route: number(route.get("margin"), -99))
            if wide.get("value", -1e9) >= best.get("value", -1e9) - 8:
                best = wide
        return best

    def choose_action(self, observation):
        tick = integer(observation.get("tick"), 0)
        if not active_frog(observation):
            self.pressed = None
            self.ready_tick = max(self.ready_tick, tick + 2)
            return "NOOP"
        row = frog_row(observation)
        if row == -1 and tick < self.retry_wait_until:
            self.pressed = None
            return "NOOP"
        if self.pressed is not None:
            moved = (
                self.press_origin is not None
                and (row != self.press_origin[0]
                     or abs(number(observation.get("frog", {}).get("x"), 80.0) - self.press_origin[1]) >= 6)
            )
            if moved or tick - self.press_tick >= 3:
                old = self.pressed
                self.pressed = None
                self.ready_tick = max(self.ready_tick, tick + 2)
                self.log("pulse_release", tick=tick, action=old, moved=moved,
                         origin=self.press_origin, frog=observation.get("frog"))
                return "NOOP"
            return self.pressed
        if tick < self.ready_tick or not -1 <= row <= 10:
            return "NOOP"

        routes = self.route_finder.find(observation)
        self.routes = routes
        if tick >= self.next_jev:
            self.request_jev(observation, routes)
        chosen = self.route_choice(routes, tick)
        if chosen is None:
            if tick - self.last_log_tick >= 60:
                self.log("no_route", tick=tick, snapshot=self.compact(observation),
                         search_ms=round(self.route_finder.last_ms, 2))
                self.last_log_tick = tick
            self.ready_tick = tick + 3
            return "NOOP"

        action = chosen.get("action") if chosen.get("action") in LEGAL else "NOOP"
        # This guard is intentionally independent of the beam's longer path.
        # Search can be stale by the time a background Jev/Astra call completes.
        latest = obj(self.ctx.observe())
        if latest and epoch(latest) == epoch(observation) and frog_row(latest) == row:
            check = Forecast(self, latest, 1)
            if not check.one_hop_safe(latest, action):
                alternatives = [move[0] for move in MOVES if check.one_hop_safe(latest, move[0])]
                replacement = next((candidate for candidate in alternatives if candidate != "DOWN"), "NOOP")
                self.log("live_guard_veto", tick=integer(latest.get("tick"), tick),
                         before=action, after=replacement, planned=chosen.get("label"))
                action = replacement
        else:
            action = "NOOP"

        chosen = dict(chosen, action=action)
        self.last_plan = {
            "tick": tick, "origin": (row, number(observation.get("frog", {}).get("x"), 80.0)),
            **chosen,
        }
        if action != "NOOP" or tick - self.last_log_tick >= 60:
            self.log("route_action", tick=tick, frog=observation.get("frog"),
                     selected=chosen, alternatives=routes[:5],
                     retry_serial=self.retry_serial, retry_wait_until=self.retry_wait_until,
                     search_ms=round(self.route_finder.last_ms, 2))
            self.last_log_tick = tick
        self.ready_tick = tick + (6 if action != "NOOP" else 3)
        if action != "NOOP":
            self.pressed = action
            self.press_tick = tick
            self.press_origin = (row, number(observation.get("frog", {}).get("x"), 80.0))
        return action

    def run_episode(self, game):
        while not self.ctx.done:
            observation = obj(self.ctx.observe())
            if not observation:
                self.ctx.act("NOOP")
                self.ctx.wait_for_tick()
                continue
            self.observe_update(observation)
            self.poll_models(observation)
            tick = integer(observation.get("tick"), 0)
            if self.goal and tick > self.goal_until:
                self.goal = None
            if self.failure_count and tick >= self.next_astra:
                self.request_astra(observation)
            action = self.choose_action(observation)
            self.last_action = action
            self.ctx.act(action)
            self.ctx.wait_for_tick()


def build(ctx):
    return Controller(ctx)
