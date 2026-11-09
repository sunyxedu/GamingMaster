"""Frogger controller: exact carrier ink plus a risk-calibrated route search.

The controller is deliberately conservative about river support.  The adaptor exposes
wrapped x_intervals for each decoded object; those intervals are used as carrier ink,
while NUSIZ expansion is used only as an additional road-car hazard envelope.  A
bounded search chooses one hop at a time, and the motor always releases a direction
before planning the next hop.
"""
from __future__ import annotations

import json
import math
import time
from collections import deque

from metaharness.core.contract import BudgetExceeded, EpisodeOver, ModelUnavailable
from metaharness.models.jev import Choice, Noul


# These are fallback sprite layouts.  Copy-bearing layouts are used to
# reconstruct separated river carriers; for road cars they are combined with
# the public interval as a conservative hazard envelope.  A gap between
# carrier copies remains water.
NUSIZ_LAYOUT = {
    0: ((0,), 8),
    1: ((0, 16), 8),
    2: ((0, 32), 8),
    3: ((0, 16, 32), 8),
    4: ((0, 64), 8),
    5: ((0,), 16),
    6: ((0, 32, 64), 8),
    7: ((0,), 24),
}

ACTIONS = ("UP", "LEFT", "RIGHT", "NOOP", "DOWN")


def finite(value, default=0.0):
    try:
        result = float(value)
        return result if math.isfinite(result) else default
    except (TypeError, ValueError):
        return default


def integer(value, default=0):
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def unpack(value):
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except (TypeError, ValueError):
            return {}
    return value if isinstance(value, dict) else {}


def json_fragment(text):
    if not isinstance(text, str):
        return {}
    try:
        left, right = text.index("{"), text.rindex("}")
        return unpack(text[left : right + 1])
    except (ValueError, TypeError):
        return {}


def frog_row(obs):
    raw = integer(obs.get("frog", {}).get("row"), 255)
    return -1 if raw == 255 else raw


def episode_epoch(obs):
    return (
        integer(obs.get("deaths"), 0),
        integer(obs.get("frogs_home"), 0),
        integer(obs.get("level"), 1),
    )


def wrapped_delta(new, old):
    delta = finite(new) - finite(old)
    while delta > 80:
        delta -= 160
    while delta < -80:
        delta += 160
    return delta


def object_key(lane, obj):
    return str(obj.get("id", "%s_%s" % (lane.get("row"), obj.get("index", 0))))


def public_intervals(obj):
    """Return the adaptor's wrapped collision/render intervals.

    A segment ending at 160 and a segment beginning at zero are kept separate;
    future translation handles the wrap with unwrapped copies.  This preserves
    gaps instead of turning a sprite group into a rectangle.
    """
    raw = obj.get("x_intervals")
    if not isinstance(raw, list) or not raw:
        raw = [obj.get("x_interval")] if obj.get("x_interval") is not None else []
    result = []
    for pair in raw:
        if not isinstance(pair, (list, tuple)) or len(pair) < 2:
            continue
        a, b = finite(pair[0]), finite(pair[1])
        if b < a:
            a, b = b, a
        if b <= a:
            b = a + max(1.0, finite(obj.get("width_px_estimate"), 1.0))
        result.append((a, b))
    return result


def layout_intervals(obj):
    """Expand a NUSIZ code into a conservative *hazard* envelope."""
    n = integer(obj.get("nusiz"), 0) & 7
    offsets, width = NUSIZ_LAYOUT.get(n, NUSIZ_LAYOUT[0])
    base = finite(obj.get("x"))
    result = []
    for offset in offsets:
        start = (base + offset) % 160
        end = start + width
        if end <= 160:
            result.append((start, end))
        else:
            result.extend(((start, 160.0), (0.0, end - 160.0)))
    return result


def carrier_base_intervals(obj):
    """Return separated carrier ink, not the decoder's total span.

    For copy-bearing NUSIZ values the public interval often spans from the
    first to the last copy (for example NUSIZ-6 at x=116 reports [116,148]),
    so using that span would turn the water between copies into a log.  The
    low-bit layout supplies the individual eight-pixel copies.  Single and
    enlarged sprites retain the public wrapped interval; NUSIZ-7 is corrected
    to its observed 24-pixel triple width.
    """
    n = integer(obj.get("nusiz"), 0) & 7
    if n in (1, 2, 3, 4, 6):
        return layout_intervals(obj)
    intervals = public_intervals(obj)
    return intervals if intervals else layout_intervals(obj)


def shifted_segments(segments, displacement):
    result = []
    for a, b in segments:
        for wrap in (-160.0, 0.0, 160.0):
            result.append((a + displacement + wrap, b + displacement + wrap))
    return result


def overlap(a, b, c, d):
    return min(b, d) - max(a, c)


def interval_gap(a, b, c, d):
    if b < c:
        return c - b
    if d < a:
        return a - d
    return 0.0


class Forecast:
    """One short public-state forecast.  It never reads emulator state."""

    def __init__(self, controller, obs):
        self.controller = controller
        self.obs = obs
        self.now = integer(obs.get("tick"), 0)
        self.frame = integer(obs.get("frame", obs.get("frame_number")), self.now)
        self.difficulty = integer(obs.get("difficulty"), 0)
        self.lanes = {}
        for lane in obs.get("road", []) + obs.get("river", []):
            self.lanes[integer(lane.get("row"), -99)] = lane
        self.homes = [
            h
            for h in obs.get("homes", [])
            if not h.get("occupied")
            and h.get("available", True)
            and h.get("crocodile") not in (True, "present", "dangerous")
        ]
        self.edge_escape = bool(obs.get("frog", {}).get("moving_with_platform"))
        self._margin_cache = {}
        self._drift_cache = {}

    def lane_shift(self, row, dt, object_id=None):
        """Predict integer activity phases, with a fractional fallback."""
        lane = self.lanes.get(row, {})
        period = max(1, integer(lane.get("activity_mask"), 3) + 1)
        direction = -1 if lane.get("direction") == "left" else 1
        phase = self.controller.phases.get(row)
        magnitude = self.controller.motion_size.get(row, 1.0)
        if phase is None:
            return direction * magnitude * float(dt) / period
        steps = 0
        for k in range(1, max(0, int(dt)) + 1):
            if (self.frame + k) % period == phase:
                steps += 1
        # A fractional remainder is less dangerous than pretending that a
        # quantized update happened on every display frame.
        return direction * magnitude * steps

    def object_shift(self, lane, obj, dt):
        row = integer(lane.get("row"), -99)
        return self.lane_shift(row, dt, object_key(lane, obj))

    def uncertainty_shifts(self, lane, obj, dt):
        base = self.object_shift(lane, obj, dt)
        row = integer(lane.get("row"), -99)
        direction = -1 if lane.get("direction") == "left" else 1
        if self.controller.phases.get(row) is None and dt:
            # Before the first observed update, timing is uncertain but the
            # carrier still moves in its decoded direction.  Never use an
            # uncertainty interval that invents a direction reversal.
            return (0.0, base, base + direction * 1.0)
        if self.controller.risk_mode == "strict" and dt:
            return (0.0, base, base + direction * 0.5)
        return (base,)

    def carrier_segments(self, lane, obj, dt):
        return shifted_segments(carrier_base_intervals(obj), self.object_shift(lane, obj, dt))

    def road_segments(self, lane, obj, dt):
        # Public intervals are authoritative where present.  The NUSIZ envelope
        # is additionally checked for cars, since a false road clearance is more
        # costly than a slightly delayed road hop.
        exact = public_intervals(obj)
        broad = layout_intervals(obj)
        segments = exact + broad
        result = []
        for shift in self.uncertainty_shifts(lane, obj, dt):
            result.extend(shifted_segments(segments, shift))
        return result

    def river_margin(self, row, frog_x, dt, need):
        lane = self.lanes.get(row)
        if not lane:
            return -100.0
        body_a, body_b = frog_x, frog_x + 8.0
        object_scores = []
        for obj in lane.get("objects", []):
            if obj.get("hazard") or obj.get("kind") not in ("log", "turtle"):
                continue
            # Every uncertain phase prediction must leave this particular
            # carrier usable; then choose the carrier with the best worst case.
            per_obj = []
            shifts = self.uncertainty_shifts(lane, obj, dt)
            for displacement in shifts:
                per_obj.append(
                    max(
                        (
                            overlap(body_a, body_b, a, b)
                            for a, b in shifted_segments(carrier_base_intervals(obj), displacement)
                        ),
                        default=-100.0,
                    )
                )
            object_scores.append(min(per_obj))
        if not object_scores:
            return -100.0
        return max(object_scores) - float(need)

    def road_margin(self, row, frog_x, dt):
        lane = self.lanes.get(row)
        if not lane:
            return -100.0
        guard = self.controller.road_guard
        body_a, body_b = frog_x - guard, frog_x + 8.0 + guard
        best_gap = 100.0
        for obj in lane.get("objects", []):
            if not obj.get("hazard") and obj.get("kind") not in ("vehicle", "car"):
                continue
            for a, b in self.road_segments(lane, obj, dt):
                if overlap(body_a, body_b, a, b) >= 0:
                    return -1.0
                best_gap = min(best_gap, interval_gap(body_a, body_b, a, b))
        return best_gap

    def drift(self, row, dt, frog_x=None):
        cache_key = (row, int(dt), None if frog_x is None else round(frog_x, 1))
        if cache_key in self._drift_cache:
            return self._drift_cache[cache_key]
        lane = self.lanes.get(row)
        if not lane or row < 6 or row > 10:
            self._drift_cache[cache_key] = 0.0
            return 0.0
        # Follow the carrier under the frog when identifiable; otherwise use
        # the lane phase.  This avoids carrying every route by the average of
        # unrelated objects.
        if frog_x is not None:
            best = -100.0
            selected = None
            body_a, body_b = frog_x, frog_x + 8.0
            for obj in lane.get("objects", []):
                if obj.get("hazard") or obj.get("kind") not in ("log", "turtle"):
                    continue
                score = max(
                    (
                        overlap(body_a, body_b, a, b)
                        for a, b in self.carrier_segments(lane, obj, 0)
                    ),
                    default=-100.0,
                )
                if score > best:
                    best, selected = score, obj
            if selected is not None and best >= 1.0:
                result = self.object_shift(lane, selected, dt)
                self._drift_cache[cache_key] = result
                return result
        result = self.lane_shift(row, dt)
        self._drift_cache[cache_key] = result
        return result

    def home_margin(self, frog_x, slot):
        home = next((h for h in self.homes if str(h.get("slot")) == str(slot)), None)
        if home is None:
            return -100.0
        center = finite(home.get("x_center"), 80.0)
        # Frog x is the left edge.  This is the observed opening, not a centre
        # point: center-5 is the preferred entry and center+1 is the edge.
        return min(frog_x - (center - 5.0), center + 1.0 - frog_x)

    def state_margin(self, row, frog_x, dt, slot=None, need=None):
        if row < -1 or row > 11:
            return -100.0
        if self.difficulty == 1 and row >= 6 and row <= 10:
            frog_x = frog_x % 160.0
        actual_need = self.controller.ride_need if need is None else need
        cache_key = (row, round(frog_x, 2), int(dt), str(slot), round(float(actual_need), 2))
        if cache_key in self._margin_cache:
            return self._margin_cache[cache_key]
        if row in (-1, 5):
            result = 25.0
        elif row == 11:
            result = self.home_margin(frog_x, slot)
        elif frog_x < 0.0 or frog_x > 152.0:
            result = -100.0
        elif row < 5:
            result = self.road_margin(row, frog_x, dt)
        else:
            result = self.river_margin(row, frog_x, dt, actual_need)
        self._margin_cache[cache_key] = result
        return result

    def transition(self, row, frog_x, action, slot=None, horizon=6):
        if action not in ACTIONS:
            return None
        if action == "UP":
            next_row = row + 1
            next_x = frog_x
        elif action == "DOWN":
            next_row = row - 1
            next_x = frog_x
        elif action == "LEFT":
            next_row = row
            next_x = frog_x - 8.0
        elif action == "RIGHT":
            next_row = row
            next_x = frog_x + 8.0
        else:
            next_row = row
            next_x = frog_x
        if next_row < -1 or next_row > 11:
            return None
        if row >= 5 and row <= 10 and next_row < 5:
            return None
        if next_row >= 0 and next_row <= 10 and (next_x < 0 or next_x > 152):
            return None

        source_margin = self.state_margin(row, frog_x, 0, slot=slot)
        lateral_escape = (
            row >= 6
            and row <= 10
            and action in ("LEFT", "RIGHT")
            and source_margin < 0
            and self.edge_escape
        )
        if source_margin < 0 and not lateral_escape:
            return None

        minimum = 100.0
        for s in range(1, horizon + 1):
            # The first frame is the joystick sampling window.  Later frames
            # are the destination/ride window.  Check every frame, not just the
            # predicted landing frame.
            if action != "NOOP" and s == 1:
                check_row = row
                check_x = frog_x + (self.drift(row, s, frog_x) if 6 <= row <= 10 else 0.0)
                check_need = self.controller.ride_need
                check_slot = slot
                if lateral_escape and source_margin < 0:
                    # The frog is already at an observed carrier edge.  Its
                    # only useful action is an immediate move toward ink.
                    continue
            else:
                check_row = next_row
                if 6 <= next_row <= 10:
                    check_x = next_x + self.drift(next_row, max(0, s - 1), next_x)
                else:
                    check_x = next_x
                check_need = self.controller.landing_need if action != "NOOP" else self.controller.ride_need
                check_slot = slot
                if action != "NOOP" and s >= 3:
                    check_need = self.controller.ride_need
            margin = self.state_margin(check_row, check_x, s, slot=check_slot, need=check_need)
            if margin < 0:
                return None
            minimum = min(minimum, margin)

        if next_row >= 6 and next_row <= 10:
            end_x = next_x + self.drift(next_row, max(0, horizon - 1), next_x)
        else:
            end_x = next_x
        if self.difficulty != 1 and (end_x < 0 or end_x > 152):
            return None
        return {
            "row": next_row,
            "x": end_x,
            "margin": minimum,
            "edge": minimum < 2.0,
        }


class Harness:
    def __init__(self, ctx):
        self.ctx = ctx
        self.previous = None
        self.last_epoch = None
        self.phases = {}
        self.motion_size = {}
        self.risk_mode = "balanced"
        self.risk_level = 0
        self.road_guard = 1.5
        self.ride_need = 1.0
        self.landing_need = 2.0
        self.home_order = "nearest"
        self.style = "safe"
        self.plan_version = 0
        self.failure_history = deque(maxlen=8)
        self.recent_outcomes = deque(maxlen=8)
        self.last_certificate = None
        self.last_routes = []
        self.last_search_ms = 0.0

        self.press = None
        self.press_origin = None
        self.press_tick = -100
        self.release_until = 0
        self.ready_tick = 0
        self.last_action = "NOOP"
        self.last_log_tick = -100

        self.jev_future = None
        self.jev_meta = None
        self.jev_lease = None
        self.next_jev = 0
        self.astra_future = None
        self.astra_meta = None
        self.next_astra = 600
        self.session_id = None

    def log(self, event, **fields):
        self.ctx.log(event, **fields)

    def compact_obs(self, obs):
        return {
            "tick": integer(obs.get("tick")),
            "frog": obs.get("frog"),
            "score": finite(obs.get("score")),
            "timer_bucket": "urgent" if finite(obs.get("timer_s"), 30) < 7 else "normal",
            "epoch": episode_epoch(obs),
        }

    def update_motion(self, obs):
        if not self.previous:
            return
        old_lanes = {
            integer(l.get("row"), -99): l
            for l in self.previous.get("road", []) + self.previous.get("river", [])
        }
        tick = integer(obs.get("tick"), 0)
        old_tick = integer(self.previous.get("tick"), tick - 1)
        if tick - old_tick != 1:
            return
        for lane in obs.get("road", []) + obs.get("river", []):
            row = integer(lane.get("row"), -99)
            old_lane = old_lanes.get(row, {})
            old_objects = {
                object_key(old_lane, z): z for z in old_lane.get("objects", [])
            }
            period = max(1, integer(lane.get("activity_mask"), 3) + 1)
            for obj in lane.get("objects", []):
                old = old_objects.get(object_key(lane, obj))
                if old is None:
                    continue
                dx = wrapped_delta(obj.get("x"), old.get("x"))
                if abs(dx) > 0.05 and abs(dx) <= 8:
                    self.phases[row] = integer(obs.get("frame", obs.get("frame_number")), tick) % period
                    self.motion_size[row] = max(1.0, min(4.0, abs(dx)))

    def update(self, obs):
        tick = integer(obs.get("tick"), 0)
        current_epoch = episode_epoch(obs)
        self.update_motion(obs)
        old = self.previous
        old_epoch = self.last_epoch
        if self.last_epoch != current_epoch:
            if self.last_epoch is not None:
                death = current_epoch[0] > self.last_epoch[0]
                record = {
                    "tick": tick,
                    "death": death,
                    "before": self.last_epoch,
                    "after": current_epoch,
                    "certificate": self.last_certificate,
                    "before_obs": self.compact_obs(old or {}),
                    "after_obs": self.compact_obs(obs),
                }
                self.recent_outcomes.append(record)
                if death:
                    self.failure_history.append(record)
                    # A death is evidence that the previous forecast's edge
                    # margin was not enough.  Tighten only this episode; do not
                    # learn a seed-specific route.
                    self.risk_level = min(3, self.risk_level + 1)
                    self.road_guard = min(4.0, self.road_guard + 0.5)
                    self.plan_version += 1
                    self.jev_lease = None
                    self.phases.clear()
                    self.motion_size.clear()
                    self.log("forecast_failure", **record)
                else:
                    self.plan_version += 1
                    self.jev_lease = None
                    self.log("home_or_level", **record)
            self.last_epoch = current_epoch
            self.press = None
            self.press_origin = None
            self.release_until = tick + 4
            self.ready_tick = tick + 4
            if old_epoch is not None and old_epoch[2] != current_epoch[2]:
                self.phases.clear()
                self.motion_size.clear()
        self.previous = obs

        if self.risk_mode == "strict":
            self.road_guard = max(self.road_guard, 2.5)
        if self.risk_level >= 1:
            self.landing_need = 3.0
        else:
            self.landing_need = 2.0
        if self.risk_level >= 2:
            self.ride_need = 2.0
        else:
            self.ride_need = 1.0

    def open_homes(self, obs, x):
        homes = [
            h
            for h in obs.get("homes", [])
            if not h.get("occupied")
            and h.get("available", True)
            and h.get("crocodile") not in (True, "present", "dangerous")
        ]
        if self.home_order == "left":
            homes.sort(key=lambda h: finite(h.get("x_center"), 80))
        elif self.home_order == "right":
            homes.sort(key=lambda h: -finite(h.get("x_center"), 80))
        else:
            homes.sort(key=lambda h: abs(finite(h.get("x_center"), 80) - (x + 4)))
        return homes[:4]

    def route_value(self, current_row, state, cost, target, complete):
        row = state["row"]
        x = state["x"]
        margin = state["margin"]
        distance = abs((x + 4.0) - target) if target is not None else 0.0
        value = row * 70.0 - cost + min(margin, 8.0) * (3.0 if self.style == "safe" else 1.0)
        value -= distance * (0.08 if row >= 8 else 0.015)
        if complete:
            value += 1200.0
        if state.get("edge"):
            value -= 12.0 if self.style == "safe" else 4.0
        return value

    def search(self, obs):
        started = time.perf_counter()
        forecast = Forecast(self, obs)
        start_row = frog_row(obs)
        start_x = finite(obs.get("frog", {}).get("x"), 80.0)
        if start_row < -1 or start_row > 10:
            return []

        deadline = started + 0.009
        if start_row < 5:
            goals = [("MEDIAN", None, 5)]
        else:
            goals = [
                ("HOME_" + str(h.get("slot")), finite(h.get("x_center"), 80.0), 11)
                for h in self.open_homes(obs, start_x)
            ]
        all_routes = []
        # A separate small beam per open home avoids a single attractive bay
        # starving safer alternatives.  The deadline bounds its real-time cost.
        for label, target, endpoint in goals:
            if time.perf_counter() > deadline:
                break
            beam = [(start_row, start_x, None, (), 25.0, 0.0)]
            best_partial = None
            exhausted = False
            for depth in range(12 if start_row >= 5 else 8):
                if time.perf_counter() > deadline:
                    break
                next_beam = []
                for rr, xx, first, path, bottleneck, cost in beam:
                    if time.perf_counter() > deadline:
                        exhausted = True
                        break
                    for action in ACTIONS:
                        if time.perf_counter() > deadline:
                            exhausted = True
                            break
                        if action == "DOWN" and rr >= 5 and rr <= 10 and rr - 1 < 5:
                            continue
                        transition = forecast.transition(rr, xx, action, None if endpoint == 5 else label[5:])
                        if transition is None:
                            continue
                        nr, nx = transition["row"], transition["x"]
                        new_path = path + (action,)
                        new_first = first or action
                        new_margin = min(bottleneck, transition["margin"])
                        new_cost = cost + {
                            "UP": 0.2,
                            "NOOP": 0.45,
                            "LEFT": 2.0,
                            "RIGHT": 2.0,
                            "DOWN": 7.0,
                        }[action]
                        complete = nr == endpoint
                        candidate_state = {
                            "row": nr,
                            "x": nx,
                            "margin": new_margin,
                            "edge": new_margin < 2.0,
                        }
                        value = self.route_value(start_row, candidate_state, new_cost, target, complete)
                        if nr > start_row and (best_partial is None or value > best_partial["value"]):
                            best_partial = {
                                "label": label if endpoint == 11 else "PROGRESS",
                                "action": new_first,
                                "path": new_path,
                                "value": value - (500.0 if not complete else 0.0),
                                "margin": round(new_margin, 2),
                                "x": round(nx, 2),
                                "landing_row": nr,
                                "complete": False,
                                "surface": "interior" if new_margin >= 3 else "edge",
                                "route": "home_reached" if complete else "forward_connected",
                            }
                        if complete:
                            route = {
                                "label": label,
                                "action": new_first,
                                "path": new_path,
                                "value": value,
                                "margin": round(new_margin, 2),
                                "x": round(nx, 2),
                                "landing_row": nr,
                                "complete": True,
                                "surface": "interior" if new_margin >= 3 else "edge",
                                "route": "home_reached",
                            }
                            all_routes.append(route)
                            continue
                        next_beam.append((nr, nx, new_first, new_path, new_margin, new_cost))
                if exhausted:
                    break
                if not next_beam:
                    break
                # Retain first-action diversity, with a two-pixel position
                # bucket.  This is a receding-horizon beam, not a fixed route.
                next_beam.sort(
                    key=lambda q: self.route_value(
                        start_row,
                        {"row": q[0], "x": q[1], "margin": q[4], "edge": q[4] < 2},
                        q[5],
                        target,
                        False,
                    ),
                    reverse=True,
                )
                kept, seen, counts = [], set(), {}
                for state in next_beam:
                    key = (state[2], state[0], round(state[1] / 2.0))
                    if key in seen or counts.get(state[2], 0) >= 7:
                        continue
                    seen.add(key)
                    counts[state[2]] = counts.get(state[2], 0) + 1
                    kept.append(state)
                    if len(kept) >= 28:
                        break
                beam = kept
            if not any(r["label"] == label and r["complete"] for r in all_routes) and best_partial:
                all_routes.append(best_partial)

        # If exact support made every long route unavailable, retain one-step
        # escape/hold candidates.  This prevents a false model certainty from
        # turning into an idle death at a carrier edge.
        if not all_routes:
            for action in ACTIONS:
                transition = forecast.transition(start_row, start_x, action)
                if transition is None:
                    continue
                all_routes.append(
                    {
                        "label": "SURVIVE",
                        "action": action,
                        "path": (action,),
                        "value": transition["margin"] - (0 if action == "NOOP" else 1),
                        "margin": round(transition["margin"], 2),
                        "x": round(transition["x"], 2),
                        "landing_row": transition["row"],
                        "complete": False,
                        "surface": "interior" if transition["margin"] >= 3 else "edge",
                        "route": "hold_or_escape",
                    }
                )
        # One representative per label/action keeps the Jev state small.
        unique = {}
        for route in all_routes:
            key = (route["label"], route["action"])
            if key not in unique or route["value"] > unique[key]["value"]:
                unique[key] = route
        result = sorted(unique.values(), key=lambda r: (r["complete"], r["value"]), reverse=True)[:8]
        self.last_search_ms = (time.perf_counter() - started) * 1000.0
        return result

    def ask_jev(self, obs, routes):
        tick = integer(obs.get("tick"), 0)
        if self.jev_future is not None or tick < self.next_jev or not routes:
            return
        if getattr(self.ctx, "system1", True) is None:
            return
        try:
            if self.ctx.budget.remaining().get("system1_calls", 0) < 2:
                return
        except Exception:
            return
        candidates = []
        options = {}
        for route in routes[:5]:
            label = str(route["label"])
            if label in options:
                continue
            margin = finite(route.get("margin"))
            bucket = "wide" if margin >= 4 else "buffered" if margin >= 2 else "edge"
            kind = "complete home" if route.get("complete") else "median/forward" if route.get("landing_row", 0) >= 5 else "hold or escape"
            candidates.append(
                {
                    "route": label,
                    "kind": kind,
                    "first_hop": route.get("action"),
                    "surface": "interior" if route.get("surface") == "interior" else "edge",
                    "clearance": bucket,
                    "connected": bool(route.get("complete") or route.get("landing_row", 0) > frog_row(obs)),
                }
            )
            options[label] = "Use this code-certified route; its first hop is rechecked immediately."
        options["UNKNOWN"] = "Do not prefer any candidate; let the code choose the safest verified fallback."
        reversed_options = dict(reversed(list(options.items())))
        state = {
            "game": "Atari 2600 Frogger",
            "rules": "Cars are hazards. River support requires overlap with decoded log/turtle x_intervals; gaps and diving or unfamiliar carriers are unsafe. Code owns all geometry and action legality.",
            "phase": obs.get("frog", {}).get("zone"),
            "timer": "urgent" if finite(obs.get("timer_s"), 30) < 7 else "normal",
            "risk_policy": {"mode": self.risk_mode, "level": self.risk_level, "style": self.style},
            "candidates": candidates,
            "open_home_count": sum(1 for h in obs.get("homes", []) if not h.get("occupied")),
            "recent_failure": bool(self.failure_history),
            "model_role": "Select a durable route preference, never invent an action or override a rejected geometry check.",
        }
        instructions = (
            "Choose one route label from `candidates`. Prefer a complete home route, "
            "then a connected forward route with buffered/interior clearance; choose UNKNOWN "
            "when the facts do not support a preference. This is a semantic route choice, "
            "not a request for arithmetic or a joystick sequence."
        )
        questions = {
            "route": Choice(instructions, options),
            "verify": Choice(instructions, reversed_options),
            "hold": Noul(
                "Do the supplied facts explicitly make the leading candidate an unsafe edge case? True only when its clearance is edge and the route is not a complete home entry."
            ),
        }
        self.jev_meta = {
            "tick": tick,
            "epoch": episode_epoch(obs),
            "version": self.plan_version,
            "zone": obs.get("frog", {}).get("zone"),
        }

        def request():
            try:
                before = self.ctx.clock.now()
                answer = self.ctx.decide(state, questions, timeout_s=3.0)
                return answer, self.ctx.clock.now() - before
            except (BudgetExceeded, EpisodeOver, ModelUnavailable):
                raise
            except (TimeoutError, RuntimeError, TypeError):
                return None, 0.0

        self.jev_future = self.ctx.spawn(request)
        self.next_jev = tick + 75

    def ask_astra(self, obs):
        tick = integer(obs.get("tick"), 0)
        if self.astra_future is not None or getattr(self.ctx, "system2", True) is None:
            return
        if tick < self.next_astra:
            return
        # Do not spend a slow call merely to restate the executable default.
        # A failure opens an immediate review window; an unusually long clean
        # run gets one periodic calibration even without a counterexample.
        if not self.failure_history and tick < 1800:
            return
        try:
            if self.ctx.budget.remaining().get("system2_calls", 0) < 1:
                return
        except Exception:
            return
        summary = {
            "frog": obs.get("frog"),
            "score": finite(obs.get("score")),
            "homes": obs.get("homes"),
            "level": integer(obs.get("level"), 1),
            "risk_mode": self.risk_mode,
            "risk_level": self.risk_level,
            "failures": list(self.failure_history)[-3:],
            "recent_outcomes": list(self.recent_outcomes)[-3:],
            "route_search_ms": round(self.last_search_ms, 2),
        }
        prompt = (
            "You are a slow Frogger route-policy reviewer, not a frame controller. "
            "The executor uses exact public x_intervals for river carrier support, "
            "a phase-aware short forecast, and neutral release after each hop. "
            "A recent death is evidence against an edge certificate, not permission "
            "to invent emulator facts. Choose only a conservative policy. Return ONLY "
            "JSON with risk_mode=balanced|strict, home_order=nearest|left|right, "
            "style=safe|progress. Strict raises support/road margins; nearest is "
            "normally best. Do not return an action, coordinates, or prose. Snapshot="
            + json.dumps(summary, separators=(",", ":"))
        )
        self.astra_meta = {"level": integer(obs.get("level"), 1), "tick": tick}
        session = self.session_id

        def request():
            try:
                return self.ctx.think(prompt, session_id=session, timeout_s=120.0)
            except (BudgetExceeded, EpisodeOver, ModelUnavailable):
                raise
            except (TimeoutError, RuntimeError):
                return None

        self.astra_future = self.ctx.spawn(request)
        self.next_astra = tick + 3600

    def poll_models(self, obs):
        tick = integer(obs.get("tick"), 0)
        if self.jev_future is not None and self.jev_future.done():
            future, self.jev_future = self.jev_future, None
            try:
                result = future.result()
            except (TimeoutError, RuntimeError, TypeError):
                result = None
            if isinstance(result, tuple) and len(result) == 2:
                answers, latency = result
                if isinstance(answers, dict):
                    first, second = answers.get("route"), answers.get("verify")
                    value = getattr(first, "value", None)
                    second_value = getattr(second, "value", None)
                    p1 = getattr(first, "probabilities", None) or {}
                    p2 = getattr(second, "probabilities", None) or {}
                    try:
                        probability = min(finite(p1.get(value), 0), finite(p2.get(value), 0))
                    except (AttributeError, TypeError):
                        probability = 0.0
                    meta = self.jev_meta or {}
                    fresh = (
                        episode_epoch(obs) == meta.get("epoch")
                        and self.plan_version == meta.get("version")
                        and obs.get("frog", {}).get("zone") == meta.get("zone")
                        and tick - integer(meta.get("tick"), tick) <= 90
                    )
                    valid = isinstance(value, str) and value == second_value and value != "UNKNOWN"
                    if fresh and valid and probability >= 0.62:
                        self.jev_lease = {
                            "route": value,
                            "until": integer(meta.get("tick"), tick) + 120,
                            "epoch": meta.get("epoch"),
                            "version": meta.get("version"),
                            "zone": meta.get("zone"),
                        }
                        self.log("jev_route_adopted", tick=tick, route=value, probability=probability, latency_s=latency)
                    else:
                        self.log("jev_route_discarded", tick=tick, fresh=fresh, probability=probability, latency_s=latency)

        if self.astra_future is not None and self.astra_future.done():
            future, self.astra_future = self.astra_future, None
            try:
                result = future.result()
            except (TimeoutError, RuntimeError, TypeError):
                result = None
            if result is not None:
                self.session_id = getattr(result, "session_id", None)
                proposed = json_fragment(getattr(result, "text", ""))
                allowed = {
                    "risk_mode": ("balanced", "strict"),
                    "home_order": ("nearest", "left", "right"),
                    "style": ("safe", "progress"),
                }
                changes = {
                    key: value
                    for key, value in proposed.items()
                    if key in allowed and isinstance(value, str) and value in allowed[key]
                }
                if changes and integer(obs.get("level"), 1) == (self.astra_meta or {}).get("level"):
                    self.risk_mode = changes.get("risk_mode", self.risk_mode)
                    self.home_order = changes.get("home_order", self.home_order)
                    self.style = changes.get("style", self.style)
                    self.plan_version += 1
                    self.jev_lease = None
                    self.log(
                        "astra_policy_adopted",
                        tick=tick,
                        policy={"risk_mode": self.risk_mode, "home_order": self.home_order, "style": self.style},
                    )
                else:
                    self.log("astra_output_rejected", tick=tick)

    def select_route(self, obs, routes):
        if not routes:
            return None
        leading = routes[0]
        lease = self.jev_lease
        if lease:
            fresh = (
                tick := integer(obs.get("tick"), 0)
            ) <= lease.get("until", -1) and episode_epoch(obs) == lease.get("epoch") and self.plan_version == lease.get("version") and obs.get("frog", {}).get("zone") == lease.get("zone")
            if not fresh:
                self.jev_lease = None
            else:
                same = [r for r in routes if r.get("label") == lease.get("route")]
                if same and same[0].get("value", -1e9) >= leading.get("value", -1e9) - 45:
                    chosen = same[0]
                    if chosen is not leading:
                        self.log("jev_changes_route", tick=tick, before=leading.get("action"), after=chosen.get("action"), route=chosen.get("label"))
                    return chosen
        # After a witnessed failure, prefer a buffered route among near-best
        # alternatives.  This is code-owned risk adaptation, not model math.
        if self.risk_level:
            near = [r for r in routes if r.get("value", -1e9) >= leading.get("value", -1e9) - 30]
            if near:
                return max(near, key=lambda r: (finite(r.get("margin"), -100), r.get("complete", False), r.get("value", -1e9)))
        return leading

    def motor_choice(self, obs, routes):
        tick = integer(obs.get("tick"), 0)
        active = integer(obs.get("frog", {}).get("state"), 128) == 128
        current_row = frog_row(obs)
        if not active or current_row < -1 or current_row > 10:
            self.press = None
            return "NOOP"
        if self.press is not None:
            origin_row, origin_x = self.press_origin or (current_row, finite(obs.get("frog", {}).get("x"), 80))
            moved = current_row != origin_row or abs(finite(obs.get("frog", {}).get("x"), origin_x) - origin_x) >= 4.0
            if moved or tick - self.press_tick >= 4:
                self.log("pulse_released", tick=tick, action=self.press, moved=moved, frog=obs.get("frog"))
                self.press = None
                self.release_until = tick + 2
                self.ready_tick = tick + 2
                return "NOOP"
            return self.press
        if tick < self.ready_tick or tick < self.release_until:
            return "NOOP"
        chosen = self.select_route(obs, routes)
        if chosen is None:
            return "NOOP"
        action = chosen.get("action") if chosen.get("action") in ACTIONS else "NOOP"
        if action != "NOOP":
            self.press = action
            self.press_origin = (current_row, finite(obs.get("frog", {}).get("x"), 80))
            self.press_tick = tick
            self.ready_tick = tick + 6
            self.last_certificate = {
                "tick": tick,
                "origin": self.press_origin,
                "action": action,
                "route": chosen.get("label"),
                "destination": (chosen.get("landing_row"), chosen.get("x")),
                "margin": chosen.get("margin"),
                "path": chosen.get("path"),
            }
            self.log("route_hop", tick=tick, certificate=self.last_certificate, alternatives=routes[:3])
        else:
            self.ready_tick = tick + 3
            if tick - self.last_log_tick >= 45:
                self.log("route_hold", tick=tick, selected=chosen, search_ms=round(self.last_search_ms, 2))
                self.last_log_tick = tick
        return action

    def run_episode(self, game):
        del game
        while not self.ctx.done:
            obs = unpack(self.ctx.observe())
            if not obs:
                self.ctx.act("NOOP")
                self.ctx.wait_for_tick()
                continue
            self.update(obs)
            self.poll_models(obs)
            self.ask_astra(obs)
            tick = integer(obs.get("tick"), 0)
            routes = []
            active = integer(obs.get("frog", {}).get("state"), 128) == 128
            if active and self.press is None and tick >= self.ready_tick and tick >= self.release_until:
                routes = self.search(obs)
                self.last_routes = routes
                self.ask_jev(obs, routes)
            action = self.motor_choice(obs, routes)
            self.last_action = action
            self.ctx.act(action)
            self.ctx.wait_for_tick()


def build(ctx):
    return Harness(ctx)
