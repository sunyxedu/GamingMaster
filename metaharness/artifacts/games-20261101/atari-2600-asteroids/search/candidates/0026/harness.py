"""Candidate 0026: acknowledged shot transactions over a calibrated turret.

Numerical occupancy and gun estimation retain the validated 0022 mechanics.
The new bridge separates requested FIRE, visible missile birth, and mandatory
release; models allocate target families and bounded aim patience, not buttons.
"""
from __future__ import annotations


import json


import math


from metaharness.core.contract import BudgetExceeded, EpisodeOver, ModelUnavailable


from metaharness.models.jev import Choice, Noul


WIDTH = 160.0


HEIGHT = 210.0


DEFAULT_ACTIONS = (
    "NOOP", "FIRE", "UP", "RIGHT", "LEFT", "DOWN", "UPRIGHT", "UPLEFT",
    "UPFIRE", "RIGHTFIRE", "LEFTFIRE", "DOWNFIRE", "UPRIGHTFIRE",
    "UPLEFTFIRE",
)


def _number(value, default=None):
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _integer(value, default=0):
    result = _number(value)
    return int(result) if result is not None else default


def _pair(value):
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        return None
    x = _number(value[0])
    y = _number(value[1])
    return (x, y) if x is not None and y is not None else None


def _dict(value):
    return value if isinstance(value, dict) else {}


def _list(value):
    return value if isinstance(value, list) else []


def _wrap_delta(value, period):
    value = _number(value, 0.0) or 0.0
    while value > period / 2.0:
        value -= period
    while value < -period / 2.0:
        value += period
    return value


def _bucket(value, cuts, names):
    if value is None:
        return names[0]
    for cut, name in zip(cuts, names[1:]):
        if value < cut:
            return name
    return names[-1]


def _observation(game):
    try:
        value = game.observe()
    except Exception:
        return {}
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except (TypeError, ValueError):
            return {}
        return value if isinstance(value, dict) else {}
    return {}


def _json_object(text):
    """Extract a JSON object from prose or an arbitrary stand-in response."""
    if not isinstance(text, str):
        return None
    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text[index:])
        except (TypeError, ValueError):
            continue
        if isinstance(value, dict):
            return value
    return None


def _answer_value(answer):
    if isinstance(answer, dict):
        return answer.get("value", answer.get("choice", answer.get("answer")))
    value = getattr(answer, "value", None)
    if value is not None:
        return value
    return getattr(answer, "choice", None)


def _answer_probability(answer, key):
    if isinstance(answer, dict):
        probabilities = answer.get("probabilities", {})
    else:
        probabilities = getattr(answer, "probabilities", {})
    if not isinstance(probabilities, dict):
        return 0.0
    value = _number(probabilities.get(key), 0.0) or 0.0
    return max(0.0, min(1.0, value))


def _yes_probability(answer):
    value = _answer_value(answer)
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, str):
        return 1.0 if value.lower() in {"true", "yes"} else 0.0
    if isinstance(value, (int, float)):
        return max(0.0, min(1.0, _number(value, 0.0) or 0.0))
    return _answer_probability(answer, "true")


_HEADING_VECTORS = tuple(
    (-math.sin(code * math.pi / 8.0), -math.cos(code * math.pi / 8.0))
    for code in range(16)
)


def _heading_vector(code):
    """Screen-coordinate nose prior; gun velocities are learned separately."""
    return _HEADING_VECTORS[int(code) % 16]


def _desired_heading(dx, dy):
    if abs(dx) + abs(dy) < 1e-8:
        return 0
    return int(round(math.atan2(-dx, -dy) / (math.pi / 8.0))) % 16


def _turn_delta(current, desired):
    if current is None or desired is None:
        return 0
    delta = (int(desired) - int(current)) % 16
    if delta > 8:
        delta -= 16
    return delta


def _action_has_thrust(action):
    return isinstance(action, str) and action.startswith("UP")


class KinematicPilot:
    """A frame-local controller with an explicit velocity-damping phase."""

    def __init__(self, game, ctx):
        self.game = game
        self.ctx = ctx
        spec = getattr(game, "action_spec", None)
        try:
            names = list(getattr(spec, "names", DEFAULT_ACTIONS)) if spec is not None else list(DEFAULT_ACTIONS)
        except Exception:
            names = list(DEFAULT_ACTIONS)
        self.names = set(names or DEFAULT_ACTIONS)
        self.noop = getattr(spec, "noop", "NOOP") if spec is not None else "NOOP"
        if self.noop not in self.names:
            self.noop = "NOOP" if "NOOP" in self.names else next(iter(self.names), "NOOP")

        self.frame = 0
        self.score = 0.0
        self.lives = 0
        self.previous_lives = None
        self.life_epoch = 0
        self.obs = {}
        self.facts = {}
        self.previous_position = None
        self.player_velocity = None
        self.previous_velocity = None
        self.last_heading = None
        self.last_action = self.noop

        # These are preferences, not commands.  The hard local gates remain in
        # force whatever a model says.
        self.plan = {
            "aim": "threat_first",
            "mobility": "anchor_then_coast",
            "fire": "normal",
            "defense": "last_resort",
        }
        self.mode_hint = None
        self.mode_until = -1
        self.target_hint = None
        self.target_hint_until = -1
        self.defense_hint = False

        # The first phase is deliberately about momentum, not scoring.  It is
        # short and can be interrupted by an observed critical contact.
        self.settle_until = 0
        self.recovery_until = 0
        self.brake_next = 0
        self.brake_pulses = 0
        self.acceleration = 0.045
        self.accel_samples = 0

        # A maneuver is a small revalidated tranche, never an action latch.
        self.maneuver = None
        self.maneuver_next = 0
        self.escape_next = 0
        self.escape_count = 0
        self.defense_used = False
        self.defense_frame = -100000
        self.defenses = 0

        self.next_fire = 0
        self.next_sweep = 0
        self.sweep_sign = 1
        self.last_log = -100000
        self.last_jev_options = []

        self.losses = 0
        self.anchor_actions = 0
        self.thrust_actions = 0
        self.maneuvers_started = 0
        self.maneuvers_cancelled = 0

    # ---------- basic helpers ----------

    def _log(self, event, **fields):
        try:
            self.ctx.log(event, **fields)
        except Exception:
            pass

    def legal(self, action):
        return action if action in self.names else self.noop

    @staticmethod
    def _center(player):
        if not bool(player.get("present")):
            return None
        x = _number(player.get("x"))
        y = _number(player.get("y"))
        if x is None or y is None:
            return None
        w = _number(player.get("width"), 6.0) or 6.0
        h = _number(player.get("height"), 10.0) or 10.0
        return x + w / 2.0, y + h / 2.0, w, h

    @staticmethod
    def _relative_bearing(dx, dy):
        horizontal = "right" if dx > 10 else "left" if dx < -10 else "center"
        vertical = "below" if dy > 10 else "above" if dy < -10 else "level"
        return horizontal + "_" + vertical

    def _read_player_velocity(self, player, finite_difference):
        observed = _pair(player.get("velocity_px_per_frame"))
        if bool(player.get("motion_discontinuity")):
            return None
        if observed is None:
            return finite_difference
        if finite_difference is not None:
            # The adaptor's velocity is a multi-frame finite difference.  On
            # quantized frames it can say zero while the decoded position has
            # visibly advanced; retain that evidence for momentum control.
            if abs(observed[0]) + abs(observed[1]) < 0.01 and abs(finite_difference[0]) + abs(finite_difference[1]) > 0.08:
                return finite_difference
            return (
                0.72 * observed[0] + 0.28 * finite_difference[0],
                0.72 * observed[1] + 0.28 * finite_difference[1],
            )
        return observed

    def _learn_acceleration(self, velocity, discontinuity):
        old = self.previous_velocity
        if old is None or velocity is None or discontinuity:
            return
        if not _action_has_thrust(self.last_action) or self.last_heading is None:
            return
        hx, hy = _heading_vector(self.last_heading)
        gain = (velocity[0] - old[0]) * hx + (velocity[1] - old[1]) * hy
        if 0.006 <= gain <= 0.16:
            self.acceleration = max(0.018, min(0.08, 0.88 * self.acceleration + 0.12 * gain))
            self.accel_samples += 1

    # ---------- object map and trajectory forecasts ----------

    def _rock_records(self, obs, player, px, py, pw, ph, pv):
        records = []
        for index, raw0 in enumerate(_list(obs.get("asteroids"))):
            raw = _dict(raw0)
            rx = _number(raw.get("x"))
            ry = _number(raw.get("y"))
            if rx is None or ry is None:
                continue
            rw = _number(raw.get("width"), 8.0) or 8.0
            rh = _number(raw.get("height"), 15.0) or 15.0
            center_x = rx + rw / 2.0
            center_y = ry + rh / 2.0
            av = _pair(raw.get("velocity_px_per_frame"))
            known = av is not None and pv is not None and not bool(raw.get("motion_discontinuity"))
            if known:
                rvx, rvy = av[0] - pv[0], av[1] - pv[1]
            elif av is not None:
                # This is a useful directional hint but not a confident
                # relative track; the projection inflates its footprint.
                rvx, rvy = av
            else:
                rvx = rvy = 0.0
            size = str(raw.get("size") or "unknown").lower()
            value = {"small": 3, "medium": 2, "large": 1}.get(size, 1)
            slot = _integer(raw.get("source_slot"), index)
            records.append({
                "id": "r" + str(slot) + "_" + size + "_" + str(index),
                "slot": slot,
                "index": index,
                "size": size,
                "value": value,
                "x": center_x,
                "y": center_y,
                "dx": _wrap_delta(center_x - px, WIDTH),
                "dy": _wrap_delta(center_y - py, HEIGHT),
                "rvx": rvx,
                "rvy": rvy,
                "known": bool(known),
                "half_x": rw / 2.0 + pw / 2.0 + 3.0 + (4.5 if not known else 0.0),
                "half_y": rh / 2.0 + ph / 2.0 + 3.0 + (4.5 if not known else 0.0),
                "distance": math.hypot(_wrap_delta(center_x - px, WIDTH), _wrap_delta(center_y - py, HEIGHT)),
            })
        return records

    def _project(self, rock, desired_code=None, thrust_frames=0, turn_delay=0, horizon=66):
        """Forecast relative occupancy for one short, physically reachable path."""
        minimum = float("inf")
        first = None
        uncertainty = 3.5 if not rock.get("known") else 0.0
        speed = math.hypot(rock.get("rvx", 0.0), rock.get("rvy", 0.0))
        # Exact samples cover the lethal near window; coarser samples keep the
        # full two-dimensional map below one native-frame deadline.  The
        # relative-speed inflation makes skipped crossings conservative.
        times = list(range(0, 7)) + list(range(9, horizon + 1, 3))
        if times[-1] != horizon:
            times.append(horizon)
        for t in times:
            if desired_code is None or thrust_frames <= 0:
                displacement = 0.0
                ux = uy = 0.0
            else:
                ux, uy = _heading_vector(desired_code)
                active = min(float(thrust_frames), max(0.0, float(t - turn_delay)))
                coast_after = max(0.0, float(t - turn_delay) - active)
                displacement = self.acceleration * (0.5 * active * active + active * coast_after)
            rx = _wrap_delta(rock["dx"] + rock["rvx"] * t - ux * displacement, WIDTH)
            ry = _wrap_delta(rock["dy"] + rock["rvy"] * t - uy * displacement, HEIGHT)
            clearance = max(abs(rx) - rock["half_x"], abs(ry) - rock["half_y"])
            clearance -= uncertainty + min(6.0, speed * 1.2)
            minimum = min(minimum, clearance)
            if first is None and abs(rx) <= rock["half_x"] and abs(ry) <= rock["half_y"]:
                first = t
        return first, minimum

    @staticmethod
    def _path_cost(rock_results):
        total = 0.0
        minimum = float("inf")
        first = None
        for rock, contact, clearance in rock_results:
            minimum = min(minimum, clearance)
            if contact is None:
                total += max(0.0, 14.0 - clearance) * (0.42 if rock.get("known") else 0.62)
            else:
                total += 42.0 + max(0.0, 38.0 - contact) * (1.65 + 0.12 * rock.get("value", 1))
                if first is None or contact < first:
                    first = contact
            if not rock.get("known"):
                total += 0.35
        return total, minimum, first

    def _path(self, path_id, desired_code, thrust, records, heading):
        if desired_code is None:
            delay = 0
        else:
            delta = abs(_turn_delta(heading, desired_code)) if heading is not None else 8
            # The cartridge turns only intermittently.  The delay is a forecast,
            # not an open-loop command timer; every frame is revalidated.
            delay = min(24, delta * 3)
        outcomes = []
        for rock in records:
            contact, clearance = self._project(
                rock, desired_code, thrust, delay, horizon=66
            )
            outcomes.append((rock, contact, clearance))
        cost, minimum, first = self._path_cost(outcomes)
        return {
            "id": path_id,
            "code": desired_code,
            "thrust": thrust,
            "delay": delay,
            "cost": cost,
            "minimum": minimum if records else 100.0,
            "first": first,
            "outcomes": outcomes,
        }

    def _build_paths(self, records, heading, speed):
        paths = [self._path("coast", None, 0, records, heading)]
        if heading is not None:
            paths.append(self._path("nose", int(heading) % 16, 2, records, heading))
        if speed is not None and speed > 0.10 and self.player_velocity is not None:
            anti = _desired_heading(-self.player_velocity[0], -self.player_velocity[1])
            paths.append(self._path("brake", anti, 2, records, heading))
        for code in (0, 2, 4, 6, 8, 10, 12, 14):
            paths.append(self._path("d" + str(code), code, 3, records, heading))
        return paths

    def _target_from(self, records, coast):
        if not records:
            return None
        contact_by_id = {}
        for rock, contact, _ in coast.get("outcomes", []):
            contact_by_id[rock["id"]] = contact

        def rank(rock):
            contact = contact_by_id.get(rock["id"])
            urgency = 100.0 if contact is not None else 0.0
            if contact is not None:
                urgency += max(0.0, 70.0 - contact) * 1.5
            elif rock["distance"] < 80:
                urgency += 10.0
            return -(urgency + 4.0 * rock["value"] + (1.0 if rock["known"] else 0.0) - rock["distance"] / 180.0)

        return min(records, key=rank)

    def _derive(self, obs, player, center, pv):
        if center is None:
            return {
                "present": False, "frame": self.frame, "score": self.score,
                "lives": self.lives, "x": None, "y": None,
                "heading": None, "vx": None, "vy": None, "speed": None,
                "rocks": [], "paths": [], "coast": None, "best": None,
                "best_gain": 0.0, "ttc": None, "clearance": None,
                "danger": "unknown", "near_unknown": False, "uncertain": True,
                "target": None, "missiles": 0, "discontinuity": True,
            }
        px, py, pw, ph = center
        heading = _integer(player.get("orientation_code"), 0) % 16
        records = self._rock_records(obs, player, px, py, pw, ph, pv)
        speed = math.hypot(*pv) if pv is not None else None
        paths = self._build_paths(records, heading, speed)
        coast = next((item for item in paths if item["id"] == "coast"), None)
        candidates = [item for item in paths if item["id"] != "coast"]
        best = min(candidates, key=lambda item: (item["cost"], -item["minimum"], item["thrust"]), default=coast)
        ttc = coast.get("first") if coast else None
        clearance = coast.get("minimum") if coast else None
        near_unknown = any((not r["known"]) and r["distance"] < 48.0 for r in records)
        # `clearance` is the minimum over the whole forecast horizon.  A rock
        # that is safely 59 frames away can therefore make it negative even
        # though the present frame is safe.  Use earliest contact for the
        # regime, and reserve clearance for ranking paths; otherwise every
        # ordinary future crossing becomes an immediate emergency.
        if ttc is None:
            danger = "watch" if near_unknown else "clear"
        elif ttc <= 5:
            danger = "critical"
        elif ttc <= 18:
            danger = "caution"
        elif ttc <= 50:
            danger = "watch"
        else:
            danger = "watch" if near_unknown else "clear"
        best_gain = (coast["cost"] - best["cost"]) if coast and best else 0.0
        target = self._target_from(records, coast or {"outcomes": []})
        missiles = sum(1 for item in _list(obs.get("player_missiles")) if isinstance(item, dict))
        return {
            "present": True, "frame": self.frame, "score": self.score, "lives": self.lives,
            "x": px, "y": py, "heading": heading,
            "vx": pv[0] if pv is not None else None,
            "vy": pv[1] if pv is not None else None,
            "speed": speed, "rocks": records, "paths": paths,
            "coast": coast, "best": best, "best_gain": best_gain,
            "ttc": ttc, "clearance": clearance, "danger": danger,
            "near_unknown": near_unknown,
            "uncertain": pv is None or near_unknown or any(not r["known"] for r in records),
            "target": target, "missiles": missiles,
            "discontinuity": bool(player.get("motion_discontinuity")),
        }

    def update(self, observation):
        data = _dict(observation)
        self.obs = data
        old_frame = self.frame
        self.frame = _integer(data.get("frame"), old_frame + 1)
        self.score = _number(data.get("score"), self.score) or 0.0
        player = _dict(data.get("player"))
        center = self._center(player)
        finite_difference = None
        if center is not None and self.previous_position is not None:
            delta_frames = max(1, self.frame - old_frame)
            dx = _wrap_delta(center[0] - self.previous_position[0], WIDTH) / delta_frames
            dy = _wrap_delta(center[1] - self.previous_position[1], HEIGHT) / delta_frames
            if abs(dx) <= 24.0 and abs(dy) <= 30.0:
                finite_difference = (dx, dy)
        discontinuity = center is None or bool(player.get("motion_discontinuity"))
        if finite_difference is None and center is not None and self.previous_position is not None:
            discontinuity = True
        velocity = self._read_player_velocity(player, finite_difference)
        self._learn_acceleration(velocity, discontinuity)

        new_lives = _integer(data.get("lives"), self.lives)
        life_changed = self.previous_lives is not None and new_lives != self.previous_lives
        if life_changed:
            self.life_epoch += 1
            if new_lives < self.previous_lives:
                self.losses += self.previous_lives - new_lives
            self.maneuver = None
            self.target_hint = None
            self.mode_hint = None
            self.defense_hint = False
            self.defense_used = False
            self.brake_next = self.frame + 8
            self.settle_until = self.frame + 42
            self.recovery_until = self.frame + 22
            self.next_fire = self.frame + 8
            self._log("life_epoch", frame=self.frame, old_lives=self.previous_lives, lives=new_lives)
        self.lives = new_lives
        self.previous_lives = new_lives

        if discontinuity:
            if self.maneuver is not None:
                self.maneuvers_cancelled += 1
            self.maneuver = None
            self.target_hint = None
            self.recovery_until = max(self.recovery_until, self.frame + (16 if center is not None else 10))
            if center is not None:
                self._log("kinematic_reset", frame=self.frame)

        self.facts = self._derive(data, player, center, velocity)
        if self.maneuver is not None:
            active = next((p for p in self.facts.get("paths", []) if p.get("id") == self.maneuver.get("path")), None)
            if active is None or self.frame > _integer(self.maneuver.get("expires"), self.frame):
                self.maneuvers_cancelled += 1
                self.maneuver = None
            elif self.maneuver.get("kind") == "escape":
                if self.facts.get("danger") == "clear" and self.frame > _integer(self.maneuver.get("started"), self.frame) + 3:
                    self.maneuvers_cancelled += 1
                    self.maneuver = None
                elif active.get("cost", 999.0) > (self.facts.get("coast") or {}).get("cost", 0.0) + 2.5:
                    self.maneuvers_cancelled += 1
                    self.maneuver = None
            elif self.facts.get("speed") is not None and self.facts.get("speed") < 0.10:
                self.maneuver = None

        self.previous_position = (center[0], center[1]) if center is not None else None
        self.previous_velocity = velocity
        self.player_velocity = velocity

    # ---------- semantic interface ----------

    @staticmethod
    def _clearance_label(value):
        if value is None:
            return "unknown"
        if value < 0:
            return "overlap"
        if value < 6:
            return "tight"
        if value < 16:
            return "limited"
        return "open"






    # ---------- local action bridge ----------

    def _moving_player(self):
        return self.facts.get("speed") is not None and self.facts.get("speed") > 0.16

    def _fire_ready(self):
        if self.frame < self.next_fire or self.facts.get("missiles", 0) >= 2:
            return False
        gap = 6 if self.plan.get("fire") == "spaced" else 5
        self.next_fire = self.frame + gap
        return True

    def _compose(self, turn=0, thrust=False, fire=False):
        if thrust:
            if turn > 0:
                name = "UPLEFTFIRE" if fire else "UPLEFT"
            elif turn < 0:
                name = "UPRIGHTFIRE" if fire else "UPRIGHT"
            else:
                name = "UPFIRE" if fire else "UP"
        elif turn > 0:
            name = "LEFTFIRE" if fire else "LEFT"
        elif turn < 0:
            name = "RIGHTFIRE" if fire else "RIGHT"
        else:
            name = "FIRE" if fire else "NOOP"
        return self.legal(name)

    def _choose_target(self):
        records = self.facts.get("rocks", [])
        if not records:
            return None
        if self.target_hint and self.frame <= self.target_hint_until:
            hinted = next((r for r in records if r.get("id") == self.target_hint), None)
            if hinted is not None:
                return hinted
        if self.plan.get("aim") == "sweep":
            return None
        if self.plan.get("aim") == "value_first":
            return min(records, key=lambda r: (-4.0 * r.get("value", 1) + r.get("distance", 999.0) / 180.0))
        target = self.facts.get("target")
        if target is not None:
            return target
        return records[0]

    def _aim_turn(self):
        facts = self.facts
        heading = facts.get("heading")
        if heading is None:
            return 0
        mode = self.mode_hint if self.frame <= self.mode_until else None
        target = self._choose_target()
        # A model's hold/anchor signal is never allowed to freeze aiming while
        # the local map calls the state caution or critical.
        if mode == "hold" and facts.get("danger") not in {"caution", "critical"}:
            return 0
        if mode == "anchor" and facts.get("danger") == "clear" and self._moving_player():
            return 0
        if target is not None:
            lead = 5.0 if target.get("known") else 2.0
            desired = _desired_heading(target.get("dx", 0.0) + target.get("rvx", 0.0) * lead,
                                       target.get("dy", 0.0) + target.get("rvy", 0.0) * lead)
            return 1 if _turn_delta(heading, desired) > 0 else -1 if _turn_delta(heading, desired) < 0 else 0
        if self.frame >= self.next_sweep:
            self.sweep_sign *= -1
            self.next_sweep = self.frame + 48
        return 1 if self.sweep_sign > 0 else -1

    def _should_defend(self):
        facts = self.facts
        if not facts.get("present") or self.defense_used or self.plan.get("defense") == "never":
            return False
        ttc = facts.get("ttc")
        if ttc is None or ttc > 6:
            return False
        best = facts.get("best") or {}
        coast = facts.get("coast") or {}
        gain = _number(facts.get("best_gain"), 0.0) or 0.0
        # A genuinely safer reachable path wins.  If every short path still
        # intersects, act before the ttc==0 failure window seen in 0017.
        path_safe = best.get("id") not in {None, "coast"} and best.get("first") is None and gain >= 1.0
        if path_safe:
            return False
        if ttc <= 1:
            return True
        if self.defense_hint and ttc <= 6:
            return True
        return ttc <= 5 and best.get("cost", 999.0) >= coast.get("cost", 0.0) - 0.5

    def _start_escape(self):
        facts = self.facts
        if self.maneuver is not None or self.frame < self.escape_next:
            return False
        danger = facts.get("danger")
        if danger not in {"watch", "caution", "critical"}:
            return False
        # A watch forecast is still a firing state by default.  Only Jev's
        # agreeing `evade` regime may open a watch-window relocation, and even
        # then it must be no more than 24 frames away.  This prevents a large
        # horizon forecast from turning the turret into continuous wandering.
        if danger == "watch":
            ttc = facts.get("ttc")
            if ttc is None or ttc > 24:
                return False
            if not (self.mode_hint == "evade" and self.frame <= self.mode_until):
                return False
        best = facts.get("best") or {}
        coast = facts.get("coast") or {}
        path_id = best.get("id")
        if path_id in {None, "coast"} or best.get("code") is None:
            return False
        gain = _number(facts.get("best_gain"), 0.0) or 0.0
        if facts.get("danger") == "caution":
            threshold = 1.3 if self.plan.get("mobility") == "reactive_escape" else 2.0
        else:
            threshold = 0.8 if facts.get("danger") == "critical" else 4.0
        if self.mode_hint == "evade" and self.frame <= self.mode_until:
            threshold = max(0.8, threshold - 0.6)
        if gain < threshold:
            return False
        if facts.get("near_unknown") and facts.get("ttc") is None:
            return False
        # At a long warning distance, do not start a maneuver whose own
        # forecast still has an early contact.
        if facts.get("ttc") is not None and facts.get("ttc") > 35 and best.get("first") is not None:
            return False
        urgent = facts.get("danger") == "critical"
        thrust = 2 if urgent else 3
        self.maneuver = {
            "kind": "escape", "path": path_id, "code": _integer(best.get("code"), 0) % 16,
            "remaining": thrust, "started": self.frame,
            "expires": self.frame + (16 if urgent else 24),
        }
        self.maneuver_next = self.frame
        self.escape_next = self.frame + (5 if urgent else 10)
        self.escape_count += 1
        self.maneuvers_started += 1
        self._log("escape_start", frame=self.frame, path=path_id, danger=facts.get("danger"),
                  ttc=facts.get("ttc"), gain=round(gain, 2))
        return True

    def _maneuver_action(self):
        move = self.maneuver
        if not isinstance(move, dict) or not self.facts.get("present"):
            return None
        if self.frame > _integer(move.get("expires"), self.frame):
            self.maneuver = None
            self.maneuvers_cancelled += 1
            return None
        current = self.facts.get("heading")
        desired = _integer(move.get("code"), current or 0) % 16
        delta = _turn_delta(current, desired)
        if delta:
            # Use thrust while turning only when already close to the desired
            # direction and a critical contact leaves no time for pure rotation.
            if abs(delta) <= 1 and self.facts.get("danger") == "critical" and _integer(move.get("remaining"), 0) > 0:
                move["remaining"] = _integer(move.get("remaining"), 0) - 1
                self.thrust_actions += 1
                return self._compose(1 if delta > 0 else -1, thrust=True, fire=False)
            return self._compose(1 if delta > 0 else -1, thrust=False, fire=False)
        remaining = _integer(move.get("remaining"), 0)
        if remaining > 0 and self.frame >= self.maneuver_next:
            move["remaining"] = remaining - 1
            self.maneuver_next = self.frame + 5
            self.thrust_actions += 1
            return self._compose(0, thrust=True, fire=False)
        if remaining <= 0:
            self.maneuver = None
        return self.noop

    def _brake_action(self):
        facts = self.facts
        speed = facts.get("speed")
        if not facts.get("present") or speed is None or speed <= 0.16:
            return None
        if facts.get("danger") == "critical" or self.frame < self.recovery_until:
            return None
        vx, vy = facts.get("vx"), facts.get("vy")
        if vx is None or vy is None:
            return None
        desired = _desired_heading(-vx, -vy)
        current = facts.get("heading")
        delta = _turn_delta(current, desired)
        if delta:
            return self._compose(1 if delta > 0 else -1, thrust=False, fire=False)
        if self.frame >= self.brake_next:
            self.brake_next = self.frame + 8
            self.brake_pulses += 1
            self.anchor_actions += 1
            self.thrust_actions += 1
            self._log("anchor_pulse", frame=self.frame, speed=round(speed, 3), heading=current, desired=desired)
            return self._compose(0, thrust=True, fire=False)
        return self.noop

    def choose(self):
        facts = self.facts
        if not facts.get("present"):
            self.last_action = self.noop
            return self.noop

        if self._should_defend():
            self.defense_used = True
            self.defenses += 1
            self.defense_frame = self.frame
            self.maneuver = None
            self.recovery_until = self.frame + 26
            self.defense_hint = False
            self._log("defense", frame=self.frame, ttc=facts.get("ttc"), clearance=round(_number(facts.get("clearance"), 0.0) or 0.0, 2))
            self.last_action = self.legal("DOWN")
            return self.last_action

        if self.frame >= self.recovery_until:
            self._start_escape()
        if self.maneuver is not None:
            action = self._maneuver_action()
            if action is not None:
                self.last_action = action
                self.last_heading = facts.get("heading")
                return action

        brake = self._brake_action()
        if brake is not None:
            self.last_action = brake
            self.last_heading = facts.get("heading")
            return brake

        # Release controls after a reset/defense, but allow a fresh shot once
        # the player is visible.  Old target/heading advice never creates thrust.
        turn = self._aim_turn()
        fire = self._fire_ready()
        if self.frame < self.recovery_until:
            turn = 0
            fire = False
        action = self._compose(turn=turn, thrust=False, fire=fire)
        self.last_action = action
        self.last_heading = facts.get("heading")
        if fire or facts.get("danger") == "critical" or self.frame - self.last_log >= 600:
            self._log("control", frame=self.frame, action=action, phase="turret",
                      danger=facts.get("danger"), ttc=facts.get("ttc"),
                      speed=round(_number(facts.get("speed"), 0.0) or 0.0, 3),
                      score=self.score, lives=self.lives)
            self.last_log = self.frame
        return action

    # ---------- Astra summary ----------

    def astra_summary(self, reason):
        facts = self.facts
        return {
            "reason": reason, "frame": self.frame,
            "score": int(max(0, self.score)), "lives": self.lives,
            "losses": self.losses, "danger": facts.get("danger"),
            "impact": facts.get("ttc"), "speed": _bucket(facts.get("speed"), (0.15, 0.35, 0.7), ("unknown", "still", "moving", "fast")),
            "anchor_pulses": self.anchor_actions,
            "escape_starts": self.maneuvers_started,
            "escape_cancelled": self.maneuvers_cancelled,
            "thrust_frames": self.thrust_actions,
            "defenses": self.defenses,
            "plan": dict(self.plan),
        }



class BallisticPilot(KinematicPilot):
    """Learn the gun from visible shots, then aim at reachable intersections.

    This is an offensive controller, not a new mobility policy. Parent occupancy,
    anchor, escape and hyperspace gates remain authoritative. Its new hypothesis
    is that clearing a threatening rock is sometimes better than spending the
    whole warning interval rotating toward an ineffective displacement.
    """

    def __init__(self, game, ctx):
        super().__init__(game, ctx)
        self.bullet_tracks = {}
        self.recent_presses = []
        self.ballistics = {}
        self.gun_solutions = []
        self.gun_selected = None
        self.gun_role = 'threat'
        self.gun_role_until = -1
        self.gun_plan = 'threat'
        self.gun_serial = 0
        self.gun_epoch = 0
        self.gun_previous_score = 0.0
        self.gun_last_reward = 0
        self.gun_last_shot = -1000
        self.gun_last_log = -1000
        self.shots_posted = 0
        self.shots_observed = 0
        self.shots_calibrated = 0
        self.aligned_shots = 0
        self.opportunity_shots = 0
        self.calibration_rejects = 0
        self.shot_wait_frames = 0
        self.last_press_heading = None
        self.gun_debug = {}

    def update(self, observation):
        super().update(observation)
        if self.life_epoch != self.gun_epoch:
            self.bullet_tracks.clear()
            self.recent_presses.clear()
            self.gun_role_until = -1
            self.gun_epoch = self.life_epoch
        if self.score > self.gun_previous_score:
            self.gun_last_reward = self.frame
        self.gun_previous_score = self.score
        self._observe_shots()
        self._solve_shots()

    def _observe_shots(self):
        f = self.facts
        if not f.get('present'):
            self.bullet_tracks.clear()
            return
        self.recent_presses = [p for p in self.recent_presses if self.frame-p['frame'] <= 8]
        visible = {}
        for raw in _list(self.obs.get('player_missiles')):
            raw = _dict(raw)
            x, y = _number(raw.get('x')), _number(raw.get('y'))
            if x is None or y is None:
                continue
            slot = _integer(raw.get('source_slot'), -1)
            visible[slot] = True
            old = self.bullet_tracks.get(slot)
            continuous = (old is not None and 0 < self.frame-old['frame'] <= 3
                          and abs(x-old['x']) <= 10 and abs(y-old['y']) <= 14)
            if not continuous:
                press = next((p for p in reversed(self.recent_presses)
                              if 0 < self.frame-p['frame'] <= 4
                              and abs(x-f['x']) < 22 and abs(y-f['y']) < 28), None)
                if press is None:
                    self.bullet_tracks.pop(slot, None)
                    continue
                # Prefer the observed heading at shot appearance. A simultaneous
                # turn can change the angle before the cartridge samples FIRE.
                heading = _integer(f.get('heading'), press['heading']) % 16
                old = {'frame': self.frame, 'x': x, 'y': y,
                       'start': self.frame, 'sx': x, 'sy': y,
                       'heading': heading, 'learned': False,
                       'turning': bool(press['turning'])}
                self.bullet_tracks[slot] = old
                self.shots_observed += 1
                continue
            old['frame'], old['x'], old['y'] = self.frame, x, y
            age = self.frame-old['start']
            if age < 6 or old['learned']:
                continue
            old['learned'] = True
            vx, vy = (x-old['sx'])/age, (y-old['sy'])/age
            code = old['heading']
            hx, hy = _heading_vector(code)
            # Reject silent slot replacement, zero/flicker segments, implausible
            # speed, and a trajectory inconsistent with the associated heading.
            dot = vx*hx+vy*hy
            speed = math.hypot(vx, vy)
            cross = abs(vx*hy-vy*hx)
            if not (0.6 <= speed <= 6.0 and dot > 0.3 and cross < speed*0.65):
                self.calibration_rejects += 1
                continue
            entry = self.ballistics.get(code)
            if entry is None:
                entry = {'vx': vx, 'vy': vy, 'n': 1, 'err': 0.0}
                self.ballistics[code] = entry
            else:
                residual = math.hypot(vx-entry['vx'], vy-entry['vy'])
                if entry['n'] >= 3 and residual > 1.1:
                    self.calibration_rejects += 1
                    continue
                weight = 1.0/min(8, entry['n']+1)
                entry['vx'] += weight*(vx-entry['vx'])
                entry['vy'] += weight*(vy-entry['vy'])
                entry['err'] = 0.8*entry['err']+0.2*residual
                entry['n'] += 1
            self.shots_calibrated += 1
            if entry['n'] in (1, 3, 8, 24):
                self._log('gun_calibration', frame=self.frame, heading=code,
                          vx=round(entry['vx'], 3), vy=round(entry['vy'], 3),
                          samples=entry['n'], error=round(entry['err'], 3))
        for slot in list(self.bullet_tracks):
            if slot not in visible:
                del self.bullet_tracks[slot]

    def _bullet_vector(self, code):
        entry = self.ballistics.get(code)
        if entry is not None and entry['n'] >= 2:
            return entry['vx'], entry['vy'], entry['n'] >= 3
        # The observations in the archive show anisotropic motion: horizontal
        # ~2 and vertical ~3 px/frame. This is a provisional prior, not a claim
        # that intermediate cartridge headings have evenly spaced angles.
        hx, hy = _heading_vector(code)
        return hx*2.1, hy*3.0, False

    def _solution(self, rock, code, delay):
        vx, vy, calibrated = self._bullet_vector(code)
        hx, hy = _heading_vector(code)
        # These are direct, short, in-screen shots. We do not invent a missile
        # wrap period from the RGB dimensions. Off-screen paths are excluded.
        x = rock['x']-self.facts['x'] + rock['rvx']*delay - hx*4.0
        y = rock['y']-self.facts['y'] + rock['rvy']*delay - hy*5.0
        rx = rock['rvx']-vx
        ry = rock['rvy']-vy
        # Parent half extents include ship size and safety inflation; remove
        # those for gun intersection rather than firing at an inflated hazard.
        margin = 3.0 + (4.5 if not rock['known'] else 0.0)
        half_x = max(2.5, rock['half_x']-margin-3.0+1.0)
        half_y = max(4.5, rock['half_y']-margin-5.0+1.0)
        den = (rx/half_x)**2+(ry/half_y)**2
        if den < 0.00001:
            return None
        flight = -(x*rx/(half_x*half_x)+y*ry/(half_y*half_y))/den
        if flight < 1.0 or flight > 38.0:
            return None
        miss = max(abs(x+rx*flight)/half_x, abs(y+ry*flight)/half_y)
        impact_x = self.facts['x']+hx*4.0+vx*flight
        impact_y = self.facts['y']+hy*5.0+vy*flight
        if not (0 <= impact_x <= 163 and 15 <= impact_y <= 201):
            return None
        return {'rock': rock, 'code': code, 'delay': delay,
                'flight': flight, 'arrival': delay+flight, 'miss': miss,
                'calibrated': calibrated, 'known': rock['known']}

    def _solve_shots(self):
        self.gun_solutions = []
        self.gun_selected = None
        f = self.facts
        if not f.get('present'):
            return
        current = f['heading']
        contacts = {r['id']: t for r, t, _ in (f.get('coast') or {}).get('outcomes', [])}
        for rock in f.get('rocks', []):
            best = None
            for code in range(16):
                delay = 4*abs(_turn_delta(current, code))
                solution = self._solution(rock, code, delay)
                if solution is None:
                    continue
                quality = (solution['miss']*14 + delay*0.28
                           + solution['flight']*0.12
                           + (0 if solution['calibrated'] else 1.5))
                if best is None or quality < best['quality']:
                    best = dict(solution, quality=quality)
            if best is None or best['miss'] > 2.5:
                continue
            contact = contacts.get(rock['id'])
            clearable = (contact is not None and best['arrival']+4 < contact
                         and best['miss'] <= 1.1 and rock['known'])
            imminent = contact is not None and contact <= 50
            best['ttc'] = contact
            best['clearable'] = clearable
            best['priority'] = (35 if clearable else 9 if imminent else 0) + 4*rock['value']-best['quality']
            self.gun_solutions.append(best)
        self.gun_solutions.sort(key=lambda s: s['priority'], reverse=True)
        if not self.gun_solutions:
            return
        role = self.gun_role if self.frame <= self.gun_role_until else self.gun_plan
        if role == 'clean':
            selected = min(self.gun_solutions, key=lambda s:s['quality']-3*s['rock']['value'])
        elif role == 'value':
            selected = max(self.gun_solutions, key=lambda s:7*s['rock']['value']-s['quality'])
        else:
            selected = self.gun_solutions[0]
        # Models cannot override a verified shot at a near crossing with a
        # distracting high-value target.
        urgent = next((s for s in self.gun_solutions if s['clearable'] and s['ttc'] <= 40), None)
        self.gun_selected = urgent or selected

    def _aim_turn(self):
        selected = self.gun_selected
        if selected is None:
            return super()._aim_turn()
        # Calibrated intersection rather than a fixed five-frame angular lead.
        delta = _turn_delta(self.facts.get('heading'), selected['code'])
        return 1 if delta > 0 else -1 if delta < 0 else 0

    def _fire_ready(self):
        if self.frame < self.next_fire or self.facts.get('missiles', 0) >= 2:
            return False
        heading = self.facts.get('heading')
        if heading is None or 'FIRE' in str(self.last_action):
            return False
        # Fire at any useful target already in this barrel corridor, not only
        # the selected future target. Reserve at most eight frames for an aim;
        # uncertain geometry never silences the gun indefinitely.
        aligned = None
        for rock in self.facts.get('rocks', []):
            shot = self._solution(rock, heading, 0)
            if shot and shot['miss'] <= (1.25 if shot['known'] else 1.65):
                aligned = shot
                break
        held = self.frame-self.gun_last_shot
        selected = self.gun_selected
        waiting_for_aim = (aligned is None and selected is not None
                           and selected['delay'] <= 8 and selected['miss'] <= 1.1
                           and held < 12)
        if waiting_for_aim:
            self.shot_wait_frames += 1
            return False
        self.next_fire = self.frame+(6 if self.plan.get('fire') == 'spaced' else 5)
        self.gun_debug = {'aligned': aligned is not None,
                          'calibrated': bool(aligned and aligned['calibrated']),
                          'target': aligned['rock']['id'] if aligned else None}
        if aligned:
            self.aligned_shots += 1
        else:
            self.opportunity_shots += 1
        return True

    def _maneuver_action(self):
        action = super()._maneuver_action()
        if action is None or not self.facts.get('present'):
            return action
        # Escape used to disarm the turret during the whole warning interval.
        # Preserve the parent's turn/thrust exactly, but allow a released shot
        # when the current nose already has an intercept opportunity.
        heading = self.facts.get('heading')
        if heading is None:
            return action
        opportunity = any((s is not None and s['miss'] <= 1.15)
                          for s in (self._solution(r, heading, 0) for r in self.facts.get('rocks', [])))
        if opportunity and self._fire_ready():
            return self.legal('FIRE' if action == 'NOOP' else action+'FIRE')
        return action








class TransactionPilot(BallisticPilot):
    """An acknowledged, one-frame press followed by an observed release.

    Geometry is not a proof of collision and lack of birth is not a proved miss.
    A FIRE transaction is acknowledged only by a locally associated new missile.
    Score windows are descriptive aggregates, never causal kill attribution.
    """

    def __init__(self, game, ctx):
        super().__init__(game, ctx)
        self.pending_shot = None
        self.release_until = -1
        self.transaction_serial = 0
        self.transaction_acks = 0
        self.transaction_unobserved = 0
        self.transaction_aborts = 0
        self.fire_frames = 0
        self.shot_blocks = []
        self.block_frame = 0
        self.block_score = 0.0
        self.block_attempts = 0
        self.block_acks = 0
        self.block_losses = 0
        self.patience_plan = 'normal'
        self.patience_hint = False
        self.patience_until = -1
        self.patience_used = 0
        self.plan_version = 0
        self.advice_changes = 0
        self.advice_actions = 0
        self.last_jev_options = []

    def _finish_transaction(self, result, born_heading=None):
        pending = self.pending_shot
        if pending is None:
            return
        if result == 'visible_birth':
            self.transaction_acks += 1
        elif result == 'unobserved':
            self.transaction_unobserved += 1
        else:
            self.transaction_aborts += 1
        self._log('shot_result', frame=self.frame, serial=pending['serial'],
                  started=pending['frame'], age=self.frame-pending['frame'],
                  outcome=result, requested_heading=pending['heading'],
                  observed_heading=born_heading, aligned=pending['aligned'],
                  target=pending['target'], emitted_frames=pending['emitted'])
        self.pending_shot = None
        # These are simulated-frame deadlines. At least one release frame
        # precedes a new press even if a new missile appeared immediately.
        self.release_until = max(self.release_until, self.frame + 1)

    def update(self, observation):
        old_epoch = self.life_epoch
        old_observed = self.shots_observed
        super().update(observation)
        p = self.pending_shot
        if p is not None:
            if self.life_epoch != old_epoch or not self.facts.get('present'):
                self._finish_transaction('life_or_absence')
            elif self.shots_observed > old_observed:
                newest = max(self.bullet_tracks.values(), key=lambda x:x['start'], default={})
                self._finish_transaction('visible_birth', newest.get('heading'))
            elif self.frame-p['frame'] >= 4:
                self._finish_transaction('unobserved')
        if self.life_epoch != old_epoch:
            self.patience_until = -1
        if self.frame-self.block_frame >= 600:
            block = {'start': self.block_frame, 'end': self.frame,
                     'score_gain': int(self.score-self.block_score),
                     'attempts': self.shots_posted-self.block_attempts,
                     'visible_births': self.transaction_acks-self.block_acks,
                     'losses': self.losses-self.block_losses}
            self.shot_blocks.append(block)
            self.shot_blocks = self.shot_blocks[-12:]
            self._log('shot_block', **block)
            self.block_frame, self.block_score = self.frame, self.score
            self.block_attempts, self.block_acks = self.shots_posted, self.transaction_acks
            self.block_losses = self.losses

    def _fire_ready(self):
        # Finish a press before asking for a new one. This is a bounded button
        # level, NOT repeated shooting while FIRE is held.
        if self.pending_shot is not None:
            return (self.frame-self.pending_shot['frame'] < 1
                    and self.facts.get('present')
                    and self.frame >= self.recovery_until
                    and self.facts.get('missiles', 0) < 2)
        if self.frame < self.release_until:
            return False
        # Jev may reserve a few more release frames for an almost-aligned clean
        # shot. It may not silence the gun indefinitely or reserve during escape.
        reserve = (self.patience_plan == 'selective' or
                   (self.patience_hint and self.frame <= self.patience_until))
        selected = self.gun_selected
        if (reserve and self.maneuver is None and self.facts.get('danger') == 'clear'
                and selected is not None and selected['known']
                and selected['miss'] <= 1.1 and 0 < selected['delay'] <= 8
                and self.frame-self.gun_last_shot < 14
                and self.frame >= self.next_fire):
            heading = self.facts.get('heading')
            aligned = any(s and s['miss'] <= 1.25 for s in
                          (self._solution(r, heading, 0) for r in self.facts.get('rocks', [])))
            if not aligned:
                self.patience_used += 1
                return False
        return super()._fire_ready()

    def choose(self):
        # Execute the established safety hierarchy. Parent BallisticPilot's
        # choose counted every pressed frame as a fresh attempt; this bridge
        # instead counts a transaction once and acknowledges visible creation.
        action = KinematicPilot.choose(self)
        if 'FIRE' in action:
            # A normal gun press reserves its nose until sampled; otherwise a
            # combined turn can change the firing direction before recognition.
            # Movement/escape and hyperspace retain their original authority.
            if self.maneuver is None and not action.startswith('UP'):
                action = self.legal('FIRE')
                self.last_action = action
            self.fire_frames += 1
            if self.pending_shot is None:
                self.transaction_serial += 1
                self.shots_posted += 1
                self.gun_last_shot = self.frame
                heading = _integer(self.facts.get('heading'), 0)
                target = self.gun_selected
                self.pending_shot = {
                    'serial': self.transaction_serial, 'frame': self.frame,
                    'heading': heading, 'emitted': 0,
                    'aligned': bool(self.gun_debug.get('aligned', False)),
                    'target': target['rock']['id'] if target else None,
                }
                self.recent_presses.append({'frame':self.frame, 'heading':heading,
                                           'turning':'LEFT' in action or 'RIGHT' in action})
                self._log('shot_request', frame=self.frame, serial=self.transaction_serial,
                          heading=heading, selected_code=target['code'] if target else None,
                          target=self.pending_shot['target'],
                          aligned=self.pending_shot['aligned'],
                          miss=round(target['miss'], 3) if target else None,
                          role=self.gun_role if self.frame<=self.gun_role_until else self.gun_plan)
            self.pending_shot['emitted'] += 1
        elif action == 'DOWN' and self.pending_shot is not None:
            self._finish_transaction('defense_preemption')
        return action

    def jev_state(self):
        f = self.facts
        opportunities = []
        for s in self.gun_solutions[:3]:
            opportunities.append({
                'size':s['rock']['size'],
                'trajectory':'calibrated' if s['calibrated'] else 'provisional',
                'intercept':'clean' if s['miss']<=1 else 'marginal',
                'turn':'aligned' if s['delay']==0 else 'short' if s['delay']<=8 else 'long',
                'threat':'clearable' if s['clearable'] else 'crossing' if s['ttc'] is not None else 'noncrossing',
                'motion':'known' if s['known'] else 'uncertain'})
        recent = self.shot_blocks[-1] if self.shot_blocks else {}
        attempts = recent.get('attempts', 0)
        births = recent.get('visible_births', 0)
        return {
            'rules': {'fire':'button level; bounded press must be released before another shot',
                      'down':'mode-0 hyperspace, never braking; local safety only',
                      'motion':'NOOP coasts; rotation does not change velocity',
                      'coverage':'enemy craft and projectiles unknown; slots may be reused'},
            'snapshot': {'frame':self.frame, 'player':'present' if f.get('present') else 'absent',
                         'danger':f.get('danger'), 'slots':'full' if f.get('missiles',0)>=2 else 'available'},
            'gun': {'opportunities':opportunities,
                    'phase':'acknowledgement_pending' if self.pending_shot else
                            'release' if self.frame<self.release_until else 'ready',
                    'realization':'unmeasured' if attempts<10 else 'usually_observed' if births>=0.8*attempts else 'many_unobserved',
                    'progress':'stalled' if self.frame-self.gun_last_reward>240 else 'scoring',
                    'calibration':'established' if sum(e['n']>=3 for e in self.ballistics.values())>=6 else 'learning'},
            'plan': {'focus':self.gun_plan, 'patience':self.patience_plan,
                     'version':self.plan_version,
                     'goal':'maximize native reward with clean useful shots; never delay local safety'},
        }

    def jev_questions(self):
        options = {
            'threat':'Clear a supplied reachable crossing before impact; do not chase unreachable threats.',
            'clean':'Prefer the cheapest clean interception when ammunition slots are scarce or motion is uncertain.',
            'value':'Prefer small valuable rocks when no supplied clearable near threat takes priority.',
            'unknown':'Insufficient evidence; keep local threat-first fallback.'}
        order = list(options)
        if (self.frame//48)%2:
            order.reverse()
        self.last_jev_options = order
        return {
            'allocation_a':Choice('Choose a gun allocation rule from gun.opportunities and plan.focus. Geometry is already computed. You cannot choose buttons, missile headings or an action sequence.',{k:options[k] for k in order}),
            'allocation_b':Choice('Independently select the most useful allocation rule. Clearable crossing outranks cleanup; approximate interception is not a guaranteed kill.',{k:options[k] for k in reversed(order)}),
            'reserve':Noul('Is a short alignment reserve useful? True only when gun.opportunities contains a clean known-motion short-turn shot, snapshot.danger is clear, and gun.progress is scoring. False for unknown, emergency, or stalled evidence.',
                           {'true':'Reserve a few release frames for this kind of clean shot, subject to current code checks.',
                            'false':'Do not add alignment delay beyond the local fallback.'})}

    def accept_jev(self, answers, frame, epoch, version):
        if (not isinstance(answers,dict) or epoch!=self.life_epoch
                or version!=self.plan_version or not 0<=self.frame-frame<=36):
            self._log('jev_drop',frame=self.frame,judged_frame=frame,reason='stale_epoch_or_plan')
            return
        old = self.gun_selected
        old_key = (old['rock']['id'], old['code']) if old else None
        a,b = _answer_value(answers.get('allocation_a')),_answer_value(answers.get('allocation_b'))
        accepted = False
        if isinstance(a,str) and a==b and a in {'threat','clean','value'}:
            p = min(_answer_probability(answers.get('allocation_a'),a),
                    _answer_probability(answers.get('allocation_b'),a))
            if p>=0.60:
                self.gun_role=a
                self.gun_role_until=self.frame+60
                accepted=True
        self.patience_hint = _yes_probability(answers.get('reserve'))>=0.80
        self.patience_until=self.frame+48
        self._solve_shots()
        new=self.gun_selected
        new_key=(new['rock']['id'],new['code']) if new else None
        changed=old_key!=new_key
        self.advice_changes+=int(changed)
        self._log('jev_allocation',frame=self.frame,judged_frame=frame,
                  role=a if accepted else 'fallback',reserve=self.patience_hint,
                  changed_target=changed,old=old_key,new=new_key)

    def astra_summary(self, reason):
        result=KinematicPilot.astra_summary(self,reason)
        result['transactions']={'attempts':self.shots_posted,'visible_births':self.transaction_acks,
                                'unobserved':self.transaction_unobserved,'aborted':self.transaction_aborts,
                                'pressed_frames':self.fire_frames,'alignment_wait':self.patience_used}
        result['calibration']={'headings':sum(e['n']>=3 for e in self.ballistics.values()),
                               'samples':self.shots_calibrated,'rejects':self.calibration_rejects}
        result['recent_blocks']=list(self.shot_blocks[-6:])
        result['allocation']={'focus':self.gun_plan,'patience':self.patience_plan,
                              'model_changed_target':self.advice_changes}
        return result

    def apply_astra(self,text):
        data=_json_object(text)
        if not isinstance(data,dict):
            return False
        focus=data.get('focus')
        patience=data.get('patience')
        valid=False
        if isinstance(focus,str) and focus in {'threat','clean','value'}:
            self.gun_plan=focus
            self.plan['aim']={'threat':'threat_first','clean':'sweep','value':'value_first'}[focus]
            valid=True
        if isinstance(patience,str) and patience in {'normal','selective'}:
            self.patience_plan=patience
            valid=True
        if valid:
            self.plan_version+=1
            self.gun_role_until=-1
            self.patience_until=-1
            self._solve_shots()
            self._log('astra_allocation',frame=self.frame,version=self.plan_version,
                      focus=self.gun_plan,patience=self.patience_plan)
        return valid


class Harness:
    def __init__(self,ctx):
        self.ctx=ctx

    def run_episode(self,game):
        ctx=self.ctx
        pilot=TransactionPilot(game,ctx)
        s1_future=s2_future=None
        s1_origin=(-1,-1,-1)
        next_s1=-1
        astra_started=False
        s1_enabled=getattr(ctx,'system1',True) is not None
        start_frame=None

        def post(action):
            # Timestamped input expires by contract; the local controller still
            # supplies a fresh action before every wait.  Deadline and cancelled
            # episode exceptions are normal ends, not harness errors.
            stamped=str(action)+'@'+str(pilot.frame)
            try:
                ctx.act(stamped)
                return True
            except (BudgetExceeded,EpisodeOver,ModelUnavailable):
                return False
            except Exception:
                try:
                    ctx.act(str(pilot.noop)+'@'+str(pilot.frame))
                    return True
                except Exception:
                    return False

        while not ctx.done and not game.done:
            observation=_observation(game)
            try:
                pilot.update(observation)
            except Exception as exc:
                pilot._log('observation_miss',kind=type(exc).__name__)
                pilot.facts={'present':False}
            if start_frame is None:
                start_frame=pilot.frame

            if s1_future is not None and s1_future.done():
                done=s1_future
                s1_future=None
                try:
                    pilot.accept_jev(done.result(),*s1_origin)
                except (BudgetExceeded,EpisodeOver,ModelUnavailable):
                    s1_enabled=False
                except Exception as exc:
                    pilot._log('advisory_miss',model='jev',kind=type(exc).__name__)
            if s2_future is not None and s2_future.done():
                done=s2_future
                s2_future=None
                try:
                    answer=done.result()
                    pilot.apply_astra(getattr(answer,'text',answer))
                except (BudgetExceeded,EpisodeOver,ModelUnavailable):
                    pass
                except Exception as exc:
                    pilot._log('advisory_miss',model='astra',kind=type(exc).__name__)

            # Expiry is a backstop, not the normal scheduler. It does not make
            # an unobserved shot successful and cannot extend a pending press.
            if not post(pilot.choose()):
                return

            try:
                remaining=ctx.budget.remaining()
            except (BudgetExceeded,EpisodeOver,ModelUnavailable):
                return
            except Exception:
                remaining={}

            if (s1_enabled and s1_future is None and pilot.frame>=next_s1
                    and _integer(remaining.get('system1_calls'),0)>1):
                state,questions=pilot.jev_state(),pilot.jev_questions()
                s1_origin=(pilot.frame,pilot.life_epoch,pilot.plan_version)
                try:
                    s1_future=ctx.spawn(lambda state=state,questions=questions:
                                        ctx.decide(state,questions,timeout_s=2.5))
                    next_s1=pilot.frame+48
                except (BudgetExceeded,EpisodeOver,ModelUnavailable):
                    s1_enabled=False
                except Exception as exc:
                    pilot._log('advisory_miss',model='jev_start',kind=type(exc).__name__)
                    next_s1=pilot.frame+54

            if (not astra_started and pilot.frame-start_frame>=1800
                    and _integer(remaining.get('system2_calls'),0)>0
                    and _integer(observation.get('frames_remaining'),18000)>3000):
                astra_started=True
                try:
                    summary=pilot.astra_summary('shot-realization review')
                    prompt=(
                        'Review a real-time Atari 2600 Asteroids mode-0/difficulty-0 gun controller. '
                        'Code owns calibrated trajectories, one-frame acknowledged FIRE presses, '
                        'mandatory release, bounded thrust and all safety. New visible missiles are '
                        'acknowledgements; unobserved means uncertain, NOT necessarily a missed shot. '
                        'Score gain is native ALE reward, not a survival bonus. Jev chooses a target '
                        'allocation rule and whether to reserve a few frames for a clean short turn. '
                        'Each Jev request gets opportunities (size, calibrated/provisional, clean/'
                        'marginal, short/long turn, clearable/crossing/noncrossing, known motion), '
                        'score-progress and realization buckets, and your durable focus/patience. '
                        'Choose from this fixed interface; do not invent headings, questions, code, '
                        'safety thresholds, actions or timings. Return JSON only, under 50 words: '
                        '{"focus":"threat|clean|value","patience":"normal|selective"}. '
                        'Normal retains the proven bounded fallback; selective can wait only for '
                        'already-computed clean short turns in clear scoring states. Prefer normal '
                        'unless blocks show a reason to trade volume for alignment. Unknown enemy '
                        'coverage remains unknown. No individual kill is attributed from aggregate score.\n'
                        +json.dumps(summary,separators=(',',':')))
                    s2_future=ctx.spawn(lambda prompt=prompt:ctx.think(prompt,timeout_s=150.0))
                except (BudgetExceeded,EpisodeOver,ModelUnavailable):
                    pass
                except Exception as exc:
                    pilot._log('advisory_miss',model='astra_start',kind=type(exc).__name__)

            try:
                ctx.wait_for_tick()
            except (BudgetExceeded,EpisodeOver,ModelUnavailable):
                return
            except Exception as exc:
                pilot._log('tick_end',kind=type(exc).__name__)
                return
        try:
            pilot._log('transaction_summary',**pilot.astra_summary('episode complete'))
        except Exception:
            pass


def build(ctx):
    return Harness(ctx)
