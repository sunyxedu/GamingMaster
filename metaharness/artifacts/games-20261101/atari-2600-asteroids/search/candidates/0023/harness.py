"""Kinematic-anchor Asteroids controller for the tier-2 real-time adaptor.

This controller separates three jobs which the earlier lane/radar controllers
mixed together: (1) code damps inherited ship velocity before asking the ship to
be a turret, (2) code builds a short spatiotemporal occupancy map and executes
only revalidated escape/brake pulses, and (3) Jev chooses a semantic target or
operating regime without being able to suppress the local aiming loop.  Astra
reviews aggregate evidence and changes only durable preferences.
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


def _heading_vector(code):
    """Screen-coordinate nose vector: code 0 up, 4 left, 8 down, 12 right."""
    angle = (int(code) % 16) * math.pi / 8.0
    return -math.sin(angle), -math.cos(angle)


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

    def _path_view(self):
        facts = self.facts
        coast = facts.get("coast") or {}
        base = _number(coast.get("cost"), 0.0) or 0.0
        paths = []
        for path in facts.get("paths", []):
            if path.get("id") == "coast":
                continue
            if path.get("id") == "brake" or path.get("cost", 999.0) <= base + 9.0 or facts.get("danger") in {"caution", "critical"}:
                paths.append({
                    "id": path.get("id"),
                    "direction": "none" if path.get("code") is None else str(path.get("code")),
                    "clearance": self._clearance_label(path.get("minimum")),
                    "impact": _bucket(path.get("first"), (5, 19, 50), ("none", "immediate", "soon", "later")),
                    "kind": "brake" if path.get("id") == "brake" else "escape",
                })
        return paths[:10]

    def jev_state(self):
        facts = self.facts
        target = facts.get("target")
        target_view = None
        if target is not None:
            target_view = {
                "id": target.get("id"), "size": target.get("size"),
                "bearing": self._relative_bearing(target.get("dx", 0.0), target.get("dy", 0.0)),
                "distance": _bucket(target.get("distance"), (30, 80, 125), ("unknown", "near", "middle", "far")),
                "motion": "known" if target.get("known") else "uncertain",
            }
        threats = []
        coast = facts.get("coast") or {}
        for rock, contact, clearance in coast.get("outcomes", [])[:6]:
            threats.append({
                "id": rock.get("id"), "size": rock.get("size"),
                "bearing": self._relative_bearing(rock.get("dx", 0.0), rock.get("dy", 0.0)),
                "impact": _bucket(contact, (6, 19, 50), ("none", "immediate", "soon", "later")),
                "clearance": self._clearance_label(clearance),
                "motion": "known" if rock.get("known") else "uncertain",
            })
        return {
            "rules": {
                "mode": 0, "difficulty": 0,
                "fire": "press_then_release; holding is not rapid fire",
                "noop": "coasts; it is not a brake",
                "down": "mode-0 hyperspace; one local last-resort pulse",
                "orientation": "LEFT increments and RIGHT decrements orientation code",
                "enemy_coverage": "unknown; null is not absence",
            },
            "snapshot": {
                "frame": int(self.frame), "lives": int(self.lives),
                "score": int(max(0, self.score)),
                "player": "present" if facts.get("present") else "absent",
                "position": "unknown" if facts.get("x") is None else _bucket(facts.get("x"), (52, 108), ("unknown", "left", "middle", "right")),
                "motion": "unknown" if facts.get("speed") is None else ("still" if facts["speed"] < 0.15 else "moving"),
                "danger": facts.get("danger", "unknown"),
                "impact": _bucket(facts.get("ttc"), (6, 19, 50), ("none", "immediate", "soon", "later")),
                "clearance": self._clearance_label(facts.get("clearance")),
                "asteroids": min(15, len(facts.get("rocks", []))),
                "missiles": min(2, int(facts.get("missiles", 0))),
                "uncertain": bool(facts.get("uncertain")),
            },
            "threats": threats,
            "target": target_view,
            "paths": self._path_view(),
            "controller": {
                "phase": "recovery" if self.frame < self.recovery_until else ("anchor" if self._moving_fast() else "turret"),
                "anchor_pulses": int(self.anchor_actions),
                "maneuver": self.maneuver.get("path") if isinstance(self.maneuver, dict) else "none",
                "defenses_this_life": int(self.defenses),
                "acceleration": _bucket(self.acceleration, (0.027, 0.046, 0.065), ("low", "normal", "high", "very_high")),
                "frame_local": True,
            },
            "plan": dict(self.plan),
        }

    def _moving_fast(self):
        value = self.facts.get("speed")
        return value is not None and value > 0.16

    def jev_questions(self):
        mode_text = {
            "anchor": "Prioritize damping inherited velocity; code chooses the opposite-velocity heading and only short thrust pulses.",
            "evade": "Prioritize a named code-generated escape path when its latest forecast is safer; never invent a direction or sequence.",
            "hunt": "Prioritize the supplied threat/value target for short aiming turns and code-gated FIRE pulses; do not add thrust.",
            "hold": "Keep a stationary firing posture when the supplied geometry is clear; this is not permission to ignore a hard local collision gate.",
        }
        modes = ["anchor", "evade", "hunt", "hold"]
        if (self.frame // 36) % 2:
            modes.reverse()
        mode_reverse = list(reversed(modes))
        target_options = {"none": "Do not override the code-selected target."}
        for rock in self.facts.get("rocks", [])[:6]:
            target_options[rock.get("id")] = "Prefer this observed rock for aiming only; code revalidates its slot, size, and current geometry."
        target_order = list(target_options)
        if (self.frame // 48) % 2:
            target_order.reverse()
        target_options = {key: target_options[key] for key in target_order}
        self.last_jev_options = list(target_options)
        return {
            "mode_a": Choice(
                "Choose the safest useful operating regime for this exact snapshot. This is semantic advice, not a raw action or an open-loop sequence; local geometry and momentum gates always win.",
                {key: mode_text[key] for key in modes},
            ),
            "mode_b": Choice(
                "Independently choose one operating regime. Treat snapshot.impact, threats, and paths as already computed; do not perform arithmetic.",
                {key: mode_text[key] for key in mode_reverse},
            ),
            "target_a": Choice(
                "Choose an observed target id or none. Prefer an imminent threat over a merely valuable rock, but never invent an id.",
                target_options,
            ),
            "target_b": Choice(
                "Independently choose the useful observed target for the code-owned aiming loop. A stale or missing id is rejected.",
                {key: target_options[key] for key in reversed(target_order)},
            ),
            "defense": Noul(
                "Should the local controller consider its one mode-0 hyperspace pulse? Say true only when snapshot.impact is immediate and paths do not show a safer revalidated escape; otherwise false.",
                {"true": "A bounded last-resort defense is warranted.", "false": "Do not invoke defense; let the local path or turret controller act."},
            ),
        }

    def accept_jev(self, answers, origin_frame, origin_epoch, origin_targets):
        if not isinstance(answers, dict):
            return
        if origin_epoch != self.life_epoch or self.frame < origin_frame or self.frame - origin_frame > 42:
            self._log("jev_drop", frame=self.frame, judged_frame=origin_frame, reason="stale_or_life")
            return
        first = _answer_value(answers.get("mode_a"))
        second = _answer_value(answers.get("mode_b"))
        valid_modes = {"anchor", "evade", "hunt", "hold"}
        if first == second and first in valid_modes:
            confidence = min(_answer_probability(answers.get("mode_a"), first), _answer_probability(answers.get("mode_b"), second))
            if confidence >= 0.30:
                self.mode_hint = first
                self.mode_until = self.frame + 54
        ta = _answer_value(answers.get("target_a"))
        tb = _answer_value(answers.get("target_b"))
        if ta == tb and isinstance(ta, str) and ta in set(origin_targets):
            confidence = min(_answer_probability(answers.get("target_a"), ta), _answer_probability(answers.get("target_b"), tb))
            if confidence >= 0.28 and ta != "none":
                current_ids = {rock.get("id") for rock in self.facts.get("rocks", [])}
                if ta in current_ids:
                    self.target_hint = ta
                    self.target_hint_until = self.frame + 54
            elif ta == "none":
                self.target_hint = None
        defense = _yes_probability(answers.get("defense"))
        self.defense_hint = defense >= 0.80
        self._log(
            "jev_advice", frame=self.frame, judged_frame=origin_frame,
            mode=first if first == second else "disagree",
            target=ta if ta == tb else "disagree", defense=round(defense, 2),
        )

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
            # Commit one small displacement pulse immediately: a later
            # observation may invalidate the forecast before the cartridge
            # has turned enough for a pure-rotation reservation to act.
            "turn_only_until": self.frame,
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
        remaining = _integer(move.get("remaining"), 0)
        if delta:
            # Orientation changes only every few native frames.  The parent
            # often spent the whole tranche rotating, so its selected escape
            # never displaced the ship.  Couple one bounded thrust pulse to
            # the first turn, then keep the five-frame pulse spacing.  This
            # remains revalidated and does not become indiscriminate thrust.
            turn_only_until = _integer(move.get("turn_only_until"), self.frame)
            grace_over = self.frame >= turn_only_until
            critical = self.facts.get("danger") == "critical"
            if remaining > 0 and self.frame >= self.maneuver_next and (grace_over or critical):
                move["remaining"] = remaining - 1
                self.maneuver_next = self.frame + 5
                self.thrust_actions += 1
                self._log(
                    "escape_pulse", frame=self.frame, path=move.get("path"),
                    desired=desired, heading=current, delta=delta,
                    danger=self.facts.get("danger"),
                )
                return self._compose(1 if delta > 0 else -1, thrust=True, fire=False)
            return self._compose(1 if delta > 0 else -1, thrust=False, fire=False)
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

    def apply_astra(self, text):
        data = _json_object(text)
        if not isinstance(data, dict):
            return False
        allowed = {
            "aim": {"threat_first", "value_first", "sweep"},
            "mobility": {"anchor_then_coast", "reactive_escape"},
            "fire": {"normal", "spaced"},
            "defense": {"last_resort", "never"},
        }
        changed = {}
        for key, values in allowed.items():
            value = data.get(key)
            if isinstance(value, str) and value in values:
                changed[key] = value
        if not changed:
            return False
        self.plan.update(changed)
        self._log("astra_plan", frame=self.frame, plan=dict(self.plan))
        return True


class Harness:
    def __init__(self, ctx):
        self.ctx = ctx

    def _start_jev(self, pilot):
        if getattr(self.ctx, "system1", True) is None:
            return None
        state = pilot.jev_state()
        questions = pilot.jev_questions()
        origin_frame = pilot.frame
        origin_epoch = pilot.life_epoch
        origin_targets = list(pilot.last_jev_options)

        def request():
            return self.ctx.decide(state, questions, timeout_s=2.25)

        try:
            future = self.ctx.spawn(request)
        except (BudgetExceeded, EpisodeOver):
            return None
        except ModelUnavailable:
            raise
        except Exception:
            return None
        return future, origin_frame, origin_epoch, origin_targets

    def _start_astra(self, pilot, reason):
        try:
            remaining = self.ctx.budget.remaining()
        except Exception:
            remaining = {}
        if _integer(_dict(remaining).get("system2_calls"), 0) <= 0:
            return None
        summary = pilot.astra_summary(reason)
        prompt = (
            "You are the slow strategic reviewer for a real-time Atari 2600 "
            "Asteroids mode-0/difficulty-0 controller. The local code owns the "
            "clock, object geometry, wrap-aware forecasts, momentum damping, "
            "bounded escape pulses, FIRE press/release, life bookkeeping, and "
            "hyperspace legality. Jev only labels a current state and selects an "
            "observed target. Review the aggregate evidence below and return one "
            "small JSON object only. Never return actions, sequences, timestamps, "
            "or prose. Allowed values are: aim='threat_first'|'value_first'|'sweep', "
            "mobility='anchor_then_coast'|'reactive_escape', fire='normal'|'spaced', "
            "defense='last_resort'|'never'. Unknown enemy coverage is not absence.\n"
            + json.dumps(summary, separators=(",", ":"))
        )

        def request():
            return self.ctx.think(prompt, timeout_s=100.0)

        try:
            return self.ctx.spawn(request)
        except (BudgetExceeded, EpisodeOver):
            return None
        except ModelUnavailable:
            raise
        except Exception:
            return None

    def run_episode(self, game) -> None:
        ctx = self.ctx
        if getattr(game, "tick_s", None) is None:
            return
        pilot = KinematicPilot(game, ctx)
        pilot.update(_observation(game))

        def post(action):
            try:
                ctx.act(action)
                return True
            except (BudgetExceeded, EpisodeOver, ModelUnavailable):
                return False
            except Exception:
                try:
                    ctx.act(pilot.noop)
                    return True
                except Exception:
                    return False

        if not post(pilot.choose()):
            return

        s1_enabled = getattr(ctx, "system1", True) is not None
        jev_future = None
        jev_frame = -100000
        jev_epoch = -1
        jev_targets = []
        next_jev = pilot.frame

        astra_future = None
        astra_started = False
        astra_due = pilot.frame + 300

        while not ctx.done and not getattr(game, "done", False):
            if s1_enabled and jev_future is None and pilot.frame >= next_jev:
                try:
                    result = self._start_jev(pilot)
                    if result is not None:
                        jev_future, jev_frame, jev_epoch, jev_targets = result
                        next_jev = pilot.frame + 36
                    else:
                        next_jev = pilot.frame + 48
                except ModelUnavailable:
                    return
                except (BudgetExceeded, EpisodeOver):
                    s1_enabled = False
                except Exception:
                    next_jev = pilot.frame + 48

            if not astra_started and astra_future is None and pilot.frame >= astra_due:
                astra_started = True
                try:
                    astra_future = self._start_astra(pilot, "post-launch review")
                except ModelUnavailable:
                    return
                except (BudgetExceeded, EpisodeOver):
                    astra_future = None
                except Exception:
                    astra_future = None

            try:
                ctx.wait_for_tick()
            except (BudgetExceeded, EpisodeOver, ModelUnavailable):
                return
            if ctx.done or getattr(game, "done", False):
                break

            pilot.update(_observation(game))

            if jev_future is not None and jev_future.done():
                finished = jev_future
                jev_future = None
                try:
                    pilot.accept_jev(finished.result(), jev_frame, jev_epoch, jev_targets)
                except (BudgetExceeded, EpisodeOver):
                    s1_enabled = False
                except ModelUnavailable:
                    return
                except Exception:
                    # Timeout, cancellation, arbitrary stand-in output, and
                    # malformed typed answers are advisory misses.
                    pass

            if astra_future is not None and astra_future.done():
                finished = astra_future
                astra_future = None
                try:
                    result = finished.result()
                    pilot.apply_astra(getattr(result, "text", result))
                except (BudgetExceeded, EpisodeOver):
                    pass
                except ModelUnavailable:
                    return
                except Exception:
                    pass

            if not post(pilot.choose()):
                return


def build(ctx):
    return Harness(ctx)
