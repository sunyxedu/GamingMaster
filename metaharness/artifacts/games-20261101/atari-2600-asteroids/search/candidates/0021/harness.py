"""Contact-rescue turret for real-time Atari 2600 Asteroids.

The controller keeps the useful sparse-fire turret from the archive, but treats a
near crossing as a different control problem.  A close rock gets a short,
observed contact-rescue manoeuvre (or one bounded defense pulse followed by a
rescue), rather than the old ``DOWN -> coast`` recovery.  All manoeuvres are
recomputed from the newest snapshot and expire quickly; Jev and Astra can
change preferences but cannot hold a control or bypass the local safety gates.
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


# ---------- defensive parsing helpers ----------


def _num(value, default=None):
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _int(value, default=0):
    result = _num(value)
    return int(result) if result is not None else default


def _pair(value):
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        return None
    x, y = _num(value[0]), _num(value[1])
    return (x, y) if x is not None and y is not None else None


def _dict(value):
    return value if isinstance(value, dict) else {}


def _list(value):
    return value if isinstance(value, list) else []


def _wrap(value, period):
    value = _num(value, 0.0) or 0.0
    while value > period / 2.0:
        value -= period
    while value < -period / 2.0:
        value += period
    return value


def _bucket(value, cuts, labels):
    if value is None:
        return labels[0]
    for cut, label in zip(cuts, labels[1:]):
        if value < cut:
            return label
    return labels[-1]


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
    """Find the first JSON object in a possibly arbitrary Astra response."""
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
    return getattr(answer, "value", None)


def _answer_probs(answer):
    if isinstance(answer, dict):
        result = answer.get("probabilities", {})
    else:
        result = getattr(answer, "probabilities", {})
    return result if isinstance(result, dict) else {}


def _prob(answer, key):
    value = _num(_answer_probs(answer).get(key), 0.0) or 0.0
    return max(0.0, min(1.0, value))


def _yes(answer):
    value = _answer_value(answer)
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, str):
        return 1.0 if value.lower() in {"true", "yes"} else 0.0
    if isinstance(value, (int, float)):
        return max(0.0, min(1.0, _num(value, 0.0) or 0.0))
    return _prob(answer, "true")


# Code 0 is up, 4 left, 8 down, 12 right.  Intermediate codes are kept in the
# same ring even though the adaptor deliberately does not claim their angles.
def _heading_vector(code):
    angle = (_int(code) % 16) * math.pi / 8.0
    return -math.sin(angle), -math.cos(angle)


def _desired_heading(dx, dy):
    if abs(dx) + abs(dy) < 1.0e-8:
        return 0
    return int(round(math.atan2(-dx, -dy) / (math.pi / 8.0))) % 16


def _turn_delta(current, desired):
    if current is None or desired is None:
        return 0
    delta = (int(desired) - int(current)) % 16
    if delta > 8:
        delta -= 16
    return delta


def _thrusting(action):
    return isinstance(action, str) and action.startswith("UP")


class ContactPilot:
    """Frame-local occupancy controller with a separate close-contact alarm."""

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
        self.old_lives = None
        self.life_epoch = 0
        self.obs = {}
        self.facts = {}
        self.previous_position = None
        self.previous_velocity = None
        self.player_velocity = None
        self.last_heading = None
        self.last_action = self.noop

        # Short-lived source-slot tracks are only used to bridge a flicker.  A
        # two-frame same-size track is not treated as a permanent identity.
        self.track_cache = {}
        # The bounded launch prelude can place a harmless rock close to the
        # freshly moving ship.  A velocity-backed TTC may override this guard,
        # but a lone flicker cannot trigger emergency motion in the opening.
        self.near_guard_until = 0

        self.plan = {
            "aim": "threat_first",
            "mobility": "contact_rescue",
            "fire": "normal",
            "defense": "last_resort",
        }
        self.mode_hint = None
        self.mode_until = -1
        self.target_hint = None
        self.target_until = -1
        self.defense_hint = False

        self.acceleration = 0.045
        self.accel_samples = 0
        self.anchor_next = 0
        self.anchor_count = 0
        self.settle_until = 0
        self.recovery_until = 0

        # ``rescue`` is a tiny receding-horizon program, not an action latch.
        # It records a desired code and a small number of one-frame thrusts.
        self.rescue = None
        self.rescue_next = 0
        self.escape_next = 0
        self.escape_count = 0
        self.defense_used = False
        self.defense_count = 0
        self.defense_frame = -100000
        self.rescue_after_defense = False
        self.rescue_failures = 0
        self.critical_streak = 0

        self.next_fire = 0
        self.next_sweep = 0
        self.sweep_sign = 1
        self.last_log = -100000
        self.last_jev_targets = []
        self.losses = 0
        self.last_loss_context = "none"

    # ---------- small utilities ----------

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
        x, y = _num(player.get("x")), _num(player.get("y"))
        if x is None or y is None:
            return None
        w = _num(player.get("width"), 6.0) or 6.0
        h = _num(player.get("height"), 10.0) or 10.0
        return x + w / 2.0, y + h / 2.0, w, h

    @staticmethod
    def _relative_bearing(dx, dy):
        horizontal = "right" if dx > 10 else "left" if dx < -10 else "center"
        vertical = "below" if dy > 10 else "above" if dy < -10 else "level"
        return horizontal + "_" + vertical

    def _read_velocity(self, player, finite_difference):
        observed = _pair(player.get("velocity_px_per_frame"))
        if bool(player.get("motion_discontinuity")):
            return None
        if observed is None:
            return finite_difference
        if finite_difference is None:
            return observed
        # The adaptor's multi-frame value can quantize a small displacement to
        # zero; retain a visible one-frame displacement in that case.
        if abs(observed[0]) + abs(observed[1]) < 0.01 and abs(finite_difference[0]) + abs(finite_difference[1]) > 0.08:
            return finite_difference
        return (
            0.72 * observed[0] + 0.28 * finite_difference[0],
            0.72 * observed[1] + 0.28 * finite_difference[1],
        )

    def _learn_acceleration(self, velocity, discontinuity):
        old = self.previous_velocity
        if old is None or velocity is None or discontinuity:
            return
        if not _thrusting(self.last_action) or self.last_heading is None:
            return
        ux, uy = _heading_vector(self.last_heading)
        gain = (velocity[0] - old[0]) * ux + (velocity[1] - old[1]) * uy
        if 0.006 <= gain <= 0.16:
            self.acceleration = max(0.018, min(0.08, 0.88 * self.acceleration + 0.12 * gain))
            self.accel_samples += 1

    # ---------- tracking and occupancy ----------

    def _rock_records(self, obs, player, px, py, pw, ph, pv):
        records = []
        seen = set()
        raw_asteroids = _list(obs.get("asteroids"))
        for index, raw0 in enumerate(raw_asteroids):
            raw = _dict(raw0)
            rx, ry = _num(raw.get("x")), _num(raw.get("y"))
            if rx is None or ry is None:
                continue
            rw = _num(raw.get("width"), 8.0) or 8.0
            rh = _num(raw.get("height"), 15.0) or 15.0
            size = str(raw.get("size") or "unknown").lower()
            slot = _int(raw.get("source_slot"), index)
            key = (slot, size)
            seen.add(key)
            cx, cy = rx + rw / 2.0, ry + rh / 2.0
            raw_v = _pair(raw.get("velocity_px_per_frame"))
            discontinuity = bool(raw.get("motion_discontinuity"))
            old = self.track_cache.get(key)
            use_old = False
            if raw_v is None and not discontinuity and isinstance(old, dict) and self.frame - old.get("frame", self.frame) <= 2:
                # Reject same-slot reuse if it jumped farther than a short
                # screen-space motion can plausibly travel.
                odx = _wrap(cx - old.get("x", cx), WIDTH)
                ody = _wrap(cy - old.get("y", cy), HEIGHT)
                use_old = abs(odx) <= 18.0 and abs(ody) <= 22.0
            velocity = raw_v if raw_v is not None and not discontinuity else (
                (old.get("vx"), old.get("vy")) if use_old else None
            )
            if velocity is not None and (velocity[0] is None or velocity[1] is None):
                velocity = None
            self.track_cache[key] = {
                "frame": self.frame, "x": cx, "y": cy, "vx": velocity[0] if velocity else 0.0,
                "vy": velocity[1] if velocity else 0.0, "w": rw, "h": rh,
            }
            records.append(self._make_rock(
                slot, index, size, cx, cy, rw, rh, velocity, pv, px, py, pw, ph,
                ghost=False, reused=use_old,
            ))

        # Do not synthesize an absent source slot.  The adaptor explicitly
        # warns that slots can be reused, so the cache is used only above to
        # stabilize a currently visible same-size record; a missing sprite is
        # uncertainty, not a new physical object.
        return records

    @staticmethod
    def _make_rock(slot, index, size, cx, cy, rw, rh, velocity, pv,
                   px, py, pw, ph, ghost=False, reused=False):
        if velocity is not None and pv is not None:
            rvx, rvy = velocity[0] - pv[0], velocity[1] - pv[1]
            known = not ghost
        elif velocity is not None:
            rvx, rvy = velocity
            known = False
        else:
            rvx = rvy = 0.0
            known = False
        extra = 5.0 if ghost or not known else 0.0
        dx, dy = _wrap(cx - px, WIDTH), _wrap(cy - py, HEIGHT)
        size_value = {"small": 3, "medium": 2, "large": 1}.get(size, 1)
        return {
            "id": "r" + str(slot) + "_" + size + "_" + str(index),
            "slot": slot, "index": index, "size": size, "value": size_value,
            "x": cx, "y": cy, "dx": dx, "dy": dy,
            "rvx": rvx, "rvy": rvy, "known": bool(known),
            "ghost": bool(ghost), "reused": bool(reused),
            "half_x": rw / 2.0 + pw / 2.0 + 3.0 + extra,
            "half_y": rh / 2.0 + ph / 2.0 + 3.0 + extra,
            "distance": math.hypot(dx, dy),
        }

    def _project(self, rock, desired_code=None, pulse=0, turn_delay=0, horizon=54):
        """Forecast one actual one-frame thrust pulse after a turn delay."""
        minimum = float("inf")
        first = None
        # Unknown and flickering objects are widened instead of being called
        # absent.  The forecast is a ranking signal, never a clearance proof.
        uncertainty = 3.0 if not rock.get("known") else 0.0
        rel_speed = math.hypot(rock.get("rvx", 0.0), rock.get("rvy", 0.0))
        if desired_code is None or pulse <= 0:
            ux = uy = 0.0
        else:
            ux, uy = _heading_vector(desired_code)
        # Exact samples cover the lethal near window; coarser samples retain
        # the longer forecast without making the frame loop miss its deadline.
        times = list(range(0, 9)) + list(range(10, horizon + 1, 3))
        if times[-1] != horizon:
            times.append(horizon)
        for t in times:
            active = max(0.0, min(float(pulse), float(t - turn_delay)))
            # A pulse adds velocity and then the ship coasts.  This is closer
            # to the observed impulse than the older consecutive-thrust lane
            # approximation, while still being cheap enough for every frame.
            travel = self.acceleration * max(0.0, float(t - turn_delay) - 0.5 * active) * active
            if pulse > 0 and t > turn_delay + pulse:
                travel = self.acceleration * (float(t - turn_delay) - 0.5 * float(pulse)) * float(pulse)
            player_dx = (self.facts.get("vx") or 0.0) * t + ux * travel
            player_dy = (self.facts.get("vy") or 0.0) * t + uy * travel
            rx = _wrap(rock["dx"] + rock["rvx"] * t - player_dx, WIDTH)
            ry = _wrap(rock["dy"] + rock["rvy"] * t - player_dy, HEIGHT)
            gap_x = abs(rx) - rock["half_x"] - uncertainty - min(5.0, rel_speed)
            gap_y = abs(ry) - rock["half_y"] - uncertainty - min(5.0, rel_speed)
            clearance = max(gap_x, gap_y)
            minimum = min(minimum, clearance)
            if first is None and gap_x <= 0.0 and gap_y <= 0.0:
                first = t
                # Retain the current contact as a useful minimum, but do not
                # spend work on a long horizon after a clear collision.
                minimum = min(minimum, max(gap_x, gap_y))
                if t <= 8:
                    break
        return first, minimum

    def _path(self, path_id, code, pulse, records, heading):
        if code is None:
            delay = 0
        else:
            delay = min(18, abs(_turn_delta(heading, code)) * 3) if heading is not None else 12
        outcomes = []
        for rock in records:
            first, clearance = self._project(rock, code, pulse, delay)
            outcomes.append((rock, first, clearance))
        total = 0.0
        minimum = float("inf")
        first_contact = None
        for rock, first, clearance in outcomes:
            minimum = min(minimum, clearance)
            if first is None:
                total += max(0.0, 12.0 - clearance) * (0.5 if rock.get("known") else 0.8)
            else:
                total += 48.0 + max(0.0, 30.0 - first) * (1.5 + 0.15 * rock.get("value", 1))
                if first_contact is None or first < first_contact:
                    first_contact = first
            if rock.get("ghost"):
                total += 1.5
        return {
            "id": path_id, "code": code, "pulse": pulse, "delay": delay,
            "outcomes": outcomes, "cost": total,
            "minimum": minimum if records else 100.0, "first": first_contact,
        }

    def _build_paths(self, records, heading):
        paths = [self._path("coast", None, 0, records, heading)]
        if heading is not None:
            paths.append(self._path("nose", int(heading) % 16, 1, records, heading))
        # Include cardinal and diagonal headings.  They are options for one
        # bounded pulse, not a promise that an intermediate angle is calibrated.
        for code in (0, 2, 4, 6, 8, 10, 12, 14):
            paths.append(self._path("d" + str(code), code, 1, records, heading))
        return paths

    @staticmethod
    def _path_by_id(paths, path_id):
        return next((p for p in paths if p.get("id") == path_id), None)

    def _target(self, records, coast):
        if not records:
            return None
        contact = {}
        for rock, first, _ in (coast or {}).get("outcomes", []):
            contact[rock.get("id")] = first

        def rank(rock):
            first = contact.get(rock.get("id"))
            urgency = 100.0 if first is not None else 0.0
            if first is not None:
                urgency += max(0.0, 50.0 - first) * 1.6
            if rock.get("distance", 999.0) < 78:
                urgency += 10.0
            return -(urgency + 4.0 * rock.get("value", 1) +
                     (1.0 if rock.get("known") else 0.0) - rock.get("distance", 999.0) / 180.0)
        return min(records, key=rank)

    def _near_alarm(self, records):
        """Detect a close corridor even when a flicker leaves velocity null."""
        candidate = None
        best_score = float("inf")
        for rock in records:
            d = rock.get("distance", 999.0)
            if d > 44.0:
                continue
            xgap = abs(rock.get("dx", 0.0)) - rock.get("half_x", 10.0)
            ygap = abs(rock.get("dy", 0.0)) - rock.get("half_y", 12.0)
            axis_corridor = min(xgap, ygap) <= 9.0
            closing = (rock.get("dx", 0.0) * rock.get("rvx", 0.0) +
                       rock.get("dy", 0.0) * rock.get("rvy", 0.0)) < -0.01
            # A side-by-side object is not automatically dangerous.  Unknown
            # motion gets the conservative interpretation only when it is
            # genuinely close; otherwise the sparse turret remains the parent
            # policy and does not panic on a launch-frame flicker.
            if rock.get("known"):
                # A known object is an alarm only while it is closing; a
                # stationary-looking nearby sprite is still a firing target.
                active = axis_corridor and closing
            else:
                # Flicker/ghost evidence is reserved for the final contact
                # corridor, not every harmless object near the launch route.
                active = axis_corridor and d <= 12.0
            if active:
                score = d + (0.0 if rock.get("known") else 3.0)
                if score < best_score:
                    best_score, candidate = score, rock
        return candidate

    def _derive(self, obs, player, center, velocity):
        if center is None:
            return {
                "present": False, "frame": self.frame, "score": self.score,
                "lives": self.lives, "x": None, "y": None, "heading": None,
                "vx": None, "vy": None, "speed": None, "rocks": [], "paths": [],
                "coast": None, "best": None, "ttc": None, "danger": "unknown",
                "near": None, "near_alarm": False, "uncertain": True,
                "target": None, "missiles": 0, "discontinuity": True,
            }
        px, py, pw, ph = center
        heading = _int(player.get("orientation_code"), 0) % 16
        records = self._rock_records(obs, player, px, py, pw, ph, velocity)
        self.facts["vx"] = velocity[0] if velocity is not None else 0.0
        self.facts["vy"] = velocity[1] if velocity is not None else 0.0
        paths = self._build_paths(records, heading)
        coast = self._path_by_id(paths, "coast")
        alternatives = [p for p in paths if p.get("id") != "coast"]
        best = min(alternatives, key=lambda p: (p.get("cost", 9999.0), -p.get("minimum", -9999.0)), default=coast)
        ttc = coast.get("first") if coast else None
        near = self._near_alarm(records)
        near_alarm = near is not None
        if ttc is None:
            if near_alarm and near.get("distance", 999.0) <= 21.0:
                danger = "critical"
            elif near_alarm:
                danger = "caution"
            else:
                danger = "clear"
        elif ttc <= 5:
            danger = "critical"
        elif ttc <= 18:
            danger = "caution"
        elif ttc <= 48:
            danger = "watch"
        else:
            danger = "clear"
        if near_alarm and danger == "clear":
            danger = "caution"
        speed = math.hypot(*velocity) if velocity is not None else None
        near_unknown = any((not r.get("known")) and r.get("distance", 999.0) < 48.0 for r in records)
        uncertain = velocity is None or near_unknown or any(not r.get("known") for r in records)
        missiles = sum(1 for item in _list(obs.get("player_missiles")) if isinstance(item, dict))
        return {
            "present": True, "frame": self.frame, "score": self.score, "lives": self.lives,
            "x": px, "y": py, "heading": heading,
            "vx": velocity[0] if velocity is not None else None,
            "vy": velocity[1] if velocity is not None else None,
            "speed": speed, "rocks": records, "paths": paths, "coast": coast, "best": best,
            "best_gain": (coast.get("cost", 0.0) - best.get("cost", 0.0)) if coast and best else 0.0,
            "ttc": ttc, "danger": danger, "near": near, "near_alarm": near_alarm,
            "near_unknown": near_unknown, "uncertain": uncertain,
            "target": self._target(records, coast), "missiles": missiles,
            "discontinuity": bool(player.get("motion_discontinuity")),
        }

    def update(self, observation):
        data = _dict(observation)
        self.obs = data
        old_frame = self.frame
        self.frame = _int(data.get("frame"), old_frame + 1)
        self.score = _num(data.get("score"), self.score) or 0.0
        player = _dict(data.get("player"))
        center = self._center(player)

        finite_difference = None
        if center is not None and self.previous_position is not None:
            frames = max(1, self.frame - old_frame)
            dx = _wrap(center[0] - self.previous_position[0], WIDTH) / frames
            dy = _wrap(center[1] - self.previous_position[1], HEIGHT) / frames
            if abs(dx) <= 24.0 and abs(dy) <= 30.0:
                finite_difference = (dx, dy)
        discontinuity = center is None or bool(player.get("motion_discontinuity"))
        if finite_difference is None and center is not None and self.previous_position is not None:
            discontinuity = True
        velocity = self._read_velocity(player, finite_difference)
        self._learn_acceleration(velocity, discontinuity)

        new_lives = _int(data.get("lives"), self.lives)
        life_changed = self.old_lives is not None and new_lives != self.old_lives
        if self.old_lives is None:
            self.near_guard_until = self.frame + 180
        if life_changed:
            self.near_guard_until = self.frame + 24
            if new_lives < self.old_lives:
                self.losses += self.old_lives - new_lives
                self.last_loss_context = "rescue" if self.rescue is not None else (
                    "defense" if self.defense_frame > self.frame - 30 else "turret"
                )
            self.life_epoch += 1
            self.rescue = None
            self.target_hint = None
            self.mode_hint = None
            self.defense_hint = False
            self.defense_used = False
            self.defense_count = 0
            self.recovery_until = self.frame + 12
            self.anchor_next = self.frame + 8
            self.next_fire = self.frame + 6
            self.rescue_after_defense = False
            self.critical_streak = 0
            self._log("life_epoch", frame=self.frame, old_lives=self.old_lives,
                      lives=new_lives, loss_context=self.last_loss_context)
        self.lives = new_lives
        self.old_lives = new_lives

        if discontinuity:
            if self.rescue is not None:
                self.rescue_failures += 1
            self.rescue = None
            self.target_hint = None
            self.critical_streak = 0
            # A visible jump may be hyperspace or respawn.  Do not carry a
            # route, but do not impose a long idle period either.
            self.recovery_until = max(self.recovery_until, self.frame + (10 if center is not None else 8))
            if center is not None:
                self._log("track_reset", frame=self.frame)
        self.facts = self._derive(data, player, center, velocity)
        self.facts["discontinuity"] = discontinuity
        if (self.facts.get("ttc") is not None and self.facts.get("ttc") <= 3):
            self.critical_streak += 1
        else:
            self.critical_streak = 0

        # Revalidate a rescue against the newest snapshot.  A clear state ends
        # it; a still-dangerous state gets another short opportunity later.
        if isinstance(self.rescue, dict):
            if self.frame > _int(self.rescue.get("expires"), self.frame) or not self.facts.get("present"):
                self.rescue = None
            elif (self.facts.get("danger") == "clear" and
                  self.frame > _int(self.rescue.get("started"), self.frame) + 2):
                self._log("rescue_end", frame=self.frame, reason="clear")
                self.rescue = None
            elif self.facts.get("near") is None and self.facts.get("ttc") is None and self.frame > _int(self.rescue.get("started"), self.frame) + 3:
                self.rescue = None

        self.previous_position = (center[0], center[1]) if center is not None else None
        self.previous_velocity = velocity
        self.player_velocity = velocity

    # ---------- short rescue and turret decisions ----------

    def _risk_record(self):
        facts = self.facts
        if facts.get("near") is not None:
            return facts["near"]
        rocks = facts.get("rocks", [])
        if not rocks:
            return None
        coast = facts.get("coast") or {}
        contacts = {r.get("id"): first for r, first, _ in coast.get("outcomes", [])}
        return min(rocks, key=lambda r: (
            (contacts.get(r.get("id")) if contacts.get(r.get("id")) is not None else 999),
            r.get("distance", 999.0), -r.get("value", 1)
        ))

    def _away_code(self, rock):
        if rock is None:
            heading = self.facts.get("heading")
            return ((heading or 0) + 4) % 16
        dx, dy = rock.get("dx", 0.0), rock.get("dy", 0.0)
        if abs(dx) + abs(dy) > 1.0:
            return _desired_heading(-dx, -dy)
        rvx, rvy = rock.get("rvx", 0.0), rock.get("rvy", 0.0)
        if abs(rvx) + abs(rvy) > 0.05:
            return _desired_heading(-rvx, -rvy)
        return ((self.facts.get("heading") or 0) + 4) % 16

    def _local_pulse_score(self, path, risk):
        """Prefer a path that changes the next few frames, not only its horizon."""
        if not isinstance(path, dict):
            return (-9999.0, -9999.0, -9999.0)
        outcomes = path.get("outcomes", [])
        own = next((item for item in outcomes if item[0].get("id") == risk.get("id")), None) if risk else None
        first = own[1] if own else path.get("first")
        clearance = own[2] if own else path.get("minimum", -999.0)
        # An action which delays the nearest crossing is useful even if another
        # rock still blocks the long horizon.  This is the deliberate escape
        # from the old all-candidates-overlap -> NOOP dead end.
        first_score = 100.0 if first is None else min(40.0, float(first))
        local = float(clearance)
        if own:
            local = max(local, own[2])
        return (first_score, local, -float(path.get("cost", 9999.0)))

    def _choose_rescue_path(self, urgent=False):
        facts = self.facts
        paths = facts.get("paths", [])
        risk = self._risk_record()
        away = self._away_code(risk)
        preferred = [p for p in paths if p.get("id") not in {None, "coast"}]
        if not preferred:
            return None
        # Keep the direct separation direction in the tie set.  Otherwise a
        # global cost can choose a far rock and ignore the object already beside
        # the ship.
        scored = []
        for p in preferred:
            score = self._local_pulse_score(p, risk)
            away_bonus = 2.0 if p.get("code") == away else 0.0
            if p.get("code") is not None and abs(_turn_delta(facts.get("heading"), p.get("code"))) <= 1:
                away_bonus += 0.5
            scored.append((score[0], score[1] + away_bonus, score[2], p))
        scored.sort(key=lambda item: (item[0], item[1], item[2]), reverse=True)
        return scored[0][3]

    def _start_rescue(self, reason, urgent=False, post_defense=False):
        if not self.facts.get("present") or self.frame < self.rescue_next:
            return False
        path = self._choose_rescue_path(urgent=urgent)
        if path is None or path.get("code") is None:
            return False
        risk = self._risk_record()
        coast = self.facts.get("coast") or {}
        own = None
        if risk:
            own = next((v for r, _, v in path.get("outcomes", []) if r.get("id") == risk.get("id")), None)
        coast_own = None
        if risk:
            coast_own = next((v for r, _, v in coast.get("outcomes", []) if r.get("id") == risk.get("id")), None)
        gain = (coast_own - own) if own is not None and coast_own is not None else _num(self.facts.get("best_gain"), 0.0) or 0.0
        ttc = self.facts.get("ttc")
        if not post_defense:
            # One noisy TTC/near label is not enough to leave the proven turret.
            # Require a confirmed multi-frame crossing, or a materially safer
            # forecast, before spending thrust on an ordinary contact.
            if ttc is None:
                return False
            if self.critical_streak < 3 and gain < 4.0:
                return False
            if urgent and ttc > 2 and gain < 4.0:
                return False
            if not urgent and not self.facts.get("near_alarm") and ttc > 20:
                return False
            if not urgent and gain < 4.0:
                return False
        near = self.facts.get("near")
        hard_unknown = (ttc is None and bool(near) and near.get("distance", 999.0) <= 12.0)
        pulses = 2 if post_defense or hard_unknown else 1
        self.rescue = {
            "code": _int(path.get("code"), 0) % 16,
            "remaining": pulses,
            "started": self.frame,
            "expires": self.frame + (20 if urgent or post_defense else 16),
            "reason": reason,
            "path": path.get("id"),
        }
        self.rescue_next = self.frame + (5 if urgent else 9)
        self.escape_count += 1
        self._log("rescue_start", frame=self.frame, reason=reason,
                  path=path.get("id"), code=self.rescue["code"], ttc=ttc,
                  gain=round(gain, 2), urgent=bool(urgent), post_defense=bool(post_defense))
        return True

    def _rescue_action(self):
        move = self.rescue
        facts = self.facts
        if not isinstance(move, dict) or not facts.get("present"):
            return None
        if self.frame > _int(move.get("expires"), self.frame):
            self.rescue = None
            return None
        desired = _int(move.get("code"), facts.get("heading") or 0) % 16
        current = facts.get("heading")
        delta = _turn_delta(current, desired)
        ttc = facts.get("ttc")
        fire = False
        if abs(delta) > 1:
            # When there is still time, turn without adding a bad trajectory.
            # At the last few frames, thrust+turn at least changes momentum now;
            # this is only active inside a close alarm and expires quickly.
            turn = 1 if delta > 0 else -1
            if ((ttc is not None and ttc <= 8) or facts.get("near_alarm")):
                # A close crossing cannot spend its whole response budget
                # rotating toward a perfect code.  The thrust+turn input gives
                # the ship an immediate impulse, then later frames can align.
                fire = self._fire_ready() if ttc is not None and ttc > 4 else False
                return self._compose(turn=turn, thrust=True, fire=fire)
            return self._compose(turn=turn, thrust=False, fire=False)
        remaining = _int(move.get("remaining"), 0)
        if remaining <= 0:
            self.rescue = None
            return None
        if self.frame < self.rescue_next:
            return self.noop
        move["remaining"] = remaining - 1
        self.rescue_next = self.frame + 5
        self._log("rescue_pulse", frame=self.frame, path=move.get("path"),
                  code=desired, ttc=ttc, remaining=move["remaining"])
        return self._compose(turn=0, thrust=True, fire=False)

    def _should_defend(self):
        facts = self.facts
        if not facts.get("present") or self.defense_used or self.plan.get("defense") == "never":
            return False
        ttc = facts.get("ttc")
        # Unknown close geometry is handled by the contact-rescue pulses.
        # Hyperspace is reserved for a velocity-backed unresolved crossing; this
        # avoids spending it on a launch-frame/flicker false alarm.
        if ttc is None:
            return False
        effective_ttc = ttc
        if self.critical_streak < 3:
            return False
        if effective_ttc > 4:
            return False
        best = facts.get("best") or {}
        coast = facts.get("coast") or {}
        safe_path = best.get("id") not in {None, "coast"} and best.get("first") is None
        if safe_path and (facts.get("best_gain", 0.0) or 0.0) >= 1.0:
            return False
        # Do not let a fresh rescue be pre-empted by DOWN at ttc 3; the rescue
        # has an actual positional effect.  If it has exhausted its pulses,
        # hyperspace becomes the last bounded option.
        if isinstance(self.rescue, dict) and _int(self.rescue.get("remaining"), 0) > 0:
            return False
        if effective_ttc <= 2:
            return True
        return effective_ttc <= 3 and self.rescue_failures == 0

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

    def _anchor_action(self):
        facts = self.facts
        speed = facts.get("speed")
        if not facts.get("present") or speed is None or speed <= 0.18:
            return None
        if self.frame < self.settle_until or self.frame < self.anchor_next:
            return None
        if facts.get("danger") in {"caution", "critical"} or facts.get("near_alarm"):
            return None
        vx, vy = facts.get("vx"), facts.get("vy")
        if vx is None or vy is None:
            return None
        desired = _desired_heading(-vx, -vy)
        delta = _turn_delta(facts.get("heading"), desired)
        if delta:
            return self._compose(turn=1 if delta > 0 else -1, thrust=False, fire=False)
        self.anchor_next = self.frame + 10
        self.anchor_count += 1
        self._log("anchor_pulse", frame=self.frame, speed=round(speed, 3), desired=desired)
        return self._compose(thrust=True, fire=False)

    def _aim_target(self):
        records = self.facts.get("rocks", [])
        if self.target_hint and self.frame <= self.target_until:
            hinted = next((r for r in records if r.get("id") == self.target_hint), None)
            if hinted is not None:
                return hinted
        if self.plan.get("aim") == "sweep":
            return None
        return self.facts.get("target")

    def _aim_turn(self):
        facts = self.facts
        heading = facts.get("heading")
        if heading is None:
            return 0
        if self.mode_hint == "hold" and self.frame <= self.mode_until and facts.get("danger") == "clear":
            return 0
        target = self._aim_target()
        if target is not None:
            lead = 5.0 if target.get("known") else 1.5
            desired = _desired_heading(
                target.get("dx", 0.0) + target.get("rvx", 0.0) * lead,
                target.get("dy", 0.0) + target.get("rvy", 0.0) * lead,
            )
            delta = _turn_delta(heading, desired)
            return 1 if delta > 0 else -1 if delta < 0 else 0
        if self.frame >= self.next_sweep:
            self.sweep_sign *= -1
            self.next_sweep = self.frame + 48
        return 1 if self.sweep_sign > 0 else -1

    def choose(self):
        facts = self.facts
        if not facts.get("present"):
            self.last_action = self.noop
            return self.noop

        # A contact rescue is checked before both offense and defense.  This is
        # the main structural change: a post-DOWN snapshot is not a license to
        # coast while the same collision remains in the corridor.
        if self.rescue is not None:
            action = self._rescue_action()
            if action is not None:
                self.last_action = action
                self.last_heading = facts.get("heading")
                return action

        near = facts.get("near_alarm")
        ttc = facts.get("ttc")
        urgent = facts.get("danger") == "critical" or (ttc is not None and ttc <= 7)
        if urgent or near:
            if self._start_rescue("contact", urgent=urgent):
                action = self._rescue_action()
                if action is not None:
                    self.last_action = action
                    self.last_heading = facts.get("heading")
                    return action

        if self._should_defend():
            self.defense_used = True
            self.defense_count += 1
            self.defense_frame = self.frame
            self.rescue = None
            # DOWN is a one-frame mode-0 defense.  Arrange a new rescue from
            # the post-defense observation, not a long NOOP quarantine.
            self.rescue_after_defense = True
            self.recovery_until = self.frame + 8
            self._log("defense", frame=self.frame, ttc=ttc,
                      near=bool(near), count=self.defense_count)
            self.last_action = self.legal("DOWN")
            return self.last_action

        if self.rescue_after_defense and self.frame >= self.defense_frame + 1:
            self.rescue_after_defense = False
            if self._start_rescue("post_defense", urgent=True, post_defense=True):
                action = self._rescue_action()
                if action is not None:
                    self.last_action = action
                    self.last_heading = facts.get("heading")
                    return action

        # A genuinely safer forecast can justify one preemptive pulse, but the
        # idle-like turret remains the default for distant/uncertain fields.
        if facts.get("danger") in {"caution", "watch"} and not facts.get("near_unknown"):
            best = facts.get("best") or {}
            gain = _num(facts.get("best_gain"), 0.0) or 0.0
            ttc = facts.get("ttc")
            threshold = 4.0 if facts.get("danger") == "caution" else 5.0
            if (best.get("id") not in {None, "coast"} and gain >= threshold and
                    (ttc is not None and ttc <= 24) and self.frame >= self.escape_next):
                if self._start_rescue("forecast", urgent=False):
                    action = self._rescue_action()
                    if action is not None:
                        self.last_action = action
                        self.last_heading = facts.get("heading")
                        return action

        anchor = self._anchor_action()
        if anchor is not None:
            self.last_action = anchor
            self.last_heading = facts.get("heading")
            return anchor

        # Recovery only suppresses an old turn/shot.  It never suppresses the
        # close-contact branch above.
        if self.frame < self.recovery_until:
            action = self.noop
        else:
            fire = self._fire_ready()
            danger = facts.get("danger")
            turn = self._aim_turn()
            if facts.get("near_unknown") and danger in {"caution", "critical"}:
                turn = 0
            action = self._compose(turn=turn, fire=fire)
        self.last_action = action
        self.last_heading = facts.get("heading")
        if (action != self.noop or facts.get("danger") in {"caution", "critical"} or
                self.frame - self.last_log >= 600):
            self._log("control", frame=self.frame, action=action,
                      danger=facts.get("danger"), ttc=facts.get("ttc"),
                      near=bool(facts.get("near_alarm")), speed=round(_num(facts.get("speed"), 0.0) or 0.0, 3),
                      score=self.score, lives=self.lives, rescue=bool(self.rescue))
            self.last_log = self.frame
        return action

    # ---------- Jev state and questions ----------

    def jev_state(self):
        facts = self.facts
        target = facts.get("target")
        target_view = None
        if target is not None:
            target_view = {
                "id": target.get("id"), "size": target.get("size"),
                "bearing": self._relative_bearing(target.get("dx", 0.0), target.get("dy", 0.0)),
                "distance": _bucket(target.get("distance"), (28, 76, 125), ("unknown", "near", "middle", "far")),
                "motion": "known" if target.get("known") else "uncertain",
            }
        threat_view = []
        coast = facts.get("coast") or {}
        for rock, first, clearance in coast.get("outcomes", [])[:7]:
            threat_view.append({
                "id": rock.get("id"), "size": rock.get("size"),
                "bearing": self._relative_bearing(rock.get("dx", 0.0), rock.get("dy", 0.0)),
                "impact": _bucket(first, (6, 19, 50), ("none", "immediate", "soon", "later")),
                "clearance": _bucket(clearance, (0, 7, 17), ("unknown", "overlap", "tight", "open")),
                "motion": "known" if rock.get("known") else "uncertain",
                "ghost": bool(rock.get("ghost")),
            })
        paths = []
        for p in facts.get("paths", []):
            if p.get("id") == "coast":
                continue
            paths.append({
                "id": p.get("id"),
                "direction": "none" if p.get("code") is None else str(p.get("code")),
                "impact": _bucket(p.get("first"), (6, 19, 50), ("none", "immediate", "soon", "later")),
                "clearance": _bucket(p.get("minimum"), (0, 7, 17), ("unknown", "overlap", "tight", "open")),
            })
        return {
            "rules": {
                "mode": 0, "difficulty": 0,
                "fire": "press then release; holding is not rapid fire",
                "noop": "coasts; it is not a brake",
                "down": "mode-0 hyperspace; one bounded last-resort pulse",
                "orientation": "LEFT increments and RIGHT decrements orientation code",
                "enemy_coverage": "unknown; null is not absence",
            },
            "snapshot": {
                "frame": int(self.frame), "lives": int(self.lives),
                "score": int(max(0, self.score)),
                "player": "present" if facts.get("present") else "absent",
                "position": "unknown" if facts.get("x") is None else _bucket(facts.get("x"), (52, 108), ("unknown", "left", "middle", "right")),
                "motion": "unknown" if facts.get("speed") is None else ("still" if facts.get("speed") < 0.15 else "moving"),
                "danger": facts.get("danger", "unknown"),
                "impact": _bucket(facts.get("ttc"), (6, 19, 50), ("none", "immediate", "soon", "later")),
                "near_corridor": bool(facts.get("near_alarm")),
                "asteroids": min(15, len(facts.get("rocks", []))),
                "missiles": min(2, int(facts.get("missiles", 0))),
                "uncertain": bool(facts.get("uncertain")),
            },
            "threats": threat_view,
            "target": target_view,
            "paths": paths[:10],
            "controller": {
                "phase": "rescue" if self.rescue is not None else ("recovery" if self.frame < self.recovery_until else "turret"),
                "rescue_path": self.rescue.get("path") if isinstance(self.rescue, dict) else "none",
                "rescue_pulses": _int(self.rescue.get("remaining"), 0) if isinstance(self.rescue, dict) else 0,
                "defenses_this_life": int(self.defense_count),
                "anchor_pulses": int(self.anchor_count),
                "frame_local": True,
            },
            "plan": dict(self.plan),
        }

    def jev_questions(self):
        modes_text = {
            "rescue": "Treat a close crossing as the priority; code chooses and revalidates a named short escape pulse.",
            "hunt": "Aim at the supplied threat-first target and fire in short press/release pulses; do not create thrust.",
            "sweep": "Use a cautious left/right aiming sweep in a clear field; do not create thrust.",
            "hold": "Keep the settled stationary turret posture; this never overrides a local contact rescue.",
        }
        modes = ["rescue", "hunt", "sweep", "hold"]
        if (self.frame // 36) % 2:
            modes.reverse()
        opts = {"none": "Keep the code-ranked target; do not invent an id."}
        for rock in self.facts.get("rocks", [])[:7]:
            opts[rock.get("id")] = "Prefer this currently observed rock for aiming only; the code rechecks its slot and geometry."
        order = list(opts)
        if (self.frame // 48) % 2:
            order.reverse()
        opts = {key: opts[key] for key in order}
        self.last_jev_targets = list(opts)
        return {
            "mode_a": Choice(
                "Choose a semantic operating regime for this exact snapshot. The code owns arithmetic, collision gates, FIRE cadence, and all actions; do not provide a sequence.",
                {key: modes_text[key] for key in modes},
            ),
            "mode_b": Choice(
                "Independently choose the safest useful regime from snapshot.danger, snapshot.near_corridor, threats, and paths. Treat those fields as computed facts.",
                {key: modes_text[key] for key in reversed(modes)},
            ),
            "target_a": Choice(
                "Choose an observed target id or none. Prefer an imminent collision threat over a distant valuable rock, and never invent an id.",
                opts,
            ),
            "target_b": Choice(
                "Independently choose the useful observed aiming target. A missing or stale id will be rejected by code.",
                {key: opts[key] for key in reversed(order)},
            ),
            "defense": Noul(
                "Should the local controller consider its one mode-0 hyperspace pulse? Say true only for a close crossing that remains after a short rescue path; otherwise false.",
                {"true": "A bounded last-resort defense is warranted by a close unresolved contact.", "false": "Do not invoke defense; let the local rescue or turret rule act."},
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
        valid_modes = {"rescue", "hunt", "sweep", "hold"}
        if first == second and first in valid_modes:
            confidence = min(_prob(answers.get("mode_a"), first), _prob(answers.get("mode_b"), second))
            if confidence >= 0.30:
                self.mode_hint = first
                self.mode_until = self.frame + 54
        first_target = _answer_value(answers.get("target_a"))
        second_target = _answer_value(answers.get("target_b"))
        if first_target == second_target and isinstance(first_target, str) and first_target in set(origin_targets):
            confidence = min(_prob(answers.get("target_a"), first_target), _prob(answers.get("target_b"), second_target))
            if confidence >= 0.28:
                if first_target == "none":
                    self.target_hint = None
                else:
                    current = {r.get("id") for r in self.facts.get("rocks", [])}
                    if first_target in current:
                        self.target_hint = first_target
                        self.target_until = self.frame + 54
        self.defense_hint = _yes(answers.get("defense")) >= 0.80
        self._log("jev_advice", frame=self.frame, judged_frame=origin_frame,
                  mode=first if first == second else "disagree",
                  target=first_target if first_target == second_target else "disagree",
                  defense=round(_yes(answers.get("defense")), 2))

    # ---------- Astra aggregate review ----------

    def astra_summary(self, reason):
        facts = self.facts
        return {
            "reason": reason, "frame": self.frame, "score": int(max(0, self.score)),
            "lives": self.lives, "losses": self.losses,
            "danger": facts.get("danger"), "ttc": facts.get("ttc"),
            "near_corridor": bool(facts.get("near_alarm")),
            "anchor_pulses": self.anchor_count, "rescue_starts": self.escape_count,
            "rescue_failures": self.rescue_failures, "defenses": self.defense_count,
            "last_loss_context": self.last_loss_context, "plan": dict(self.plan),
        }

    def apply_astra(self, text):
        data = _json_object(text)
        if not isinstance(data, dict):
            return False
        allowed = {
            "aim": {"threat_first", "value_first", "sweep"},
            "mobility": {"contact_rescue", "conservative"},
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
        origin_frame, origin_epoch = pilot.frame, pilot.life_epoch
        origin_targets = list(pilot.last_jev_targets)

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
        if _int(_dict(remaining).get("system2_calls"), 0) <= 0:
            return None
        summary = pilot.astra_summary(reason)
        prompt = (
            "You are the slow strategic reviewer for a real-time Atari 2600 "
            "Asteroids mode-0/difficulty-0 controller. Code owns the native "
            "clock, observed RAM geometry, source-slot tracking, wrap-aware "
            "occupancy, short contact-rescue pulses, FIRE press/release, life "
            "bookkeeping, and the one bounded hyperspace gate. Jev only labels "
            "the current state and selects an observed target. Review the aggregate "
            "evidence and return one small JSON object only. Never return actions, "
            "directions, sequences, or timestamps. Allowed values are: "
            "aim='threat_first'|'value_first'|'sweep', "
            "mobility='contact_rescue'|'conservative', fire='normal'|'spaced', "
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
        pilot = ContactPilot(game, ctx)
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
                    astra_future = self._start_astra(pilot, "post-launch contact review")
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
                    # Timeout, cancellation, provider garbage, and malformed
                    # stand-in answers are all advisory misses.
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
