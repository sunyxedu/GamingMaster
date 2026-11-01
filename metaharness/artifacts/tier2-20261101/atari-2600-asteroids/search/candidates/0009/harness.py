"""Two-stage Asteroids controller: predictive lane fire with cross-axis escape.

The local controller owns the native clock.  It uses the high-scoring parent
pattern (short FIRE presses, target-aware rotation, and rare horizontal lane
relocation), but invokes a second, cardinal-sector escape calculation only when
all useful horizontal lanes are poor or the measured drift is the real danger.
System-1 can select among code-generated postures; System-2 can revise only
bounded policy knobs.  Neither model owns a joystick level or a native frame.
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
CARDINALS = (0, 4, 8, 12)  # documented cardinal codes: up, left, down, right
LANES = (55.0, 70.0, 85.0, 100.0, 115.0, 125.0)


def _number(value, default=None):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    return value if math.isfinite(value) else default


def _integer(value, default=0):
    value = _number(value)
    return int(value) if value is not None else default


def _dict(value):
    return value if isinstance(value, dict) else {}


def _list(value):
    return value if isinstance(value, list) else []


def _pair(value):
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        return None
    x = _number(value[0])
    y = _number(value[1])
    return (x, y) if x is not None and y is not None else None


def _wrap(value, period):
    value = _number(value, 0.0) or 0.0
    return (value + period / 2.0) % period - period / 2.0


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


def _answer_prob(answer, key):
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
    value = _number(value)
    if value is not None:
        return max(0.0, min(1.0, value))
    return _answer_prob(answer, "true")


def _heading_vector(code):
    """The public cardinal calibration, extended only for internal ranking."""
    angle = (int(code) % 16) * math.pi / 8.0
    return -math.sin(angle), -math.cos(angle)


def _heading_for_vector(dx, dy):
    if abs(dx) + abs(dy) < 1e-8:
        return 0
    return int(round(math.atan2(-dx, -dy) / (math.pi / 8.0))) % 16


def _short_turn(current, desired):
    if current is None or desired is None:
        return 0
    delta = (int(desired) - int(current)) % 16
    return delta - 16 if delta > 8 else delta


def _bucket(value, cuts, names):
    if value is None:
        return names[0]
    for cut, name in zip(cuts, names[1:]):
        if value < cut:
            return name
    return names[-1]


class EscapeArbiter:
    """Frame-local predictor and bounded action bridge.

    The predictor deliberately has two resolutions.  Horizontal lanes are cheap
    and conservative, so they remain the default.  Cardinal escape directions
    are evaluated with thrust/inertia only after the lane alternatives fail a
    margin test.  This prevents the broad proactive-thrust behavior from 0008
    while retaining its missing cross-axis option.
    """

    FIRE_GAP = 5
    SPACED_FIRE_GAP = 7
    NORMAL_PULSE = 2
    URGENT_PULSE = 3
    NORMAL_MOVE_COOLDOWN = 105
    URGENT_MOVE_COOLDOWN = 34
    RESPAWN_GRACE = 46
    THRUST_ACCEL = 0.035

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
            self.noop = "NOOP" if "NOOP" in self.names else (next(iter(self.names), "NOOP"))

        self.frame = 0
        self.score = 0.0
        self.lives = 0
        self.previous_lives = None
        self.life_epoch = 0
        self.life_started = 0
        self.obs = {}
        self.facts = {}

        self.policy = {
            "movement": "hybrid",       # hybrid or turret
            "escape": "conservative",   # conservative or early
            "fire": "normal",           # normal or spaced
            "defense": "hyperspace_last",  # hyperspace_last or never
        }
        self.warning_streak = 0
        self.last_danger = "unknown"
        self.last_move_frame = -100000
        self.move_cooldown_until = 0
        self.move_plan = None
        self.last_discontinuity = False

        self.fire_ready_frame = 0
        self.fire_release = 0
        self.sweep_sign = 1
        self.sweep_until = 0
        self.defense_used = False
        self.last_defense_frame = -100000

        self.tactical = "aim"
        self.tactical_until = -1
        self.model_escape = None
        self.model_escape_until = -1
        self.model_defense_hint = False
        self.judged_frame = -100000
        self.judged_epoch = -1
        self.last_action = self.noop
        self.last_log = -100000
        self.current_candidates = []

    # ---------- geometry ----------

    @staticmethod
    def _player(obs):
        return _dict(_dict(obs).get("player"))

    @staticmethod
    def _asteroids(obs):
        value = _dict(obs).get("asteroids")
        return value if isinstance(value, list) else []

    @staticmethod
    def _clearance(dx, dy, half_x, half_y):
        # Negative means overlapping approximate boxes; 0 is the conservative
        # contact boundary.  This is safer than using a circular distance for
        # the tall Atari reference boxes.
        return max(abs(dx) - half_x, abs(dy) - half_y)

    def _project(self, rock, base_dx=None, base_dy=None, heading=None,
                 thrust_frames=0, horizon=72):
        """Return min clearance and first contact for one candidate path."""
        dx0 = rock["dx"] if base_dx is None else base_dx
        dy0 = rock["dy"] if base_dy is None else base_dy
        rvx, rvy = rock["rvx"], rock["rvy"]
        hx, hy = rock["half_x"], rock["half_y"]
        if heading is None or thrust_frames <= 0:
            ax = ay = 0.0
        else:
            ax, ay = _heading_vector(heading)
        best = float("inf")
        best_t = None
        first_contact = None
        # Six-frame samples are paired with a speed-scaled clearance margin so
        # a crossing between samples is not treated as a clean miss.
        times = list(range(0, horizon + 1, 6))
        if times[-1] != horizon:
            times.append(horizon)
        speed_bound = max(abs(rvx), abs(rvy))
        if heading is not None and thrust_frames > 0:
            speed_bound += self.THRUST_ACCEL * thrust_frames
        sampling_margin = 3.0 * speed_bound
        for t in times:
            if heading is None or thrust_frames <= 0:
                player_dx = player_dy = 0.0
            else:
                active = min(float(t), float(thrust_frames))
                after = max(0.0, float(t) - active)
                player_dx = self.THRUST_ACCEL * ax * (0.5 * active * active + active * after)
                player_dy = self.THRUST_ACCEL * ay * (0.5 * active * active + active * after)
            rx = _wrap(dx0 + rvx * t - player_dx, WIDTH)
            ry = _wrap(dy0 + rvy * t - player_dy, HEIGHT)
            clear = self._clearance(rx, ry, hx, hy)
            # Unknown finite-difference velocity is not zero motion.  Treat its
            # forecast as an uncertainty band, without fabricating a direction.
            adjusted = clear - sampling_margin - (5.0 if not rock["velocity_known"] else 0.0)
            if adjusted < best:
                best = adjusted
                best_t = t
            if first_contact is None and adjusted <= 0.0:
                first_contact = max(0, t - 3)
        return best, best_t, first_contact

    def _risk_cost(self, minimum, contact, unknown=False):
        cost = max(0.0, 16.0 - minimum) * 0.55
        if contact is not None:
            cost += 17.0 + max(0.0, 34.0 - contact) * 1.45
        if unknown:
            cost *= 1.25
        return cost

    def _derive(self, obs):
        player = self._player(obs)
        px0 = _number(player.get("x"))
        py0 = _number(player.get("y"))
        pw = _number(player.get("width"), 6.0) or 6.0
        ph = _number(player.get("height"), 10.0) or 10.0
        present = bool(player.get("present")) and px0 is not None and py0 is not None
        pv = _pair(player.get("velocity_px_per_frame"))
        pvx, pvy = pv if pv is not None else (0.0, 0.0)
        player_velocity_known = pv is not None and not bool(player.get("motion_discontinuity"))
        px = (px0 + pw / 2.0) if present else WIDTH / 2.0
        py = (py0 + ph / 2.0) if present else HEIGHT / 2.0
        heading = _integer(player.get("orientation_code"), 0) % 16 if present else None

        rocks = []
        if present:
            for index, raw0 in enumerate(self._asteroids(obs)):
                raw = _dict(raw0)
                rx0 = _number(raw.get("x"))
                ry0 = _number(raw.get("y"))
                if rx0 is None or ry0 is None:
                    continue
                rw = _number(raw.get("width"), 8.0) or 8.0
                rh = _number(raw.get("height"), 15.0) or 15.0
                av = _pair(raw.get("velocity_px_per_frame"))
                motion_break = bool(raw.get("motion_discontinuity"))
                known = av is not None and player_velocity_known and not motion_break
                avx, avy = av if av is not None else (0.0, 0.0)
                # Relative motion is the only safe use of the decoded velocity.
                # A wrap or split is marked unknown and receives a larger box.
                rvx = avx - pvx if known else 0.0
                rvy = avy - pvy if known else 0.0
                rcx = rx0 + rw / 2.0
                rcy = ry0 + rh / 2.0
                size = str(raw.get("size") or "unknown").lower()
                value = {"large": 1, "medium": 2, "small": 3}.get(size, 1)
                half_x = rw / 2.0 + pw / 2.0 + 3.0
                half_y = rh / 2.0 + ph / 2.0 + 3.0
                rock = {
                    "slot": _integer(raw.get("source_slot"), index),
                    "size": size,
                    "value": value,
                    "x": rcx,
                    "y": rcy,
                    "dx": _wrap(rcx - px, WIDTH),
                    "dy": _wrap(rcy - py, HEIGHT),
                    "rvx": rvx,
                    "rvy": rvy,
                    "velocity_known": bool(known),
                    "half_x": half_x + (3.0 if not known else 0.0),
                    "half_y": half_y + (3.0 if not known else 0.0),
                    "distance": math.hypot(_wrap(rcx - px, WIDTH), _wrap(rcy - py, HEIGHT)),
                }
                rock["current_clear"], rock["closest_t"], rock["contact_t"] = self._project(rock)
                rocks.append(rock)

        current_min = min((r["current_clear"] for r in rocks), default=90.0)
        contact_rocks = [r for r in rocks if r["contact_t"] is not None]
        nearest = min(
            rocks,
            key=lambda r: (
                r["contact_t"] if r["contact_t"] is not None else 999,
                r["current_clear"],
                -r["value"],
            ),
            default=None,
        )
        ttc = nearest["contact_t"] if nearest is not None else None
        if not present:
            danger = "unknown"
        elif ttc is not None and ttc <= 5:
            danger = "emergency"
        elif (ttc is not None and ttc <= 22) or current_min <= 1.0:
            danger = "caution"
        elif current_min <= 17.0 or (ttc is not None and ttc <= 42):
            danger = "watch"
        else:
            danger = "clear"

        # Evaluate horizontal relocation without pretending that a lane change
        # teleports the ship.  This is only a conservative ranking; actual
        # movement is two thrust frames at most and is rechecked each frame.
        lanes = []
        for lane_x in LANES:
            lane_dx_shift = _wrap(lane_x - px, WIDTH)
            total = 0.0
            min_clear = 90.0
            first = None
            for rock in rocks:
                base_dx = _wrap(rock["dx"] - lane_dx_shift, WIDTH)
                value, _, contact = self._project(rock, base_dx=base_dx, base_dy=rock["dy"], horizon=64)
                min_clear = min(min_clear, value)
                total += self._risk_cost(value, contact, not rock["velocity_known"])
                if contact is not None and (first is None or contact < first):
                    first = contact
            # Keep centrality as a soft tie-breaker, not a hard wall.
            edge = min(lane_x, WIDTH - lane_x)
            total += max(0.0, 18.0 - edge) * 0.18
            lanes.append({"x": lane_x, "cost": total, "min_clear": min_clear, "first": first})
        current_lane_cost = 0.0
        for rock in rocks:
            current_lane_cost += self._risk_cost(rock["current_clear"], rock["contact_t"], not rock["velocity_known"])

        # Cardinal sector forecast: this is the cross-axis escape fallback, not a
        # patrol.  Use a short thrust pulse and score both near and later contact.
        sectors = []
        for code in CARDINALS:
            total = 0.0
            short_min = 90.0
            long_min = 90.0
            first = None
            for rock in rocks:
                short, _, short_contact = self._project(rock, heading=code, thrust_frames=2, horizon=18)
                long, _, long_contact = self._project(rock, heading=code, thrust_frames=2, horizon=58)
                short_min = min(short_min, short)
                long_min = min(long_min, long)
                contact = short_contact if short_contact is not None else long_contact
                total += self._risk_cost(long, contact, not rock["velocity_known"])
                if contact is not None and (first is None or contact < first):
                    first = contact
            # A safe sector with substantial turn distance is still useful for a
            # caution window, but not as a claim that the turn is instantaneous.
            turn_distance = abs(_short_turn(heading, code)) if heading is not None else 8
            total += 0.22 * turn_distance
            score = long_min * 1.5 + short_min * 0.8 - total * 0.08
            sectors.append({
                "code": code,
                "cost": total,
                "score": score,
                "short_min": short_min,
                "min_clear": long_min,
                "first": first,
                "turn": turn_distance,
            })

        best_lane = min(lanes, key=lambda item: (item["cost"], abs(item["x"] - px)), default=None)
        best_sector = max(sectors, key=lambda item: (item["score"], -item["turn"]), default=None)
        current_sector_score = None
        if heading is not None:
            # No-thrust current heading is the baseline for deciding whether a
            # cross-axis pulse is genuinely better, not a forced movement choice.
            current_sector_score = 0.0
            for rock in rocks:
                current_sector_score += rock["current_clear"]

        horizontal_gain = current_lane_cost - (best_lane["cost"] if best_lane else current_lane_cost)
        cross_gain = (best_sector["score"] - current_sector_score) if best_sector is not None and current_sector_score is not None else 0.0
        all_lanes_hot = bool(lanes) and (best_lane["cost"] >= 9.0)
        # Compare like with like here: min clearance is the useful common
        # quantity.  The diagnostic cross_gain is intentionally retained, but
        # its aggregate score is not mixed into this hard gate.
        cross_axis_needed = bool(best_sector) and (
            (all_lanes_hot and best_sector["min_clear"] > current_min + 3.0)
            or (current_min <= 2.0 and best_sector["short_min"] > current_min + 3.0)
        )

        missiles = sum(1 for item in _list(obs.get("player_missiles")) if isinstance(item, dict))
        return {
            "frame": self.frame,
            "present": present,
            "x": px if present else None,
            "y": py if present else None,
            "vx": pvx if present else None,
            "vy": pvy if present else None,
            "speed": math.hypot(pvx, pvy) if present else None,
            "velocity_known": player_velocity_known,
            "heading": heading,
            "rocks": rocks,
            "rock_count": len(rocks),
            "small_count": sum(r["size"] == "small" for r in rocks),
            "medium_count": sum(r["size"] == "medium" for r in rocks),
            "large_count": sum(r["size"] == "large" for r in rocks),
            "missiles": missiles,
            "danger": danger,
            "current_min": current_min,
            "nearest": nearest,
            "ttc": ttc,
            "lanes": lanes,
            "current_lane_cost": current_lane_cost,
            "best_lane": best_lane,
            "sectors": sectors,
            "best_sector": best_sector,
            "horizontal_gain": horizontal_gain,
            "cross_gain": cross_gain,
            "cross_axis_needed": cross_axis_needed,
            "motion_discontinuity": bool(player.get("motion_discontinuity")) if present else True,
        }

    def update(self, obs):
        data = _dict(obs)
        old_lives = self.lives
        self.obs = data
        self.frame = _integer(data.get("frame"), self.frame + 1)
        self.score = _number(data.get("score"), self.score) or 0.0
        new_lives = _integer(data.get("lives"), self.lives)
        changed = self.previous_lives is not None and new_lives != self.previous_lives
        if self.previous_lives is None:
            self.life_started = self.frame
        elif changed:
            self.life_epoch += 1
            self.life_started = self.frame
            self.move_plan = None
            self.move_cooldown_until = self.frame + self.RESPAWN_GRACE
            self.defense_used = False
            self.last_defense_frame = -100000
            self.model_escape = None
            self.model_escape_until = -1
            self.model_defense_hint = False
            self.tactical = "aim"
            self.tactical_until = self.frame + 40
            self.fire_release = max(self.fire_release, 3)
            self._log("life_change", frame=self.frame, old_lives=old_lives, lives=new_lives, score=self.score)
        self.lives = new_lives
        self.previous_lives = new_lives
        self.facts = self._derive(data)
        self.last_discontinuity = bool(self.facts.get("motion_discontinuity"))
        if self.last_discontinuity:
            self.move_plan = None
            self.model_escape = None
            self.model_escape_until = -1
            self.move_cooldown_until = max(self.move_cooldown_until, self.frame + 20)

        danger = self.facts.get("danger", "unknown")
        if danger in {"watch", "caution", "emergency"}:
            self.warning_streak += 1
        else:
            self.warning_streak = max(0, self.warning_streak - 2)
        self.last_danger = danger

    # ---------- derived escape decisions ----------

    def _escape_choice(self, urgent=False):
        f = self.facts
        lanes = f.get("lanes", [])
        best_lane = f.get("best_lane")
        best_sector = f.get("best_sector")
        if not lanes and not best_sector:
            return None
        horizontal_ok = bool(best_lane) and (
            f.get("horizontal_gain", 0.0) > (2.8 if urgent else 4.0)
            or (f.get("current_lane_cost", 0.0) > 12.0 and best_lane.get("cost", 99.0) < 10.0)
        )
        cross_ok = bool(best_sector) and bool(f.get("cross_axis_needed"))
        if urgent:
            # If a horizontal lane is materially safer in the short horizon,
            # keep the proven parent behavior.  Otherwise use the cross-axis
            # forecast, including up/down, rather than repeatedly turning in
            # place at the center.
            if horizontal_ok and best_lane.get("cost", 99.0) <= (best_sector.get("cost", 99.0) + 5.0 if best_sector else 999.0):
                return {"kind": "horizontal", "desired": self._horizontal_heading(best_lane["x"]), "label": "urgent safer horizontal lane"}
            if best_sector is not None:
                return {"kind": "sector", "desired": best_sector["code"], "label": "urgent cross-axis sector"}
            if best_lane is not None:
                return {"kind": "horizontal", "desired": self._horizontal_heading(best_lane["x"]), "label": "urgent fallback lane"}
        if cross_ok and (not horizontal_ok or best_lane.get("cost", 99.0) >= 9.0):
            return {"kind": "sector", "desired": best_sector["code"], "label": "cross-axis sector because lanes are hot"}
        if horizontal_ok:
            return {"kind": "horizontal", "desired": self._horizontal_heading(best_lane["x"]), "label": "safer horizontal lane"}
        return None

    def _horizontal_heading(self, lane_x):
        x = self.facts.get("x")
        if x is None:
            return 4
        return 4 if _wrap(lane_x - x, WIDTH) < 0 else 12

    def _movement_due(self):
        f = self.facts
        if not f.get("present") or self.policy.get("movement") == "turret":
            return False
        if self.frame < self.move_cooldown_until or self.frame - self.life_started < self.RESPAWN_GRACE:
            return False
        danger = f.get("danger")
        if danger == "emergency":
            return True
        if danger == "caution":
            return True
        # A watch window has to persist.  This is the only proactive case; it
        # cannot create the repeated 0008 patrol in an otherwise clear field.
        if danger == "watch" and self.policy.get("escape") == "early" and self.warning_streak >= 10:
            return True
        if self.model_escape is not None and self.frame <= self.model_escape_until:
            return bool(f.get("horizontal_gain", 0.0) > 4.0 or f.get("cross_axis_needed"))
        return False

    def _start_move(self, choice, urgent=False):
        if not isinstance(choice, dict) or self.facts.get("heading") is None:
            return False
        desired = _integer(choice.get("desired"), self.facts.get("heading")) % 16
        self.move_plan = {
            "kind": choice.get("kind", "horizontal"),
            "desired": desired,
            "pulses": self.URGENT_PULSE if urgent else self.NORMAL_PULSE,
            "started": self.frame,
            "expires": self.frame + (15 if urgent else 22),
            "urgent": bool(urgent),
        }
        self.last_move_frame = self.frame
        self.move_cooldown_until = self.frame + (self.URGENT_MOVE_COOLDOWN if urgent else self.NORMAL_MOVE_COOLDOWN)
        self._log("escape_start", frame=self.frame, kind=choice.get("kind"), desired=desired, urgent=urgent)
        return True

    # ---------- action construction ----------

    def legal(self, action):
        return action if action in self.names else self.noop

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

    def _fire_ready(self):
        if self.frame < self.fire_ready_frame or self.fire_release > 0:
            return False
        if self.facts.get("missiles", 0) >= 2:
            return False
        return True

    def _take_fire(self):
        gap = self.SPACED_FIRE_GAP if self.policy.get("fire") == "spaced" else self.FIRE_GAP
        self.fire_ready_frame = self.frame + gap
        self.fire_release = 1
        return True

    def _aim_target(self):
        rocks = self.facts.get("rocks", [])
        if not rocks:
            return None
        def key(rock):
            threat = 7.0 if rock.get("contact_t") is not None and rock["contact_t"] <= 55 else 0.0
            # Small rocks are valuable, but safety/threat comes first.
            return (-threat - 2.7 * rock.get("value", 1) + rock.get("distance", 100.0) / 150.0)
        return min(rocks, key=key)

    def _aim_turn(self):
        heading = self.facts.get("heading")
        if heading is None:
            return 0
        if self.frame > self.tactical_until:
            self.tactical = "aim"
            self.tactical_until = self.frame + 44
        if self.tactical == "sweep_left":
            return 1
        if self.tactical == "sweep_right":
            return -1
        if self.tactical == "hold":
            return 0
        target = self._aim_target()
        if target is not None:
            lead_x = target["dx"] + target["rvx"] * 7.0
            lead_y = target["dy"] + target["rvy"] * 7.0
            desired = _heading_for_vector(lead_x, lead_y)
            delta = _short_turn(heading, desired)
            if abs(delta) > 0:
                return 1 if delta > 0 else -1
        if self.frame >= self.sweep_until:
            self.sweep_sign *= -1
            self.sweep_until = self.frame + 48
        return self.sweep_sign

    def _move_action(self):
        plan = self.move_plan
        if not isinstance(plan, dict):
            return None
        if self.frame > _integer(plan.get("expires"), self.frame):
            self.move_plan = None
            return None
        if not self.facts.get("present") or self.facts.get("heading") is None:
            self.move_plan = None
            return self.noop
        desired = _integer(plan.get("desired"), self.facts["heading"]) % 16
        delta = _short_turn(self.facts["heading"], desired)
        urgent = bool(plan.get("urgent"))
        if delta != 0:
            # A critical window may need a single turn+thrust sample.  Normal
            # movement rotates first, so a delayed answer cannot inject thrust
            # in an arbitrary direction.
            if urgent and abs(delta) > 2 and self.frame == _integer(plan.get("started"), self.frame):
                return self._compose(1 if delta > 0 else -1, thrust=True, fire=False)
            fire = False
            if not urgent and self._fire_ready():
                fire = self._take_fire()
            return self._compose(1 if delta > 0 else -1, thrust=False, fire=fire)
        pulses = _integer(plan.get("pulses"), 0)
        if pulses <= 0:
            self.move_plan = None
            return None
        plan["pulses"] = pulses - 1
        fire = False
        if not urgent and self._fire_ready():
            fire = self._take_fire()
        action = self._compose(0, thrust=True, fire=fire)
        if plan["pulses"] <= 0:
            self.move_plan = None
        return action

    def _current_heading_escape(self):
        """Whether one immediate UP pulse beats coasting at a critical contact."""
        heading = self.facts.get("heading")
        if heading is None:
            return False
        baseline = self.facts.get("current_min", -999.0)
        best = 90.0
        for rock in self.facts.get("rocks", []):
            value, _, _ = self._project(rock, heading=heading, thrust_frames=2, horizon=10)
            best = min(best, value)
        return best > baseline + 2.0

    def _should_defend(self):
        f = self.facts
        if not f.get("present") or self.defense_used or self.policy.get("defense") == "never":
            return False
        if f.get("danger") != "emergency":
            return False
        if self.frame - self.last_defense_frame < 90:
            return False
        ttc = f.get("ttc")
        # Do not spend hyperspace while a current-nose thrust pulse has a
        # positive forecast.  The parent often defended at ttc=2 and still died
        # shortly afterwards; use defense only when the local escape is not
        # feasible or the boxes are already overlapping.
        if ttc is not None and ttc <= 1 and self.facts.get("current_min", 0.0) <= 0.0:
            return not self._current_heading_escape()
        if ttc is not None and ttc <= 3 and self.facts.get("current_min", 0.0) < -2.0:
            return not self._current_heading_escape()
        return False

    def choose(self):
        f = self.facts
        if not f.get("present"):
            self.move_plan = None
            self.fire_release = max(self.fire_release, 2)
            self.last_action = self.noop
            return self.noop
        if self.fire_release > 0:
            self.fire_release -= 1

        # Safety is code-owned.  A one-frame defense is considered only after
        # testing whether an immediately available thrust direction helps.
        if self._should_defend():
            self.defense_used = True
            self.last_defense_frame = self.frame
            self.move_plan = None
            self.model_defense_hint = False
            self.last_action = self.legal("DOWN")
            self._log("defense", frame=self.frame, ttc=f.get("ttc"), clearance=f.get("current_min"))
            return self.last_action

        urgent = f.get("danger") == "emergency"
        if urgent and self.move_plan is None:
            choice = self._escape_choice(True)
            if choice is not None:
                self._start_move(choice, urgent=True)
        elif self.move_plan is None and self._movement_due():
            choice = self._escape_choice(False)
            if choice is not None:
                self._start_move(choice, urgent=False)

        if self.move_plan is not None:
            action = self._move_action()
            if action is not None:
                self.last_action = action
                return action

        turn = self._aim_turn()
        fire = False
        if self._fire_ready() and f.get("danger") not in {"emergency", "caution"}:
            fire = self._take_fire()
        action = self._compose(turn, thrust=False, fire=fire)
        self.last_action = action
        if fire or self.frame - self.last_log >= 600:
            self._log("control", frame=self.frame, action=action, danger=f.get("danger"),
                      clearance=round(_number(f.get("current_min"), 0.0) or 0.0, 2),
                      score=self.score, lives=self.lives)
            self.last_log = self.frame
        return action

    # ---------- Jev interface ----------

    @staticmethod
    def _clearance_bucket(value):
        if value is None:
            return "unknown"
        if value < 0:
            return "overlap"
        if value < 7:
            return "tight"
        if value < 18:
            return "limited"
        return "clear"

    @staticmethod
    def _sector_id(code):
        return "S" + str(int(code) % 16)

    def _candidate_bundle(self):
        candidates = [
            {"id": "aim", "kind": "aim", "label": "track the code-ranked rock and fire only on the local release cadence"},
            {"id": "sweep_left", "kind": "posture", "label": "short counterclockwise aiming sweep with no thrust"},
            {"id": "sweep_right", "kind": "posture", "label": "short clockwise aiming sweep with no thrust"},
            {"id": "hold", "kind": "posture", "label": "keep the current heading; release controls except an allowed fire pulse"},
        ]
        choice = self._escape_choice(self.facts.get("danger") == "emergency")
        if choice is not None:
            cid = "H" + str(int(round(choice.get("desired", 0)))) if choice.get("kind") == "horizontal" else self._sector_id(choice.get("desired", 0))
            candidates.append({"id": cid, "kind": "escape", "desired": int(choice.get("desired", 0)), "label": choice.get("label", "bounded escape") + "; two or three thrust frames maximum"})
        if self.facts.get("danger") == "emergency" and not self.defense_used:
            candidates.append({"id": "defend", "kind": "defend", "label": "one mode-0 hyperspace pulse only for a current critical collision"})
        self.current_candidates = candidates
        return candidates

    def jev_state(self):
        f = self.facts
        target = self._aim_target()
        target_summary = None
        if target is not None:
            target_summary = {
                "size": target.get("size", "unknown"),
                "bearing": "right" if target.get("dx", 0.0) > 10 else "left" if target.get("dx", 0.0) < -10 else "center",
                "vertical": "below" if target.get("dy", 0.0) > 10 else "above" if target.get("dy", 0.0) < -10 else "level",
                "distance": _bucket(target.get("distance"), (30, 80, 125), ("unknown", "near", "middle", "far")),
                "velocity": "known" if target.get("velocity_known") else "uncertain",
            }
        lane_view = [
            {"id": "L" + str(int(item["x"])), "risk": self._clearance_bucket(item.get("min_clear")),
             "relation": "left" if item["x"] < (f.get("x") or 80) - 8 else "right" if item["x"] > (f.get("x") or 80) + 8 else "current"}
            for item in f.get("lanes", [])
        ]
        sector_view = [
            {"id": self._sector_id(item["code"]), "direction": {0: "up", 4: "left", 8: "down", 12: "right"}.get(item["code"], "cardinal"),
             "risk": self._clearance_bucket(item.get("min_clear"))}
            for item in f.get("sectors", [])
        ]
        return {
            "rules": {
                "mode": 0,
                "fire": "press_then_release; no automatic rapid fire",
                "noop": "preserves inertia; not a brake",
                "down": "mode_0_hyperspace",
                "orientation": "LEFT increments and RIGHT decrements code",
                "enemy_coverage": "null means unknown, not absent",
            },
            "snapshot": {
                "frame": int(f.get("frame", 0)),
                "lives": int(self.lives),
                "score": int(max(0, self.score)),
                "player_present": bool(f.get("present")),
                "position": "unknown" if f.get("x") is None else _bucket(f.get("x"), (55, 105), ("unknown", "left", "middle", "right")),
                "speed": "unknown" if f.get("speed") is None else ("still" if f["speed"] < 0.25 else "moving"),
                "danger": f.get("danger", "unknown"),
                "clearance": self._clearance_bucket(f.get("current_min")),
                "impact": _bucket(f.get("ttc"), (3, 12, 30, 60), ("none", "critical", "soon", "watch", "later")),
                "asteroids": int(f.get("rock_count", 0)),
                "small": int(f.get("small_count", 0)),
                "missiles": int(f.get("missiles", 0)),
                "warning_streak": min(30, int(self.warning_streak)),
                "motion_discontinuity": bool(f.get("motion_discontinuity")),
            },
            "target": target_summary,
            "lanes": lane_view,
            "sectors": sector_view,
            "candidates": [{k: v for k, v in item.items() if k in {"id", "kind", "label"}} for item in self._candidate_bundle()],
            "controller": {
                "tactical": self.tactical,
                "active_escape": bool(self.move_plan),
                "defense_used_this_life": bool(self.defense_used),
                "policy": dict(self.policy),
                "frame_local": True,
            },
        }

    def jev_questions(self):
        candidates = self.current_candidates or self._candidate_bundle()
        labels = {item["id"]: item["label"] for item in candidates}
        order = [item["id"] for item in candidates]
        if (self.frame // 32) % 2:
            order.reverse()
        order_b = list(reversed(order))
        risk_order = ["clear", "watch", "caution", "emergency", "unknown"]
        if (self.frame // 48) % 2:
            risk_order.reverse()
        risk_text = {
            "clear": "No derived near-term contact; keep the bounded firing posture.",
            "watch": "A crossing is possible later; do not begin thrust unless the code safety gate agrees.",
            "caution": "A near crossing exists; prefer the named bounded escape over scoring greed.",
            "emergency": "Contact is critical; survival and the code-generated escape/defense gate override aiming.",
            "unknown": "Tracks are insufficient; use conservative release and no invented enemy absence.",
        }
        return {
            "choice_a": Choice(
                "Choose one candidate for this exact snapshot. The code computed geometry, directions and pulse lengths. Never invent an action or hold thrust.",
                {key: labels[key] for key in order},
            ),
            "choice_b": Choice(
                "Independently choose the safest useful candidate. An escape is a bounded pulse and aim/sweep never includes thrust.",
                {key: labels[key] for key in order_b},
            ),
            "risk": Choice(
                "Classify risk using snapshot.danger, snapshot.clearance and snapshot.impact. Do not calculate numbers and do not treat unknown enemy coverage as empty.",
                {key: risk_text[key] for key in risk_order},
            ),
            "defense": Noul(
                "Should one hyperspace pulse be considered now? Say true only if the snapshot is emergency and the named local defense candidate is present; otherwise false.",
                {"true": "A current critical collision justifies one bounded mode-0 defense.", "false": "Do not use hyperspace; retain the local escape or firing fallback."},
            ),
        }

    def accept_jev(self, answers, origin_frame, origin_epoch, origin_candidates):
        if not isinstance(answers, dict):
            return
        if origin_epoch != self.life_epoch or self.frame < origin_frame or self.frame - origin_frame > 46:
            self._log("jev_discard", frame=self.frame, judged_frame=origin_frame, reason="stale_or_life")
            return
        a = answers.get("choice_a")
        b = answers.get("choice_b")
        va, vb = _answer_value(a), _answer_value(b)
        valid = {item.get("id") for item in origin_candidates if isinstance(item, dict)}
        if va != vb or va not in valid:
            return
        selected = next((item for item in origin_candidates if item.get("id") == va), None)
        if not isinstance(selected, dict):
            return
        threshold = 0.42 if selected.get("kind") in {"escape", "defend"} else 0.17
        if min(_answer_prob(a, va), _answer_prob(b, vb)) < threshold:
            return
        kind = selected.get("kind")
        if kind == "posture":
            self.tactical = va
            self.tactical_until = self.frame + 50
        elif kind == "aim":
            self.tactical = "aim"
            self.tactical_until = self.frame + 50
        elif kind == "escape":
            # This is only a hint.  _movement_due and _escape_choice recheck the
            # current geometry, so Jev cannot make a clear field thrust.
            self.model_escape = dict(selected)
            self.model_escape_until = self.frame + 34
        elif kind == "defend":
            self.model_defense_hint = True
        self.judged_frame = origin_frame
        self.judged_epoch = origin_epoch
        self._log("jev_advice", frame=self.frame, judged_frame=origin_frame, choice=va,
                  risk=_answer_value(answers.get("risk")), defense=_yes_probability(answers.get("defense")))

    # ---------- Astra policy ----------

    def astra_summary(self, reason):
        f = self.facts
        nearest = f.get("nearest") or {}
        return {
            "reason": reason,
            "frame": self.frame,
            "score": self.score,
            "lives": self.lives,
            "danger": f.get("danger"),
            "warning_streak": self.warning_streak,
            "clearance": round(_number(f.get("current_min"), 0.0) or 0.0, 2),
            "impact": nearest.get("contact_t"),
            "horizontal_gain": round(_number(f.get("horizontal_gain"), 0.0) or 0.0, 2),
            "cross_gain": round(_number(f.get("cross_gain"), 0.0) or 0.0, 2),
            "active_escape": bool(self.move_plan),
            "policy": dict(self.policy),
        }

    def apply_astra(self, text):
        data = _json_object(text)
        if not isinstance(data, dict):
            return False
        allowed = {
            "movement": {"hybrid", "turret"},
            "escape": {"conservative", "early"},
            "fire": {"normal", "spaced"},
            "defense": {"hyperspace_last", "never"},
        }
        changed = {}
        for key, values in allowed.items():
            value = data.get(key)
            if isinstance(value, str) and value in values:
                changed[key] = value
        if not changed:
            return False
        self.policy.update(changed)
        self._log("astra_policy", frame=self.frame, policy=dict(self.policy))
        return True

    def _log(self, event, **fields):
        try:
            self.ctx.log(event, **fields)
        except Exception:
            pass


class Harness:
    def __init__(self, ctx):
        self.ctx = ctx

    def _start_jev(self, controller):
        if getattr(self.ctx, "system1", True) is None:
            return None
        state = controller.jev_state()
        questions = controller.jev_questions()
        origin_frame = controller.frame
        origin_epoch = controller.life_epoch
        origin_candidates = [dict(item) for item in controller.current_candidates]

        def request():
            return self.ctx.decide(state, questions, timeout_s=2.4)

        try:
            future = self.ctx.spawn(request)
        except (BudgetExceeded, EpisodeOver):
            return None
        except ModelUnavailable:
            raise
        except Exception:
            return None
        return future, origin_frame, origin_epoch, origin_candidates

    def _start_astra(self, controller, reason):
        try:
            remaining = self.ctx.budget.remaining()
        except Exception:
            remaining = {}
        if _integer(_dict(remaining).get("system2_calls"), 0) <= 0:
            return None
        summary = controller.astra_summary(reason)
        prompt = (
            "You are the slow strategic reviewer for a real-time Atari 2600 Asteroids "
            "mode-0/difficulty-0 controller. The local code owns every native frame, "
            "RAM-derived collision forecast, target arithmetic, FIRE release, bounded "
            "cardinal escape pulses, defense legality, and life resets. Do not give an "
            "action, sequence, timestamp, or prose plan. Review the aggregate evidence "
            "below and return one small JSON object only.\n" +
            json.dumps(summary, separators=(",", ":")) +
            "\nAllowed exact fields and values: "
            '{"movement":"hybrid"|"turret",'
            '"escape":"conservative"|"early",'
            '"fire":"normal"|"spaced",'
            '"defense":"hyperspace_last"|"never"}. '
            "Hybrid means lane-first with cross-axis escape only when lanes are unsafe; "
            "never means do not use DOWN. Unknown enemy coverage is not absence."
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
        controller = EscapeArbiter(game, ctx)
        controller.update(_observation(game))
        try:
            ctx.act(controller.choose())
        except (BudgetExceeded, EpisodeOver, ModelUnavailable):
            return
        except Exception:
            try:
                ctx.act(controller.noop)
            except Exception:
                return

        s1_enabled = getattr(ctx, "system1", True) is not None
        jev_future = None
        jev_origin = -100000
        jev_epoch = -1
        jev_candidates = []
        next_jev = controller.frame

        astra_future = None
        astra_started = False
        astra_due = controller.frame + 520

        while not ctx.done and not getattr(game, "done", False):
            if s1_enabled and jev_future is None and controller.frame >= next_jev:
                try:
                    result = self._start_jev(controller)
                    if result is not None:
                        jev_future, jev_origin, jev_epoch, jev_candidates = result
                        next_jev = controller.frame + 30
                    else:
                        next_jev = controller.frame + 42
                except ModelUnavailable:
                    return
                except (BudgetExceeded, EpisodeOver):
                    s1_enabled = False
                except Exception:
                    next_jev = controller.frame + 42

            if not astra_started and astra_future is None and controller.frame >= astra_due:
                try:
                    astra_future = self._start_astra(controller, "first aggregate review")
                    astra_started = True
                except ModelUnavailable:
                    return
                except (BudgetExceeded, EpisodeOver):
                    astra_started = True
                except Exception:
                    astra_started = True

            try:
                ctx.wait_for_tick()
            except (BudgetExceeded, EpisodeOver, ModelUnavailable):
                return
            if ctx.done or getattr(game, "done", False):
                break

            controller.update(_observation(game))

            if jev_future is not None and jev_future.done():
                future = jev_future
                jev_future = None
                try:
                    controller.accept_jev(future.result(), jev_origin, jev_epoch, jev_candidates)
                except (BudgetExceeded, EpisodeOver):
                    s1_enabled = False
                except ModelUnavailable:
                    return
                except Exception:
                    # Arbitrary preflight answers, cancellation, and malformed
                    # provider data are ordinary tactical misses.
                    pass

            if astra_future is not None and astra_future.done():
                future = astra_future
                astra_future = None
                try:
                    result = future.result()
                    controller.apply_astra(getattr(result, "text", result))
                except (BudgetExceeded, EpisodeOver):
                    pass
                except ModelUnavailable:
                    return
                except Exception:
                    pass

            try:
                ctx.act(controller.choose())
            except (BudgetExceeded, EpisodeOver, ModelUnavailable):
                return
            except Exception:
                try:
                    ctx.act(controller.noop)
                except (BudgetExceeded, EpisodeOver, ModelUnavailable):
                    return
                except Exception:
                    return


def build(ctx):
    return Harness(ctx)
