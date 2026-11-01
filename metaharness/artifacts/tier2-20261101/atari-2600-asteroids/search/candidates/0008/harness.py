"""Proactive safe-sector Asteroids controller.

The local controller owns every native frame.  It uses the RAM-derived tracks to
choose a short, bounded lane-change pulse before a collision is imminent, while
keeping the parent candidate's press/release firing and asynchronous Jev bridge.
Astra is a slow policy reviewer and cannot post controls directly.
"""
from __future__ import annotations

import json
import math

from metaharness.core.contract import BudgetExceeded, EpisodeOver, ModelUnavailable
from metaharness.models.jev import Choice, Noul


WIDTH = 160.0
HEIGHT = 210.0
CENTER_X = 80.0
CENTER_Y = 105.0
ACTIONS = (
    "NOOP", "FIRE", "UP", "RIGHT", "LEFT", "DOWN", "UPRIGHT", "UPLEFT",
    "UPFIRE", "RIGHTFIRE", "LEFTFIRE", "DOWNFIRE", "UPRIGHTFIRE",
    "UPLEFTFIRE",
)


def _num(value, default=None):
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _int(value, default=0):
    result = _num(value)
    return int(result) if result is not None else default


def _dict(value):
    return value if isinstance(value, dict) else {}


def _wrap(value, period):
    # Return the shortest signed displacement on a torus.
    while value > period / 2.0:
        value -= period
    while value < -period / 2.0:
        value += period
    return value


def _heading_for_vector(dx, dy):
    if abs(dx) + abs(dy) < 1e-6:
        return 0
    # Public calibration: 0=up, 4=left, 8=down, 12=right.
    return int(round(math.atan2(-dx, -dy) / (math.pi / 8.0))) % 16


def _heading_vector(code):
    angle = (int(code) % 16) * math.pi / 8.0
    return -math.sin(angle), -math.cos(angle)


def _answer_value(answer):
    if isinstance(answer, dict):
        return answer.get("value", answer.get("choice", answer.get("answer")))
    return getattr(answer, "value", None)


def _answer_probs(answer):
    if isinstance(answer, dict):
        value = answer.get("probabilities", {})
    else:
        value = getattr(answer, "probabilities", {})
    return value if isinstance(value, dict) else {}


def _prob(answer, key):
    value = _num(_answer_probs(answer).get(key), 0.0)
    return max(0.0, value or 0.0)


def _yes(answer):
    value = _answer_value(answer)
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, str):
        return 1.0 if value.lower() in {"true", "yes"} else 0.0
    number = _num(value)
    if number is not None:
        return max(0.0, min(1.0, number))
    return max(0.0, min(1.0, _prob(answer, "true")))


def _observation(game):
    try:
        value = game.observe()
    except Exception:
        return {}
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            decoded = json.loads(value)
        except (TypeError, ValueError):
            return {}
        return decoded if isinstance(decoded, dict) else {}
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


class SafeSectorController:
    """Frame-local movement, aiming, firing and one-shot defense."""

    # This is only a conservative planning model, not a claim about exact ALE
    # acceleration.  It is used to rank directions, and is refreshed every frame.
    THRUST_ACCEL = 0.035
    FIRE_GAP = 4
    MOVE_PULSE = 5
    EMERGENCY_PULSE = 7
    MOVE_COOLDOWN = 150
    RESPAWN_GRACE = 55

    def __init__(self, game, ctx):
        self.game = game
        self.ctx = ctx
        spec = getattr(game, "action_spec", None)
        try:
            names = list(getattr(spec, "names", ACTIONS)) if spec is not None else list(ACTIONS)
        except Exception:
            names = list(ACTIONS)
        self.names = set(names or ACTIONS)
        self.noop = getattr(spec, "noop", "NOOP") if spec is not None else "NOOP"
        if self.noop not in self.names:
            self.noop = "NOOP" if "NOOP" in self.names else (next(iter(self.names), "NOOP"))

        self.frame = 0
        self.score = 0.0
        self.lives = 0
        self.previous_lives = None
        self.life_started = 0
        self.features = {}
        self.last_action = self.noop
        self.last_score = 0.0

        self.fire_cooldown = 0
        self.sweep_sign = 1
        self.sweep_until = 0
        self.last_move_frame = -100000
        self.move_cooldown_until = 0
        self.move_plan = None
        self.defense_used_life = False
        self.last_defense_frame = -100000
        self.model_choice = None
        self.model_until = -1
        self.model_answer_frame = -100000
        self.model_lives = None
        self.last_discontinuity = False

        self.plan = {
            "movement": "safe_lane",
            "risk": "cautious",
            "fire": "pulse",
            "defense": "emergency_only",
        }
        self.current_candidates = []

    # ----- observation and geometry -----

    @staticmethod
    def _player(obs):
        return _dict(_dict(obs).get("player"))

    @staticmethod
    def _rock_list(obs):
        value = _dict(obs).get("asteroids")
        return value if isinstance(value, list) else []

    @staticmethod
    def _finite_pair(value):
        if isinstance(value, (list, tuple)) and len(value) >= 2:
            x = _num(value[0])
            y = _num(value[1])
            if x is not None and y is not None:
                return x, y
        return None, None

    def update(self, obs):
        data = _dict(obs)
        self.frame = _int(data.get("frame"), self.frame + 1)
        self.score = _num(data.get("score"), self.score) or 0.0
        new_lives = _int(data.get("lives"), self.lives)
        if self.previous_lives is None:
            self.life_started = self.frame
        elif new_lives != self.previous_lives:
            # A respawn, and especially a final missing-player frame, invalidates
            # every old heading and source-slot selection.
            self.life_started = self.frame
            self.move_plan = None
            self.model_choice = None
            self.model_until = -1
            self.fire_cooldown = max(self.fire_cooldown, 5)
            self.defense_used_life = False
            self.last_defense_frame = -100000
            self.move_cooldown_until = self.frame + self.RESPAWN_GRACE
            self.sweep_until = self.frame + 36
            self._log("life_change", frame=self.frame, lives=new_lives, score=self.score)
        self.lives = new_lives
        self.previous_lives = new_lives
        self.features = self._derive(data)
        self.last_discontinuity = bool(self.features.get("motion_discontinuity", False))
        if self.features.get("present") and self.last_discontinuity:
            # A teleport, wrap discontinuity, or respawn makes both the old
            # trajectory and a delayed maneuver unsafe to reuse.
            self.move_plan = None
            self.model_choice = None
            self.model_until = -1
            self.move_cooldown_until = max(self.move_cooldown_until, self.frame + 30)

    def _derive(self, obs):
        player = self._player(obs)
        present = bool(player.get("present"))
        px = _num(player.get("x"))
        py = _num(player.get("y"))
        pw = _num(player.get("width"), 6.0) or 6.0
        ph = _num(player.get("height"), 10.0) or 10.0
        pvx, pvy = self._finite_pair(player.get("velocity_px_per_frame"))
        if px is None or py is None:
            present = False
        if pvx is None or pvy is None:
            pvx = pvy = 0.0
            velocity_known = False
        else:
            velocity_known = True

        rocks = []
        if present:
            player_cx = px + pw / 2.0
            player_cy = py + ph / 2.0
            for raw0 in self._rock_list(obs):
                raw = _dict(raw0)
                ax = _num(raw.get("x"))
                ay = _num(raw.get("y"))
                if ax is None or ay is None:
                    continue
                aw = _num(raw.get("width"), 8.0) or 8.0
                ah = _num(raw.get("height"), 15.0) or 15.0
                avx, avy = self._finite_pair(raw.get("velocity_px_per_frame"))
                known = avx is not None and avy is not None and velocity_known
                if avx is None or avy is None:
                    avx = avy = 0.0
                dx = _wrap(ax + aw / 2.0 - player_cx, WIDTH)
                dy = _wrap(ay + ah / 2.0 - player_cy, HEIGHT)
                rvx = avx - pvx
                rvy = avy - pvy
                radius = max(aw, ah) / 2.0 + max(pw, ph) / 2.0 + 3.0
                min_clear, min_t = self._swept_clearance(dx, dy, rvx, rvy, radius, 36.0)
                # A simple collision time is deliberately a bucket input for Jev,
                # not something the model is asked to calculate.
                ttc = min_t if min_clear <= 20.0 else None
                size = str(raw.get("size") or "unknown").lower()
                lead_t = min(12.0, max(2.0, math.hypot(dx, dy) / 5.0))
                lead_x = dx + rvx * lead_t
                lead_y = dy + rvy * lead_t
                rocks.append({
                    "slot": _int(raw.get("source_slot"), -1),
                    "size": size,
                    "dx": dx,
                    "dy": dy,
                    "rvx": rvx,
                    "rvy": rvy,
                    "distance": math.hypot(dx, dy),
                    "radius": radius,
                    "min_clearance": min_clear,
                    "ttc": ttc,
                    "velocity_known": bool(known),
                    "aim_code": _heading_for_vector(lead_x, lead_y),
                })

        if rocks:
            emergency = [r for r in rocks if r["min_clearance"] <= 5.0 or (
                r["ttc"] is not None and r["ttc"] <= 9.0 and r["min_clearance"] <= 16.0
            )]
            caution = [r for r in rocks if r["min_clearance"] <= 25.0 or (
                r["ttc"] is not None and r["ttc"] <= 28.0 and r["min_clearance"] <= 35.0
            )]
            if emergency:
                danger = "emergency"
            elif caution:
                danger = "caution"
            else:
                danger = "clear"
            nearest = min(rocks, key=lambda r: (r["min_clearance"], r["distance"]))
            # Safety ordering first, then small-rock value, then distance.
            value = {"small": 3, "medium": 2, "large": 1}
            target = min(
                rocks,
                key=lambda r: (
                    0 if r["min_clearance"] < 0 else 1,
                    r["ttc"] if r["ttc"] is not None else 99,
                    -value.get(r["size"], 0),
                    r["distance"],
                ),
            )
        else:
            danger = "unknown" if present else "clear"
            nearest = None
            target = None

        speed = math.hypot(pvx, pvy) if present else 0.0
        central_dx = _wrap((px if px is not None else CENTER_X) - CENTER_X, WIDTH)
        central_dy = _wrap((py if py is not None else CENTER_Y) - CENTER_Y, HEIGHT)
        return {
            "frame": self.frame,
            "score": round(self.score, 2),
            "lives": self.lives,
            "present": present,
            "x": px,
            "y": py,
            "vx": pvx,
            "vy": pvy,
            "velocity_known": velocity_known,
            "speed": speed,
            "heading": (_int(player.get("orientation_code"), 0) % 16) if present else None,
            "central_dx": central_dx,
            "central_dy": central_dy,
            "rocks": rocks,
            "rock_count": len(rocks),
            "small_count": sum(r["size"] == "small" for r in rocks),
            "medium_count": sum(r["size"] == "medium" for r in rocks),
            "large_count": sum(r["size"] == "large" for r in rocks),
            "danger": danger,
            "nearest": nearest,
            "target": target,
            "motion_discontinuity": bool(player.get("motion_discontinuity")) if present else True,
            "missile_count": len(_dict(obs).get("player_missiles")) if isinstance(_dict(obs).get("player_missiles"), list) else 0,
        }

    @staticmethod
    def _swept_clearance(dx, dy, rvx, rvy, radius, horizon):
        vv = rvx * rvx + rvy * rvy
        if vv > 1e-9:
            closest = max(0.0, min(horizon, -(dx * rvx + dy * rvy) / vv))
        else:
            closest = 0.0
        samples = (0.0, closest, min(horizon, closest + 4.0), horizon)
        best_clear = float("inf")
        best_t = None
        for t in samples:
            rx = _wrap(dx + rvx * t, WIDTH)
            ry = _wrap(dy + rvy * t, HEIGHT)
            clear = math.hypot(rx, ry) - radius
            if clear < best_clear:
                best_clear = clear
                best_t = t
        return best_clear, best_t

    def _candidate_projection_score(self, heading, thrust):
        """Rank a short acceleration direction by future swept clearances."""
        f = self.features
        if not f.get("present"):
            return -1000.0
        hx, hy = _heading_vector(heading)
        accel = self.THRUST_ACCEL if thrust else 0.0
        samples = (0.0, 6.0, 12.0, 20.0, 32.0, 46.0)
        minimum = float("inf")
        total = 0.0
        for rock in f.get("rocks", []):
            local_min = float("inf")
            for t in samples:
                # Relative rock position already subtracts current player velocity.
                rx = _wrap(
                    rock["dx"] + rock["rvx"] * t - 0.5 * accel * hx * t * t,
                    WIDTH,
                )
                ry = _wrap(
                    rock["dy"] + rock["rvy"] * t - 0.5 * accel * hy * t * t,
                    HEIGHT,
                )
                local_min = min(local_min, math.hypot(rx, ry) - rock["radius"])
            minimum = min(minimum, local_min)
            total += local_min
        if not f.get("rocks"):
            minimum = 80.0
            total = 80.0
        # Discourage deliberate edge commitment without treating a wrapped edge
        # as an invalid state.  This is a soft tie-breaker only.
        x = (f.get("x") or CENTER_X) + f.get("vx", 0.0) * 24.0 + 0.5 * accel * hx * 24.0 * 24.0
        y = (f.get("y") or CENTER_Y) + f.get("vy", 0.0) * 24.0 + 0.5 * accel * hy * 24.0 * 24.0
        x = x % WIDTH
        y = y % HEIGHT
        edge = min(x, WIDTH - x, y, HEIGHT - y)
        edge_penalty = max(0.0, 18.0 - edge) * 1.5
        return minimum * 1.8 + total * 0.12 - edge_penalty

    def _safe_heading(self, emergency=False):
        f = self.features
        choices = [0, 2, 4, 6, 8, 10, 12, 14]
        # In an emergency, the direction opposite the most dangerous track gets
        # an explicit tie-breaker, but still competes with all other sectors.
        threat = f.get("nearest")
        preferred = None
        preference_weight = 0.0
        if isinstance(threat, dict):
            preferred = _heading_for_vector(-threat["dx"], -threat["dy"])
            preference_weight = 3.0 if emergency else 0.5
        # With inertia, repeated lane changes can otherwise turn a survival
        # controller into a slow edge-seeking controller.  When speed is already
        # substantial, prefer a short pulse opposite the measured drift unless a
        # critical threat has supplied a stronger away vector.
        if not emergency and f.get("speed", 0.0) > 0.45:
            preferred = _heading_for_vector(-f.get("vx", 0.0), -f.get("vy", 0.0))
            preference_weight = max(preference_weight, 1.5)
        scored = []
        for code in choices:
            score = self._candidate_projection_score(code, True)
            if preferred is not None:
                turn = abs((code - preferred + 8) % 16 - 8)
                score += preference_weight * (8 - turn) / 8.0
            scored.append((score, code))
        scored.sort(reverse=True)
        return scored[0][1] if scored else 0

    # ----- local action construction -----

    def _legal(self, name):
        if name in self.names:
            return name
        return self.noop

    def _turn(self, sign, fire=False):
        if sign > 0:
            return self._legal("LEFTFIRE" if fire else "LEFT")
        return self._legal("RIGHTFIRE" if fire else "RIGHT")

    @staticmethod
    def _delta(current, desired):
        value = (int(desired) - int(current)) % 16
        return value - 16 if value > 8 else value

    def _fire_ready(self):
        return self.fire_cooldown <= 0 and self.plan.get("fire") != "sparse"

    def _mark_fire(self):
        self.fire_cooldown = self.FIRE_GAP

    def _aim_action(self):
        f = self.features
        if self.fire_cooldown > 0:
            self.fire_cooldown -= 1
        target = f.get("target")
        heading = f.get("heading")
        if heading is None:
            return self.noop
        if target is None:
            if self.frame >= self.sweep_until:
                self.sweep_sign *= -1
                self.sweep_until = self.frame + 48
            fire = self._fire_ready()
            if fire:
                self._mark_fire()
            return self._turn(self.sweep_sign, fire)
        delta = self._delta(heading, target.get("aim_code", heading))
        if abs(delta) > 1:
            fire = self._fire_ready()
            if fire:
                self._mark_fire()
            return self._turn(1 if delta > 0 else -1, fire)
        if self._fire_ready():
            self._mark_fire()
            return self._legal("FIRE")
        return self.noop

    def _start_move(self, heading, emergency=False):
        self.move_plan = {
            "heading": int(heading) % 16,
            "thrust_left": self.EMERGENCY_PULSE if emergency else self.MOVE_PULSE,
            "started": self.frame,
            "deadline": self.frame + (18 if emergency else 24),
            "emergency": bool(emergency),
        }
        self.last_move_frame = self.frame
        self.move_cooldown_until = self.frame + (70 if emergency else self.MOVE_COOLDOWN)

    def _move_action(self):
        plan = self.move_plan
        if not isinstance(plan, dict) or plan.get("thrust_left", 0) <= 0:
            self.move_plan = None
            return None
        if self.frame > int(plan.get("deadline", self.frame)):
            # An orientation estimate that never converges must not turn into
            # an accidental long-held LEFT/RIGHT action.
            self.move_plan = None
            return None
        heading = self.features.get("heading")
        if heading is None:
            self.move_plan = None
            return self.noop
        delta = self._delta(heading, plan.get("heading", heading))
        emergency = bool(plan.get("emergency"))
        if abs(delta) > 1:
            # Planned repositioning rotates first, so its pulse does not inject
            # acceleration in an arbitrary direction.  Emergency movement may
            # thrust while turning because a lethal window is already short.
            if emergency:
                plan["thrust_left"] -= 1
                return self._legal("UPLEFT" if delta > 0 else "UPRIGHT")
            # Rotation is safe during a planned lane change; retain the normal
            # press/release firing cadence while waiting for the heading.
            if self.fire_cooldown > 0:
                self.fire_cooldown -= 1
            fire = self._fire_ready()
            if fire:
                self._mark_fire()
            return self._turn(1 if delta > 0 else -1, fire)
        plan["thrust_left"] -= 1
        if self._fire_ready() and not emergency:
            self._mark_fire()
            return self._legal("UPFIRE")
        return self._legal("UP")

    def _critical_defense(self):
        f = self.features
        if not f.get("present") or f.get("danger") != "emergency":
            return False
        if self.plan.get("defense") == "never" or self.defense_used_life:
            return False
        if self.frame - self.last_defense_frame < 180:
            return False
        nearest = f.get("nearest") or {}
        # Use the cartridge defense only for a genuinely short collision window,
        # not for every caution label.  It is one button-level pulse, never held.
        return (
            nearest.get("min_clearance", 999.0) <= 3.0
            or (nearest.get("ttc") is not None and nearest.get("ttc") <= 7.0)
        )

    def _movement_due(self):
        f = self.features
        if not f.get("present") or self.plan.get("movement") == "stationary":
            return False
        if self.frame < self.move_cooldown_until:
            return False
        if self.frame - self.life_started < self.RESPAWN_GRACE:
            return False
        if f.get("danger") == "caution":
            return True
        if f.get("danger") == "emergency":
            return True
        # A field with several tracks eventually makes a fixed central lane a
        # bad stationary policy even when the instantaneous clearance is benign.
        return f.get("rock_count", 0) >= 4 and self.frame - self.last_move_frame >= self.MOVE_COOLDOWN

    def fallback_kind(self):
        f = self.features
        if f.get("danger") == "emergency":
            return "move"
        if self._movement_due():
            return "move"
        return "aim"

    def candidates(self):
        f = self.features
        candidates = []
        if f.get("present"):
            if f.get("danger") in {"emergency", "caution"} or self._movement_due():
                heading = self._safe_heading(f.get("danger") == "emergency")
                candidates.append({
                    "id": "move",
                    "kind": "move",
                    "heading": heading,
                    "label": "bounded thrust toward a code-ranked safe sector; no long hold",
                })
            candidates.extend([
                {"id": "aim", "kind": "aim", "label": "turn toward the code-ranked useful rock and pulse FIRE"},
                {"id": "sweep_left", "kind": "sweep", "sign": 1, "label": "short counterclockwise firing sweep"},
                {"id": "sweep_right", "kind": "sweep", "sign": -1, "label": "short clockwise firing sweep"},
                {"id": "coast", "kind": "coast", "label": "keep the current heading, release controls except a legal fire pulse"},
            ])
            if f.get("danger") == "emergency":
                candidates.append({
                    "id": "defend",
                    "kind": "defend",
                    "label": "one mode-0 hyperspace pulse for a current critical collision only",
                })
        else:
            candidates.append({"id": "coast", "kind": "coast", "label": "release controls while the ship is absent or respawning"})
        self.current_candidates = candidates
        return candidates

    def _execute_choice(self, choice):
        if not isinstance(choice, dict):
            return None
        kind = choice.get("kind")
        if kind == "move":
            if self.move_plan is None:
                self._start_move(choice.get("heading", self._safe_heading()), self.features.get("danger") == "emergency")
            return self._move_action()
        if kind == "sweep":
            self.sweep_sign = 1 if choice.get("sign", 1) > 0 else -1
            if self.fire_cooldown > 0:
                self.fire_cooldown -= 1
            fire = self._fire_ready()
            if fire:
                self._mark_fire()
            return self._turn(self.sweep_sign, fire)
        if kind == "coast":
            if self.fire_cooldown > 0:
                self.fire_cooldown -= 1
            if self._fire_ready() and self.features.get("present"):
                self._mark_fire()
                return self._legal("FIRE")
            return self.noop
        if kind == "aim":
            return self._aim_action()
        if kind == "defend" and self._critical_defense():
            self.defense_used_life = True
            self.last_defense_frame = self.frame
            self.move_plan = None
            self.fire_cooldown = max(self.fire_cooldown, 3)
            return self._legal("DOWN")
        return None

    def choose(self):
        f = self.features
        if not f.get("present"):
            self.move_plan = None
            self.fire_cooldown = max(self.fire_cooldown, 2)
            self.last_action = self.noop
            return self.noop

        # The deterministic critical gate is intentionally stronger than Jev: a
        # late or confused answer can never suppress immediate survival handling.
        if self._critical_defense():
            self.defense_used_life = True
            self.last_defense_frame = self.frame
            self.move_plan = None
            self.fire_cooldown = max(self.fire_cooldown, 3)
            self.last_action = self._legal("DOWN")
            self._log("defense", frame=self.frame, score=self.score)
            return self.last_action

        if f.get("danger") == "emergency":
            if self.move_plan is None:
                self._start_move(self._safe_heading(True), True)
            action = self._move_action()
            self.last_action = action or self.noop
            return self.last_action

        # A planned lane change is allowed to finish, but it is short and is
        # abandoned if the current observation invalidates the player state.
        if self.move_plan is not None:
            action = self._move_action()
            self.last_action = action or self.noop
            return self.last_action

        # Jev may choose a safe candidate, but local due/safety rules retain veto.
        if self._movement_due():
            selected = self.model_choice if self.model_until >= self.frame else None
            if not isinstance(selected, dict) or selected.get("kind") != "move":
                selected = {"kind": "move", "heading": self._safe_heading(False)}
            self._start_move(selected.get("heading", self._safe_heading(False)), False)
            action = self._move_action()
            self.last_action = action or self.noop
            return self.last_action

        selected = self.model_choice if self.model_until >= self.frame else None
        if isinstance(selected, dict) and selected.get("kind") in {"aim", "sweep", "coast"}:
            action = self._execute_choice(selected)
        else:
            action = self._aim_action()
        self.last_action = action or self.noop
        return self.last_action

    # ----- model-facing state -----

    @staticmethod
    def _band_distance(value):
        if value is None:
            return "unknown"
        if value < 35:
            return "near"
        if value < 80:
            return "middle"
        return "far"

    @staticmethod
    def _band_clearance(value):
        if value is None:
            return "unknown"
        if value < 0:
            return "overlapping"
        if value < 10:
            return "tight"
        if value < 28:
            return "limited"
        return "clear"

    def _compact(self, candidates=None):
        f = self.features
        if candidates is None:
            candidates = self.candidates()
        target = f.get("target")
        target_summary = None
        if isinstance(target, dict):
            target_summary = {
                "size": target.get("size", "unknown"),
                "bearing": "right" if target.get("dx", 0) > 8 else "left" if target.get("dx", 0) < -8 else "center",
                "vertical": "below" if target.get("dy", 0) > 8 else "above" if target.get("dy", 0) < -8 else "level",
                "distance": self._band_distance(target.get("distance")),
                "clearance": self._band_clearance(target.get("min_clearance")),
                "velocity": "known" if target.get("velocity_known") else "unknown",
            }
        return {
            "rules": {
                "mode": 0,
                "fire_requires_release": True,
                "noop_does_not_brake": True,
                "down_is_hyperspace": True,
                "left_increments_orientation": True,
                "right_decrements_orientation": True,
                "enemy_coverage_unknown_is_not_absence": True,
            },
            "observation": {
                "frame": int(f.get("frame", 0)),
                "lives": int(f.get("lives", 0)),
                "score": int(max(0, f.get("score", 0))),
                "player_present": bool(f.get("present")),
                "centrality": self._band_distance(math.hypot(f.get("central_dx", 0), f.get("central_dy", 0))),
                "speed": "still" if f.get("speed", 0) < 0.25 else "moving",
                "danger": f.get("danger", "unknown"),
                "nearest_clearance": self._band_clearance((f.get("nearest") or {}).get("min_clearance")),
                "nearest_ttc": "imminent" if (f.get("nearest") or {}).get("ttc") is not None and (f.get("nearest") or {}).get("ttc") < 10 else "soon" if (f.get("nearest") or {}).get("ttc") is not None else "unknown",
                "asteroid_count": int(f.get("rock_count", 0)),
                "small_count": int(f.get("small_count", 0)),
                "motion_discontinuity": bool(f.get("motion_discontinuity")),
            },
            "target": target_summary,
            "candidate_maneuvers": [
                {key: value for key, value in c.items() if key in {"id", "kind", "label", "heading"}}
                for c in candidates
            ],
            "controller": {
                "movement_policy": self.plan.get("movement"),
                "active_move": self.move_plan is not None,
                "defense_used_this_life": bool(self.defense_used_life),
                "fire_release_frames": max(0, self.fire_cooldown),
                "last_action": self.last_action,
            },
        }

    def question_bundle(self):
        candidates = self.candidates()
        options = {c["id"]: c["label"] for c in candidates}
        order = [c["id"] for c in candidates]
        if self.frame // 30 % 2:
            order.reverse()
        criteria_a = {key: options[key] for key in order}
        criteria_b = {key: options[key] for key in reversed(order)}
        risk_options = {
            "clear": "No derived near collision is present; a bounded firing action is acceptable.",
            "caution": "A derived near-term crossing exists; prefer a short safe-sector move over scoring greed.",
            "emergency": "The derived swept clearance is critical; survival and a bounded escape override firing.",
            "unknown": "The derived tracks are insufficient; choose the conservative release/aim fallback.",
        }
        return {
            "maneuver_a": Choice(
                "Choose one candidate_maneuvers entry for this exact snapshot. Code already computed geometry, safe-sector scores, turn direction and durations. Do not invent actions or do arithmetic.",
                criteria_a,
            ),
            "maneuver_b": Choice(
                "Independently choose the safest useful candidate_maneuvers entry. Treat a move as a short bounded pulse, never an instruction to hold thrust.",
                criteria_b,
            ),
            "risk": Choice(
                "Classify the current danger using observation.danger and the named clearance/ttc buckets. Do not infer that missing enemy coverage is safe.",
                risk_options,
            ),
            "defense": Noul(
                "Should the local controller consider one mode-0 hyperspace pulse now? Say yes only when observation.danger is emergency and nearest_ttc is imminent; otherwise no.",
                {
                    "true": "A single bounded defensive pulse is justified by a current critical collision.",
                    "false": "Do not use hyperspace; local steering and firing remain sufficient.",
                },
            ),
        }, candidates

    def accept_jev(self, answers, origin_frame, origin_lives, origin_candidates):
        if not isinstance(answers, dict):
            return False
        age = self.frame - origin_frame
        if age < 0 or age > 42 or origin_lives != self.lives:
            return False
        a = answers.get("maneuver_a")
        b = answers.get("maneuver_b")
        va = _answer_value(a)
        vb = _answer_value(b)
        valid = {c.get("id") for c in origin_candidates if isinstance(c, dict)}
        if va != vb or va not in valid:
            return False
        chosen = next((c for c in origin_candidates if c.get("id") == va), None)
        if not isinstance(chosen, dict):
            return False
        threshold = 0.38 if chosen.get("kind") in {"move", "defend"} else 0.16
        if min(_prob(a, va), _prob(b, vb)) < threshold:
            return False
        self.model_choice = dict(chosen)
        self.model_answer_frame = origin_frame
        self.model_lives = origin_lives
        self.model_until = self.frame + 36
        # Defense is advisory only.  The deterministic critical gate in choose()
        # remains the final authority and cannot be weakened by a model answer.
        defense = answers.get("defense")
        if _yes(defense) >= 0.78 and self.features.get("danger") == "emergency":
            self.model_choice = {"id": "defend", "kind": "defend", "label": "model-approved critical defense"}
        self._log(
            "jev_policy",
            frame=self.frame,
            judged_frame=origin_frame,
            age=age,
            maneuver=va,
            risk=_answer_value(answers.get("risk")),
        )
        return True

    def accept_astra(self, text):
        data = _json_object(text)
        if not isinstance(data, dict):
            return False
        allowed = {
            "movement": {"safe_lane", "stationary", "escape_first"},
            "risk": {"cautious", "balanced"},
            "fire": {"pulse", "sparse"},
            "defense": {"emergency_only", "never"},
        }
        changed = {}
        for key, values in allowed.items():
            value = data.get(key)
            if isinstance(value, str) and value in values:
                changed[key] = value
        if not changed:
            return False
        self.plan.update(changed)
        self._log("astra_policy", frame=self.frame, plan=dict(self.plan))
        return True

    def astra_summary(self):
        f = self.features
        nearest = f.get("nearest") or {}
        return {
            "frame": self.frame,
            "score": self.score,
            "lives": self.lives,
            "danger": f.get("danger"),
            "rock_count": f.get("rock_count"),
            "small_count": f.get("small_count"),
            "player_speed": round(f.get("speed", 0.0), 3),
            "central_offset": [round(f.get("central_dx", 0.0), 1), round(f.get("central_dy", 0.0), 1)],
            "nearest_clearance": round(nearest.get("min_clearance", 999.0), 2),
            "active_move": self.move_plan is not None,
            "plan": dict(self.plan),
        }

    def _log(self, event, **fields):
        try:
            self.ctx.log(event, **fields)
        except Exception:
            pass


class Harness:
    def __init__(self, ctx):
        self.ctx = ctx

    def _jev_future(self, state, questions):
        return self.ctx.spawn(lambda: self.ctx.decide(state, questions, timeout_s=2.5))

    def _astra_future(self, controller, reason):
        summary = controller.astra_summary()
        prompt = (
            "You are the slow strategic supervisor for a real-time Atari 2600 Asteroids "
            "mode-0 difficulty-0 harness. The local controller owns every native frame, "
            "computes wrap-aware swept clearances, performs short safe-sector thrust pulses, "
            "and handles critical hyperspace. Do not give actions or sequences. Review this "
            "aggregate evidence and return one small JSON object only.\n"
            + json.dumps({"reason": reason, "evidence": summary}, separators=(",", ":"))
            + "\nAllowed exact fields and values: "
            '{"movement":"safe_lane"|"stationary"|"escape_first",'
            '"risk":"cautious"|"balanced",'
            '"fire":"pulse"|"sparse",'
            '"defense":"emergency_only"|"never"}. '
            "Choose a durable policy for the next several hundred frames. "
            "Stationary firing is safe only when the observed swept collision pattern "
            "supports it; unknown enemy coverage is not absence."
        )
        return self.ctx.spawn(lambda: self.ctx.think(prompt, timeout_s=100.0))

    def run_episode(self, game) -> None:
        ctx = self.ctx
        if getattr(game, "tick_s", None) is None:
            return
        controller = SafeSectorController(game, ctx)
        controller.update(_observation(game))
        try:
            ctx.act(controller.choose())
        except (BudgetExceeded, EpisodeOver):
            return

        s1_enabled = getattr(ctx, "system1", True) is not None
        jev_future = None
        jev_origin_frame = None
        jev_origin_lives = None
        jev_origin_candidates = None
        next_jev_frame = controller.frame

        astra_future = None
        astra_started = False
        astra_due = controller.frame + 240

        while not ctx.done and not getattr(game, "done", False):
            if s1_enabled and jev_future is None and controller.frame >= next_jev_frame:
                try:
                    questions, origin_candidates = controller.question_bundle()
                    jev_origin_frame = controller.frame
                    jev_origin_lives = controller.lives
                    jev_origin_candidates = [dict(c) for c in origin_candidates]
                    jev_future = self._jev_future(controller._compact(origin_candidates), questions)
                    next_jev_frame = controller.frame + 30
                except (BudgetExceeded, EpisodeOver):
                    s1_enabled = False
                except ModelUnavailable:
                    raise
                except Exception:
                    s1_enabled = False

            if not astra_started and astra_future is None and controller.frame >= astra_due:
                try:
                    remaining = ctx.budget.remaining()
                    if _int(_dict(remaining).get("system2_calls"), 0) > 0:
                        astra_future = self._astra_future(controller, "scheduled review")
                    astra_started = True
                except (BudgetExceeded, EpisodeOver):
                    astra_started = True
                except ModelUnavailable:
                    raise
                except Exception:
                    astra_started = True

            try:
                ctx.wait_for_tick()
            except (BudgetExceeded, EpisodeOver):
                return
            if ctx.done or getattr(game, "done", False):
                break

            controller.update(_observation(game))

            if jev_future is not None and jev_future.done():
                future = jev_future
                jev_future = None
                try:
                    controller.accept_jev(
                        future.result(),
                        jev_origin_frame,
                        jev_origin_lives,
                        jev_origin_candidates or [],
                    )
                except (BudgetExceeded, EpisodeOver):
                    s1_enabled = False
                except ModelUnavailable:
                    raise
                except Exception:
                    # Stand-in models, malformed typed answers and late futures
                    # are ordinary events; the local controller remains active.
                    pass

            if astra_future is not None and astra_future.done():
                future = astra_future
                astra_future = None
                try:
                    result = future.result()
                    controller.accept_astra(getattr(result, "text", result))
                except (BudgetExceeded, EpisodeOver):
                    pass
                except ModelUnavailable:
                    raise
                except Exception:
                    pass

            try:
                ctx.act(controller.choose())
            except (BudgetExceeded, EpisodeOver):
                return


def build(ctx):
    return Harness(ctx)
