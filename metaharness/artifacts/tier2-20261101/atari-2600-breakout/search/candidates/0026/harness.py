"""Observation-validated native-wall prediction with asynchronous evidence review.

The proven 0022 early motor and impact controller are retained. Geometry is
computed at native-frame boundaries, while models review evidence and decide
whether a stalled wall merits a bounded measured contact experiment.
"""
from __future__ import annotations

import math
import json
from typing import Any

from metaharness.core.contract import BudgetExceeded, EpisodeOver
from metaharness.models.jev import Choice


NOOP = "NOOP"
FIRE = "FIRE"
RIGHT = "RIGHT"
LEFT = "LEFT"
ACTIONS = (NOOP, FIRE, RIGHT, LEFT)
DIRECTIONS = (RIGHT, LEFT)


class Interceptor:
    def __init__(self, ctx: Any):
        self.ctx = ctx

        # The latest complete RAM observation is the only history used by the
        # controller.  All globals are intentionally per-episode.
        self.previous: dict[str, Any] | None = None
        self.last_frame = -1
        self.previous_paddle_x: float | None = None
        self.paddle_speed = 0.0
        self.noop_drift = 0.0
        self.last_requested = NOOP
        self.last_effective = NOOP

        # The RAM position cadence alternates adjacent displacements on ordinary
        # flights.  Average same-sign current/previous samples; cache the result
        # because _flight and _make_return_contract both query it.
        self.velocity_frame = -1
        self.velocity_valid = False
        self.velocity_warm = False
        self.filtered_dx = 0.0
        self.filtered_dy = 0.0

        # A return contract is a paddle-left interval which overlaps the ball at
        # the predicted contact.  It is held through a clean descent rather
        # than chasing a new point on every frame.
        self.contract_valid = False
        self.contract_low = 64.0
        self.contract_high = 88.0
        self.contract_center = 76.0
        self.contract_landing = None
        self.contract_eta = None
        self.contract_frame = -1
        self.contract_vx = 0.0
        self.contract_vy = 0.0
        self.contract_velocity_warm = False
        self.contract_epoch = 0
        self.flight_phase = "home"

        # Motor primitive state.  A lease prevents a one-pixel estimate from
        # reversing a command.  A brake is a single observed-motion correction,
        # followed by a neutral cooldown; it is never a standing second servo.
        self.lease_action = NOOP
        self.lease_until = -1
        self.brake_until = -1
        self.neutral_until = -1
        self.last_brake_frame = -10000
        self.last_action_frame = -1

        # Long-lived contract selected by the advisory layers.  It is allowed
        # to move the early/home anchor only.  It cannot alter a catch interval
        # during the final seven frames of a return.
        self.rally_contract = "PRESERVE"
        self.rally_version = 0
        self.astra_future = None
        self.astra_started = False
        self.astra_plan = "PRESERVE"

        # Jev is consulted at meaningful boundaries, rather than every 20
        # frames.  Two reordered choices are used for high-consequence advice.
        self.jev_enabled = getattr(ctx, "system1", None) is not None
        self.jev_future = None
        self.jev_request_frame = -100000
        self.jev_request_version = -1
        self.jev_serial = 0
        self.jev_answer = None
        self.jev_answer_frame = -100000
        self.jev_answer_version = -1
        self.next_jev_frame = 0

    # ---------------- defensive and logging helpers ----------------
    @staticmethod
    def _num(value: Any, default: float = 0.0) -> float:
        try:
            x = float(value)
            return x if math.isfinite(x) else default
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _int(value: Any, default: int = 0) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _clamp(value: float, low: float, high: float) -> float:
        return max(low, min(high, value))

    @staticmethod
    def _ball(obs: Any) -> dict[str, Any] | None:
        if isinstance(obs, dict) and isinstance(obs.get("ball"), dict):
            return obs["ball"]
        return None

    @staticmethod
    def _paddle(obs: Any) -> dict[str, Any]:
        if isinstance(obs, dict) and isinstance(obs.get("paddle"), dict):
            return obs["paddle"]
        return {}

    def _frame(self, obs: dict[str, Any]) -> int:
        return self._int(obs.get("frame"), self.last_frame + 1)

    def _log(self, name: str, **fields: Any) -> None:
        try:
            self.ctx.log(name, **fields)
        except Exception:
            pass

    # ---------------- observation facts and event boundaries ----------------
    def _event(self, obs: dict[str, Any]) -> str:
        old = self.previous
        if old is None:
            return "start"
        if self._int(obs.get("lives"), 0) < self._int(old.get("lives"), 0):
            return "life_loss"
        if self._int(obs.get("wall_index"), 0) != self._int(old.get("wall_index"), 0):
            return "wall_transition"
        if self._num(obs.get("score"), 0.0) > self._num(old.get("score"), 0.0):
            return "brick_hit"
        old_ball = self._ball(old)
        ball = self._ball(obs)
        if old_ball is None and ball is not None:
            return "serve"
        if old_ball is not None and ball is None:
            return "ball_lost"
        if old_ball is not None and ball is not None:
            old_dy = self._num(old_ball.get("dy"), 0.0)
            new_dy = self._num(ball.get("dy"), 0.0)
            old_y = self._num(old_ball.get("y"), 0.0)
            new_y = self._num(ball.get("y"), old_y)
            # This is useful as a plan boundary, but only near the paddle. A
            # dy sign change in the wall is a brick collision, not a catch.
            if old_dy > 0.0 and new_dy < 0.0 and max(old_y, new_y) > 145.0:
                return "paddle_bounce"
            old_dx = self._num(old_ball.get("dx"), 0.0)
            new_dx = self._num(ball.get("dx"), old_dx)
            if old_dx * new_dx < 0.0 and new_y > 95.0:
                return "side_bounce"
        return "none"

    def _update_paddle_motion(self, obs: dict[str, Any]) -> None:
        px = self._num(self._paddle(obs).get("x"), 99.0)
        if self.previous_paddle_x is not None:
            frame_gap = max(1, self._frame(obs) - self.last_frame)
            raw = self._clamp((px - self.previous_paddle_x) / frame_gap, -10.0, 10.0)
            self.paddle_speed = 0.58 * self.paddle_speed + 0.42 * raw
            if self.last_requested == NOOP:
                self.noop_drift = 0.90 * self.noop_drift + 0.10 * (1.0 if abs(raw) > 0.4 else 0.0)
        self.previous_paddle_x = px

        effective = obs.get("last_action")
        requested = obs.get("requested_action")
        if isinstance(effective, str) and effective in ACTIONS:
            self.last_effective = effective
        if isinstance(requested, str) and requested in ACTIONS:
            # This is diagnostic only; the paddle displacement remains
            # authoritative when sticky actions disagree with the request.
            self.last_requested = requested

    def _reset_return(self, reason: str, frame: int) -> None:
        self.contract_valid = False
        self.contract_landing = None
        self.contract_eta = None
        self.flight_phase = "home"
        self.contract_epoch += 1
        self.lease_action = NOOP
        self.lease_until = frame
        self.brake_until = -1
        self.neutral_until = frame + 1
        self.last_brake_frame = -10000
        self.velocity_frame = -1
        self.velocity_valid = False
        self.velocity_warm = False
        self.contract_velocity_warm = False
        self._log("return_contract_reset", frame=frame, reason=reason, epoch=self.contract_epoch)

    def _apply_event(self, obs: dict[str, Any], event: str, frame: int) -> None:
        if event in ("start", "serve", "life_loss", "ball_lost", "wall_transition", "paddle_bounce"):
            if getattr(self, "impact_lock", None) is not None:
                self._log("impact_lock_reset", frame=frame, reason=event,
                          phase=self.impact_lock.get("phase"))
            self.impact_lock = None
            # A model profile is scoped to the return after the boundary; a
            # missing/late answer therefore returns to the safe code policy.
            if event in ("start", "serve", "life_loss", "wall_transition", "paddle_bounce"):
                self.return_profile = "TRACK"
            self._reset_return(event, frame)
            if event in ("life_loss", "wall_transition"):
                self.rally_version += 1
                self.rally_contract = "PRESERVE"
                self.jev_answer = None
                self.next_jev_frame = frame
        elif event in ("brick_hit", "side_bounce"):
            # A top-wall collision can change either velocity component.  Do
            # not keep a contract made from the previous trajectory.
            self._reset_return(event, frame)

    # ---------------- exact local geometry ----------------
    @staticmethod
    def _reflect_x(x: float, dx: float, frames: float) -> float:
        left, right = 8.0, 150.0
        span = right - left
        if span <= 0.0:
            return x
        phase = (x - left + dx * max(0.0, frames)) % (2.0 * span)
        return left + phase if phase <= span else right - (phase - span)

    def _velocity(self, obs: dict[str, Any]) -> tuple[float, float, bool]:
        frame = self._frame(obs)
        if frame == self.velocity_frame:
            return self.filtered_dx, self.filtered_dy, self.velocity_valid

        ball = self._ball(obs)
        if ball is None:
            self.velocity_frame = frame
            self.velocity_valid = False
            self.velocity_warm = False
            self.filtered_dx = 0.0
            self.filtered_dy = 0.0
            return 0.0, 0.0, False

        dx = self._num(ball.get("dx"), 0.0)
        dy = self._num(ball.get("dy"), 0.0)
        valid = bool(obs.get("ball_velocity_valid", False))
        old_ball = self._ball(self.previous) if self.previous is not None else None
        old_valid = bool(
            isinstance(self.previous, dict) and self.previous.get("ball_velocity_valid", False)
        )
        if old_ball is not None:
            inferred_dx = self._clamp(
                self._num(ball.get("x"), 0.0) - self._num(old_ball.get("x"), 0.0), -8.0, 8.0
            )
            inferred_dy = self._clamp(
                self._num(ball.get("y"), 0.0) - self._num(old_ball.get("y"), 0.0), -8.0, 8.0
            )
            if not valid and (abs(inferred_dx) > 0.0 or abs(inferred_dy) > 0.0):
                dx, dy = inferred_dx, inferred_dy
                valid = True
        if not valid and abs(dx) + abs(dy) == 0.0:
            self.velocity_frame = frame
            self.velocity_valid = False
            self.velocity_warm = False
            self.filtered_dx = 0.0
            self.filtered_dy = 0.0
            return 0.0, 0.0, False

        filtered_dx = dx
        filtered_dy = dy
        same_dx = same_dy = False
        if old_valid and old_ball is not None:
            old_dx = self._num(old_ball.get("dx"), 0.0)
            old_dy = self._num(old_ball.get("dy"), 0.0)
            # A sign change is a collision/flight clue, so never smear it.
            if old_dx * dx > 0.0:
                filtered_dx = 0.5 * (old_dx + dx)
                same_dx = True
            if old_dy * dy > 0.0:
                filtered_dy = 0.5 * (old_dy + dy)
                same_dy = True

        self.velocity_frame = frame
        self.velocity_valid = True
        self.velocity_warm = same_dx and same_dy
        self.filtered_dx = filtered_dx
        self.filtered_dy = filtered_dy
        return filtered_dx, filtered_dy, True

    def _home_interval(self, obs: dict[str, Any]) -> tuple[float, float, float]:
        # A center home is deliberately a *range*: it avoids constantly moving
        # the paddle while the ball is still in the wall.  The optional rally
        # contract only shifts this range when the return is far away and lives
        # are plentiful.
        lives = self._int(obs.get("lives"), 0)
        center = 76.0
        if lives >= 4 and self.rally_contract == "LEFT":
            center = 62.0
        elif lives >= 4 and self.rally_contract == "RIGHT":
            center = 90.0
        low = self._clamp(center - 15.0, 8.0, 144.0)
        high = self._clamp(center + 15.0, 8.0, 144.0)
        return low, high, center

    def _make_return_contract(self, obs: dict[str, Any], frame: int) -> bool:
        ball = self._ball(obs)
        if ball is None or not bool(obs.get("ball_in_play", False)):
            return False
        dx, dy, valid = self._velocity(obs)
        if not valid or dy <= 0.0:
            return False
        paddle = self._paddle(obs)
        py = self._num(paddle.get("y"), 189.0)
        pw = self._num(paddle.get("w"), 16.0)
        bx = self._num(ball.get("x"), 80.0)
        by = self._num(ball.get("y"), 100.0)
        bw = self._num(ball.get("w"), 2.0)
        bh = self._num(ball.get("h"), 4.0)
        eta = max(0.0, (py - 1.0 - by) / max(0.25, dy))
        landing_left = self._reflect_x(bx, dx, eta)

        # Do not extrapolate through a still-occupied brick band.  The lower
        # row ends just below 90 in this decoder; 94 leaves a small collision
        # ordering guard.  Once below it, the reflected path is deterministic.
        if by < 94.0:
            return False

        speed = abs(dx) + abs(dy)
        margin = 3.0
        # Paddle-left coordinates that produce interior overlap with the ball.
        low = landing_left + margin - pw
        high = landing_left + bw - margin
        pmin, pmax = 8.0, max(8.0, 160.0 - pw)
        low = self._clamp(low, pmin, pmax)
        high = self._clamp(high, pmin, pmax)
        if high < low:
            mid = self._clamp(landing_left + bw / 2.0 - pw / 2.0, pmin, pmax)
            low = high = mid

        # A large uncertainty guard is applied as an action policy, not by
        # shrinking the physical overlap interval.  Keeping the full set is
        # what gives the motor room to settle under sticky actions.
        self.contract_valid = True
        self.contract_low = low
        self.contract_high = high
        self.contract_center = (low + high) / 2.0
        self.contract_landing = landing_left + bw / 2.0
        self.contract_eta = eta
        self.contract_frame = frame
        self.contract_vx = dx
        self.contract_vy = dy
        self.contract_velocity_warm = self.velocity_warm
        self.flight_phase = "descending"
        self._log(
            "return_contract",
            frame=frame,
            low=round(low, 2),
            high=round(high, 2),
            landing=round(self.contract_landing, 2),
            eta=round(eta, 2),
        )
        return True

    def _flight(self, obs: dict[str, Any], event: str, frame: int) -> dict[str, Any]:
        ball = self._ball(obs)
        if ball is None or not bool(obs.get("ball_in_play", False)):
            low, high, center = self._home_interval(obs)
            return {
                "phase": "serve",
                "urgency": "serve",
                "eta": None,
                "low": low,
                "high": high,
                "center": center,
                "landing": None,
                "speed": "unknown",
            }

        dx, dy, valid = self._velocity(obs)
        by = self._num(ball.get("y"), 100.0)
        paddle = self._paddle(obs)
        py = self._num(paddle.get("y"), 189.0)
        bh = self._num(ball.get("h"), 4.0)
        eta = None
        if valid and dy > 0.0:
            eta = max(0.0, (py - 1.0 - by) / max(0.25, dy))

        if dy < 0.0 or not valid:
            if self.contract_valid:
                self._reset_return("ascending", frame)
            low, high, center = self._home_interval(obs)
            self.flight_phase = "ascending" if dy < 0.0 else "unknown"
            return {
                "phase": self.flight_phase,
                "urgency": "early",
                "eta": None,
                "low": low,
                "high": high,
                "center": center,
                "landing": None,
                "speed": "fast" if abs(dx) + abs(dy) >= 4.0 else "normal",
            }

        if by < 94.0:
            if self.contract_valid:
                self._reset_return("collision_zone", frame)
            low, high, center = self._home_interval(obs)
            return {
                "phase": "collision_zone",
                "urgency": "early",
                "eta": None if eta is None else round(eta, 1),
                "low": low,
                "high": high,
                "center": center,
                "landing": None,
                "speed": "fast" if abs(dx) + abs(dy) >= 4.0 else "normal",
            }

        if not self.contract_valid:
            self._make_return_contract(obs, frame)
        else:
            # With filtered velocity, a fresh interval is a stable receding
            # horizon rather than the raw-cadence flip seen in the parent.
            self._make_return_contract(obs, frame)

        if not self.contract_valid:
            low, high, center = self._home_interval(obs)
            return {
                "phase": "uncertain_descent",
                "urgency": "early",
                "eta": None if eta is None else round(eta, 1),
                "low": low,
                "high": high,
                "center": center,
                "landing": None,
                "speed": "unknown",
            }

        if eta is None or eta <= 6.0:
            urgency = "imminent"
        elif eta <= 18.0:
            urgency = "near"
        else:
            urgency = "early"
        return {
            "phase": "descending",
            "urgency": urgency,
            "eta": None if eta is None else round(eta, 1),
            "low": round(self.contract_low, 2),
            "high": round(self.contract_high, 2),
            "center": round(self.contract_center, 2),
            "landing": None if self.contract_landing is None else round(self.contract_landing, 2),
            "speed": "fast" if abs(dx) + abs(dy) >= 4.0 else "normal",
        }

    # ---------------- interval motor controller ----------------
    def _opposite(self, action: str) -> str:
        return LEFT if action == RIGHT else RIGHT

    def _steer_interval(self, obs: dict[str, Any], flight: dict[str, Any], frame: int) -> str:
        if bool(obs.get("terminated", False)) or bool(obs.get("truncated", False)):
            return NOOP
        if bool(obs.get("serve_required", False)):
            return FIRE if self._int(obs.get("lives"), 0) > 0 else NOOP
        ball = self._ball(obs)
        if ball is None:
            return NOOP

        paddle = self._paddle(obs)
        px = self._num(paddle.get("x"), 99.0)
        low = self._num(flight.get("low"), 62.0)
        high = self._num(flight.get("high"), 90.0)
        eta_value = flight.get("eta")
        eta = self._num(eta_value, 12.0) if eta_value is not None else 12.0
        urgency = flight.get("urgency")

        # A command lease is a motor primitive, not a model decision.  Stop a
        # lease as soon as the paddle enters the interval; otherwise retain it
        # until expiry so the servo cannot chatter around a moving estimate.
        if self.lease_action in DIRECTIONS and frame < self.lease_until:
            inside = low - 1.0 <= px <= high + 1.0
            if not inside:
                return self.lease_action
            self.lease_action = NOOP
            self.lease_until = frame
            self.neutral_until = max(self.neutral_until, frame + 1)

        horizon = self._clamp(eta, 1.0, 5.0)
        predicted_px = px + self.paddle_speed * horizon
        guard = 1.0 if urgency in ("near", "imminent") else 2.0

        # If the paddle is outside the physical catch set, approach the nearer
        # interior edge.  A robust interval matters more than its midpoint.
        if px < low - guard or predicted_px < low - guard:
            desired = RIGHT
        elif px > high + guard or predicted_px > high + guard:
            desired = LEFT
        else:
            desired = NOOP

        # If a current directional command is still effective but the new
        # target is only a small opposite error, release it instead of mirroring
        # it.  A single opposite pulse is reserved for measured overshoot.
        if desired == NOOP:
            moving_out = (self.paddle_speed > 0.45 and predicted_px > high + guard) or (
                self.paddle_speed < -0.45 and predicted_px < low - guard
            )
            if moving_out and frame >= self.last_brake_frame + 3 and frame >= self.neutral_until:
                action = self._opposite(RIGHT if self.paddle_speed > 0.0 else LEFT)
                self.last_brake_frame = frame
                self.brake_until = frame + 1
                self.neutral_until = frame + 2
                self.lease_action = NOOP
                self.lease_until = frame
                self._log("brake_pulse", frame=frame, action=action)
                return action
            self.lease_action = NOOP
            self.lease_until = frame + (1 if urgency in ("near", "imminent") else 2)
            return NOOP

        # Do not add a direction during the neutral cooldown after a brake
        # unless the paddle is genuinely outside the interval.
        if frame < self.neutral_until and low <= px <= high:
            return NOOP

        # A direction is leased for two frames early and one near impact.  A
        # far-side error can interrupt an opposite lease, but only when the
        # paddle is outside by a full guard, never for a small target change.
        if self.lease_action in DIRECTIONS and self.lease_action != desired:
            if low - 3.0 <= px <= high + 3.0:
                self.lease_action = NOOP
                self.lease_until = frame
            else:
                self.lease_action = desired

        if self.lease_action not in DIRECTIONS:
            self.lease_action = desired
            self.lease_until = frame + (1 if urgency in ("near", "imminent") else 2)
        return self.lease_action

    # ---------------- derived state and model contracts ----------------
    def _side_facts(self, obs: dict[str, Any]) -> dict[str, Any]:
        bitmap = obs.get("brick_bitmap") if isinstance(obs, dict) else None
        left = right = 0
        rows = 0
        if isinstance(bitmap, list):
            for row in bitmap:
                if not isinstance(row, list):
                    continue
                rows += 1
                if len(row) >= 2:
                    left += int(bool(row[0])) + int(bool(row[1]))
                if len(row) >= 18:
                    right += int(bool(row[16])) + int(bool(row[17]))
        return {
            "left_edge": "open" if left == 0 else "partial" if left <= 3 else "blocked",
            "right_edge": "open" if right == 0 else "partial" if right <= 3 else "blocked",
            "left_count": min(12, left),
            "right_count": min(12, right),
            "wall_index": self._int(obs.get("wall_index"), 0),
            "bricks_remaining": self._int(obs.get("bricks_remaining"), 108),
        }

    @staticmethod
    def _answer_value(answer: Any) -> str | None:
        value = getattr(answer, "value", None)
        return value if isinstance(value, str) else None

    @staticmethod
    def _answer_probability(answer: Any, value: str) -> float:
        probs = getattr(answer, "probabilities", None)
        if isinstance(probs, dict):
            try:
                p = float(probs.get(value, 0.0))
                return p if math.isfinite(p) else 0.0
            except (TypeError, ValueError):
                return 0.0
        return 1.0 if Harness._answer_value(answer) == value else 0.0

    # ---------------- episode loop ----------------
    @staticmethod
    def _observe(game: Any) -> dict[str, Any]:
        try:
            value = game.observe()
        except Exception:
            return {}
        return value if isinstance(value, dict) else {}

    def run_episode(self, game: Any) -> None:
        realtime = bool(getattr(game, "tick_s", 0.0))
        ctx = self.ctx
        while not ctx.done and not bool(getattr(game, "done", False)):
            if not self._one(game, realtime):
                return


class ReferenceController(Interceptor):
    """Stable one-flight objectives over a continuously refreshed catch set.

    A trajectory replan is not a new contact experiment.  This distinction is
    the crossover: 0015 supplies filtered interception, 0016 supplies endpoint
    objectives, and only an observed bounce or loss resolves a trial.
    """

    def __init__(self, ctx):
        super().__init__(ctx)
        self.contact_id = 0
        self.rally_bounces = 0
        self.trial = None
        self.trial_considered = -1
        self.trials_this_life = 0
        self.trial_cooldown = 0
        self.intent = "AUTO"
        self.intent_contact = -1
        self.endpoint_stats = {"low": [0, 0], "high": [0, 0]}
        self.contact_window = []
        self.recent_contacts = []
        self.plan_revision = 0
        self.model_epoch = 0
        self.astra_wall = -1
        self.astra_request_wall = -1
        self.astra_request_revision = -1
        self.astra_calls = 0
        self.astra_plan_keys = set()
        self.jev_tag = None
        self.jev_request_state = None
        self.jev_serial = 0
        self.next_jev_frame = 0
        self.last_progress_frame = 0
        self.last_trial_outcome = "none"
        # Measured first-order paddle response with one queued input. Defaults
        # are a fit to this candidate's public seed-1/5 contact observations.
        # Every episode estimates the two coefficients again from live motion.
        self.motor_drag = 0.60
        self.motor_gain = 2.40
        self.motor_raw_v = 0.0
        self.motor_previous_v = 0.0
        self.motor_effective = 0.0
        self.motor_queued = 0.0
        self.motor_previous_input = 0.0
        self.motor_fit = [0.0] * 5
        self.motor_samples = 0
        self.cadence_frame = -1
        self.cadence_x = []
        self.cadence_y = []
        self.pending_contact = None
        self.fast_endpoint_stats = {"low": [0, 0], "high": [0, 0]}
        self.fast_speeds = []
        self.calibration_trials = 0

        # A profile is advisory; the impact lock below is always available and
        # is never disabled by a model answer.
        self.return_profile = "TRACK"
        self.astra_profile = "TRACK"
        self.pending_profile = None
        self.impact_lock = None
        self.impact_lock_count = 0

    def _velocity(self, obs):
        frame = self._frame(obs)
        if frame == self.cadence_frame:
            return self.filtered_dx, self.filtered_dy, self.velocity_valid
        ball = self._ball(obs)
        if ball is None:
            self.cadence_x, self.cadence_y = [], []
            self.cadence_frame = frame
            self.filtered_dx = self.filtered_dy = 0.0
            self.velocity_valid = self.velocity_warm = False
            return 0.0, 0.0, False
        # Same-sign period-six means cover both observed native cadences:
        # 1/2 at low speed and 3/3/4 at high speed. Sign events are immediate.
        dx = self._num(ball.get("dx"))
        dy = self._num(ball.get("dy"))
        if self.previous is None or frame != self.last_frame + 1:
            self.cadence_x, self.cadence_y = [], []
        valid = bool(obs.get("ball_velocity_valid")) or abs(dx) + abs(dy) > 0.0
        if not valid:
            self.cadence_x, self.cadence_y = [], []
        for values, component in ((self.cadence_x, dx), (self.cadence_y, dy)):
            nonzero = next((value for value in reversed(values) if value != 0.0), 0.0)
            if values and (nonzero * component < 0.0 or abs(values[-1]-component) > 1.0):
                values.clear()
            values.append(component)
            del values[:-6]
        self.filtered_dx = sum(self.cadence_x) / len(self.cadence_x)
        self.filtered_dy = sum(self.cadence_y) / len(self.cadence_y)
        self.velocity_valid = valid
        self.velocity_warm = valid and min(len(self.cadence_x), len(self.cadence_y)) >= 2
        self.cadence_frame = self.velocity_frame = frame
        return self.filtered_dx, self.filtered_dy, valid

    def _home_interval(self, obs):
        # Endpoint objectives must not perturb the tested home/survival motor.
        return 61.0, 91.0, 76.0

    def _contact_event(self, obs, event, frame):
        if event == "brick_hit":
            self.last_progress_frame = frame
        if event == "paddle_bounce":
            old_ball = self._ball(self.previous) or {}
            old_paddle = self._paddle(self.previous)
            ball = self._ball(obs) or {}
            # Record actual overlap at the last descending sample, not merely
            # the requested endpoint.  A request that missed its endpoint is
            # not evidence for reversing the paddle-section mapping.
            offset = (self._num(old_ball.get("x")) + 1.0 -
                      self._num(old_paddle.get("x")) -
                      self._num(old_paddle.get("w"), 16.0) / 2.0)
            outdx = self._num(ball.get("dx"))
            outcome = {
                "frame": frame, "contact": self.contact_id,
                "offset": round(offset, 2), "out_dx": outdx,
                "in_dx": self._num(old_ball.get("dx")),
                "in_dy": self._num(old_ball.get("dy")),
                "ball_y_before": self._num(old_ball.get("y")),
                "ball_y_after": self._num(ball.get("y")),
                "paddle_x": self._num(old_paddle.get("x")),
                "trial": None if self.trial is None else dict(self.trial),
            }
            self.recent_contacts.append(outcome)
            self.recent_contacts = self.recent_contacts[-12:]
            self._log("contact_observed", **outcome)
            self.pending_contact = dict(outcome)
            self.pending_contact["x"] = self._num(ball.get("x"))
            self.pending_contact["stable_paddle"] = abs(
                self._num(self._paddle(obs).get("x")) - self._num(old_paddle.get("x"))) <= 2.0
            self.pending_contact["samples"] = []
            if self.trial is not None:
                self.trial_cooldown = 1
            elif self.trial_cooldown:
                self.trial_cooldown -= 1
            self.rally_bounces += 1
            self.contact_id += 1
            self.trial = None
            self.contact_window = []
            self.intent = "AUTO"
        elif event in ("start", "serve", "ball_lost", "life_loss", "wall_transition"):
            self.pending_contact = None
            if event == "ball_lost":
                self._log("miss_window", frame=frame, contact=self.contact_id,
                          trial=self.trial, samples=self.contact_window)
                if self.trial is not None:
                    self.last_trial_outcome = "miss"
                    self.trial_cooldown = 2
            self.contact_id += 1
            self.model_epoch += 1
            self.trial = None
            self.contact_window = []
            self.rally_bounces = 0
            self.intent = "AUTO"
            if event in ("start", "serve"):
                self.trials_this_life = 0
            if event == "wall_transition":
                self.calibration_trials = 0
                self.astra_plan = "PRESERVE"
                self.plan_revision += 1
                self.trial_cooldown = 1

    def _confirm_outgoing(self, obs, frame):
        item = self.pending_contact
        ball = self._ball(obs)
        if item is None or ball is None:
            return
        age = frame - item["frame"]
        if 1 <= age <= 4 and self._num(ball.get("dy")) < 0:
            item["samples"].append([self._num(ball.get("dx")), self._num(ball.get("dy"))])
        if age < 4:
            return
        self.pending_contact = None
        samples = item["samples"]
        if len(samples) < 3:
            return
        vx = sum(v[0] for v in samples) / len(samples)
        vy = sum(v[1] for v in samples) / len(samples)
        outgoing = LEFT if vx < -0.1 else RIGHT if vx > 0.1 else "unknown"
        offset = item["offset"]
        endpoint = "high" if offset < -2.0 else "low" if offset > 2.0 else None
        reliable = item["stable_paddle"] and 20 <= item["x"] <= 138 and 2 < abs(offset) <= 6
        if reliable and endpoint is not None and outgoing in DIRECTIONS:
            index = 0 if outgoing == LEFT else 1
            self.endpoint_stats[endpoint][index] += 1
            if abs(item["in_dx"])+abs(item["in_dy"]) >= 5 and abs(vx) > 0.5:
                self.fast_endpoint_stats[endpoint][index] += 1
                self.fast_speeds.append([abs(vx), abs(vy)])
                self.fast_speeds = self.fast_speeds[-12:]
        trial = item["trial"]
        if trial is not None:
            achieved = endpoint == trial["endpoint"]
            self.last_trial_outcome = (
                "confirmed" if achieved and outgoing == trial["side"] else
                "different_direction" if achieved else "endpoint_not_reached")
            self._log("contact_trial_resolved", frame=frame, contact=item["contact"],
                      outcome=self.last_trial_outcome, outgoing=outgoing,
                      offset=offset, actual_endpoint=achieved, requested=trial["side"],
                      endpoint=trial["endpoint"], reliable_mapping_sample=reliable,
                      measured_velocity=[round(vx, 3), round(vy, 3)])

    def _calibration_stage(self, obs):
        # A perfectly centered motor can enter a no-progress orbit without
        # ever observing an endpoint. Permit bounded interior calibration only
        # with a spare third life; never pretend an unmeasured angle is known.
        missing = any(v[i] < 2 for v, i in ((self.fast_endpoint_stats["high"], 0),
                                          (self.fast_endpoint_stats["low"], 1)))
        return (self._int(obs.get("lives")) >= 3 and self.calibration_trials < 4 and
                0 < self._int(obs.get("bricks_remaining"), 108) <= 12 and
                self._frame(obs)-self.last_progress_frame >= 420 and missing and
                abs(self.filtered_dy) >= 2.8 and abs(self.filtered_dx) >= 1.5)

    def _cleanup_stage(self, obs):
        return (self._int(obs.get("lives")) >= 2 and
                0 < self._int(obs.get("bricks_remaining"), 108) <= 12 and
                self._frame(obs)-self.last_progress_frame >= 420 and
                abs(self.filtered_dy) >= 2.8 and abs(self.filtered_dx) >= 1.5 and
                len(self.fast_speeds) >= 2)

    def _cleanup_side(self, obs, flight):
        if not self._cleanup_stage(obs) or flight.get("phase") != "descending":
            return None
        landing = flight.get("landing")
        bricks = obs.get("bricks")
        if landing is None or not isinstance(bricks, list):
            return None
        vx = sum(v[0] for v in self.fast_speeds)/len(self.fast_speeds)
        vy = sum(v[1] for v in self.fast_speeds)/len(self.fast_speeds)
        if vy < 1.0:
            return None
        best, best_value = None, -1e9
        for side, sign in ((LEFT, -1), (RIGHT, 1)):
            endpoint = self._endpoint(side)
            evidence = self.fast_endpoint_stats[endpoint]
            index = 0 if side == LEFT else 1
            if evidence[index] < 2 or evidence[index] < 2*evidence[1-index]:
                continue
            value = -1e9
            for brick in bricks:
                if not isinstance(brick, dict):
                    continue
                # Exact reflection is arithmetic; measured outgoing speed is
                # evidence, not an imported generic Breakout angle table.
                top = self._num(brick.get("y"), 60.0)
                height = self._num(brick.get("h"), 6.0)
                eta = max(0.0, (188.0-top-height)/vy)
                x = self._reflect_x(float(landing)-1.0, sign*vx, eta)+1.0
                bx = self._num(brick.get("x")); bw = self._num(brick.get("w"), 8.0)
                gap = max(bx-x, 0.0, x-(bx+bw))
                quality = (12.0 if gap == 0 else 0.0) + self._num(brick.get("value"), 1.0) - gap
                value = max(value, quality)
            if value > best_value:
                best, best_value = side, value
        return best

    def _endpoint(self, side):
        expected = "high" if side == LEFT else "low"
        other = "low" if expected == "high" else "high"
        index = 0 if side == LEFT else 1
        evidence = self.endpoint_stats[other]
        # Unlike 0016, one unachieved request cannot flip the mapping. Require
        # two observed opposite-endpoint successes before using that mapping.
        if evidence[index] >= 2 and evidence[index] >= evidence[1 - index] + 2:
            return other
        return expected

    def _candidate(self, obs, flight):
        cleanup = self._cleanup_side(obs, flight)
        if cleanup is not None:
            if self.intent_contact == self.contact_id and self.intent in DIRECTIONS:
                evidence = self.fast_endpoint_stats[self._endpoint(self.intent)]
                index = 0 if self.intent == LEFT else 1
                if evidence[index] >= 2 and evidence[index] >= 2*evidence[1-index]:
                    return self.intent
            return cleanup
        if self._calibration_stage(obs):
            return LEFT if self.calibration_trials % 2 == 0 else RIGHT
        side = self._side_facts(obs)
        if side["left_edge"] == "open" and side["right_edge"] == "open":
            return None
        if self.intent_contact == self.contact_id and self.intent in DIRECTIONS:
            choice = self.intent
        elif self.astra_plan in DIRECTIONS:
            choice = self.astra_plan
        elif side["left_edge"] == "open":
            choice = RIGHT
        elif side["right_edge"] == "open":
            choice = LEFT
        elif side["left_count"] < side["right_count"]:
            choice = LEFT
        elif side["right_count"] < side["left_count"]:
            choice = RIGHT
        else:
            choice = LEFT if self.filtered_dx < 0 else RIGHT
        if side["left_edge" if choice == LEFT else "right_edge"] == "open":
            return None
        return choice

    def _shape_ready(self, obs, flight):
        width = self._num(flight.get("high")) - self._num(flight.get("low"))
        common = (flight.get("phase") == "descending" and
                  self._num(flight.get("eta")) > 14.0 and self.velocity_warm and
                  self.trial_cooldown == 0 and width >= 9.0)
        ordinary = (flight.get("speed") == "normal" and
                    self._int(obs.get("lives")) >= 4 and self.rally_bounces >= 2 and
                    (self.trial is not None or self.trials_this_life < 3))
        cleanup = self._cleanup_side(obs, flight) is not None
        # Profile choice changes only the terminal lock timing; the code-owned
        # tunnel gate remains available so a slow Jev answer cannot turn the
        # scoring policy into a permanent survival-only mode.
        return common and (ordinary or cleanup or self._calibration_stage(obs))

    def _shape_flight(self, obs, flight, frame):
        if (self.trial is None and self.trial_considered != self.contact_id and
                self._shape_ready(obs, flight)):
            side = self._candidate(obs, flight)
            # SURVIVE applies to one future contact, not all remaining lives.
            veto = self.intent == "SURVIVE" and self.intent_contact == self.contact_id
            if side is not None and not veto:
                self.trial = {"side": side, "endpoint": self._endpoint(side),
                              "frame": frame, "contact": self.contact_id,
                              "purpose": "cleanup" if self._cleanup_side(obs, flight) is not None else
                                         "calibration" if self._calibration_stage(obs) else "tunnel",
                              "source": "jev" if self.intent_contact == self.contact_id and self.intent in DIRECTIONS
                                        else "astra_prior" if self.astra_plan in DIRECTIONS else "code"}
                self.trials_this_life += 1
                if self.trial["purpose"] == "calibration":
                    self.calibration_trials += 1
                self._log("contact_trial_started", **self.trial,
                          filtered_velocity=[self.filtered_dx, self.filtered_dy])
            self.trial_considered = self.contact_id

        # Only the early positioning band is narrowed. Every impact is still
        # governed by the original full physical interval and original motor.
        if (self.trial is not None and self._shape_ready(obs, flight) and
                not (self.intent == "SURVIVE" and self.intent_contact == self.contact_id)):
            low, high = flight["low"], flight["high"]
            endpoint = self.trial["endpoint"]
            target = high - 2.0 if endpoint == "high" else low + 2.0
            shaped = dict(flight)
            shaped["low"] = max(low, target - 2.0)
            shaped["high"] = min(high, target + 2.0)
            shaped["center"] = target
            shaped["shape"] = self.trial["side"]
            return shaped
        if (self.trial is not None and flight["phase"] == "descending" and
                ((flight["speed"] == "normal" and self._int(obs.get("lives")) >= 4) or
                 (self.trial.get("purpose") in ("cleanup", "calibration") and self._int(obs.get("lives")) >= 2)) and
                not (self.intent == "SURVIVE" and self.intent_contact == self.contact_id)):
            soft = dict(flight)
            soft["preferred_target"] = (flight["high"]-2.0 if self.trial["endpoint"] == "high"
                                        else flight["low"]+2.0)
            return soft
        return flight

    def _terminal_risk(self, obs, flight):
        if flight.get("phase") != "descending" or flight.get("eta") is None:
            return "not_terminal"
        eta = self._num(flight.get("eta"), 99.0)
        if eta > 8.0:
            return "early"
        px = self._num(self._paddle(obs).get("x"), 99.0)
        low = self._num(flight.get("low"), 61.0)
        high = self._num(flight.get("high"), 91.0)
        v = self._num(self.motor_raw_v, 0.0)
        forecast = px + v * max(1.0, min(3.0, eta))
        if low <= px <= high and low-1.0 <= forecast <= high+1.0:
            return "settled"
        if low <= px <= high:
            return "moving_away_from_set"
        center = (low + high) / 2.0
        if (px < low and v > 0.0) or (px > high and v < 0.0):
            return "moving_toward_set"
        if (px-center) * v > 0.0:
            return "moving_away_from_set"
        return "unknown"

    def _model_state(self, obs, flight, event):
        side = self._side_facts(obs)
        return {
            "game": "Atari 2600 Breakout mode 0 difficulty 0; native 60 Hz, sticky actions",
            "rules": {"priority": "preserve every catch before shaping",
                      "bridge": "models choose a future contact objective, never a frame action",
                      "candidate": "code computed a reachable interior endpoint; code rechecks safety",
                      "late": "a changed life, wall, contact, or plan rejects the answer",
                      "cleanup": "verified_cleanup uses observed endpoint directions and ray projections to remaining bricks; an open edge or fast speed is not a veto in this mode, but code still requires a wide safe interval and at least two lives",
                      "calibration": "endpoint_calibration is a bounded interior-offset trial after no score progress, with at least three lives and a wide catch interval; it measures direction instead of assuming an angle"},
            "observation": {
                "frame": self._frame(obs), "event": event,
                "lives": self._int(obs.get("lives")), "score": self._num(obs.get("score")),
                "wall": side["wall_index"], "wall_index": side["wall_index"], "bricks_remaining": side["bricks_remaining"],
                "phase": flight["phase"], "urgency": flight["urgency"], "eta_bucket": flight["urgency"], "speed": flight["speed"],
                "landing_lane": ("left" if self._num(flight.get("landing"), 76.0) < 56.0 else
                                 "right" if self._num(flight.get("landing"), 76.0) > 104.0 else "center"),
                "catch_width": "wide" if self._num(flight.get("high")) - self._num(flight.get("low")) >= 10.0 else "narrow",
                "shape_ready": self._shape_ready(obs, flight),
                "next_contact_eligible": ((self._int(obs.get("lives")) >= 4 and self.rally_bounces >= 2
                                         and flight["speed"] == "normal" and self.trial_cooldown == 0)
                                         or self._cleanup_stage(obs) or self._calibration_stage(obs)),
                "objective_kind": "verified_cleanup" if self._cleanup_stage(obs) else
                                  "endpoint_calibration" if self._calibration_stage(obs) else "tunnel",
                "safe_catches_since_serve": "several" if self.rally_bounces >= 2 else "few",
                "cleanup_candidate": self._cleanup_side(obs, flight) or "awaiting_landing_geometry",
                "candidate_side": self._candidate(obs, flight) or "none",
                "left_edge": side["left_edge"], "right_edge": side["right_edge"],
                "current_objective": self.intent, "return_profile": self.return_profile,
                "astra_profile": self.astra_profile, "astra_proposal": self.astra_plan,
                "terminal_risk": self._terminal_risk(obs, flight),
                "impact_lock_state": None if self.impact_lock is None else self.impact_lock.get("phase"),
                "paddle_motion": "right" if self.motor_raw_v > 0.45 else "left" if self.motor_raw_v < -0.45 else "settled",
                "last_contact_outcome": self.last_trial_outcome, "trial_outcome": self.last_trial_outcome,
                "return_epoch": self.contract_epoch,
                "endpoint_evidence": {k: ("measured" if sum(v) >= 2 else "unverified")
                                      for k, v in self.endpoint_stats.items()},
                "contact_id": self.contact_id, "plan_revision": self.plan_revision,
            },
        }

    def _questions(self):
        options = {
            "DAMP": {"what": "Use the conservative return profile: preserve the full catch set and commit to the measured brake-then-adaptive impact lock.",
                     "when": "terminal risk is high, speed or geometry is uncertain, or lives are scarce"},
            "TRACK": {"what": "Use ordinary catch-first interception; the code still installs the terminal brake-and-adaptive lock.",
                      "when": "the return is known and there is no evidence for deliberate shaping"},
            "SHAPE": {"what": "Allow the code-generated early contact objective for a side tunnel or verified cleanup ray; never trade away the terminal catch set.",
                      "when": "the code reports an eligible early return, normal speed, a spare life, and a non-open candidate edge"},
            "NO_CHANGE": {"what": "Keep the current profile and let the deterministic controller decide.",
                          "when": "ambiguous, late, or no profile change is justified"},
        }
        keys = list(options)
        shift = self.jev_serial % 4
        order = keys[shift:] + keys[:shift]
        text = ("Choose the return profile for the NEXT paddle contact using only the named evidence. "
                "These are subgoals, not frame actions; do not calculate coordinates. DAMP is the "
                "safe short-brake/adaptive-lock profile, TRACK is ordinary catch-first control, and SHAPE "
                "only authorizes a code-generated early tunnel/cleanup preference. The code always "
                "rechecks lives, speed, candidate edge, overlap, and terminal risk. Prefer DAMP or "
                "NO_CHANGE for an imminent/uncertain return, a moving-away paddle, or fewer than four "
                "lives. A past miss does not permanently veto a later safe SHAPE opportunity. "
                "The impact lock is code-owned and cannot be replaced by a language-model action.")
        return {"objective_a": Choice(text, {k: options[k] for k in order}),
                "objective_b": Choice(text, {k: options[k] for k in reversed(order)})}

    @staticmethod
    def _answer_probability(answer, option):
        probs = getattr(answer, "probabilities", None)
        if not isinstance(probs, dict):
            return 0.0
        try:
            p = float(probs.get(option, 0.0))
            return p if math.isfinite(p) else 0.0
        except (TypeError, ValueError):
            return 0.0

    def _update_motor(self, obs, frame):
        px = self._num(self._paddle(obs).get("x"), 99.0)
        previous_px = self._num(self._paddle(self.previous).get("x"), px)
        gap = frame - self.last_frame
        raw_v = px - previous_px
        if gap == 1 and 9.0 < min(px, previous_px) and max(px, previous_px) < 143.0:
            estimated = (raw_v-self.motor_drag*self.motor_raw_v)/max(1.2, self.motor_gain)
            self.motor_effective = min((-1.0, 0.0, 1.0), key=lambda u: abs(u-estimated))
        if gap == 1 and 12.0 < min(px, previous_px) and max(px, previous_px) < 140.0:
            v, u, y = self.motor_raw_v, self.motor_previous_input, raw_v
            if abs(y) <= 9.0:
                additions = [v*v, v*u, u*u, v*y, u*y]
                self.motor_fit = [0.997 * x + z for x, z in zip(self.motor_fit, additions)]
                self.motor_samples += 1
                vv, vu, uu, vy, uy = self.motor_fit
                det = vv*uu - vu*vu
                if self.motor_samples >= 35 and det > 12.0:
                    drag = (vy*uu - uy*vu) / det
                    gain = (uy*vv - vy*vu) / det
                    if 0.35 <= drag <= 0.85 and 1.2 <= gain <= 3.6:
                        self.motor_drag = 0.97*self.motor_drag + 0.03*drag
                        self.motor_gain = 0.97*self.motor_gain + 0.03*gain
        self.motor_raw_v = self._clamp(raw_v / max(1, gap), -9.0, 9.0)
        if gap > 1:
            # Unknown intervening mailbox repeats: do not fit that transition.
            self.motor_previous_input = self.motor_queued
        if frame % 600 == 0:
            self._log("motor_response_fit", frame=frame, samples=self.motor_samples,
                      drag=round(self.motor_drag, 3), gain=round(self.motor_gain, 3))

    def _legacy_steer_interval(self, obs, flight, frame):
        if bool(obs.get("terminated")) or bool(obs.get("truncated")):
            return NOOP
        if bool(obs.get("serve_required")):
            return FIRE if self._int(obs.get("lives")) > 0 else NOOP
        if self._ball(obs) is None:
            return NOOP
        px = self._num(self._paddle(obs).get("x"), 99.0)
        pw = self._num(self._paddle(obs).get("w"), 16.0)
        pmin, pmax = 8.0, max(8.0, 160.0-pw)
        low, high = flight["low"], flight["high"]
        center = self._clamp(flight.get("preferred_target", (low + high) / 2.0), low, high)
        eta = flight.get("eta")
        horizon = 10 if eta is None else max(2, min(10, int(math.ceil(eta)) + 2))
        best, best_cost = NOOP, float("inf")
        for action, pulse in ((NOOP, 0.0), (LEFT, -1.0), (RIGHT, 1.0)):
            # Model both the already queued input and this candidate's sticky
            # alternative. The previous effective input is inferred from the
            # latest displacement, never from the requested-action label.
            expected = worst = 0.0
            for queued, qweight in ((self.motor_queued, 0.75), (self.motor_effective, 0.25)):
                for applied, aweight in ((pulse, 0.75), (queued, 0.25)):
                    x, v, tail_input = px, self.motor_raw_v, applied
                    for step in range(horizon):
                        if step == 0:
                            u = queued
                        elif step == 1:
                            u = applied
                        else:
                            # Future release can itself be sticky. Its expected
                            # residual input decays geometrically, not instantly.
                            tail_input *= 0.25
                            u = tail_input
                        v = self.motor_drag*v + self.motor_gain*u
                        nx = self._clamp(x+v, pmin, pmax)
                        if nx == pmin and v < 0 or nx == pmax and v > 0:
                            v = 0.0
                        x = nx
                    distance = max(low-x, 0.0, x-high)
                    cost = 20.0*distance*distance + 0.05*(x-center)**2 + 0.20*v*v
                    expected += qweight*aweight*cost
                    worst = max(worst, distance*distance)
            total = expected + 2.0*worst + (0.12 if pulse else 0.0)
            if total < best_cost:
                best, best_cost = action, total
        return best

    def _clear_impact(self, reason, frame):
        if self.impact_lock is not None:
            self._log("impact_lock_clear", frame=frame, reason=reason,
                      phase=self.impact_lock.get("phase"),
                      age=frame-self.impact_lock.get("start", frame))
        self.impact_lock = None

    def _impact_action(self, obs, flight, frame):
        """Return a committed terminal action, or None before the lock."""
        ball = self._ball(obs)
        if ball is None or not bool(obs.get("ball_in_play", False)):
            self._clear_impact("no_ball", frame)
            return None

        # A post-paddle dy reversal is the authoritative end of the lock.
        # Until then, the interval is fixed but the actuator is observed: one
        # short committed pulse overcomes sticky input, then a hysteretic
        # terminal servo may continue that same direction or release.
        if self.impact_lock is not None:
            if self._num(ball.get("dy")) < 0.0:
                self._clear_impact("bounce_seen", frame)
                return None
            lock = self.impact_lock
            age = frame - lock.get("start", frame)
            if age > lock.get("max_age", 18):
                self._clear_impact("terminal_window_expired", frame)
                # Do not fall back to a newly moving target at y>=190.
                return NOOP
            if lock.get("phase") == "brake" and frame <= lock.get("commit_until", frame):
                return lock.get("brake", NOOP)
            if lock.get("phase") == "brake":
                lock["phase"] = "adaptive"
                self._log("impact_lock_adaptive", frame=frame,
                          interval=[lock.get("physical_low"), lock.get("physical_high")])

            low = self._num(lock.get("physical_low"), 61.0)
            high = self._num(lock.get("physical_high"), 91.0)
            px = self._num(self._paddle(obs).get("x"), 99.0)
            v = self._num(getattr(self, "motor_raw_v", 0.0))
            if abs(v) < 0.25:
                v = self._num(getattr(self, "paddle_speed", 0.0))
            eta_now = self._num(flight.get("eta"), 0.0)
            look = max(1.0, min(2.0, eta_now if eta_now > 0.0 else 1.0))
            forecast = px + v * look
            desired = NOOP
            if px < low-1.0 or forecast < low-1.0:
                desired = RIGHT
            elif px > high+1.0 or forecast > high+1.0:
                desired = LEFT
            elif abs(v) > 1.0:
                # Inside the set, damp only velocity that is carrying toward an
                # edge.  A direction toward the interval's interior is safe.
                if (px-(low+high)/2.0) * v > 0.0:
                    desired = LEFT if v > 0.0 else RIGHT

            previous = lock.get("last_action", NOOP)
            if desired != previous and previous in DIRECTIONS:
                # Do not mirror a one-frame measurement while still inside the
                # fixed set. A reversal is admitted when outside, or after two
                # frames of observed evidence.
                outside = px < low-1.0 or px > high+1.0
                if not outside and frame - lock.get("last_switch", frame) < 2:
                    desired = previous
            if desired != previous:
                lock["last_switch"] = frame
            lock["last_action"] = desired
            return desired

        if flight.get("phase") != "descending":
            return None
        eta_value = flight.get("eta")
        if eta_value is None:
            return None
        eta = self._num(eta_value, 99.0)
        lock_horizon = 10.0 if self.return_profile == "DAMP" else 8.0
        if eta > lock_horizon or eta < -0.5:
            return None

        paddle = self._paddle(obs)
        px = self._num(paddle.get("x"), 99.0)
        pw = self._num(paddle.get("w"), 16.0)
        pmin, pmax = 8.0, max(8.0, 160.0-pw)
        # Use the physical interval, not a shaped preferred endpoint.
        physical_low = self._num(flight.get("low"), 61.0)
        physical_high = self._num(flight.get("high"), 91.0)
        low = self._clamp(physical_low-2.0, pmin, pmax)
        high = self._clamp(physical_high+2.0, pmin, pmax)
        if px < low or px > high:
            return None

        raw_v = self._num(getattr(self, "motor_raw_v", 0.0))
        smooth_v = self._num(getattr(self, "paddle_speed", 0.0))
        if abs(raw_v) < 0.25:
            raw_v = smooth_v
        look = max(1.0, min(3.0, eta))
        forecast = px + raw_v * look
        center = (physical_low + physical_high) / 2.0
        brake = NOOP
        # Brake carry that is predicted to leave the physical overlap; do not
        # sign-flip merely because the interval moved by one RAM pixel.
        if forecast < physical_low-1.0 and raw_v < -0.45:
            brake = RIGHT
        elif forecast > physical_high+1.0 and raw_v > 0.45:
            brake = LEFT
        elif abs(raw_v) > 2.6 and (px-center) * raw_v > 0.0:
            brake = LEFT if raw_v > 0.0 else RIGHT

        # High carry needs only one committed correction; a second/third
        # correction is chosen from the observed fixed-set error below.
        commit_frames = 1 if abs(raw_v) >= 3.5 else 2
        max_age = max(14, int(math.ceil(max(0.0, eta))) + 8)
        self.impact_lock_count += 1
        self.impact_lock = {
            "start": frame, "phase": "brake" if brake != NOOP else "adaptive",
            "brake": brake, "commit_until": frame + commit_frames - 1,
            "max_age": max_age, "last_action": brake, "last_switch": frame,
            "low": round(low, 2), "high": round(high, 2),
            "physical_low": round(physical_low, 2),
            "physical_high": round(physical_high, 2),
            "eta": round(eta, 2), "paddle_x": round(px, 2),
            "paddle_v": round(raw_v, 2),
        }
        self._log("impact_lock", frame=frame, phase=self.impact_lock["phase"],
                  brake=brake, interval=[round(low, 2), round(high, 2)],
                  physical=[round(physical_low, 2), round(physical_high, 2)],
                  eta=round(eta, 2), paddle_x=round(px, 2),
                  paddle_v=round(raw_v, 2), lock_id=self.impact_lock_count)
        return brake

    def _steer_interval(self, obs, flight, frame):
        if bool(obs.get("terminated")) or bool(obs.get("truncated")):
            return NOOP
        if bool(obs.get("serve_required")):
            self._clear_impact("serve", frame)
            return FIRE if self._int(obs.get("lives")) > 0 else NOOP
        locked = self._impact_action(obs, flight, frame)
        if locked is not None:
            return locked
        return self._legacy_steer_interval(obs, flight, frame)

    def _one(self, game, realtime):
        obs = self._observe(game)
        frame = self._frame(obs)
        event = self._event(obs)
        self._update_motor(obs, frame)
        self._update_paddle_motion(obs)
        self._confirm_outgoing(obs, frame)
        self._contact_event(obs, event, frame)
        self._apply_event(obs, event, frame)
        flight = self._flight(obs, event, frame)
        self._models(obs, flight, event, frame)
        motor_flight = self._shape_flight(obs, flight, frame)
        action = self._steer_interval(obs, motor_flight, frame)
        if self.trial is not None and motor_flight is not flight and self.impact_lock is None:
            # Diagnostic comparison is skipped once the terminal contract owns
            # the actuator; calling a second motor would mutate its lease.
            baseline_action = self._legacy_steer_interval(obs, flight, frame)
            if baseline_action != action:
                self._log("contact_objective_changed_action", frame=frame, contact=self.contact_id,
                          action=action, survival_action=baseline_action,
                          side=self.trial["side"], source=self.trial["source"])
        if action not in ACTIONS:
            action = NOOP
        ball = self._ball(obs)
        if ball is not None and self._num(ball.get("y")) >= 162:
            sample = {"frame": frame, "ball": dict(ball), "px": self._paddle(obs).get("x"),
                      "paddle_v": round(self.paddle_speed, 2), "action": action,
                      "interval": [flight["low"], flight["high"]], "eta": flight["eta"],
                      "filtered_v": [self.filtered_dx, self.filtered_dy],
                      "shape": None if self.trial is None else self.trial["side"]}
            self.contact_window.append(sample)
            self.contact_window = self.contact_window[-32:]
            self._log("contact_tick", **sample)
        self.motor_previous_input = self.motor_queued
        self.motor_queued = 1.0 if action == RIGHT else -1.0 if action == LEFT else 0.0
        self.last_requested = action
        self.last_action_frame = frame
        try:
            self.ctx.act(action)
            self.previous = obs
            self.last_frame = frame
            if realtime:
                self.ctx.wait_for_tick()
        except (BudgetExceeded, EpisodeOver):
            return False
        return True



class Harness(ReferenceController):
    """Separate numerical transition identification from strategic evidence review."""

    def __init__(self, ctx):
        super().__init__(ctx)
        self.native_enabled = True
        self.wall_evidence = []
        self.wall_checks = 0
        self.wall_failures = 0
        self.native_last_log = -10000
        self.review_future = None
        self.review_tag = None
        self.review_frame = -10000
        self.review_next = 60
        self.review_serial = 0
        self.reviewed_wall_checks = 0
        self.strategy = "MEASURED"
        self.strategy_revision = 0
        self.strategy_wall = 0
        self.strategy_until = -1
        self.slow_future = None
        self.slow_tag = None
        self.slow_keys = set()
        self.slow_calls = 0
        self.slow_prior = "MEASURED"
        self.bridge_action_changes = 0
        self.diagonal_trials = 0
        self.diagonal_cooldown = 0
        self.diagonal_observed = []
        self.diagonal_pending = False

    @staticmethod
    def _native_projection(x, dx, frames):
        """Measured discrete side rule, not a claim about private ROM state.

        Observed integer-step flights visit 152 before reversing to 150, and
        6 before reversing to 8. The outbound threshold is crossed on a native
        frame; it is NOT a continuously reflecting mathematical boundary.
        Only call this for a stable integer horizontal displacement.
        """
        remaining = max(0.0, min(240.0, frames))
        turns = 0
        while remaining > 1e-9:
            if (x < 8.0 and dx < 0.0) or (x > 150.0 and dx > 0.0):
                dx = -dx
                turns += 1
            part = min(1.0, remaining)
            x += part * dx
            remaining -= part
        return x, turns

    def _observe_wall_law(self, obs, frame):
        old = self._ball(self.previous)
        ball = self._ball(obs)
        if old is None or ball is None or frame != self.last_frame + 1:
            return
        x = self._num(old.get("x"))
        dx = self._num(old.get("dx"))
        dy = self._num(old.get("dy"))
        y = self._num(old.get("y"))
        # Avoid bricks, paddle-angle changes, and quantized noninteger speeds.
        if not (96 <= y <= 180 and dy > 0 and self._num(ball.get("dy")) > 0):
            return
        if abs(dx) not in (1.0, 2.0, 3.0) or not (x <= 10 or x >= 148):
            return
        if len(self.cadence_x) < 4 or any(v != dx for v in self.cadence_x[-4:]):
            return
        native, _ = self._native_projection(x, dx, 1.0)
        continuous = self._reflect_x(x, dx, 1.0)
        if abs(native - continuous) < 0.5:
            return
        actual = self._num(ball.get("x"))
        a, b = abs(native - actual), abs(continuous - actual)
        self.wall_evidence.append([a, b])
        self.wall_evidence = self.wall_evidence[-24:]
        self.wall_checks += 1
        self.wall_failures = self.wall_failures + 1 if a > 2.0 else 0
        self._log("native_wall_check", frame=frame, previous_x=x, dx=dx,
                  observed=actual, native=native, continuous=continuous,
                  native_error=a, continuous_error=b)
        if self.wall_failures >= 3:
            self.native_enabled = False
            self._log("native_wall_suspended", frame=frame,
                      reason="three_consecutive_observed_contradictions")

    def _make_return_contract(self, obs, frame):
        made = super()._make_return_contract(obs, frame)
        if not made or not self.native_enabled:
            return made
        ball = self._ball(obs) or {}
        dx = self.filtered_dx
        # Preserve the verified parent for fractional/uncertain cadence and
        # all trajectories that do not encounter a side wall.
        if (len(self.cadence_x) < 4 or abs(dx) not in (1.0, 2.0, 3.0)
                or any(v != dx for v in self.cadence_x[-4:])):
            return made
        eta = self.contract_eta
        if eta is None:
            return made
        bx = self._num(ball.get("x"))
        native, turns = self._native_projection(bx, dx, eta)
        old_landing = self.contract_landing
        bw = self._num(ball.get("w"), 2.0)
        correction = native + bw / 2.0 - old_landing
        if abs(correction) < 0.05:
            return made
        paddle = self._paddle(obs)
        pw = self._num(paddle.get("w"), 16.0)
        low = self._clamp(native + 3.0 - pw, 8.0, 160.0 - pw)
        high = self._clamp(native + bw - 3.0, 8.0, 160.0 - pw)
        if high < low:
            low = high = self._clamp(native + bw / 2.0 - pw / 2.0, 8.0, 160.0 - pw)
        self.contract_low, self.contract_high = low, high
        self.contract_center = (low + high) / 2.0
        self.contract_landing = native + bw / 2.0
        if frame - self.native_last_log >= 15 or eta <= 8:
            self.native_last_log = frame
            self._log("native_landing", frame=frame, contact=self.contact_id,
                      continuous=round(old_landing, 3), native=round(self.contract_landing, 3),
                      correction=round(correction, 3), turns=turns, eta=round(eta, 3),
                      interval=[round(low, 3), round(high, 3)])
        return made

    def _diagonal_regime(self):
        return (1.75 <= abs(self.filtered_dx) <= 2.25 and
                1.75 <= abs(self.filtered_dy) <= 2.25 and self.velocity_warm)

    def _diagonal_probe_eligible(self, obs):
        # This is an experiment, not reuse of the high-speed angle table.
        return (self._diagonal_regime() and self._int(obs.get("lives")) >= 2
                and 0 < self._int(obs.get("bricks_remaining"), 108) <= 12
                and self._frame(obs) - self.last_progress_frame >= 900
                and self.rally_bounces >= 3 and self.diagonal_trials < 4
                and self.diagonal_cooldown == 0 and self.trial_cooldown == 0)

    def _contact_event(self, obs, event, frame):
        was_diagonal = self._diagonal_regime()
        trial = self.trial
        super()._contact_event(obs, event, frame)
        if event == "wall_transition":
            self.diagonal_trials = 0
            self.diagonal_cooldown = 0
        if event == "paddle_bounce":
            if self.pending_contact is not None:
                self.pending_contact["diagonal_regime"] = was_diagonal
            if trial is not None and trial.get("regime") == "diagonal_probe":
                self.diagonal_cooldown = 3
            elif self.diagonal_cooldown:
                self.diagonal_cooldown -= 1
        elif event in ("serve", "life_loss", "ball_lost"):
            self.diagonal_cooldown = 0
            self.strategy = "MEASURED"
            self.strategy_until = -1
            self.slow_prior = "MEASURED"

    def _confirm_outgoing(self, obs, frame):
        pending = self.pending_contact
        super()._confirm_outgoing(obs, frame)
        if pending is None or self.pending_contact is not None or not pending.get("diagonal_regime"):
            return
        samples = pending.get("samples", [])
        if len(samples) < 3:
            return
        if (pending.get("stable_paddle") and 20 <= pending.get("x", 0) <= 138
                and 2 < abs(pending["offset"]) <= 6):
            vx = sum(v[0] for v in samples) / len(samples)
            vy = sum(v[1] for v in samples) / len(samples)
            self.diagonal_observed.append([pending["offset"], vx, vy])
            self.diagonal_observed = self.diagonal_observed[-16:]
            self._log("diagonal_contact_evidence", frame=frame, offset=pending["offset"],
                      outgoing=[round(vx, 3), round(vy, 3)],
                      experiment=pending.get("trial") is not None)

    def _calibration_stage(self, obs):
        eligible = super()._calibration_stage(obs)
        # A new speed regime requires fresh Jev authorization. A delayed or
        # absent model does not enable an unverified low-life experiment.
        authorized = (self.strategy == "PROBE" and self._frame(obs) <= self.strategy_until)
        diagonal = authorized and self._diagonal_probe_eligible(obs)
        hold = self.strategy == "PRESERVE" and self._frame(obs) <= self.strategy_until
        return (eligible or diagonal) and not hold

    def _shape_flight(self, obs, flight, frame):
        prior_trial = self.trial
        diagonal = (self._diagonal_probe_eligible(obs) and self.strategy == "PROBE"
                    and frame <= self.strategy_until)
        shaped = super()._shape_flight(obs, flight, frame)
        if diagonal and prior_trial is None and self.trial is not None:
            self.trial["regime"] = "diagonal_probe"
            self.trial["source"] = "jev_regime_probe"
            self.diagonal_trials += 1
            self._log("diagonal_probe_started", frame=frame, contact=self.contact_id,
                      trial=self.diagonal_trials, side=self.trial["side"],
                      lives=self._int(obs.get("lives")),
                      observed_diagonal_contacts=len(self.diagonal_observed))
        if (self.trial is not None and self.trial.get("regime") == "diagonal_probe"
                and flight.get("phase") == "descending"
                and self._int(obs.get("lives")) >= 2 and self._diagonal_regime()):
            low, high = flight["low"], flight["high"]
            target = (low + high) / 2.0 + (3.0 if self.trial["endpoint"] == "high" else -3.0)
            target = self._clamp(target, low + 1.0, high - 1.0)
            shaped = dict(flight)
            if self._num(flight.get("eta")) > 14:
                shaped["low"], shaped["high"] = max(low, target-1.5), min(high, target+1.5)
                shaped["center"] = target
            else:
                shaped["preferred_target"] = target
            return shaped
        return shaped

    def _cleanup_stage(self, obs):
        eligible = super()._cleanup_stage(obs)
        hold = self.strategy == "PRESERVE" and self._frame(obs) <= self.strategy_until
        return eligible and not hold

    def _evidence_state(self, obs, flight, event):
        n = len(self.wall_evidence)
        native_error = sum(v[0] for v in self.wall_evidence) / max(1, n)
        baseline_error = sum(v[1] for v in self.wall_evidence) / max(1, n)
        verdict = ("insufficient" if n < 3 else
                   "corroborated" if native_error <= 0.5 and baseline_error >= native_error + 1.0 else
                   "contradicted" if native_error > 2.0 and native_error > baseline_error else "mixed")
        frame_stall = self._frame(obs) - self.last_progress_frame
        sparse = 0 < self._int(obs.get("bricks_remaining"), 108) <= 12
        measured = len(self.fast_speeds) >= 2
        regime_measured = ((self._diagonal_regime() and len(self.diagonal_observed) >= 2)
                           or (not self._diagonal_regime() and measured))
        probe_eligible = (bool(super()._calibration_stage(obs)) or
                          bool(self._diagonal_probe_eligible(obs)))
        return {
            "task": "Atari 2600 Breakout: preserve catches, then clear both walls for 864 points",
            "rules": {
                "models": "review evidence and future experiment policy, never control a native-frame action",
                "geometry": "code compares two predictions with actual next positions; choose only the supplied verdict, not arithmetic",
                "cleanup": "MEASURED uses supported endpoint rays; PROBE allows an existing bounded interior calibration only with spare lives",
                "safety": "no review can override full physical catch interval, life gates, cooldown or impact motor",
            },
            "observation": {
                "event": event, "frame": self._frame(obs),
                "wall": self._int(obs.get("wall_index")), "lives": self._int(obs.get("lives")),
                "phase": flight["phase"], "urgency": flight["urgency"],
                "progress": "long_stall" if frame_stall >= 1200 else "stalled" if frame_stall >= 420 else "scoring",
                "wall_density": "sparse" if sparse else "dense",
                "geometry_evidence": verdict, "native_predictor_active": self.native_enabled,
                "endpoint_evidence": "measured" if regime_measured else "missing",
                "speed_regime": "diagonal" if self._diagonal_regime() else "other",
                "calibration_eligible": probe_eligible,
                "cleanup_eligible": bool(super()._cleanup_stage(obs)),
                "last_trial": self.last_trial_outcome,
                "recent_catches": "several" if self.rally_bounces >= 3 else "few",
                "current_policy": self.strategy, "astra_prior": self.slow_prior,
                "candidate_side": self._candidate(obs, flight) or "none",
            },
            "evidence": {"samples": n, "native_mae": round(native_error, 3),
                         "continuous_mae": round(baseline_error, 3)},
        }

    def _review_questions(self):
        choices = {
            "MEASURED": "Continue supported cleanup rays and ordinary catch-first play; no unsupported direction claims.",
            "PROBE": "A sparse wall is stalled, endpoint evidence is missing, and calibration_eligible is true: permit the code's bounded calibration.",
            "PRESERVE": "Defer calibration/cleanup for one short review window after an unresolved miss or unsupported evidence; keep catching.",
            "NO_CHANGE": "Ambiguous, scoring normally, or no justified policy revision; retain the executable fallback.",
        }
        keys = list(choices)
        shift = self.review_serial % len(keys)
        order = keys[shift:] + keys[:shift]
        text = ("Select an experiment policy, not a direction or frame action. Use observation.progress, "
                "wall_density, endpoint_evidence, calibration_eligible, cleanup_eligible and recent_catches. "
                "PROBE requires explicitly eligible missing calibration on a stalled sparse wall. "
                "MEASURED is appropriate with supported cleanup or ordinary catch-first play. "
                "A historical miss is not an unresolved failure after several catches. "
                "Astra is a prior, not permission to bypass code's safety gates. If evidence is ambiguous choose NO_CHANGE.")
        geometry = {
            "ACCEPT": "geometry_evidence is corroborated: actual observations favor native-step wall crossing.",
            "REJECT": "geometry_evidence is contradicted: actual observations refute the native-step hypothesis.",
            "UNKNOWN": "geometry_evidence is insufficient or mixed; do not change the numerical predictor.",
        }
        return {
            "policy_a": Choice(text, {k: choices[k] for k in order}),
            "policy_b": Choice(text, {k: choices[k] for k in reversed(order)}),
            "geometry": Choice("Report the supplied observation.geometry_evidence. Do not calculate errors or infer hidden physics.", geometry),
        }

    def _models(self, obs, flight, event, frame):
        self._observe_wall_law(obs, frame)
        wall = self._int(obs.get("wall_index"))
        epoch = (self.model_epoch, wall)
        if self.strategy_wall != wall:
            self.strategy, self.slow_prior = "MEASURED", "MEASURED"
            self.strategy_until = -1
            self.strategy_wall = wall
        if self.review_future is not None and self.review_future.done():
            future, self.review_future = self.review_future, None
            try:
                answers = future.result()
                if not isinstance(answers, dict):
                    answers = {}
                a, b = answers.get("policy_a"), answers.get("policy_b")
                av, bv = self._answer_value(a), self._answer_value(b)
                p = min(self._answer_probability(a, av), self._answer_probability(b, bv))
                fresh = self.review_tag == (epoch, self.strategy_revision) and frame - self.review_frame <= 180
                current = self._evidence_state(obs, flight, event)["observation"]
                if fresh and av == bv and av in ("MEASURED", "PROBE", "PRESERVE") and p >= 0.65:
                    if av != "PROBE" or current["calibration_eligible"]:
                        old = self.strategy
                        self.strategy = av
                        self.strategy_until = frame + 360
                        self._log("jev_experiment_policy", frame=frame, policy=av,
                                  previous=old, probability=round(p, 3), expires=self.strategy_until)
                g = answers.get("geometry")
                gv = self._answer_value(g)
                if fresh and self._answer_probability(g, gv) >= 0.80:
                    if (gv == "ACCEPT" and current["geometry_evidence"] == "corroborated"
                            and self.wall_failures == 0):
                        self.native_enabled = True
                    elif gv == "REJECT" and current["geometry_evidence"] == "contradicted":
                        self.native_enabled = False
                    self._log("jev_geometry_review", frame=frame, verdict=gv,
                              code_verdict=current["geometry_evidence"], active=self.native_enabled)
            except Exception as exc:
                self._log("review_unavailable", error=type(exc).__name__)
        if self.slow_future is not None and self.slow_future.done():
            future, self.slow_future = self.slow_future, None
            try:
                result = future.result()
                token = str(getattr(result, "text", "")).strip().upper().strip('` .\n\r\t')
                if token in ("MEASURED", "PROBE", "PRESERVE") and self.slow_tag == epoch:
                    self.slow_prior = token
                    self.strategy_revision += 1
                    self.review_next = min(self.review_next, frame)
                    self._log("astra_experiment_prior", frame=frame, policy=token,
                              revision=self.strategy_revision)
            except Exception as exc:
                self._log("astra_review_unavailable", error=type(exc).__name__)
        stalled = frame - self.last_progress_frame >= 420
        sparse = 0 < self._int(obs.get("bricks_remaining"), 108) <= 12
        boundary = event in ("serve", "life_loss", "wall_transition", "paddle_bounce")
        new_evidence = self.wall_checks >= self.reviewed_wall_checks + 4
        should_review = new_evidence or (boundary and sparse and stalled)
        if (self.jev_enabled and self.review_future is None and frame >= self.review_next
                and should_review and self._ball(obs) is not None and not self.ctx.done):
            rem = self.ctx.budget.remaining()
            if rem.get("system1_calls", 0) > 2 and rem.get("wall_s", 0) > 3.0:
                state = self._evidence_state(obs, flight, event)
                questions = self._review_questions()
                self.review_serial += 1
                self.review_frame, self.review_tag = frame, (epoch, self.strategy_revision)
                self.review_next = frame + 180
                self.reviewed_wall_checks = self.wall_checks
                self.review_future = self.ctx.spawn(
                    lambda s=state, q=questions: self.ctx.decide(s, q, timeout_s=2.0))
        # Astra is called only when a wall stalls, not for every serve or brake.
        # One initial review and one changed-evidence review per wall suffice.
        key = (wall, "measured" if len(self.fast_speeds) >= 2 else "unmeasured")
        if (sparse and stalled and self.slow_future is None and key not in self.slow_keys
                and self.slow_calls < 4 and self._ball(obs) is not None and not self.ctx.done):
            rem = self.ctx.budget.remaining()
            if rem.get("system2_calls", 0) > 1 and rem.get("wall_s", 0) > 100:
                state = self._evidence_state(obs, flight, event)
                prompt = (
                    "Review a stalled ordinary Atari 2600 Breakout wall. Return exactly ONE token: "
                    "MEASURED, PROBE, or PRESERVE. Code controls every 60-Hz action and tests numerical "
                    "side-wall predictions against actual observations. Do not change its motor. "
                    "MEASURED favors supported contact-to-direction cleanup rays. PROBE favors the "
                    "bounded existing interior calibration when code says eligible and evidence is missing. "
                    "PRESERVE temporarily defers experiments after an unresolved failure. A historical "
                    "miss followed by several catches is not unresolved. Jev will receive your token "
                    "with current evidence and two reordered typed policy Choices; both must agree "
                    "with probability >=0.65, within the same life/wall and 180 frames. Your token is "
                    "only a prior, not an action or authorization to relax life/overlap gates. "
                    "No new numerical angle mapping is established by a requested but unreached endpoint. "
                    "State: " + json.dumps(state, separators=(",", ":")))
                self.slow_keys.add(key)
                self.slow_calls += 1
                self.slow_tag = epoch
                self.slow_future = self.ctx.spawn(lambda p=prompt: self.ctx.think(p, timeout_s=90.0))


def build(ctx):
    return Harness(ctx)
