"""Evidence-driven Breakout control, candidate 0018.

IntervalFallback retains the parent's tested home motor and geometry helpers.
Harness adds clean-flight displacement tracking, measured command-lag dynamics,
short sequence search, and actual contact-outcome records. Jev audits an
empirical contact band; Astra revises that audit's evidence requirements.
Neither model is on the native-frame action path or emits a frame action.
"""
from __future__ import annotations

import math
import re
import json
import statistics
import itertools
from collections import deque
from typing import Any

from metaharness.core.contract import BudgetExceeded, EpisodeOver
from metaharness.models.jev import Choice, Noul


NOOP = "NOOP"
FIRE = "FIRE"
RIGHT = "RIGHT"
LEFT = "LEFT"
ACTIONS = (NOOP, FIRE, RIGHT, LEFT)
DIRECTIONS = (RIGHT, LEFT)


class IntervalFallback:
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
        eta = max(0.0, (py - bh - by) / max(0.25, dy))
        landing_left = self._reflect_x(bx, dx, eta)

        # Do not extrapolate through a still-occupied brick band.  The lower
        # row ends just below 90 in this decoder; 94 leaves a small collision
        # ordering guard.  Once below it, the reflected path is deterministic.
        if by < 94.0:
            return False

        speed = abs(dx) + abs(dy)
        margin = 1.0 if speed < 4.0 else 0.5
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
            eta = max(0.0, (py - bh - by) / max(0.25, dy))

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


class Harness(IntervalFallback):
    """Empirical contact certificates, with models reviewing evidence not actions."""

    def __init__(self, ctx):
        super().__init__(ctx)
        self.samples = deque(maxlen=64)
        self.contact_window = deque(maxlen=24)
        self.pending_contact = None
        self.crossing = None
        self.last_life = None
        self.last_wall = None
        self.hit_count = 0
        self.last_progress_frame = 0
        self.safety_epoch = 0
        self.profile = "CORE"
        self.plan = "VERIFY_COVERAGE"
        self.plan_version = 0
        self.audit_future = None
        self.audit_stamp = None
        self.audit_due = False
        self.audit_serial = 0
        self.review_future = None
        self.review_stamp = None
        self.review_count = 0
        self.review_due = False
        self.last_review_frame = -10000
        self.last_diagnostic = None
        self.noop_steps = deque(maxlen=80)
        self.left_steps = deque(maxlen=80)
        self.right_steps = deque(maxlen=80)
        self.raw_motion = 0.0
        self.last_post = NOOP
        self.pre_post = NOOP
        self.last_landing = None
        self.last_audit_frame = -1000
        self.next_model_frame = 0
        self.transitions = {}
        self.transition_medians = {}
        self.sequences = tuple(itertools.product((NOOP, LEFT, RIGHT), repeat=4))
        self.motor_coef = [.65, .3, 2.2]
        self.motor_xx = [[50., 0., 0.], [0., 150., 0.], [0., 0., 150.]]
        self.motor_xy = [32.5, 45., 330.]
        self.motor_samples = 0
        self.flight_samples = deque(maxlen=9)
        self.stable_velocity_frame = -1
        self.stable_velocity = (0., 0., False)
        self.geometry_epoch = 0

    def _capture(self, obs, frame, event):
        """Observe contact outcomes; do not label an attempted return as a catch."""
        ball = self._ball(obs)
        px = self._num(self._paddle(obs).get("x"), 99)
        if event in ("start", "serve", "brick_hit", "paddle_bounce", "side_bounce", "ball_lost", "life_loss", "wall_transition"):
            self.flight_samples.clear()
            self.stable_velocity_frame = -1
        if event == "wall_transition":
            self.geometry_epoch += 1
        old = self.previous
        if old is not None:
            opx = self._num(self._paddle(old).get("x"), px)
            gap = frame - self.last_frame
            if gap == 1:
                prior_velocity = self.raw_motion
                self.raw_motion = px - opx
                if 9 < px < 142 and abs(self.raw_motion) <= 8:
                    u = lambda a: 1.0 if a == RIGHT else -1.0 if a == LEFT else 0.0
                    features = [prior_velocity, u(self.last_post), u(self.pre_post)]
                    for i in range(3):
                        self.motor_xy[i] += features[i]*self.raw_motion
                        for j in range(3):
                            self.motor_xx[i][j] += features[i]*features[j]
                    self.motor_samples += 1
                    if self.motor_samples % 120 == 0:
                        self._fit_motor(frame)
                hist = {NOOP: self.noop_steps, LEFT: self.left_steps,
                        RIGHT: self.right_steps}.get(self.last_post)
                # Samples from held inputs disentangle ordinary response from
                # sticky transitions; endpoints do not teach a zero motor speed.
                if hist is not None and self.pre_post == self.last_post:
                    if 10 < px < 140 or self.last_post == NOOP:
                        hist.append(self.raw_motion)
        if event == "brick_hit":
            self.last_progress_frame = frame
        if event in ("life_loss", "wall_transition"):
            self.safety_epoch += 1
            self.crossing = None
            self.contact_window.clear()
            self.hit_count = 0
            self.profile = "CORE"
            self.audit_due = True
        if ball is not None:
            by = self._num(ball.get("y"))
            if self._num(ball.get("dy")) > 0 and by >= 156:
                row = {
                    "frame": frame, "x": self._num(ball.get("x")),
                    "y": by, "px": px, "dx": self._num(ball.get("dx")),
                    "dy": self._num(ball.get("dy")), "action": self.last_post,
                    "motion": self.raw_motion,
                    "offset": self._num(ball.get("x")) + 1 - px,
                    "predicted": self.last_landing,
                }
                self.contact_window.append(row)
                if by <= 188 and (self.crossing is None or abs(by-186) < abs(self.crossing["y"]-186)):
                    self.crossing = row
        if event == "paddle_bounce":
            self.hit_count += 1
            contact = self.crossing
            if contact is not None and frame - contact["frame"] <= 4:
                self._outcome(obs, frame, True, contact)
            self.crossing = None
            self.contact_window.clear()
        elif event == "ball_lost":
            if self.crossing is not None:
                self._outcome(obs, frame, False, self.crossing)
            self.crossing = None
            self.contact_window.clear()

    def _outcome(self, obs, frame, caught, crossing):
        # A stable observed crossing is a measurement of coverage, not a
        # language-model guess. Store both misses and catches with provenance.
        rows = [r for r in self.contact_window if r["frame"] <= crossing["frame"]]
        offset = crossing["offset"]
        drift = max((abs(r["motion"]) for r in rows[-5:]), default=0)
        pred = crossing.get("predicted")
        residual = None if pred is None else abs(crossing["x"] + 1 - pred)
        inside = 1 <= offset <= self._num(self._paddle(obs).get("w"), 16) - 1
        evidence = {
            "frame": frame, "caught": caught, "offset": round(offset, 2),
            "y": crossing["y"], "drift": round(drift, 2),
            "prediction_error": None if residual is None else round(residual, 2),
            "inside_reported_rectangle": inside,
            "stable_paddle": drift <= 1,
            "prediction_supported": residual is not None and residual <= 3,
            "wall": self._int(obs.get("wall_index")),
            "life": self._int(obs.get("lives")),
        }
        self.samples.append(evidence)
        self._log("contact_evidence", **evidence, window=rows[-8:])
        if not caught:
            # Fallback is conservative immediately; the audit is never awaited.
            self.profile = "CORE"
            self.audit_due = True
            self.review_due = self.review_count > 0
        elif self.hit_count in (3, 8) or frame - self.last_audit_frame > 900:
            self.audit_due = True

    def _velocity(self, obs):
        frame = self._frame(obs)
        if frame == self.stable_velocity_frame:
            return self.stable_velocity
        dx, dy, valid = super()._velocity(obs)
        ball = self._ball(obs)
        if ball is not None and valid:
            raw_dx = self._num(ball.get("dx"))
            raw_dy = self._num(ball.get("dy"))
            x, y = self._num(ball.get("x")), self._num(ball.get("y"))
            if self.flight_samples:
                prior = self.flight_samples[-1]
                if (frame-prior[0] != 1 or raw_dx*prior[3] < 0 or raw_dy*prior[4] < 0):
                    self.flight_samples.clear()
            self.flight_samples.append((frame, x, y, raw_dx, raw_dy))
            if len(self.flight_samples) >= 3:
                prior = self.flight_samples[0]
                elapsed = frame-prior[0]
                dx, dy = (x-prior[1])/elapsed, (y-prior[2])/elapsed
                self.velocity_warm = True
            self.filtered_dx, self.filtered_dy = dx, dy
        else:
            self.flight_samples.clear()
        self.stable_velocity_frame = frame
        self.stable_velocity = (dx, dy, valid)
        return self.stable_velocity

    def _contact_band(self, obs):
        """A certifiable inner overlap band, rather than any rectangle overlap.

        The decoder's width is a reported bound, not a verified collision mask.
        Start in an interior left-of-center region; only broaden using measured
        stationary successful contacts and a current model evidence audit.
        """
        pw = self._num(self._paddle(obs).get("w"), 16)
        low, high = 2.0, min(8.0, pw - 2)
        if self.profile == "MEASURED":
            good = [s["offset"] for s in self.samples
                    if s["caught"] and s["stable_paddle"] and s["prediction_supported"]]
            if len(good) >= 6:
                good.sort()
                low = max(1.5, min(3.0, good[len(good)//5]))
                high = min(pw-1.5, max(7.0, good[4*len(good)//5]))
        if high < low:
            low = high = max(1.0, pw / 2)
        return low, high

    def _make_return_contract(self, obs, frame):
        ok = super()._make_return_contract(obs, frame)
        if not ok:
            return False
        b = self._ball(obs)
        dx, dy, _ = self._velocity(obs)
        # Archived observations show successful reversal after y=186, rather
        # than immediately at the rectangle-overlap plane y=185.
        contact_y = self._num(self._paddle(obs).get("y"), 189) - 3.0
        eta = max(0.0, (contact_y - self._num(b.get("y"))) / max(.25, dy))
        land = self._reflect_x(self._num(b.get("x")), dx, eta) + 1.0
        loff, hoff = self._contact_band(obs)
        pw = self._num(self._paddle(obs).get("w"), 16)
        self.contract_low = self._clamp(land-hoff, 8.0, 160.0-pw)
        self.contract_high = self._clamp(land-loff, 8.0, 160.0-pw)
        self.contract_center = (self.contract_low+self.contract_high)/2
        self.contract_landing = land
        self.contract_eta = eta
        self.last_landing = land
        return True

    def _motor(self, obs, flight, frame):
        if obs.get("serve_required"):
            return FIRE if self._int(obs.get("lives")) > 0 else NOOP
        if self._ball(obs) is None:
            return NOOP
        if flight.get("phase") != "descending":
            return super()._steer_interval(obs, flight, frame)
        px = self._num(self._paddle(obs).get("x"), 99)
        low, high = self.contract_low, self.contract_high
        target = (low+high)/2
        eta = max(0.0, self._num(self.contract_eta))
        # Reachability invariant: a fitted zero-displacement startup cannot
        # make a directional action an absorbing NOOP state. Motor calibration
        # may include sticky/delayed starts, but sustained input must still be
        # tested when the measured paddle is far outside the current catch set.
        if px < low - 14 and self.raw_motion < 4.0:
            return RIGHT
        if px > high + 14 and self.raw_motion > -4.0:
            return LEFT
        # Four-step receding-horizon search through empirical velocity-conditioned
        # transitions.  In particular, NOOP may preserve motion: this is observed
        # in the local first probe and is never assigned an automatic brake.
        if eta > 7:
            horizon = 4
        else:
            horizon = max(1, min(4, int(math.ceil(eta)) + 1))
        sequences = self.sequences if horizon == 4 else itertools.product((NOOP, LEFT, RIGHT), repeat=horizon)
        best, best_cost = NOOP, float("inf")
        for seq in sequences:
            total = 0.0
            # The second branch includes a stale/sticky first input. Both are
            # replanned next frame; no open-loop action sequence is posted.
            for delayed, weight in ((False, .75), (True, .25)):
                x, v = px, self.raw_motion
                previous_action = self.last_post if self.last_post in (NOOP, LEFT, RIGHT) else NOOP
                path_cost = 0.0
                for i, action in enumerate(seq):
                    applied = previous_action if delayed and i == 0 else action
                    v = self._transition(v, applied, previous_action)
                    next_x = self._clamp(x+v, 8, 144)
                    v = next_x - x
                    x = next_x
                    path_cost += .012 * v*v + .1*max(low-x, 0, x-high)**2 + (.15 if action != NOOP else 0)
                    previous_action = action
                # The last command is still in the actuator pipeline. Score
                # its predicted neutral continuation, not only visible speed.
                release_v = self._transition(v, NOOP, previous_action)
                if eta > horizon + 2:
                    x = self._clamp(x + release_v + .65*release_v, 8, 144)
                distance = max(low-x, 0.0, x-high)
                # Zero terminal velocity matters: overlap reached at five
                # pixels/frame is not a viable contact certificate.
                total += weight * (4*distance*distance + 1.2*v*v + 1.8*release_v*release_v +
                                   path_cost)
            if seq[0] != NOOP:
                total += .2
            if total < best_cost:
                best, best_cost = seq[0], total
        return best

    def _transition(self, velocity, action, prior_action):
        u = 1 if action == RIGHT else -1 if action == LEFT else 0
        old_u = 1 if prior_action == RIGHT else -1 if prior_action == LEFT else 0
        a, b, c = self.motor_coef
        return self._clamp(a*velocity + b*u + c*old_u, -5.8, 5.8)

    def _fit_motor(self, frame):
        # Ridge-regularized three-variable least squares from observed effects
        # of posted actions. This estimates actuator lag rather than treating
        # delayed zero displacement as an absorbing state.
        matrix = [self.motor_xx[i][:] + [self.motor_xy[i]] for i in range(3)]
        for i in range(3):
            pivot = max(range(i, 3), key=lambda k: abs(matrix[k][i]))
            matrix[i], matrix[pivot] = matrix[pivot], matrix[i]
            d = matrix[i][i]
            if abs(d) < 1e-8:
                return
            matrix[i] = [v/d for v in matrix[i]]
            for j in range(3):
                if j == i:
                    continue
                d = matrix[j][i]
                matrix[j] = [v-d*w for v,w in zip(matrix[j],matrix[i])]
        coef = [matrix[i][3] for i in range(3)]
        if .25 < coef[0] < .95 and -.6 < coef[1] < 1.4 and .8 < coef[2] < 3.8:
            self.motor_coef = coef
            self._log("motor_identified", frame=frame, coefficients=coef,
                      samples=self.motor_samples)

    def _facts(self, obs, frame):
        recent = list(self.samples)[-12:]
        misses = [s for s in recent if not s["caught"]]
        coverage = any(s["inside_reported_rectangle"] and s["stable_paddle"]
                       and s["prediction_supported"] for s in misses)
        motion = any(not s["stable_paddle"] for s in misses)
        projection = any(not s["prediction_supported"] for s in misses)
        good = [s for s in recent if s["caught"] and s["stable_paddle"]
                and s["prediction_supported"]]
        minimum = 8 if self.plan == "VERIFY_COVERAGE" else 6
        broad = len(good) >= minimum and not misses
        return {
            "game": "Atari 2600 Breakout, ordinary mode 0, native 60 Hz; sticky actions 25%",
            "policy": {"astra_review": self.plan, "active_certificate": self.profile,
                       "fallback": "CORE: interior overlap, code-owned measured motor"},
            "evidence": {"coverage_contradiction": coverage,
                         "motor_motion_at_miss": motion,
                         "projection_error_at_miss": projection,
                         "repeatable_stationary_successes": broad,
                         "no_recent_misses": not misses,
                         "progress_stalled": frame-self.last_progress_frame > 1800},
            "episodes_are_not_reset_on_life_loss": True,
            "recent_contact_outcomes": recent[-6:],
            "candidates": {
                "CORE": "Use conservative interior contact band; do not trust full decoded width.",
                "MEASURED": "Broaden only to stationary successful offsets already observed in code.",
                "UNKNOWN": "Insufficient or contradictory evidence: keep CORE."},
            "review_meanings": {
                "COVERAGE": "Stable paddle and small prediction residual but miss within reported width.",
                "MOTOR": "Paddle moved substantially at a failed crossing.",
                "PROJECTION": "Observed crossing disagreed with the code landing prediction.",
                "UNKNOWN": "No failure or evidence does not isolate one mechanism."},
        }

    @staticmethod
    def _prob(answer, value):
        probs = getattr(answer, "probabilities", {})
        if not isinstance(probs, dict):
            return 0.0
        try:
            p = float(probs.get(value, 0))
            return p if math.isfinite(p) else 0.0
        except (TypeError, ValueError):
            return 0.0

    def _poll_models(self, obs, frame):
        if self.audit_future is not None and self.audit_future.done():
            future, stamp = self.audit_future, self.audit_stamp
            self.audit_future = None
            try:
                ans = future.result()
                if not isinstance(ans, dict):
                    ans = {}
                a, b = ans.get("certificate_a"), ans.get("certificate_b")
                av, bv = getattr(a, "value", None), getattr(b, "value", None)
                current = stamp == (self.safety_epoch, self.plan_version)
                fresh = frame-self.last_audit_frame <= 240
                facts = self._facts(obs, frame)["evidence"]
                if current and fresh and av == bv and av in ("CORE", "MEASURED"):
                    if min(self._prob(a,av), self._prob(b,bv)) >= .65:
                        accepted = "MEASURED" if av == "MEASURED" and facts["repeatable_stationary_successes"] else "CORE"
                        old = self.profile
                        self.profile = accepted
                        self._log("certificate_audit", frame=frame, old=old, selected=accepted,
                                  changed=old != accepted, epoch=self.safety_epoch)
                review_answer = ans.get("needs_review")
                review_value = getattr(review_answer, "value", None)
                if current and fresh and isinstance(review_value, (int, float)) and review_value >= .75:
                    if facts["coverage_contradiction"] or facts["projection_error_at_miss"]:
                        self.review_due = True
                diagnostic = ans.get("cause")
                d = getattr(diagnostic, "value", None)
                if current and fresh and d in ("COVERAGE", "MOTOR", "PROJECTION"):
                    if self._prob(diagnostic, d) >= .65:
                        self.last_diagnostic = d
                        self.review_due = True
                        self._log("failure_classified", frame=frame, diagnosis=d)
            except (TimeoutError, RuntimeError, BudgetExceeded, EpisodeOver) as exc:
                self._log("audit_unavailable", kind=type(exc).__name__)
            except Exception as exc:
                self._log("audit_parse_ignored", kind=type(exc).__name__)
        if self.review_future is not None and self.review_future.done():
            future, stamp = self.review_future, self.review_stamp
            self.review_future = None
            try:
                result = future.result()
                text = getattr(result, "text", "")
                match = re.fullmatch(r"\s*(VERIFY_COVERAGE|VERIFY_MOTOR|VERIFY_PROJECTION|USE_MEASURED)\s*[.!]?\s*", text)
                if match is not None and stamp == self.geometry_epoch:
                    self.plan = match.group(1)
                    self.plan_version += 1
                    self.audit_due = True
                    self._log("astra_review_installed", frame=frame, plan=self.plan,
                              version=self.plan_version)
            except (TimeoutError, RuntimeError, BudgetExceeded, EpisodeOver) as exc:
                self._log("review_unavailable", kind=type(exc).__name__)
            except Exception as exc:
                self._log("review_parse_ignored", kind=type(exc).__name__)

    def _launch_models(self, obs, frame):
        if frame < self.next_model_frame or self.ctx.done:
            return
        remain = self.ctx.budget.remaining()
        if self.audit_due and self.audit_future is None and remain.get("system1_calls", 0) > 4:
            state = self._facts(obs, frame)
            criteria = state["candidates"]
            instruction = (
                "Select a contact certificate from evidence, not a paddle action. "
                "Choose CORE whenever any miss, coverage contradiction, projection error, "
                "or motor motion is present. MEASURED requires repeatable_stationary_successes=true "
                "and no_recent_misses=true. UNKNOWN for missing evidence. Astra's review is an "
                "audit priority, not permission to ignore these conditions."
            )
            qs = {
                "certificate_a": Choice(instruction, criteria),
                "certificate_b": Choice(instruction, dict(reversed(list(criteria.items())))),
                "cause": Choice(
                    "Classify the most directly supported failure mechanism. Read the precomputed "
                    "evidence flags. COVERAGE requires coverage_contradiction=true; otherwise "
                    "PROJECTION if projection_error_at_miss=true, MOTOR if motor_motion_at_miss=true, "
                    "UNKNOWN if no cause is isolated. Do no coordinate arithmetic.",
                    state["review_meanings"]),
                "needs_review": Noul("Does evidence contain a coverage contradiction or projection error at a miss?")
            }
            self.audit_due = False
            self.last_audit_frame = frame
            self.audit_stamp = (self.safety_epoch, self.plan_version)
            self.audit_future = self.ctx.spawn(lambda: self.ctx.decide(state, qs, timeout_s=3.0))
            self.next_model_frame = frame + 90
        # A slow call reviews several completed contacts, never a current ball.
        initial = self.review_count == 0 and len(self.samples) >= 5
        if (initial or self.review_due) and self.review_future is None and self.review_count < 3:
            if frame-self.last_review_frame < 1800 or remain.get("system2_calls", 0) <= 2 or remain.get("wall_s", 0) < 150:
                return
            summary = self._facts(obs, frame)
            prompt = (
                "Review an empirical Atari 2600 Breakout controller. Code keeps playing at 60 Hz. "
                "The reported paddle rectangle is not a proven collision mask. Successful/failed "
                "crossings, displacement, and projection residuals are measured from public RAM "
                "observations. Do not invent physics or frame actions. Return exactly one token: "
                "VERIFY_COVERAGE if stable rectangle-overlap misses contradict coverage; "
                "VERIFY_MOTOR if failed crossings show late paddle motion; VERIFY_PROJECTION if "
                "predictions disagree with observed crossings; USE_MEASURED only if stationary "
                "contacts repeatedly support the empirically measured band. Your token changes "
                "the next Jev evidence audit, not a live action. A full collision band will never "
                "be inferred from prose. Prefer the most falsifiable remaining mechanism.\n" +
                json.dumps(summary, separators=(",", ":"))
            )
            self.review_stamp = self.geometry_epoch
            self.review_count += 1
            self.review_due = False
            self.last_review_frame = frame
            self.review_future = self.ctx.spawn(lambda: self.ctx.think(prompt, timeout_s=150.0))

    def run_episode(self, game):
        while not self.ctx.done and not bool(getattr(game, "done", False)):
            obs = self.ctx.observe()
            if not isinstance(obs, dict):
                self.ctx.act(NOOP)
                self.ctx.wait_for_tick()
                continue
            frame = self._frame(obs)
            if frame == self.last_frame:
                self.ctx.wait_for_tick()
                continue
            event = self._event(obs)
            self._capture(obs, frame, event)
            self._update_paddle_motion(obs)
            self._apply_event(obs, event, frame)
            flight = self._flight(obs, event, frame)
            self._poll_models(obs, frame)
            action = self._motor(obs, flight, frame)
            if action not in ACTIONS:
                action = NOOP
            self.ctx.act(action)
            b = self._ball(obs)
            if b is not None and self._num(b.get("y")) >= 158 and self._num(b.get("dy")) > 0:
                self._log("capture_control", frame=frame, ball=b, paddle=self._paddle(obs),
                          action=action, low=round(self.contract_low,2),
                          high=round(self.contract_high,2), landing=self.contract_landing,
                          eta=self.contract_eta, profile=self.profile, raw_motion=self.raw_motion)
            self.pre_post, self.last_post = self.last_post, action
            self.previous = obs
            self.last_frame = frame
            self._launch_models(obs, frame)
            self.ctx.wait_for_tick()


def build(ctx):
    return Harness(ctx)
