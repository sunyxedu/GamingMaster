"""Transition-aware event-contract Breakout harness.

The emulator is driven by a small deterministic motor controller.  A return is
represented as a *catch interval* for the paddle, not as a point target.  The
interval is planned once after the ball has left the brick band and is driven
with a leased motor pulse; a late horizontal wall bounce preserves that
interval instead of replacing it with a zero-horizon estimate.  Model answers
can choose only a long-lived rally contract at event boundaries.
"""
from __future__ import annotations

from collections import deque
import math
import re
from typing import Any

from metaharness.core.contract import BudgetExceeded, EpisodeOver
from metaharness.models.jev import Choice


NOOP = "NOOP"
FIRE = "FIRE"
RIGHT = "RIGHT"
LEFT = "LEFT"
ACTIONS = (NOOP, FIRE, RIGHT, LEFT)
DIRECTIONS = (RIGHT, LEFT)


class Harness:
    def __init__(self, ctx: Any):
        self.ctx = ctx

        # The latest complete RAM observation is the only history used by the
        # controller.  All globals are intentionally per-episode.
        self.previous: dict[str, Any] | None = None
        self.last_frame = -1
        self.previous_paddle_x: float | None = None
        self.paddle_speed = 0.0
        self.paddle_history = deque(maxlen=8)
        self.noop_drift = 0.0
        self.last_requested = NOOP
        self.last_effective = NOOP

        # Keep a temporal estimate separate from the adaptor's one-frame dx/dy.
        # Breakout's RAM coordinates are quantized, so a zero sample is not a
        # new direction and a one-frame change is not by itself a new regime.
        self.ball_x_history = deque(maxlen=8)
        self.ball_y_history = deque(maxlen=8)
        self.ball_x_sign = 0
        self.ball_y_sign = 0
        self.ball_vx = 0.0
        self.ball_vy = 0.0
        self.velocity_samples = 0
        self.velocity_quarantine_until = -1
        self.velocity_reversal = False

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
        self.contract_quality = "unknown"
        self.contract_locked = False
        self.contract_lock_frame = -1
        self.contract_epoch = 0
        self.flight_phase = "home"

        # A late horizontal wall transition is not allowed to replace a
        # nearly-contacting corridor with a post-bounce zero-horizon estimate.
        # This flag lasts until the next real return event or paddle contact.
        self.side_bounce_hold = False
        self.side_bounce_hold_until = -1
        self.side_bounce_count = 0

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
            if frame_gap <= 2 and abs(raw) <= 10.0:
                self.paddle_history.append(raw)
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
            # authoritative when sticky actions disagree.
            self.last_requested = requested

    def _clear_velocity(self, reason: str, frame: int, quarantine: int = 0) -> None:
        self.ball_x_history.clear()
        self.ball_y_history.clear()
        self.ball_x_sign = 0
        self.ball_y_sign = 0
        self.ball_vx = 0.0
        self.ball_vy = 0.0
        self.velocity_samples = 0
        self.velocity_reversal = False
        self.velocity_quarantine_until = max(self.velocity_quarantine_until, frame + quarantine)
        self._log("velocity_reset", frame=frame, reason=reason, quarantine=quarantine)

    @staticmethod
    def _sign(value: float) -> int:
        return 1 if value > 0.0 else -1 if value < 0.0 else 0

    def _update_ball_motion(self, obs: dict[str, Any], event: str, frame: int) -> None:
        ball = self._ball(obs)
        old_ball = self._ball(self.previous)
        if ball is None or not bool(obs.get("ball_in_play", False)):
            if old_ball is not None:
                self._clear_velocity("ball_absent", frame)
            return
        if old_ball is None:
            self._clear_velocity("new_ball", frame)
            return
        old_frame = self._frame(self.previous) if isinstance(self.previous, dict) else frame - 1
        gap = frame - old_frame
        if gap < 1 or gap > 2:
            self._clear_velocity("observation_gap", frame)
            return

        dx = (self._num(ball.get("x"), 0.0) - self._num(old_ball.get("x"), 0.0)) / gap
        dy = (self._num(ball.get("y"), 0.0) - self._num(old_ball.get("y"), 0.0)) / gap
        if event in ("brick_hit", "side_bounce", "paddle_bounce", "life_loss", "wall_transition", "serve"):
            # The displacement containing a collision is not a flight sample.
            self._clear_velocity(event, frame, quarantine=2 if event in ("brick_hit", "side_bounce") else 0)
            return

        # A zero quantized coordinate preserves the last direction; a genuine
        # sign change is a new flight epoch and invalidates the old estimate.
        sx, sy = self._sign(dx), self._sign(dy)
        if sx and self.ball_x_sign and sx != self.ball_x_sign:
            self.ball_x_history.clear()
            self.velocity_reversal = True
        if sy and self.ball_y_sign and sy != self.ball_y_sign:
            self.ball_y_history.clear()
            self.velocity_reversal = True
        if abs(dx) < 15.0 and sx:
            self.ball_x_history.append((dx, gap))
            self.ball_x_sign = sx
        if abs(dy) < 15.0 and sy:
            self.ball_y_history.append((dy, gap))
            self.ball_y_sign = sy

        def average(history: Any) -> float:
            if not history:
                return 0.0
            total = sum(weight for _, weight in history)
            return sum(value * weight for value, weight in history) / max(1, total)

        self.ball_vx = average(self.ball_x_history)
        self.ball_vy = average(self.ball_y_history)
        self.velocity_samples = min(len(self.ball_x_history), len(self.ball_y_history))
        if self.velocity_reversal:
            self.velocity_quarantine_until = max(self.velocity_quarantine_until, frame + 1)

    def _reset_return(self, reason: str, frame: int) -> None:
        self.contract_valid = False
        self.contract_landing = None
        self.contract_eta = None
        self.contract_quality = "unknown"
        self.contract_locked = False
        self.contract_lock_frame = -1
        self.side_bounce_hold = False
        self.side_bounce_hold_until = -1
        self.flight_phase = "home"
        self.contract_epoch += 1
        self.lease_action = NOOP
        self.lease_until = frame
        self.brake_until = -1
        self.neutral_until = frame + 1
        self.last_brake_frame = -10000
        self._log("return_contract_reset", frame=frame, reason=reason, epoch=self.contract_epoch)

    def _apply_event(self, obs: dict[str, Any], event: str, frame: int) -> None:
        if event in ("start", "serve", "life_loss", "ball_lost", "wall_transition", "paddle_bounce"):
            self._reset_return(event, frame)
            if event == "start":
                self._clear_velocity("start", frame)
            if event in ("life_loss", "wall_transition"):
                self.rally_version += 1
                self.rally_contract = "PRESERVE"
                self.jev_answer = None
                self.next_jev_frame = frame
        elif event == "brick_hit":
            # A brick can change speed and vertical phase, so it always starts
            # a fresh return estimate.
            self._reset_return(event, frame)
        elif event == "side_bounce":
            ball = self._ball(obs)
            current_dy = self._num(ball.get("dy"), 0.0) if ball is not None else 0.0
            current_eta = 99.0
            if ball is not None and current_dy > 0.0:
                paddle = self._paddle(obs)
                py = self._num(paddle.get("y"), 189.0)
                bh = self._num(ball.get("h"), 4.0)
                by = self._num(ball.get("y"), 100.0)
                current_eta = max(0.0, (py - bh - by) / max(0.25, current_dy))
            late = self.contract_valid and (
                self.contract_locked or current_eta <= 8.0
            ) and current_dy > 0.0
            if late:
                # The reflected projection was already a viable paddle overlap.
                # Preserve it through this horizontal transition and add only a
                # small edge margin.  The next post-bounce sample has almost no
                # time to move the paddle, so replacing the contract here is
                # strictly worse than keeping a conservative envelope.
                pmax = 144.0
                self.contract_low = max(8.0, self.contract_low - 2.0)
                self.contract_high = min(pmax, self.contract_high + 2.0)
                self.contract_center = (self.contract_low + self.contract_high) / 2.0
                self.contract_eta = current_eta
                self.contract_locked = True
                self.side_bounce_hold = True
                self.side_bounce_hold_until = frame + 8
                self.side_bounce_count += 1
                self._log(
                    "side_bounce_continuity",
                    frame=frame,
                    low=round(self.contract_low, 2),
                    high=round(self.contract_high, 2),
                    eta=round(current_eta, 2),
                    count=self.side_bounce_count,
                )
            else:
                # An early side bounce genuinely changes the future landing;
                # let the proven quarantine-and-replan path handle it.
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
        """Use the ROM's current displacement, with temporal fallback/diagnostics."""
        ball = self._ball(obs)
        if ball is None:
            return 0.0, 0.0, False
        raw_dx = self._num(ball.get("dx"), 0.0)
        raw_dy = self._num(ball.get("dy"), 0.0)
        raw_valid = bool(obs.get("ball_velocity_valid", False))

        # The Atari motion can alternate between quantized one-frame
        # displacements.  The current decoded sample is consequently the
        # authoritative projection input; smoothing it into a new trajectory
        # was refuted by the short smoke trial.  History is retained as a
        # corroboration/fallback for missing startup samples and for Jev facts.
        if raw_valid or abs(raw_dx) + abs(raw_dy) > 0.0:
            return raw_dx, raw_dy, True
        if self.velocity_samples > 0 and abs(self.ball_vx) + abs(self.ball_vy) > 0.0:
            return self.ball_vx, self.ball_vy, True
        old_ball = self._ball(self.previous) if self.previous is not None else None
        if old_ball is not None:
            inferred_dx = self._clamp(
                self._num(ball.get("x"), 0.0) - self._num(old_ball.get("x"), 0.0), -8.0, 8.0
            )
            inferred_dy = self._clamp(
                self._num(ball.get("y"), 0.0) - self._num(old_ball.get("y"), 0.0), -8.0, 8.0
            )
            if abs(inferred_dx) + abs(inferred_dy) > 0.0:
                return inferred_dx, inferred_dy, True
        return 0.0, 0.0, False

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

        # A collision can leave an apparently valid dx/dy for one or two
        # frames.  If the adaptor has a fresh decoded displacement, it is the
        # best local evidence; quarantine is only for a missing raw sample.
        raw_confident = bool(obs.get("ball_velocity_valid", False)) or (
            abs(self._num(ball.get("dx"), 0.0)) + abs(self._num(ball.get("dy"), 0.0)) > 0.0
        )
        if frame <= self.velocity_quarantine_until and eta > 12.0 and not raw_confident:
            return False
        if by < 94.0:
            return False
        landing_left = self._reflect_x(bx, dx, eta)

        speed = abs(dx) + abs(dy)
        margin = 1.0 if speed < 4.0 else 0.5
        # Estimate uncertainty from the temporal samples, not from a single
        # quantized displacement.  Expand the safe set modestly so the motor
        # can settle under sticky actions without chasing its midpoint.
        spread = 0.0
        if self.ball_x_history:
            values = [value for value, _ in self.ball_x_history]
            spread = max(values) - min(values)
        uncertainty = self._clamp(0.75 + 0.45 * spread + (0.5 if self.velocity_samples < 3 else 0.0), 0.75, 2.75)
        low = landing_left + margin - pw - uncertainty
        high = landing_left + bw - margin + uncertainty
        pmin, pmax = 8.0, max(8.0, 160.0 - pw)
        low = self._clamp(low, pmin, pmax)
        high = self._clamp(high, pmin, pmax)
        if high < low:
            mid = self._clamp(landing_left + bw / 2.0 - pw / 2.0, pmin, pmax)
            low = high = mid

        self.contract_valid = True
        self.contract_low = low
        self.contract_high = high
        self.contract_center = (low + high) / 2.0
        self.contract_landing = landing_left + bw / 2.0
        self.contract_eta = eta
        self.contract_frame = frame
        self.contract_vx = dx
        self.contract_vy = dy
        self.contract_quality = "corroborated" if self.velocity_samples >= 3 else "settling"
        self.contract_locked = eta <= 4.0
        self.contract_lock_frame = frame if self.contract_locked else -1
        self.flight_phase = "locked" if self.contract_locked else "descending"
        self._log(
            "return_contract",
            frame=frame,
            low=round(low, 2),
            high=round(high, 2),
            landing=round(self.contract_landing, 2),
            eta=round(eta, 2),
            quality=self.contract_quality,
            samples=self.velocity_samples,
            locked=self.contract_locked,
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
            if self.contract_valid and not self.contract_locked:
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

        # Keep the parent's responsive raw projection while the landing is far
        # enough away to correct.  Temporal history is a collision diagnostic;
        # the final contact lock is the only place where a small raw change is
        # intentionally ignored.
        if self.velocity_reversal and not self.contract_locked:
            self._reset_return("unannounced_reversal", frame)
            self.velocity_reversal = False
        if not self.contract_valid:
            self._make_return_contract(obs, frame)
        elif not self.contract_locked and (eta is None or eta > 4.0):
            if abs(dx - self.contract_vx) > 0.75 or abs(dy - self.contract_vy) > 0.75:
                self._make_return_contract(obs, frame)

        if not self.contract_valid:
            low, high, center = self._home_interval(obs)
            return {
                "phase": "quarantine" if frame <= self.velocity_quarantine_until else "uncertain_descent",
                "urgency": "early",
                "eta": None if eta is None else round(eta, 1),
                "low": low,
                "high": high,
                "center": center,
                "landing": None,
                "speed": "fast" if abs(dx) + abs(dy) >= 4.0 else "unknown",
            }

        if eta is not None and eta <= 4.0 and not self.contract_locked:
            self.contract_locked = True
            self.contract_lock_frame = frame
            self.flight_phase = "locked"
            self._log("return_contract_locked", frame=frame, eta=round(eta, 2))
        urgency = "imminent" if eta is None or eta <= 6.0 else "near" if eta <= 18.0 else "early"
        return {
            "phase": "locked" if self.contract_locked else "descending",
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

    def _derived(self, obs: dict[str, Any], event: str, flight: dict[str, Any]) -> dict[str, Any]:
        side = self._side_facts(obs)
        frame = self._frame(obs)
        lives = self._int(obs.get("lives"), 0)
        motion = "right" if self.paddle_speed > 0.45 else "left" if self.paddle_speed < -0.45 else "still"
        return {
            "frame": frame,
            "event": event,
            "lives": lives,
            "score": self._int(self._num(obs.get("score"), 0.0)),
            "serve_required": bool(obs.get("serve_required", False)),
            "phase": flight.get("phase", "unknown"),
            "urgency": flight.get("urgency", "unknown"),
            "speed": flight.get("speed", "unknown"),
            "estimate": self.contract_quality if self.contract_valid else (
                "settling" if self.velocity_samples else "unknown"
            ),
            "catch_interval": "locked" if self.contract_locked else (
                "wide" if self.contract_valid and self.contract_high - self.contract_low >= 8.0 else "narrow"
            ),
            "flight_epoch": self.contract_epoch,
            "landing_lane": (
                "left" if self._num(flight.get("landing"), 80.0) < 56.0
                else "right" if self._num(flight.get("landing"), 80.0) > 104.0
                else "center"
            ),
            "catch_interval": "wide" if self._num(flight.get("high"), 0.0) - self._num(flight.get("low"), 0.0) >= 8.0 else "narrow",
            "paddle_motion": motion,
            "eta": flight.get("eta"),
            # Do not call the mutating motor selector while constructing the
            # model state. The actual action is selected exactly once below.
            "fallback_action": "FIRE" if bool(obs.get("serve_required", False)) else "INTERVAL_MOTOR",
            "rally_contract": self.rally_contract,
            "astra_plan": self.astra_plan,
            "left_edge": side["left_edge"],
            "right_edge": side["right_edge"],
            "left_count": side["left_count"],
            "right_count": side["right_count"],
            "bricks_remaining": side["bricks_remaining"],
            "wall_index": side["wall_index"],
            "contract_epoch": self.contract_epoch,
            "rally_version": self.rally_version,
            "side_bounce_hold": self.side_bounce_hold,
            "side_bounce_count": self.side_bounce_count,
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

    def _jev_questions(self, state: dict[str, Any]) -> dict[str, Choice]:
        options = {
            "PRESERVE": "Keep a centered, catch-first rally. Never deliberately move away from the code catch interval.",
            "LEFT": "For a plentiful-life, normal-speed, early return, a left-side shaping preference may move the home anchor left; code still owns every catch.",
            "RIGHT": "For a plentiful-life, normal-speed, early return, a right-side shaping preference may move the home anchor right; code still owns every catch.",
            "NO_CHANGE": "The state is ambiguous or high risk; retain the current contract and let the deterministic fallback act.",
        }
        keys = list(options)
        shift = self.jev_serial % len(keys)
        order_a = keys[shift:] + keys[:shift]
        order_b = list(reversed(order_a))
        instruction = (
            "Choose one long-lived rally contract for this exact derived state. "
            "Do not calculate coordinates or invent a frame action. `fallback_action`, "
            "the catch interval, and all arithmetic are code-owned. Choose PRESERVE or "
            "NO_CHANGE whenever lives are low, speed is fast, urgency is near/imminent, "
            "phase is uncertain/quarantine/locked, the estimate is settling, or the "
            "requested edge is open. LEFT/RIGHT is only an "
            "early home-anchor preference and is ignored during a catch. State: " + repr(state)
        )
        return {
            "contract_a": Choice(instruction, {k: options[k] for k in order_a}),
            "contract_b": Choice(instruction, {k: options[k] for k in order_b}),
        }

    def _launch_jev(self, obs: dict[str, Any], derived: dict[str, Any]) -> None:
        if not self.jev_enabled or self.jev_future is not None:
            return
        frame = derived["frame"]
        event = derived["event"]
        # Boundary-triggered requests: a new serve/life/wall or a clean paddle
        # bounce.  A cooldown avoids spending S1 calls on the same return.
        boundary = event in ("start", "serve", "life_loss", "wall_transition", "paddle_bounce")
        if not boundary or frame < self.next_jev_frame or derived["serve_required"]:
            return
        if self._ball(obs) is None:
            return
        state = {
            "game": "Atari 2600 Breakout ordinary cartridge mode 0 difficulty 0 at native 60 Hz",
            "observation": derived,
            "rules": {
                "actions": list(ACTIONS),
                "FIRE": "only when serve_required is true",
                "catch_set": "code computes a paddle-left interval; preserve overlap before scoring strategy",
                "bridge": "the selected contract is rechecked on later observations and never directly becomes an action",
            },
        }
        questions = self._jev_questions(state)
        self.jev_serial += 1
        self.jev_request_frame = frame
        self.jev_request_version = self.rally_version
        self.next_jev_frame = frame + 36

        def call():
            return self.ctx.decide(state, questions, timeout_s=1.25)

        try:
            self.jev_future = self.ctx.spawn(call)
        except Exception as exc:
            self.jev_future = None
            self._log("jev_spawn_failed", error=type(exc).__name__)

    def _collect_jev(self, frame: int, obs: dict[str, Any]) -> None:
        future = self.jev_future
        if future is None or not getattr(future, "done", lambda: False)():
            return
        self.jev_future = None
        try:
            answers = future.result()
            if not isinstance(answers, dict):
                return
            a = answers.get("contract_a")
            b = answers.get("contract_b")
            av = self._answer_value(a)
            bv = self._answer_value(b)
            allowed = {"PRESERVE", "LEFT", "RIGHT", "NO_CHANGE"}
            if av not in allowed or av != bv:
                return
            probability = min(self._answer_probability(a, av), self._answer_probability(b, bv))
            if probability < 0.55:
                return
            if self.jev_request_version != self.rally_version:
                return
            # Recheck lethal/high-speed invariants on the current observation.
            lives = self._int(obs.get("lives"), 0)
            if lives <= 2 or av == "NO_CHANGE":
                av = "PRESERVE"
            if av in ("LEFT", "RIGHT") and self._side_facts(obs)["left_edge" if av == "LEFT" else "right_edge"] == "open":
                av = "PRESERVE"
            self.jev_answer = av
            self.jev_answer_frame = self.jev_request_frame
            self.jev_answer_version = self.rally_version
            self.rally_contract = av
            # Acceptance does not create a new invalidation version; the
            # answer is stamped with the boundary version it judged.
            self._log("rally_contract_accepted", frame=frame, contract=av, probability=round(probability, 3))
        except (TimeoutError, RuntimeError, BudgetExceeded, EpisodeOver) as exc:
            self._log("jev_unavailable", error=type(exc).__name__)
        except Exception as exc:
            self._log("jev_parse_ignored", error=type(exc).__name__)

    def _astra_prompt(self, obs: dict[str, Any], derived: dict[str, Any]) -> str:
        side = self._side_facts(obs)
        return (
            "You are the slow strategic layer for ordinary Atari 2600 Breakout mode 0, "
            "difficulty 0. A deterministic controller predicts a paddle catch interval "
            "on every native frame. Return exactly one token: PRESERVE, LEFT, or RIGHT. "
            "LEFT/RIGHT is only a cautious home-anchor preference while lives are plentiful; "
            "never give frame actions. Pick PRESERVE for fast/uncertain flight, low lives, "
            "or an open side. A later Jev boundary judgment and code safety gates decide "
            "whether the token is used.\nState=" + repr({
                "derived": derived,
                "side": side,
                "ball": obs.get("ball"),
                "paddle": obs.get("paddle"),
            })
        )

    def _launch_astra(self, obs: dict[str, Any], derived: dict[str, Any]) -> None:
        if self.astra_started or self.astra_future is not None:
            return
        if derived["frame"] < 45 or self._ball(obs) is None:
            return
        try:
            if int(self.ctx.budget.remaining().get("system2_calls", 0)) <= 0:
                return
        except Exception:
            pass
        self.astra_started = True
        prompt = self._astra_prompt(obs, derived)

        def call():
            return self.ctx.think(prompt, timeout_s=12.0)

        try:
            self.astra_future = self.ctx.spawn(call)
        except Exception as exc:
            self.astra_future = None
            self._log("astra_spawn_failed", error=type(exc).__name__)

    def _collect_astra(self, frame: int, obs: dict[str, Any]) -> None:
        future = self.astra_future
        if future is None or not getattr(future, "done", lambda: False)():
            return
        self.astra_future = None
        try:
            result = future.result()
            text = getattr(result, "text", "")
            if not isinstance(text, str):
                return
            match = re.search(r"\b(PRESERVE|LEFT|RIGHT|SURVIVE|SAFE)\b", text.upper())
            if match is None:
                return
            token = match.group(1)
            plan = "PRESERVE" if token in ("PRESERVE", "SURVIVE", "SAFE") else token
            lives = self._int(obs.get("lives"), 0)
            if lives <= 2:
                plan = "PRESERVE"
            side = self._side_facts(obs)
            if plan == "LEFT" and side["left_edge"] == "open":
                plan = "PRESERVE"
            if plan == "RIGHT" and side["right_edge"] == "open":
                plan = "PRESERVE"
            self.astra_plan = plan
            # Astra changes the proposal version; Jev must validate it at the
            # next boundary, rather than applying a stale slow answer directly.
            self.rally_version += 1
            self.jev_answer = None
            self._log("astra_contract_received", frame=frame, contract=plan)
        except (TimeoutError, RuntimeError, BudgetExceeded, EpisodeOver) as exc:
            self._log("astra_unavailable", error=type(exc).__name__)
        except Exception as exc:
            self._log("astra_parse_ignored", error=type(exc).__name__)

    def _apply_advice(self, obs: dict[str, Any], flight: dict[str, Any]) -> None:
        # Astra is a prior, not authority.  Before Jev answers, its proposal is
        # intentionally kept out of the motor loop.  At a safe home phase, the
        # accepted proposal can become the default contract for the next Jev
        # boundary; a catch interval is never changed here.
        if self.jev_answer in ("PRESERVE", "LEFT", "RIGHT") and self.jev_answer_version == self.rally_version:
            if flight.get("phase") in ("ascending", "collision_zone", "unknown"):
                self.rally_contract = self.jev_answer
        elif self.rally_contract == "PRESERVE" and self.astra_plan in ("LEFT", "RIGHT"):
            # The Astra token is only a low-stakes prior while four or five
            # lives remain; Jev can replace it at the next event boundary.
            if self._int(obs.get("lives"), 0) >= 4 and flight.get("phase") in ("ascending", "collision_zone"):
                self.rally_contract = self.astra_plan

    # ---------------- episode loop ----------------
    @staticmethod
    def _observe(game: Any) -> dict[str, Any]:
        try:
            value = game.observe()
        except Exception:
            return {}
        return value if isinstance(value, dict) else {}

    def _one(self, game: Any, realtime: bool) -> bool:
        obs = self._observe(game)
        frame = self._frame(obs)
        event = self._event(obs)
        self._update_paddle_motion(obs)
        self._update_ball_motion(obs, event, frame)
        self._apply_event(obs, event, frame)
        flight = self._flight(obs, event, frame)

        self._collect_jev(frame, obs)
        self._collect_astra(frame, obs)
        self._launch_astra(obs, self._derived(obs, event, flight))
        self._launch_jev(obs, self._derived(obs, event, flight))
        self._apply_advice(obs, flight)
        action = self._steer_interval(obs, flight, frame)
        if action not in ACTIONS:
            action = NOOP
        self.last_action_frame = frame
        try:
            self.ctx.act(action)
        except (BudgetExceeded, EpisodeOver):
            return False
        self.previous = obs
        self.last_frame = frame
        if realtime:
            try:
                self.ctx.wait_for_tick()
            except (BudgetExceeded, EpisodeOver):
                return False
        return True

    def run_episode(self, game: Any) -> None:
        realtime = bool(getattr(game, "tick_s", 0.0))
        ctx = self.ctx
        while not ctx.done and not bool(getattr(game, "done", False)):
            if not self._one(game, realtime):
                return


def build(ctx: Any) -> Harness:
    return Harness(ctx)
