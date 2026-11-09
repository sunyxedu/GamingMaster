"""Committed, phase-gated interception for Atari 2600 Breakout.

The controller treats the wall as an unreliable part of the flight path.  It
waits until a descending ball has cleared the brick band, makes one reflected
trajectory plan, and then steers into an overlap interval with hysteresis.
This is intentionally different from a point-target servo: the target is
latched until an actual collision or a large measured change invalidates it.
Jev is advisory and asynchronous; the frame loop is entirely executable
without it.
"""
from __future__ import annotations

import math
from typing import Any

from metaharness.core.contract import BudgetExceeded, EpisodeOver
from metaharness.models.jev import Choice


ACTIONS = ("NOOP", "FIRE", "RIGHT", "LEFT")
DIRECTIONS = {"RIGHT", "LEFT"}
OPPOSITE = {"RIGHT": "LEFT", "LEFT": "RIGHT"}


class Harness:
    """A real-time controller with a latched landing interval."""

    def __init__(self, ctx: Any):
        self.ctx = ctx

        # Model mailbox.  Jev never sits on the native-frame loop.
        self.s1_enabled = getattr(ctx, "system1", object()) is not None
        self.s1_future = None
        self.s1_request_frame = -100000
        self.s1_request_version = -1
        self.s1_serial = 0
        self.s1_answer = None
        self.s1_answer_frame = -100000
        self.s1_answer_version = -1
        self.next_s1_frame = 30

        # Latest complete observation and decoded motion.
        self.previous: dict[str, Any] | None = None
        self.last_frame = -1
        self.last_command = "NOOP"
        self.last_effective_action = "NOOP"
        self.ball_vx = 0.0
        self.ball_vy = 0.0
        self.paddle_v = 0.0
        self.paddle_v_avg = 0.0
        self.position_samples = 0
        self.sticky_mismatch = 0

        # A trajectory plan is born only after the ball clears the wall on a
        # downward flight.  It persists through ordinary side-wall bounces.
        self.plan_version = 0
        self.plan_target = 76.0
        self.plan_low = 60.0
        self.plan_high = 92.0
        self.plan_landing = None
        self.plan_uncertainty = 20.0
        self.plan_frame = -100000
        self.plan_valid = False
        self.plan_phase = "home"
        self.last_ball_direction = 0

        # Actuator latch.  It prevents a target estimate changing by one pixel
        # from producing the LEFT/RIGHT alternation seen in earlier traces.
        self.latch_action = "NOOP"
        self.latch_until = -100000
        self.last_switch_frame = -100000
        self.brake_pulses = 0
        self.neutral_until = -100000
        self.action_count = 0

        self.last_lives = None
        self.last_score = 0.0
        self.last_wall = 0
        self.last_event = "start"

    # ------------------------------------------------------------------
    # Defensive observation and scalar helpers
    @staticmethod
    def _number(value: Any, default: float = 0.0) -> float:
        try:
            result = float(value)
            return result if math.isfinite(result) else default
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _integer(value: Any, default: int = 0) -> int:
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
        return self._integer(obs.get("frame"), self.last_frame + 1)

    def _safe_observe(self, game: Any) -> dict[str, Any]:
        try:
            obs = game.observe()
        except Exception:
            return {}
        return obs if isinstance(obs, dict) else {}

    def _log(self, name: str, **fields: Any) -> None:
        try:
            self.ctx.log(name, **fields)
        except Exception:
            # Logging is diagnostic and must never be allowed to lose a ball.
            pass

    # ------------------------------------------------------------------
    # Exact geometry, with a deliberately conservative wall-band gate
    @staticmethod
    def _reflect_x(x: float, vx: float, frames: float) -> float:
        left, right = 8.0, 150.0
        span = right - left
        if span <= 0.0:
            return x
        phase = (x - left + vx * max(0.0, frames)) % (2.0 * span)
        if phase <= span:
            return left + phase
        return right - (phase - span)

    def _side_summary(self, obs: dict[str, Any]) -> dict[str, Any]:
        bitmap = obs.get("brick_bitmap")
        left = right = 999
        if isinstance(bitmap, list):
            l = r = 0
            rows = 0
            for row in bitmap:
                if not isinstance(row, list):
                    continue
                rows += 1
                if len(row) > 0 and bool(row[0]):
                    l += 1
                if len(row) > 1 and bool(row[1]):
                    l += 1
                if len(row) > 16 and bool(row[16]):
                    r += 1
                if len(row) > 17 and bool(row[17]):
                    r += 1
            if rows:
                left, right = l, r
        def bucket(n: int) -> str:
            if n <= 0:
                return "open"
            if n <= 3:
                return "partial"
            return "blocked"
        return {
            "left": bucket(left),
            "right": bucket(right),
            "wall": self._integer(obs.get("wall_index"), 0),
            "walls_cleared": self._integer(obs.get("walls_cleared"), 0),
            "bricks_remaining": self._integer(obs.get("bricks_remaining"), 108),
        }

    def _speed_name(self) -> str:
        total = abs(self.ball_vx) + abs(self.ball_vy)
        if total < 0.5:
            return "unknown"
        return "fast" if total >= 3.0 else "normal"

    def _update_motion(self, obs: dict[str, Any]) -> None:
        old = self.previous
        paddle = self._paddle(obs)
        px = self._number(paddle.get("x"), 99.0)
        if old is not None:
            old_paddle = self._paddle(old)
            old_px = self._number(old_paddle.get("x"), px)
            dp = self._clamp(px - old_px, -10.0, 10.0)
            self.paddle_v = dp
            self.paddle_v_avg = 0.72 * self.paddle_v_avg + 0.28 * dp
            if abs(dp) > 0.0:
                self.position_samples += 1

            # The adaptor exposes both the effective last action and the
            # requested action.  They are useful diagnostics under stickiness,
            # but position delta remains authoritative for steering.
            effective = obs.get("last_action")
            requested = obs.get("requested_action")
            if isinstance(effective, str) and effective in ACTIONS:
                self.last_effective_action = effective
            if (
                isinstance(effective, str)
                and isinstance(requested, str)
                and effective in ACTIONS
                and requested in ACTIONS
                and effective != requested
            ):
                self.sticky_mismatch += 1

        ball = self._ball(obs)
        old_ball = self._ball(old) if old is not None else None
        if ball is None:
            self.ball_vx = 0.0
            self.ball_vy = 0.0
            return

        raw_vx = self._number(ball.get("dx"), 0.0)
        raw_vy = self._number(ball.get("dy"), 0.0)
        valid = bool(obs.get("ball_velocity_valid", False))
        inferred_x = inferred_y = 0.0
        if old_ball is not None:
            bx = self._number(ball.get("x"), 0.0)
            by = self._number(ball.get("y"), 0.0)
            obx = self._number(old_ball.get("x"), bx)
            oby = self._number(old_ball.get("y"), by)
            inferred_x = self._clamp(bx - obx, -8.0, 8.0)
            inferred_y = self._clamp(by - oby, -8.0, 8.0)

        if valid and (raw_vx != 0.0 or raw_vy != 0.0):
            self.ball_vx, self.ball_vy = raw_vx, raw_vy
        elif inferred_x != 0.0 or inferred_y != 0.0:
            self.ball_vx, self.ball_vy = inferred_x, inferred_y
        elif raw_vx != 0.0 or raw_vy != 0.0:
            self.ball_vx, self.ball_vy = raw_vx, raw_vy

    def _event(self, obs: dict[str, Any]) -> str:
        old = self.previous
        if old is None:
            return "start"
        old_lives = self._integer(old.get("lives"), self._integer(obs.get("lives"), 0))
        new_lives = self._integer(obs.get("lives"), old_lives)
        if new_lives < old_lives:
            return "life_loss"
        old_wall = self._integer(old.get("wall_index"), 0)
        new_wall = self._integer(obs.get("wall_index"), old_wall)
        if new_wall != old_wall:
            return "wall_transition"
        old_score = self._number(old.get("score"), 0.0)
        new_score = self._number(obs.get("score"), old_score)
        if new_score > old_score:
            return "brick_hit"
        old_ball = self._ball(old)
        ball = self._ball(obs)
        if old_ball is None and ball is not None:
            return "serve"
        if old_ball is not None and ball is None:
            return "ball_lost"

        # A down-to-up transition near the paddle is an observed impact.  A
        # plan must not survive it, even when the score did not change.
        old_dy = self._number(old_ball.get("dy"), 0.0) if old_ball else 0.0
        new_dy = self._number(ball.get("dy"), self.ball_vy) if ball else 0.0
        old_y = self._number(old_ball.get("y"), 0.0) if old_ball else 0.0
        new_y = self._number(ball.get("y"), old_y) if ball else 0.0
        if old_ball and ball and old_dy > 0 and new_dy < 0 and (old_y > 145 or new_y > 145):
            return "paddle_hit"
        if old_ball and ball:
            old_dx = self._number(old_ball.get("dx"), 0.0)
            new_dx = self._number(ball.get("dx"), self.ball_vx)
            if abs(old_dx) >= 0.5 and abs(new_dx) >= 0.5 and old_dx * new_dx < 0:
                # This can be a side-wall bounce or a paddle-angle change.  It
                # is cheap to refresh the plan; the next frame identifies the
                # new, stable descent if necessary.
                if new_y > 95.0:
                    return "horizontal_bounce"
        return "none"

    def _invalidate_plan(self, reason: str, frame: int) -> None:
        self.plan_valid = False
        self.plan_landing = None
        self.plan_version += 1
        self.plan_phase = "home"
        self.last_ball_direction = 0
        self.latch_action = "NOOP"
        self.latch_until = frame
        self.neutral_until = frame
        self.brake_pulses = 0
        self._log("landing_plan_invalidated", frame=frame, reason=reason, version=self.plan_version)

    def _apply_event(self, obs: dict[str, Any], event: str, frame: int) -> None:
        if event in ("start", "serve", "life_loss", "ball_lost", "paddle_hit", "wall_transition"):
            self._invalidate_plan(event, frame)
        elif event in ("brick_hit", "horizontal_bounce"):
            # A brick can alter dx/dy.  Do not steer against an old target
            # while still inside the brick band.
            self.plan_valid = False
            self.plan_version += 1
            self.latch_action = "NOOP"
            self.latch_until = frame
            self.neutral_until = frame
        self.last_event = event

    # ------------------------------------------------------------------
    # Phase-gated landing interval
    def _home_flight(self, px: float, pw: float, reason: str = "home") -> dict[str, Any]:
        lo = self._clamp(76.0 - 18.0, 8.0, 144.0)
        hi = self._clamp(76.0 + 18.0, 8.0, 144.0)
        return {
            "phase": reason,
            "urgency": "early",
            "eta": None,
            "landing": None,
            "landing_lo": None,
            "landing_hi": None,
            "target": 76.0,
            "safe_lo": lo,
            "safe_hi": hi,
            "uncertainty": 18.0,
        }

    def _build_landing_plan(self, obs: dict[str, Any], frame: int) -> None:
        ball = self._ball(obs)
        paddle = self._paddle(obs)
        if ball is None:
            return
        px = self._number(paddle.get("x"), 99.0)
        py = self._number(paddle.get("y"), 189.0)
        pw = self._number(paddle.get("w"), 16.0)
        bx = self._number(ball.get("x"), 80.0)
        by = self._number(ball.get("y"), 100.0)
        bw = self._number(ball.get("w"), 2.0)
        bh = self._number(ball.get("h"), 4.0)
        vy = self.ball_vy
        vx = self.ball_vx
        if vy <= 0.0:
            return
        eta = max(0.0, (py - bh - by) / max(0.25, vy))

        # Use a small ensemble instead of asking Jev to do arithmetic.  The
        # uncertainty is intentionally wider when sticky actions have been
        # observed or the ball is in the fast regime.
        speed = self._speed_name()
        velocity_error = 0.65 if speed == "fast" else 0.4
        if self.sticky_mismatch > 0:
            velocity_error += 0.15
        x_values = []
        for xoff in (-1.0, 0.0, 1.0):
            for voff in (-velocity_error, 0.0, velocity_error):
                for eoff in (-0.75, 0.0, 0.75):
                    e = max(0.0, eta + eoff)
                    x_values.append(self._reflect_x(bx + xoff, vx + voff, e) + bw / 2.0)
        landing_lo = min(x_values)
        landing_hi = max(x_values)
        landing = self._reflect_x(bx, vx, eta) + bw / 2.0
        uncertainty = max(2.0, (landing_hi - landing_lo) / 2.0)
        uncertainty += 1.5 if speed == "fast" else 0.75

        # A paddle interval that overlaps the ball with an interior margin.
        # This is a catch set, not a point target.  It remains useful when the
        # one-frame RAM velocity is temporarily noisy.
        margin = 2.5 if speed != "fast" else 3.5
        safe_lo = landing - pw + margin
        safe_hi = landing - margin
        # Clip to legal paddle-left positions.  If the uncertainty makes the
        # interval empty, retain a point target at the projected center.
        safe_lo = self._clamp(safe_lo, 8.0, 144.0)
        safe_hi = self._clamp(safe_hi, 8.0, 144.0)
        if safe_hi < safe_lo:
            midpoint = self._clamp(landing - pw / 2.0, 8.0, 144.0)
            safe_lo = safe_hi = midpoint
        target = self._clamp((safe_lo + safe_hi) / 2.0, 8.0, 144.0)

        self.plan_valid = True
        self.plan_landing = landing
        self.plan_low = safe_lo
        self.plan_high = safe_hi
        self.plan_target = target
        self.plan_uncertainty = uncertainty
        self.plan_frame = frame
        self.plan_phase = "descending"
        self.last_ball_direction = -1 if vx < 0 else 1 if vx > 0 else 0
        self._log(
            "landing_plan_created",
            frame=frame,
            landing=round(landing, 2),
            safe_lo=round(safe_lo, 2),
            safe_hi=round(safe_hi, 2),
            eta=round(eta, 2),
            uncertainty=round(uncertainty, 2),
        )

    def _flight(self, obs: dict[str, Any], frame: int) -> dict[str, Any]:
        ball = self._ball(obs)
        paddle = self._paddle(obs)
        px = self._number(paddle.get("x"), 99.0)
        pw = self._number(paddle.get("w"), 16.0)
        if ball is None or not bool(obs.get("ball_in_play", False)):
            return self._home_flight(px, pw, "serve")

        by = self._number(ball.get("y"), 100.0)
        py = self._number(paddle.get("y"), 189.0)
        bh = self._number(ball.get("h"), 4.0)
        eta = None
        if self.ball_vy > 0.0:
            eta = max(0.0, (py - bh - by) / max(0.25, self.ball_vy))

        # The bottom brick row ends near y=93.  Until the descending ball is
        # below it, its next vertical collision is unknown; home is safer than
        # committing to a stale projected return.  The explicit invalidation
        # also covers a missed paddle-transition event in a sparse observation.
        if self.ball_vy <= 0.0:
            if self.ball_vy < 0.0 and self.plan_valid:
                self._invalidate_plan("ascending", frame)
            home = self._home_flight(px, pw, "ascending" if self.ball_vy < 0 else "unknown")
            home["eta"] = None if eta is None else round(eta, 2)
            return home
        if by < 96.0:
            home = self._home_flight(px, pw, "collision_zone")
            home["eta"] = None if eta is None else round(eta, 2)
            return home

        # Replan once, then hold the target.  Only a large inconsistency in
        # the measured path is allowed to trigger a new plan mid-return.
        current_landing = self._reflect_x(
            self._number(ball.get("x"), 80.0),
            self.ball_vx,
            eta if eta is not None else 0.0,
        ) + self._number(ball.get("w"), 2.0) / 2.0
        if not self.plan_valid:
            self._build_landing_plan(obs, frame)
        elif (
            self.plan_landing is not None
            and abs(current_landing - self.plan_landing) > max(8.0, self.plan_uncertainty * 1.8)
        ):
            self._build_landing_plan(obs, frame)

        if not self.plan_valid:
            home = self._home_flight(px, pw, "descending_uncertain")
            home["eta"] = None if eta is None else round(eta, 2)
            return home

        if eta is None:
            urgency = "unknown"
        elif eta <= 7.0:
            urgency = "imminent"
        elif eta <= 20.0:
            urgency = "near"
        else:
            urgency = "early"
        center = self.plan_landing if self.plan_landing is not None else self.plan_target
        lane = "left" if center < 56 else "right" if center > 104 else "center"
        # `plan_low`/`plan_high` are paddle-left coordinates.  Keep the
        # diagnostic relation in the same coordinate system as the actuator.
        if px < self.plan_low:
            relation = "left_of_interval"
        elif px > self.plan_high:
            relation = "right_of_interval"
        else:
            relation = "overlap_possible"
        return {
            "phase": "descending",
            "urgency": urgency,
            "eta": None if eta is None else round(eta, 2),
            "landing": round(center, 2),
            "landing_lo": round(center - self.plan_uncertainty, 2),
            "landing_hi": round(center + self.plan_uncertainty, 2),
            "target": round(self.plan_target, 2),
            "safe_lo": round(self.plan_low, 2),
            "safe_hi": round(self.plan_high, 2),
            "uncertainty": round(self.plan_uncertainty, 2),
            "lane": lane,
            "relation": relation,
        }

    # ------------------------------------------------------------------
    # Latched actuator controller
    def _choose_direction(self, px: float, safe_lo: float, safe_hi: float) -> str:
        if px < safe_lo:
            return "RIGHT"
        if px > safe_hi:
            return "LEFT"
        return "NOOP"

    def _movement_action(self, obs: dict[str, Any], flight: dict[str, Any], frame: int) -> str:
        if bool(obs.get("terminated", False)) or bool(obs.get("truncated", False)):
            return "NOOP"
        if bool(obs.get("serve_required", False)):
            self._invalidate_plan("serve_required", frame)
            return "FIRE"
        ball = self._ball(obs)
        if ball is None or not bool(obs.get("ball_in_play", False)):
            return "NOOP"

        paddle = self._paddle(obs)
        px = self._number(paddle.get("x"), 99.0)
        pw = self._number(paddle.get("w"), 16.0)
        phase = flight.get("phase", "unknown")
        safe_lo = self._number(flight.get("safe_lo"), 58.0)
        safe_hi = self._number(flight.get("safe_hi"), 94.0)
        target = self._number(flight.get("target"), 76.0)
        eta = flight.get("eta")
        eta_n = self._number(eta, 12.0) if eta is not None else 12.0
        urgent = flight.get("urgency") in ("near", "imminent")
        advisory = self._current_advisory(frame)

        # Small neutral hysteresis in the collision band.  A model answer can
        # never override this branch.
        if frame < self.neutral_until and not bool(obs.get("serve_required", False)):
            self.last_command = "NOOP"
            return "NOOP"

        # Estimate where the paddle would be if the current momentum persists.
        # The horizon is capped because a long-horizon velocity estimate is less
        # reliable than a fresh position observation.
        horizon = self._clamp(eta_n, 0.0, 6.0)
        predicted_px = px + self.paddle_v_avg * horizon
        # Expand the latch's acceptance interval by a small amount early in
        # flight, but tighten it as contact approaches.
        guard = 3.0 if not urgent else 1.0
        inside = (safe_lo - guard) <= px <= (safe_hi + guard)
        predicted_inside = (safe_lo - guard) <= predicted_px <= (safe_hi + guard)

        if phase in ("ascending", "collision_zone", "unknown", "descending_uncertain"):
            # No reliable brick-free landing exists yet.  Park in a wide home
            # interval, but do not reverse every frame around its center.
            safe_lo, safe_hi, target = 58.0, 94.0, 76.0
            inside = safe_lo <= px <= safe_hi
            predicted_inside = safe_lo <= predicted_px <= safe_hi

        # Jev may select the conservative PARK_HOME branch only in a cheap,
        # low-life, non-imminent window.  Intersect it with the computed catch
        # set; if the intersection is empty, preserve the only legal catch
        # rather than obeying a model preference.  TRACK_RETURN and
        # NO_DECISION therefore fall through to the same code geometry.
        if advisory == "PARK_HOME" and phase == "descending" and not urgent:
            lives = self._integer(obs.get("lives"), 0)
            if lives <= 1:
                home_lo = max(safe_lo, 58.0)
                home_hi = min(safe_hi, 94.0)
                if home_lo <= home_hi:
                    safe_lo, safe_hi = home_lo, home_hi
                    target = (home_lo + home_hi) / 2.0

        desired = self._choose_direction(px, safe_lo, safe_hi)

        # If current motion will enter the catch set, neutralize rather than
        # adding another directional frame.  This is the main anti-overshoot
        # difference from a point-target bang-bang servo.  Because NOOP is not
        # promised to brake instantly, allow one measured counter-pulse when
        # the short prediction crosses the far edge of the interval.  The
        # single-pulse gate is important: it cannot become a new oscillator.
        if desired == "NOOP" or predicted_inside:
            if abs(self.paddle_v_avg) > 0.8:
                moving_right = self.paddle_v_avg > 0
                projected_short = px + self.paddle_v_avg * max(1.0, min(2.5, horizon))
                would_cross = (
                    (moving_right and projected_short > safe_hi + guard)
                    or ((not moving_right) and projected_short < safe_lo - guard)
                )
                if would_cross and self.brake_pulses < 1 and (urgent or eta_n <= 12.0):
                    action = OPPOSITE["RIGHT" if moving_right else "LEFT"]
                    self.brake_pulses += 1
                    self.latch_action = action
                    self.latch_until = frame + 1
                    self.neutral_until = frame + 1
                    self.last_command = action
                    return action
            self.latch_action = "NOOP"
            self.latch_until = frame + (1 if urgent else 2)
            self.last_command = "NOOP"
            return "NOOP"

        # Keep a committed direction for at least two frames unless it is now
        # demonstrably driving away from the catch interval.  This prevents
        # the 1-frame LEFT/RIGHT reversals visible at earlier misses.
        if (
            self.latch_action in DIRECTIONS
            and frame < self.latch_until
            and desired != OPPOSITE.get(self.latch_action)
        ):
            action = self.latch_action
        else:
            action = desired
            if action != self.latch_action:
                self.last_switch_frame = frame
            self.latch_action = action
            commit = 1 if urgent else 2
            self.latch_until = frame + commit
            self.brake_pulses = 0
        if action not in ACTIONS:
            action = "NOOP"
        self.last_command = action
        return action

    # ------------------------------------------------------------------
    # Jev advisory interface
    @staticmethod
    def _answer_value(answer: Any) -> str | None:
        value = getattr(answer, "value", None)
        return value if isinstance(value, str) else None

    @staticmethod
    def _answer_probability(answer: Any, value: str) -> float:
        probs = getattr(answer, "probabilities", None)
        if isinstance(probs, dict) and probs:
            try:
                p = float(probs.get(value, 0.0))
                return p if math.isfinite(p) else 0.0
            except (TypeError, ValueError):
                return 0.0
        return 1.0 if getattr(answer, "value", None) == value else 0.0

    def _jev_state(self, obs: dict[str, Any], flight: dict[str, Any], event: str) -> dict[str, Any]:
        side = self._side_summary(obs)
        lives = self._integer(obs.get("lives"), 0)
        motion = "still"
        if self.paddle_v_avg > 0.5:
            motion = "right"
        elif self.paddle_v_avg < -0.5:
            motion = "left"
        # All numbers sent to Jev are named/bucketed facts; code owns the
        # comparisons and the reflected-coordinate arithmetic.
        eta = flight.get("eta")
        eta_bucket = "unknown" if eta is None else (
            "imminent" if self._number(eta) <= 7 else "near" if self._number(eta) <= 20 else "early"
        )
        return {
            "game": "Atari 2600 Breakout ordinary mode 0 difficulty 0",
            "observation": {
                "frame": self._frame(obs),
                "event": event,
                "lives": lives,
                "lives_low": lives <= 1,
                "serve_required": bool(obs.get("serve_required", False)),
                "flight_phase": flight.get("phase", "unknown"),
                "landing_urgency": eta_bucket,
                "landing_lane": flight.get("lane", "unknown"),
                "catch_interval": "overlap" if flight.get("relation") == "overlap_possible" else "outside_or_unknown",
                "paddle_motion": motion,
                "speed_regime": self._speed_name(),
                "plan_stability": "latched" if self.plan_valid else "not_latched",
                "left_edge": side["left"],
                "right_edge": side["right"],
                "wall_index": side["wall"],
                "bricks_remaining": side["bricks_remaining"],
            },
            "controller": {
                "fallback": "code keeps paddle in the computed catch interval",
                "mode_options": ["TRACK_RETURN", "PARK_HOME", "NO_DECISION"],
                "priority": "preserve a catch; never replace a code landing plan with prose",
            },
        }

    def _jev_questions(self, state: dict[str, Any]) -> dict[str, Choice]:
        descriptions = {
            "TRACK_RETURN": "The descending return is below the brick band and its code-computed landing plan is trustworthy; keep tracking that latched catch interval.",
            "PARK_HOME": "The ball is ascending or still in the brick collision band; keep a broad centered home position until code observes a stable descent.",
            "NO_DECISION": "The state is ambiguous or a life is at risk; make no strategic change and let the deterministic fallback act.",
        }
        keys = list(descriptions)
        shift = self.s1_serial % len(keys)
        order_a = keys[shift:] + keys[:shift]
        order_b = list(reversed(order_a))
        instruction = (
            "Classify the already-derived Breakout flight phase. Do not calculate coordinates, "
            "compare raw numbers, or invent an action. Choose TRACK_RETURN only when "
            "`observation.flight_phase` is descending and `observation.plan_stability` is latched. "
            "Choose PARK_HOME only for ascending or collision_band. Choose NO_DECISION for "
            "imminent, lives_low, unknown, or contradictory facts. The controller still owns "
            "every frame and will ignore this answer near impact. State: " + repr(state)
        )
        return {
            "mode_a": Choice(instruction, {k: descriptions[k] for k in order_a}),
            "mode_b": Choice(instruction, {k: descriptions[k] for k in order_b}),
        }

    def _launch_jev(self, obs: dict[str, Any], flight: dict[str, Any], event: str) -> None:
        if not self.s1_enabled or self.s1_future is not None:
            return
        frame = self._frame(obs)
        if frame < self.next_s1_frame or bool(obs.get("serve_required", False)):
            return
        if self._ball(obs) is None:
            return
        state = self._jev_state(obs, flight, event)
        questions = self._jev_questions(state)
        request_version = self.plan_version
        self.s1_request_frame = frame
        self.s1_request_version = request_version
        self.s1_serial += 1
        self.next_s1_frame = frame + (36 if event != "none" else 90)

        def call():
            return self.ctx.decide(state, questions, timeout_s=1.25)

        try:
            self.s1_future = self.ctx.spawn(call)
        except Exception as exc:
            self.s1_future = None
            self._log("jev_spawn_failed", error=type(exc).__name__)

    def _collect_jev(self, frame: int) -> None:
        future = self.s1_future
        if future is None or not future.done():
            return
        self.s1_future = None
        try:
            if frame - self.s1_request_frame > 75 or self.s1_request_version != self.plan_version:
                return
            answers = future.result()
            if not isinstance(answers, dict):
                return
            first = answers.get("mode_a")
            second = answers.get("mode_b")
            a = self._answer_value(first)
            b = self._answer_value(second)
            allowed = {"TRACK_RETURN", "PARK_HOME", "NO_DECISION"}
            if a not in allowed or a != b:
                return
            probability = min(self._answer_probability(first, a), self._answer_probability(second, b))
            if probability < 0.55:
                return
            self.s1_answer = a
            self.s1_answer_frame = self.s1_request_frame
            self.s1_answer_version = self.s1_request_version
            self._log("jev_phase_accepted", frame=frame, phase=a, probability=round(probability, 3))
        except (TimeoutError, RuntimeError, BudgetExceeded, EpisodeOver) as exc:
            self._log("jev_unavailable", error=type(exc).__name__)
        except Exception as exc:
            self._log("jev_parse_ignored", error=type(exc).__name__)

    def _current_advisory(self, frame: int) -> str | None:
        """Return a still-current Jev policy label, never a raw action."""
        if self.s1_answer not in {"TRACK_RETURN", "PARK_HOME", "NO_DECISION"}:
            return None
        if self.s1_answer_version != self.plan_version:
            return None
        if frame - self.s1_answer_frame > 75:
            return None
        return self.s1_answer

    # ------------------------------------------------------------------
    # Episode loop
    def _realtime(self, game: Any) -> None:
        ctx = self.ctx
        while not ctx.done and not bool(getattr(game, "done", False)):
            obs = self._safe_observe(game)
            frame = self._frame(obs)
            self._update_motion(obs)
            event = self._event(obs)
            self._apply_event(obs, event, frame)
            flight = self._flight(obs, frame)

            self._collect_jev(frame)
            self._launch_jev(obs, flight, event)
            action = self._movement_action(obs, flight, frame)
            if action not in ACTIONS:
                action = "NOOP"
            self.action_count += 1
            self.last_frame = frame
            try:
                ctx.act(action)
            except (BudgetExceeded, EpisodeOver):
                return
            self.previous = obs
            try:
                ctx.wait_for_tick()
            except (BudgetExceeded, EpisodeOver):
                return

    def _turn(self, game: Any) -> None:
        # Defensive generic path for pre-flight contract checks.
        ctx = self.ctx
        while not ctx.done and not bool(getattr(game, "done", False)):
            obs = self._safe_observe(game)
            frame = self._frame(obs)
            self._update_motion(obs)
            event = self._event(obs)
            self._apply_event(obs, event, frame)
            flight = self._flight(obs, frame)
            action = self._movement_action(obs, flight, frame)
            if action not in ACTIONS:
                action = "NOOP"
            try:
                ctx.act(action)
            except (BudgetExceeded, EpisodeOver):
                return
            self.previous = obs

    def run_episode(self, game: Any) -> None:
        if bool(getattr(game, "tick_s", 0.0)):
            self._realtime(game)
        else:
            self._turn(game)


def build(ctx: Any) -> Harness:
    return Harness(ctx)
