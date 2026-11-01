"""Hysteretic reflected-intercept controller for ordinary Atari Breakout.

The emulator-facing controller is deliberately code-owned.  It projects the
next descending paddle crossing, learns the paddle's actual displacement from
RAM, and commits to short directional pulses instead of reversing on every
new estimate.  Astra and Jev can select a cautious edge-contact mode, but
neither model is allowed to block or directly drive a native frame.
"""
from __future__ import annotations

import math
import re
from typing import Any

from metaharness.core.contract import BudgetExceeded, EpisodeOver
from metaharness.models.jev import Choice


ACTIONS = ("NOOP", "FIRE", "RIGHT", "LEFT")
DIRECTIONAL = {"RIGHT", "LEFT"}


class Harness:
    def __init__(self, ctx: Any):
        self.ctx = ctx

        # Model mailboxes.  Futures are only polled; the real-time loop never
        # waits for either model.
        self.s1_enabled = getattr(ctx, "system1", None) is not None
        self.s1_future = None
        self.s1_request_frame = -100000
        self.s1_request_version = -1
        self.s1_request_id = 0
        self.s1_next_frame = 24
        self.mode = "SAFE"
        self.mode_frame = -100000
        self.mode_version = -1

        self.astra_future = None
        self.astra_started = False
        self.astra_plan = "UNKNOWN"

        # The strategic version changes only at events that invalidate a model
        # judgment.  Brick hits invalidate geometry but not the long-lived
        # edge preference.
        self.plan_version = 0
        self.previous: dict[str, Any] | None = None
        self.last_frame = -1
        self.last_action = "NOOP"

        # Measured actuator/ball state.  No-op is not assumed to be a brake;
        # the current paddle displacement is measured from successive RAM
        # observations.
        self.paddle_v = 0.0
        self.paddle_v_ema = 0.0
        self.move_step = 1.5
        self.ball_vx = 0.0
        self.ball_vy = 0.0
        self.old_ball_vy = 0.0
        self.noop_drift = 0.0

        # Target lock and pulse hysteresis.  A target may move a little every
        # frame, but a direction is not reversed for a small error change.
        self.target_center: float | None = None
        self.target_epoch = -1
        self.last_phase = "unknown"
        self.command = "NOOP"
        self.command_until = -1
        self.brake_release = -1
        self.brake_frame = -1

        # Contact-side calibration.  LEFT means put the ball on the paddle's
        # right half (initially), which normally sends it left; observations
        # can reverse that sign for this ROM/build.
        self.shape_sign = {"LEFT": 1.0, "RIGHT": 1.0}
        self.pending_shape: str | None = None
        self.pending_shape_frame = -100000
        self.descent_epoch = 0
        self.contact_samples = {"LEFT": 0, "RIGHT": 0}
        self.last_speed = "unknown"

    # ------------------------------------------------------------------
    # Defensive observation and numeric helpers
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

    @classmethod
    def _ball(cls, obs: Any) -> dict[str, Any] | None:
        if isinstance(obs, dict) and isinstance(obs.get("ball"), dict):
            return obs["ball"]
        return None

    @classmethod
    def _paddle(cls, obs: Any) -> dict[str, Any]:
        if isinstance(obs, dict) and isinstance(obs.get("paddle"), dict):
            return obs["paddle"]
        return {}

    def _frame(self, obs: dict[str, Any]) -> int:
        return self._integer(obs.get("frame"), self.last_frame + 1)

    @staticmethod
    def _direction_sign(action: str) -> float:
        return 1.0 if action == "RIGHT" else -1.0

    @staticmethod
    def _opposite(action: str) -> str:
        return "LEFT" if action == "RIGHT" else "RIGHT"

    # ------------------------------------------------------------------
    # Public-state facts and event detection
    def _event(self, obs: dict[str, Any]) -> str:
        old = self.previous
        if old is None:
            return "start"
        old_lives = self._integer(old.get("lives"), 0)
        lives = self._integer(obs.get("lives"), old_lives)
        if lives < old_lives:
            return "life_loss"
        old_wall = self._integer(old.get("wall_index"), 0)
        wall = self._integer(obs.get("wall_index"), old_wall)
        if wall != old_wall:
            return "wall_transition"
        old_score = self._number(old.get("score"), 0.0)
        score = self._number(obs.get("score"), old_score)
        if score > old_score:
            return "brick_hit"
        old_ball = self._ball(old)
        ball = self._ball(obs)
        if old_ball is None and ball is not None:
            return "serve"
        if old_ball is not None and ball is None:
            return "ball_lost"
        return "none"

    def _side_status(self, obs: dict[str, Any]) -> dict[str, Any]:
        """Bucket the two outer brick columns without asking Jev to count."""
        bitmap = obs.get("brick_bitmap") if isinstance(obs, dict) else None
        left = right = None
        if isinstance(bitmap, list):
            left_count = 0
            right_count = 0
            rows = 0
            for row in bitmap:
                if not isinstance(row, list):
                    continue
                rows += 1
                if len(row) >= 2:
                    left_count += int(bool(row[0])) + int(bool(row[1]))
                if len(row) >= 18:
                    right_count += int(bool(row[16])) + int(bool(row[17]))
            if rows:
                left = left_count
                right = right_count

        def bucket(count: int | None) -> str:
            if count is None:
                return "unknown"
            if count <= 0:
                return "open"
            if count <= 3:
                return "partial"
            return "blocked"

        return {
            "left": bucket(left),
            "right": bucket(right),
            "left_count": 0 if left is None else min(12, max(0, left)),
            "right_count": 0 if right is None else min(12, max(0, right)),
            "wall": self._integer(obs.get("wall_index"), 0),
            "walls_cleared": self._integer(obs.get("walls_cleared"), 0),
            "bricks_remaining": self._integer(obs.get("bricks_remaining"), 108),
        }

    def _speed(self) -> str:
        # The fast regime in the traces has a visibly larger vertical step;
        # avoid classifying every ordinary (2,1) diagonal as fast.
        total = abs(self.ball_vx) + abs(self.ball_vy)
        if total < 0.5:
            return "unknown"
        return "fast" if total >= 4.0 else "normal"

    def _measure(self, obs: dict[str, Any]) -> None:
        """Update velocity estimates using only current and prior observations."""
        old = self.previous
        old_paddle = self._paddle(old) if old is not None else {}
        paddle = self._paddle(obs)
        px = self._number(paddle.get("x"), 99.0)
        old_px = self._number(old_paddle.get("x"), px)
        dp = self._clamp(px - old_px, -12.0, 12.0)
        self.paddle_v = dp
        self.paddle_v_ema = 0.70 * self.paddle_v_ema + 0.30 * dp
        if abs(dp) >= 0.5:
            self.move_step = self._clamp(
                0.85 * self.move_step + 0.15 * abs(dp), 0.75, 5.0
            )
        if self.last_action == "NOOP" and old is not None:
            self.noop_drift = 0.88 * self.noop_drift + 0.12 * (
                1.0 if abs(dp) >= 0.5 else 0.0
            )

        old_vx, old_vy = self.ball_vx, self.ball_vy
        ball = self._ball(obs)
        old_ball = self._ball(old) if old is not None else None
        self.old_ball_vy = old_vy
        if ball is None:
            self.ball_vx = 0.0
            self.ball_vy = 0.0
            return

        raw_present = "dx" in ball or "dy" in ball
        raw_vx = self._number(ball.get("dx"), 0.0)
        raw_vy = self._number(ball.get("dy"), 0.0)
        valid = bool(obs.get("ball_velocity_valid", False))
        inferred_vx = inferred_vy = 0.0
        if old_ball is not None:
            bx = self._number(ball.get("x"), 0.0)
            by = self._number(ball.get("y"), 0.0)
            obx = self._number(old_ball.get("x"), bx)
            oby = self._number(old_ball.get("y"), by)
            inferred_vx = self._clamp(bx - obx, -8.0, 8.0)
            inferred_vy = self._clamp(by - oby, -8.0, 8.0)

        if valid or (raw_present and old_ball is not None):
            self.ball_vx, self.ball_vy = raw_vx, raw_vy
        elif abs(inferred_vx) > 0.0 or abs(inferred_vy) > 0.0:
            self.ball_vx, self.ball_vy = inferred_vx, inferred_vy
        elif abs(raw_vx) > 0.0 or abs(raw_vy) > 0.0:
            self.ball_vx, self.ball_vy = raw_vx, raw_vy
        else:
            self.ball_vx, self.ball_vy = old_vx, old_vy

    def _reset_prediction(self, major: bool = False) -> None:
        self.target_center = None
        self.target_epoch = -1
        self.pending_shape = None
        self.brake_release = -1
        self.brake_frame = -1
        self.command = "NOOP"
        self.command_until = -1
        if major:
            self.plan_version += 1
            self.mode = "SAFE"
            self.mode_frame = -100000
            self.mode_version = -1

    def _record_contact(self, obs: dict[str, Any], frame: int) -> None:
        """Learn whether a selected contact mode produced its intended dx."""
        ball = self._ball(obs)
        if ball is None or self.pending_shape is None:
            return
        if frame - self.pending_shape_frame > 42:
            self.pending_shape = None
            return
        # A brick can reverse dy elsewhere, so require a low-field descent to
        # ascent transition.  RAM y is the public decoded ball rectangle.
        by = self._number(ball.get("y"), 0.0)
        old_ball = self._ball(self.previous)
        old_y = self._number(old_ball.get("y"), by) if old_ball else by
        transitioned = self.old_ball_vy > 0.0 and self.ball_vy < 0.0
        if not transitioned or min(by, old_y) < 145.0:
            return
        vx = self.ball_vx
        if abs(vx) < 0.5:
            return
        mode = self.pending_shape
        desired = -1.0 if mode == "LEFT" else 1.0
        observed = -1.0 if vx < 0.0 else 1.0
        self.contact_samples[mode] += 1
        if observed != desired:
            self.shape_sign[mode] *= -1.0
            self.ctx.log(
                "contact_direction_relearned",
                frame=frame,
                mode=mode,
                observed_dx=round(vx, 2),
            )
        self.pending_shape = None

    def _update_events(self, obs: dict[str, Any], event: str, frame: int) -> None:
        if event in ("life_loss", "ball_lost", "wall_transition"):
            self._reset_prediction(major=True)
        elif event == "brick_hit":
            # A brick collision can change dx or the speed regime.  Retain the
            # strategic edge preference but force a new geometric target.
            self.target_center = None
            self.target_epoch = -1
            self.pending_shape = None
            self.command_until = min(self.command_until, frame)

        speed = self._speed()
        if speed == "fast" and self.last_speed != "fast":
            self._reset_prediction(major=True)
        self.last_speed = speed

        side = self._side_status(obs)
        if self.mode == "LEFT" and side["left"] == "open":
            self.mode = "SAFE"
            self.mode_version = -1
        if self.mode == "RIGHT" and side["right"] == "open":
            self.mode = "SAFE"
            self.mode_version = -1
        if self.mode in DIRECTIONAL and frame - self.mode_frame > 180:
            self.mode = "SAFE"
            self.mode_version = -1

    # ------------------------------------------------------------------
    # Reflected projection and target construction
    @staticmethod
    def _reflect_x(x: float, velocity: float, frames: float) -> float:
        left, right = 8.0, 150.0
        span = right - left
        if span <= 0.0:
            return x
        phase = (x - left + velocity * max(0.0, frames)) % (2.0 * span)
        return left + phase if phase <= span else right - (phase - span)

    def _flight(self, obs: dict[str, Any], frame: int) -> dict[str, Any]:
        ball = self._ball(obs)
        paddle = self._paddle(obs)
        px = self._number(paddle.get("x"), 99.0)
        py = self._number(paddle.get("y"), 189.0)
        pw = self._number(paddle.get("w"), 16.0)
        center = px + pw / 2.0
        if ball is None or not bool(obs.get("ball_in_play", False)):
            self.target_center = center
            self.last_phase = "serve"
            return {
                "phase": "serve",
                "urgency": "serve",
                "eta": None,
                "landing": None,
                "uncertainty": 0.0,
                "lane": "unknown",
                "relation": "unknown",
                "target": center,
                "safety_target": center,
                "shape_allowed": False,
                "speed": "unknown",
            }

        bx = self._number(ball.get("x"), 80.0)
        by = self._number(ball.get("y"), 100.0)
        bw = self._number(ball.get("w"), 2.0)
        bh = self._number(ball.get("h"), 4.0)
        vx, vy = self.ball_vx, self.ball_vy
        target_y = py - bh
        speed = self._speed()
        uncertainty = 4.0 if speed != "fast" else 8.0
        phase = "unknown"
        eta: float | None = None
        nominal = center

        if vy > 0.0:
            phase = "descending"
            eta = max(0.0, (target_y - by) / max(0.1, vy))
            nominal = self._reflect_x(bx, vx, eta) + bw / 2.0
            # The reflected estimate is stable during a clean descent.  A
            # small low-pass protects against one sticky/collision outlier
            # without freezing a genuine direction change.
            if self.target_center is None or self.last_phase != "descending":
                self.target_center = nominal
                self.descent_epoch += 1
                self.pending_shape = None
            else:
                self.target_center = 0.65 * self.target_center + 0.35 * nominal
            nominal = self.target_center
            if eta <= 7.0:
                urgency = "imminent"
            elif eta <= 20.0:
                urgency = "near"
            else:
                urgency = "early"
        elif vy < 0.0:
            # While the ball is ascending, bricks may change its eventual
            # return lane.  Use the parent's conservative central anchor,
            # rather than chasing an old or speculative landing; measured
            # velocity/braking keeps that anchor from causing chatter.
            phase = "ascending"
            urgency = "early"
            eta = None
            self.target_center = 84.0
        else:
            phase = "unknown"
            urgency = "uncertain"
            eta = None
            self.target_center = 84.0

        safe_target = self._clamp(self.target_center or center, 16.0, 152.0)
        target = safe_target
        shape_allowed = False
        side = self._side_status(obs)
        lives = self._integer(obs.get("lives"), 0)
        if (
            self.mode in DIRECTIONAL
            and phase == "descending"
            and urgency == "early"
            and eta is not None
            and eta >= 11.0
            and speed == "normal"
            and lives >= 3
            and side[self.mode.lower()] == "blocked"
            and uncertainty <= 4.0
        ):
            # Move the paddle a few pixels to the opposite side of the ball's
            # center, a soft contact-section experiment, never a hard miss.
            sign = self.shape_sign[self.mode]
            offset = 3.0 * sign * (1.0 if self.mode == "LEFT" else -1.0)
            target = safe_target + offset
            shape_allowed = True
            if self.pending_shape is None:
                self.pending_shape = self.mode
                self.pending_shape_frame = frame
        else:
            target = safe_target

        target = self._clamp(target, 16.0, 152.0)
        if target < 56.0:
            lane = "left"
        elif target > 104.0:
            lane = "right"
        else:
            lane = "center"
        gap = target - center
        overlap_margin = max(4.0, (pw + bw) / 3.0)
        if abs(gap) <= overlap_margin:
            relation = "overlap"
        elif gap < 0.0:
            relation = "right_of_landing"
        else:
            relation = "left_of_landing"
        self.last_phase = phase
        return {
            "phase": phase,
            "urgency": urgency,
            "eta": None if eta is None else round(eta, 2),
            "landing": round(safe_target, 2),
            "uncertainty": round(uncertainty, 2),
            "lane": lane,
            "relation": relation,
            "target": round(target, 2),
            "safety_target": round(safe_target, 2),
            "shape_allowed": shape_allowed,
            "speed": speed,
        }

    # ------------------------------------------------------------------
    # Hysteretic measured-velocity servo
    def _movement_action(self, obs: dict[str, Any], flight: dict[str, Any], frame: int) -> str:
        if bool(obs.get("terminated", False)) or bool(obs.get("truncated", False)):
            return "NOOP"
        if bool(obs.get("serve_required", False)):
            return "FIRE"
        ball = self._ball(obs)
        if ball is None or not bool(obs.get("ball_in_play", False)):
            return "NOOP"

        paddle = self._paddle(obs)
        px = self._number(paddle.get("x"), 99.0)
        pw = self._number(paddle.get("w"), 16.0)
        center = px + pw / 2.0
        target = self._number(flight.get("target"), center)
        eta = flight.get("eta")
        eta_n = self._number(eta, 4.0) if eta is not None else 3.0
        urgency = flight.get("urgency")
        if urgency in ("near", "imminent"):
            target = self._number(flight.get("safety_target"), target)

        # Predict only a short paddle drift.  The measured velocity is useful
        # for braking, but it is not allowed to create a long extrapolation.
        horizon = self._clamp(eta_n, 0.0, 6.0)
        drift_gain = 0.60 if self.noop_drift >= 0.25 else 0.35
        predicted_center = center + self.paddle_v_ema * horizon * drift_gain
        error = target - predicted_center

        if urgency == "imminent":
            deadband = 2.0
            reverse_limit = 6.0
        elif urgency == "near":
            deadband = 3.0
            reverse_limit = 8.0
        else:
            deadband = 4.0
            reverse_limit = 10.0

        desired = "NOOP"
        if error > deadband:
            desired = "RIGHT"
        elif error < -deadband:
            desired = "LEFT"

        # The first frame after a deliberate brake is a pulse; release it for
        # a couple of frames rather than immediately issuing the mirror pulse.
        if frame < self.brake_release:
            self.command = "NOOP"
            return "NOOP"

        # Commit to a short pulse.  A small opposite error is handled by
        # releasing the direction, not by a one-frame LEFT/RIGHT oscillation.
        if frame < self.command_until and self.command in DIRECTIONAL:
            if desired == self.command:
                return self.command
            if desired in DIRECTIONAL and abs(error) > reverse_limit + 3.0:
                self.command_until = frame
            else:
                self.command = "NOOP"
                return "NOOP"

        if desired == "NOOP":
            # Counter-steer only when observed motion would carry the paddle
            # clearly outside the overlap corridor.  It has a cooldown so the
            # controller cannot chatter around a target.
            overshooting = (
                abs(self.paddle_v_ema) >= 0.75
                and (target - center) * self.paddle_v_ema < 0.0
                and abs(self.paddle_v_ema) * max(1.0, min(4.0, horizon))
                > deadband + 2.0
            )
            if overshooting and frame >= self.brake_frame + 3:
                action = "LEFT" if self.paddle_v_ema > 0.0 else "RIGHT"
                self.command = action
                self.brake_frame = frame
                self.brake_release = frame + 2
                self.command_until = frame + 1
                return action
            self.command = "NOOP"
            self.command_until = frame + 1
            return "NOOP"

        if self.command in DIRECTIONAL and self.command != desired:
            # Require a meaningful miss-side error before reversing.  Paddle
            # width makes this safe and removes the parent trace's alternating
            # LEFT/RIGHT sequence.
            if abs(error) <= reverse_limit:
                self.command = "NOOP"
                self.command_until = frame + 1
                return "NOOP"

        self.command = desired
        hold = 1 if urgency == "imminent" else 2
        self.command_until = frame + hold
        return desired if desired in ACTIONS else "NOOP"

    # ------------------------------------------------------------------
    # Jev: recognition/choice only, with a code-owned bridge
    @staticmethod
    def _answer_value(answer: Any) -> str | None:
        value = getattr(answer, "value", None)
        return value if isinstance(value, str) else None

    @staticmethod
    def _answer_probability(answer: Any, value: str) -> float:
        probs = getattr(answer, "probabilities", None)
        if isinstance(probs, dict) and probs:
            try:
                return float(probs.get(value, 0.0))
            except (TypeError, ValueError):
                return 0.0
        return 1.0 if getattr(answer, "value", None) == value else 0.0

    def _jev_state(
        self, obs: dict[str, Any], flight: dict[str, Any], event: str, frame: int
    ) -> dict[str, Any]:
        side = self._side_status(obs)
        lives = self._integer(obs.get("lives"), 0)
        if self.paddle_v_ema > 0.5:
            motion = "right"
        elif self.paddle_v_ema < -0.5:
            motion = "left"
        else:
            motion = "still"
        eligible = bool(flight.get("shape_allowed", False))
        return {
            "game": "Atari 2600 Breakout ordinary cartridge mode 0 difficulty 0",
            "observation": {
                "frame": frame,
                "event": event,
                "score": self._integer(self._number(obs.get("score"), 0.0)),
                "lives": lives,
                "low_lives": lives <= 2,
                "serve_required": bool(obs.get("serve_required", False)),
                "flight_phase": flight.get("phase", "unknown"),
                "landing_urgency": flight.get("urgency", "unknown"),
                "landing_lane": flight.get("lane", "unknown"),
                "paddle_relation": flight.get("relation", "unknown"),
                "paddle_motion": motion,
                "speed_regime": flight.get("speed", "unknown"),
                "return_certainty": "wide" if flight.get("uncertainty", 99) <= 4 else "uncertain",
                "left_edge": side["left"],
                "right_edge": side["right"],
                "wall_index": side["wall"],
                "walls_cleared": side["walls_cleared"],
                "bricks_remaining": side["bricks_remaining"],
                "shape_is_code_eligible": eligible,
            },
            "current_code_mode": self.mode,
            "astra_plan": self.astra_plan,
            "rules": {
                "legal_actions": ["NOOP", "FIRE", "RIGHT", "LEFT"],
                "decision": "choose SAFE or an optional edge-contact preference; code owns geometry and action",
                "priority": "preserve paddle overlap before tunnel shaping",
                "safe_conditions": "near/imminent, fast, uncertain, low_lives, or open requested edge means SAFE",
            },
        }

    def _jev_questions(self, state: dict[str, Any]) -> dict[str, Choice]:
        descriptions = {
            "SAFE": "Use the code-computed safety landing and do not deliberately shape the next paddle contact.",
            "LEFT": "If code marks an early, normal-speed, certain return and the left edge is blocked, prefer the calibrated contact that sends the ball left; otherwise code falls back to SAFE.",
            "RIGHT": "If code marks an early, normal-speed, certain return and the right edge is blocked, prefer the calibrated contact that sends the ball right; otherwise code falls back to SAFE.",
        }
        keys = ["SAFE", "LEFT", "RIGHT"]
        shift = self.s1_request_id % len(keys)
        order_a = keys[shift:] + keys[:shift]
        order_b = list(reversed(order_a))
        instruction = (
            "Choose one tactical contact preference for this exact state. "
            "Do not calculate distances or invent an action. The code already "
            "projects the landing and will reject an unsafe preference. Choose "
            "SAFE when landing_urgency is near or imminent, speed_regime is fast, "
            "return_certainty is uncertain, low_lives is true, shape_is_code_eligible "
            "is false, or the requested edge is open. The Astra plan is only a "
            "proposal and must be supported by the current observation. State: "
            + repr(state)
        )
        return {
            "preference_a": Choice(
                instruction, {key: descriptions[key] for key in order_a}
            ),
            "preference_b": Choice(
                instruction, {key: descriptions[key] for key in order_b}
            ),
        }

    def _launch_jev(
        self, obs: dict[str, Any], flight: dict[str, Any], event: str, frame: int
    ) -> None:
        if not self.s1_enabled or self.s1_future is not None:
            return
        if frame < self.s1_next_frame:
            return
        if bool(obs.get("serve_required", False)) or self._ball(obs) is None:
            return
        if flight.get("speed") == "unknown":
            return
        state = self._jev_state(obs, flight, event, frame)
        questions = self._jev_questions(state)
        request_version = self.plan_version
        self.s1_request_frame = frame
        self.s1_request_version = request_version
        self.s1_request_id += 1
        # Event boundaries get a quick refresh; otherwise avoid wasting calls
        # on a preference that is deliberately long-lived.
        self.s1_next_frame = frame + (24 if event in ("life_loss", "wall_transition") else 42)

        def call():
            return self.ctx.decide(state, questions, timeout_s=1.5)

        try:
            self.s1_future = self.ctx.spawn(call)
        except Exception as exc:
            self.s1_future = None
            self.ctx.log("jev_spawn_failed", error=type(exc).__name__)

    def _collect_jev(self, frame: int, obs: dict[str, Any]) -> None:
        future = self.s1_future
        if future is None or not future.done():
            return
        self.s1_future = None
        try:
            if (
                self.s1_request_version != self.plan_version
                or frame - self.s1_request_frame > 100
            ):
                return
            answers = future.result()
            if not isinstance(answers, dict):
                return
            first = answers.get("preference_a")
            second = answers.get("preference_b")
            one = self._answer_value(first)
            two = self._answer_value(second)
            allowed = {"SAFE", "LEFT", "RIGHT"}
            if one not in allowed or one != two:
                return
            probability = min(
                self._answer_probability(first, one),
                self._answer_probability(second, two),
            )
            if probability < 0.50:
                return
            side = self._side_status(obs)
            if one == "LEFT" and side["left"] == "open":
                one = "SAFE"
            if one == "RIGHT" and side["right"] == "open":
                one = "SAFE"
            self.mode = one
            self.mode_frame = self.s1_request_frame
            self.mode_version = self.s1_request_version
            self.ctx.log(
                "jev_preference_accepted",
                frame=frame,
                preference=one,
                probability=round(probability, 3),
            )
        except (TimeoutError, RuntimeError, BudgetExceeded, EpisodeOver) as exc:
            self.ctx.log("jev_unavailable", error=type(exc).__name__)
        except Exception as exc:
            # Pre-flight stand-ins may return arbitrary objects.  Advice is
            # optional, so malformed answers are simply ignored.
            self.ctx.log("jev_parse_ignored", error=type(exc).__name__)

    # ------------------------------------------------------------------
    # Astra: slow strategic proposal, fed back through Jev
    def _astra_prompt(self, obs: dict[str, Any], flight: dict[str, Any]) -> str:
        side = self._side_status(obs)
        return (
            "You are the slow strategic advisor for Atari 2600 Breakout, "
            "ordinary mode 0 difficulty 0. The code controller predicts the "
            "paddle crossing at 60 Hz and protects the ball. Return exactly one "
            "token: SAFE, LEFT, or RIGHT. LEFT/RIGHT is only a long-lived edge "
            "contact preference for a normal-speed, safe, blocked edge; it is "
            "not a frame-by-frame action. Choose SAFE if lives are low, the ball "
            "is fast/uncertain, or no blocked edge is credible. Do not give math "
            "or a trace.\nPublic summary:\n"
            + repr(
                {
                    "frame": self._frame(obs),
                    "lives": self._integer(obs.get("lives"), 0),
                    "score": self._integer(self._number(obs.get("score"), 0.0)),
                    "flight_phase": flight.get("phase"),
                    "landing_urgency": flight.get("urgency"),
                    "speed_regime": flight.get("speed"),
                    "left_edge": side["left"],
                    "right_edge": side["right"],
                    "wall_index": side["wall"],
                    "bricks_remaining": side["bricks_remaining"],
                }
            )
        )

    def _launch_astra(self, obs: dict[str, Any], flight: dict[str, Any], frame: int) -> None:
        if self.astra_started or self.astra_future is not None:
            return
        if frame < 30 or self._ball(obs) is None or bool(obs.get("serve_required", False)):
            return
        try:
            remaining = self.ctx.budget.remaining()
            if int(remaining.get("system2_calls", 0)) <= 0:
                return
        except Exception:
            pass
        self.astra_started = True
        prompt = self._astra_prompt(obs, flight)

        def call():
            return self.ctx.think(prompt, timeout_s=65.0)

        try:
            self.astra_future = self.ctx.spawn(call)
        except Exception as exc:
            self.astra_future = None
            self.ctx.log("astra_spawn_failed", error=type(exc).__name__)

    def _collect_astra(self, frame: int) -> None:
        future = self.astra_future
        if future is None or not future.done():
            return
        self.astra_future = None
        try:
            result = future.result()
            text = getattr(result, "text", "")
            if not isinstance(text, str):
                return
            match = re.search(r"(?im)^\s*(SAFE|SURVIVE|LEFT|RIGHT)\b", text)
            if match is None:
                return
            token = match.group(1).upper()
            self.astra_plan = "SAFE" if token in ("SAFE", "SURVIVE") else token
            # Jev must validate the proposal against a new observation.
            self.plan_version += 1
            self.mode = "SAFE"
            self.mode_version = -1
            self.s1_next_frame = min(self.s1_next_frame, frame + 1)
            self.ctx.log("astra_plan_received", frame=frame, plan=self.astra_plan)
        except (TimeoutError, RuntimeError, BudgetExceeded, EpisodeOver) as exc:
            self.ctx.log("astra_unavailable", error=type(exc).__name__)
        except Exception as exc:
            self.ctx.log("astra_parse_ignored", error=type(exc).__name__)

    # ------------------------------------------------------------------
    # Episode loop
    @staticmethod
    def _safe_observe(game: Any) -> dict[str, Any]:
        try:
            value = game.observe()
        except Exception:
            return {}
        return value if isinstance(value, dict) else {}

    def _run_realtime(self, game: Any) -> None:
        ctx = self.ctx
        while not ctx.done and not bool(getattr(game, "done", False)):
            obs = self._safe_observe(game)
            frame = self._frame(obs)
            event = self._event(obs)
            self._measure(obs)
            self._record_contact(obs, frame)
            self._update_events(obs, event, frame)
            flight = self._flight(obs, frame)

            self._collect_jev(frame, obs)
            self._collect_astra(frame)
            self._launch_astra(obs, flight, frame)
            self._launch_jev(obs, flight, event, frame)

            action = self._movement_action(obs, flight, frame)
            if action not in ACTIONS:
                action = "NOOP"
            self.last_action = action
            self.last_frame = frame
            try:
                ctx.act(action)
            except (BudgetExceeded, EpisodeOver):
                return
            except Exception as exc:
                # A malformed internal decision should degrade to a legal
                # no-op rather than escape the episode contract.
                self.ctx.log("action_post_failed", error=type(exc).__name__)
                return
            self.previous = obs
            try:
                ctx.wait_for_tick()
            except (BudgetExceeded, EpisodeOver):
                return

    def run_episode(self, game: Any) -> None:
        if bool(getattr(game, "tick_s", 0.0)):
            self._run_realtime(game)
            return

        # The task is tier 2, but keep a legal fallback for generic pre-flight.
        ctx = self.ctx
        while not ctx.done and not bool(getattr(game, "done", False)):
            obs = self._safe_observe(game)
            frame = self._frame(obs)
            self._measure(obs)
            flight = self._flight(obs, frame)
            action = self._movement_action(obs, flight, frame)
            if action not in ACTIONS:
                action = "NOOP"
            self.last_action = action
            try:
                ctx.act(action)
            except (BudgetExceeded, EpisodeOver):
                return
            except Exception as exc:
                self.ctx.log("action_post_failed", error=type(exc).__name__)
                return
            self.previous = obs


def build(ctx: Any) -> Harness:
    return Harness(ctx)
