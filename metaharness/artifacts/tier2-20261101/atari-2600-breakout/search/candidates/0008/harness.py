"""Velocity-aware Breakout controller with cautious contact shaping.

The fast part of this harness is deliberately deterministic.  It predicts a
landing interval, estimates the paddle's actual one-frame motion from RAM, and
uses a small bang-bang servo rather than treating NOOP as an instantaneous
brake.  Jev chooses only a long-lived contact mode; Astra, when it finishes,
provides a strategic proposal that Jev must still validate.
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

        # Mailboxes.  A model result is never allowed to block the frame loop.
        self.s1_enabled = getattr(ctx, "system1", object()) is not None
        self.s1_future = None
        self.s1_request_frame = -100000
        self.s1_request_version = -1
        self.s1_request_number = 0
        self.s1_answer = "SAFE"
        self.s1_answer_frame = -100000
        self.s1_answer_version = -1
        self.next_s1_frame = 20

        self.astra_future = None
        self.astra_started = False
        self.astra_plan = "UNSET"
        self.astra_calls = 0

        # Per-episode state.  It is intentionally not put in ctx.memory: the
        # evaluator starts each episode cold.
        self.previous: dict[str, Any] | None = None
        self.last_frame = -1
        self.last_action = "NOOP"
        self.plan_version = 0
        self.last_event = "start"
        self.mode = "SAFE"
        self.mode_frame = -100000
        self.last_speed = "unknown"

        # Measured motion, not a hard-coded paddle physics assumption.
        self.paddle_v = 0.0
        self.paddle_v_avg = 0.0
        self.move_step = 2.0
        self.noop_drift = 0.0
        self.noop_samples = 0

        # The decoder exposes velocity, but this also survives transient
        # invalid velocity fields around a serve or a collision.
        self.ball_vx = 0.0
        self.ball_vy = 0.0

        # Contact calibration: the manual does not specify the numeric section
        # table, so learn whether a small offset actually sends the ball left
        # or right in this ROM.
        # Positive offset means move the paddle center to the right of the
        # ball.  That is the initial left-going contact guess; RIGHT is the
        # mirror image below.  Each entry is a learned sign multiplier.
        self.side_offset_sign = {"LEFT": 1.0, "RIGHT": 1.0}
        self.last_contact_mode = "SAFE"
        self.last_contact_offset = 0.0
        self.last_contact_frame = -100000
        # A shaping choice is made before the approach becomes imminent; keep
        # it pending until the subsequent paddle contact, even though the
        # lethal-window controller itself switches back to SAFE.
        self.pending_contact_mode = "SAFE"
        self.pending_contact_frame = -100000

    # ------------------------------------------------------------------
    # Defensive and numeric helpers
    @staticmethod
    def _number(value: Any, default: float = 0.0) -> float:
        try:
            result = float(value)
            if not math.isfinite(result):
                return default
            return result
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

    # ------------------------------------------------------------------
    # Geometry and state extraction
    @staticmethod
    def _reflect_x(x: float, velocity: float, frames: float) -> float:
        """Reflect a ball-left coordinate between the Atari side walls."""
        left, right = 8.0, 150.0
        if right <= left:
            return x
        phase = (x - left + velocity * max(0.0, frames)) % (2.0 * (right - left))
        if phase <= right - left:
            return left + phase
        return right - (phase - (right - left))

    def _side_status(self, obs: dict[str, Any]) -> dict[str, Any]:
        bitmap = obs.get("brick_bitmap")
        counts = {"left": 999, "right": 999}
        if isinstance(bitmap, list):
            left = 0
            right = 0
            valid_rows = 0
            for row in bitmap:
                if not isinstance(row, list):
                    continue
                valid_rows += 1
                if len(row) >= 2:
                    left += int(bool(row[0])) + int(bool(row[1]))
                if len(row) >= 18:
                    right += int(bool(row[16])) + int(bool(row[17]))
            if valid_rows:
                counts = {"left": left, "right": right}

        def bucket(count: int) -> str:
            if count <= 0:
                return "open"
            if count <= 3:
                return "partial"
            return "blocked"

        return {
            "left": bucket(counts["left"]),
            "right": bucket(counts["right"]),
            "left_count": min(12, max(0, counts["left"])),
            "right_count": min(12, max(0, counts["right"])),
            "wall": self._integer(obs.get("wall_index"), 0),
            "walls_cleared": self._integer(obs.get("walls_cleared"), 0),
            "bricks_remaining": self._integer(obs.get("bricks_remaining"), 108),
        }

    def _update_tracker(self, obs: dict[str, Any]) -> None:
        """Learn one-frame motion and repair transient RAM velocities."""
        old = self.previous
        paddle = self._paddle(obs)
        px = self._number(paddle.get("x"), 99.0)
        if old is not None:
            old_paddle = self._paddle(old)
            old_px = self._number(old_paddle.get("x"), px)
            dp = self._clamp(px - old_px, -12.0, 12.0)
            self.paddle_v = dp
            self.paddle_v_avg = 0.65 * self.paddle_v_avg + 0.35 * dp
            if abs(dp) >= 0.5:
                self.move_step = self._clamp(
                    0.85 * self.move_step + 0.15 * abs(dp), 0.75, 5.0
                )
            if self.last_action == "NOOP":
                self.noop_samples += 1
                drift = 1.0 if abs(dp) >= 0.5 else 0.0
                self.noop_drift = 0.88 * self.noop_drift + 0.12 * drift

        ball = self._ball(obs)
        old_ball = self._ball(old) if old is not None else None
        if ball is None:
            self.ball_vx = 0.0
            self.ball_vy = 0.0
        else:
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

            if valid and (abs(raw_vx) > 0.0 or abs(raw_vy) > 0.0):
                self.ball_vx, self.ball_vy = raw_vx, raw_vy
            elif old_ball is not None and (
                abs(inferred_vx) > 0.0 or abs(inferred_vy) > 0.0
            ):
                self.ball_vx, self.ball_vy = inferred_vx, inferred_vy
            elif abs(raw_vx) > 0.0 or abs(raw_vy) > 0.0:
                self.ball_vx, self.ball_vy = raw_vx, raw_vy

    def _event(self, obs: dict[str, Any]) -> str:
        old = self.previous
        if old is None:
            return "start"
        old_lives = self._integer(old.get("lives"), 0)
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
        new_ball = self._ball(obs)
        if old_ball is None and new_ball is not None:
            return "serve"
        if old_ball is not None and new_ball is None:
            return "ball_lost"
        return "none"

    def _speed(self) -> str:
        total = abs(self.ball_vx) + abs(self.ball_vy)
        if total < 0.5:
            return "unknown"
        return "fast" if total >= 3.0 else "normal"

    def _contact_seen(self, obs: dict[str, Any], frame: int) -> None:
        """Use observed post-paddle dx to correct the contact-section sign."""
        if self.previous is None or self.pending_contact_mode not in ("LEFT", "RIGHT"):
            return
        if frame - self.pending_contact_frame > 24:
            self.pending_contact_mode = "SAFE"
            return
        old_ball = self._ball(self.previous)
        ball = self._ball(obs)
        if old_ball is None or ball is None:
            return
        old_y = self._number(old_ball.get("y"), 0.0)
        new_y = self._number(ball.get("y"), old_y)
        old_dy = self._number(old_ball.get("dy"), 0.0)
        new_dy = self._number(ball.get("dy"), self.ball_vy)
        # A descent-to-ascent transition close to the paddle is a contact.
        # Do not treat ordinary descending frames as contacts merely because
        # the old velocity was positive.
        transitioned_up = (old_dy > 0.0 and new_dy < 0.0) or new_y < old_y
        if old_y < 140.0 or not transitioned_up:
            return
        vx = self.ball_vx
        if abs(vx) < 0.5:
            return
        mode = self.pending_contact_mode
        desired = -1.0 if mode == "LEFT" else 1.0
        observed = -1.0 if vx < 0.0 else 1.0
        if observed != desired:
            self.side_offset_sign[mode] *= -1.0
            self.ctx.log(
                "contact_sign_reversed",
                frame=frame,
                mode=mode,
                observed_dx=round(vx, 2),
            )
        self.pending_contact_mode = "SAFE"
        self.last_contact_mode = "SAFE"

    def _apply_event(self, obs: dict[str, Any], event: str, frame: int) -> None:
        speed = self._speed()
        if event in ("life_loss", "ball_lost", "wall_transition"):
            self.mode = "SAFE"
            self.mode_frame = -100000
            self.pending_contact_mode = "SAFE"
            self.last_contact_mode = "SAFE"
            self.plan_version += 1
            self.next_s1_frame = min(self.next_s1_frame, frame + 18)
        if speed != self.last_speed and speed == "fast":
            self.mode = "SAFE"
            self.plan_version += 1
            self.next_s1_frame = min(self.next_s1_frame, frame + 24)
        if event == "wall_transition":
            self.astra_plan = "UNSET"
        self.last_speed = speed

        side = self._side_status(obs)
        if self.mode in ("LEFT", "RIGHT") and frame - self.mode_frame > 240:
            self.mode = "SAFE"
        if self.mode == "LEFT" and side["left"] == "open":
            self.mode = "SAFE"
        if self.mode == "RIGHT" and side["right"] == "open":
            self.mode = "SAFE"

    def _flight(self, obs: dict[str, Any]) -> dict[str, Any]:
        """Build a compact landing interval and an executable contact target."""
        ball = self._ball(obs)
        paddle = self._paddle(obs)
        px = self._number(paddle.get("x"), 99.0)
        py = self._number(paddle.get("y"), 189.0)
        pw = self._number(paddle.get("w"), 16.0)
        if ball is None or not bool(obs.get("ball_in_play", False)):
            return {
                "phase": "serve",
                "urgency": "serve",
                "eta": None,
                "landing": None,
                "landing_lo": None,
                "landing_hi": None,
                "uncertainty": 0.0,
                "lane": "unknown",
                "relation": "unknown",
                "target": px + pw / 2.0,
                "safety_target": px + pw / 2.0,
                "contact_mode": "SAFE",
            }

        bx = self._number(ball.get("x"), 80.0)
        by = self._number(ball.get("y"), 100.0)
        bw = self._number(ball.get("w"), 2.0)
        bh = self._number(ball.get("h"), 4.0)
        vx = self.ball_vx
        vy = self.ball_vy
        target_y = py - bh
        top_y = 50.0
        eta: float | None = None
        nominal = bx + bw / 2.0
        uncertainty = 12.0
        phase = "unknown"

        if vy > 0.0:
            phase = "descending"
            eta = max(0.0, (target_y - by) / vy)
            nominal_left = self._reflect_x(bx, vx, eta)
            nominal = nominal_left + bw / 2.0
            # A small velocity/x envelope covers sticky-action and collision
            # timing uncertainty without asking Jev to do arithmetic.
            candidates = []
            for xoff in (-1.0, 0.0, 1.0):
                for voff in (-0.5, 0.0, 0.5):
                    candidates.append(
                        self._reflect_x(bx + xoff, vx + voff, eta) + bw / 2.0
                    )
            low, high = min(candidates), max(candidates)
            uncertainty = self._clamp((high - low) / 2.0, 1.0, 18.0)
        elif vy < 0.0:
            phase = "ascending"
            speed = max(1.0, abs(vy))
            to_top = max(0.0, (by - top_y) / speed)
            from_top = max(0.0, (target_y - top_y) / speed)
            eta = to_top + from_top
            nominal = self._reflect_x(bx, vx, eta) + bw / 2.0
            current = bx + bw / 2.0
            # Bricks can change the return lane, so only move part-way toward
            # the no-brick return estimate.  Descending observations take over
            # as soon as the estimate is trustworthy.
            blend = 0.28 if by >= 100.0 else 0.38
            nominal = current + blend * (nominal - current)
            uncertainty = self._clamp(10.0 + abs(nominal - current) * 0.25, 10.0, 24.0)
        else:
            phase = "unknown"
            eta = None
            nominal = px + pw / 2.0
            uncertainty = 20.0

        if eta is None:
            urgency = "unknown"
        elif eta <= 6.0:
            urgency = "imminent"
        elif eta <= 20.0:
            urgency = "near"
        else:
            urgency = "early"

        if nominal < 56.0:
            lane = "left"
        elif nominal > 104.0:
            lane = "right"
        else:
            lane = "center"
        paddle_center = px + pw / 2.0
        difference = nominal - paddle_center
        if abs(difference) <= max(3.0, pw / 3.0):
            relation = "overlap"
        elif difference < 0.0:
            # The paddle center is to the right when landing - paddle < 0.
            relation = "right_of_landing"
        else:
            relation = "left_of_landing"

        target = nominal
        contact_mode = "SAFE"
        side = self._side_status(obs)
        lives = self._integer(obs.get("lives"), 0)
        # Contact shaping is optional and fenced off from the lethal phase.
        if (
            self.mode in ("LEFT", "RIGHT")
            and phase == "descending"
            and urgency not in ("near", "imminent")
            and eta is not None
            and eta >= 5.0
            and self.last_speed == "normal"
            and lives >= 2
            and side[self.mode.lower()] != "open"
            and uncertainty <= 9.0
            and 25.0 <= nominal <= 135.0
        ):
            contact_mode = self.mode
            amount = 4.0 * self.side_offset_sign[contact_mode]
            if contact_mode == "RIGHT":
                amount *= -1.0
            target = nominal + amount
            self.last_contact_mode = contact_mode
            self.last_contact_offset = amount
            self.last_contact_frame = self._frame(obs)
            self.pending_contact_mode = contact_mode
            self.pending_contact_frame = self._frame(obs)

        # The paddle-left bounds observed in the adaptor are 8..144.  Keep the
        # target center within the corresponding center interval.
        lo_center = 8.0 + pw / 2.0
        hi_center = 144.0 + pw / 2.0
        target = self._clamp(target, lo_center, hi_center)
        return {
            "phase": phase,
            "urgency": urgency,
            "eta": None if eta is None else round(eta, 2),
            "landing": round(nominal, 2),
            "landing_lo": round(nominal - uncertainty, 2),
            "landing_hi": round(nominal + uncertainty, 2),
            "uncertainty": round(uncertainty, 2),
            "lane": lane,
            "relation": relation,
            "target": round(target, 2),
            "safety_target": round(nominal, 2),
            "contact_mode": contact_mode,
        }

    # ------------------------------------------------------------------
    # Fast servo
    def _movement_action(self, obs: dict[str, Any], flight: dict[str, Any]) -> str:
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
        eta_n = self._number(eta, 8.0) if eta is not None else 8.0
        urgency = flight.get("urgency")

        # Never trade an imminent catch for a tunnel contact section.
        if urgency in ("near", "imminent"):
            target = self._number(flight.get("safety_target"), target)

        # Estimate where the paddle will be before the next meaningful
        # correction.  If NOOP has empirically drifted, trust more of the
        # measured velocity; otherwise use only a short inertia allowance.
        horizon = self._clamp(eta_n, 0.0, 8.0)
        drift_factor = 0.75 if self.noop_drift >= 0.25 else 0.25
        predicted = center + self.paddle_v_avg * horizon * drift_factor
        error = target - predicted
        deadband = max(1.5, min(5.0, self._number(flight.get("uncertainty"), 4.0) * 0.18))

        # At the impact window, use the safety lane and the currently measured
        # paddle position.  A one-frame counter-steer is safer than assuming
        # NOOP brakes, but only when drift has actually been observed.
        if urgency == "imminent":
            safety = self._number(flight.get("safety_target"), target)
            if abs(safety - center) > max(2.0, pw * 0.30):
                error = safety - center
            else:
                error = target - predicted

        if error > deadband:
            action = "RIGHT"
        elif error < -deadband:
            action = "LEFT"
        else:
            action = "NOOP"
            if self.noop_drift >= 0.25 and abs(self.paddle_v) >= 0.75:
                # Brake only if current motion would carry the center past the
                # target.  This is the structural change over candidate 0005.
                future = center + self.paddle_v * max(1.0, min(4.0, horizon))
                if future > target + deadband:
                    action = "LEFT"
                elif future < target - deadband:
                    action = "RIGHT"

        # `_flight` records a side attempt in pending_contact_mode.  Do not
        # overwrite it here when urgency changes to SAFE; the next few frames
        # are still the same approach to the intended contact.
        return action if action in ACTIONS else "NOOP"

    # ------------------------------------------------------------------
    # Jev interface
    @staticmethod
    def _answer_value(answer: Any) -> str | None:
        value = getattr(answer, "value", None)
        return value if isinstance(value, str) else None

    @staticmethod
    def _answer_probability(answer: Any, value: str) -> float:
        probabilities = getattr(answer, "probabilities", None)
        if isinstance(probabilities, dict) and probabilities:
            try:
                return float(probabilities.get(value, 0.0))
            except (TypeError, ValueError):
                return 0.0
        # Pre-flight stand-ins often provide only a value.
        return 1.0 if getattr(answer, "value", None) == value else 0.0

    def _jev_questions(self, state: dict[str, Any]) -> dict[str, Choice]:
        descriptions = {
            "SAFE": {
                "what": "Keep the paddle centered on the code-computed safety landing and preserve the current ball.",
                "not_for": "Do not attempt a deliberate paddle-section angle.",
            },
            "LEFT": {
                "what": "When the code marks the landing early and safe, bias contact toward the paddle section that empirically sends the ball left to work on the left edge.",
                "not_for": "Never override an imminent catch, a fast ball, a low-life state, or an open left edge.",
            },
            "RIGHT": {
                "what": "When the code marks the landing early and safe, bias contact toward the paddle section that empirically sends the ball right to work on the right edge.",
                "not_for": "Never override an imminent catch, a fast ball, a low-life state, or an open right edge.",
            },
        }
        keys = ["SAFE", "LEFT", "RIGHT"]
        shift = self.s1_request_number % len(keys)
        order_a = keys[shift:] + keys[:shift]
        order_b = list(reversed(order_a))
        instruction = (
            "Choose the tactical contact mode for this exact Breakout state. "
            "The fields are already bucketed by code; do not calculate distances. "
            "If landing_urgency is near or imminent, speed_regime is fast, "
            "lives_are_low is true, or the requested edge is open, choose SAFE. "
            "A side choice is advisory: the code still owns the catch and may ignore it. "
            "The Astra proposal is only a suggestion and must be supported by the current fields. "
            "State: " + repr(state)
        )
        return {
            "mode_a": Choice(instruction, {key: descriptions[key] for key in order_a}),
            "mode_b": Choice(instruction, {key: descriptions[key] for key in order_b}),
        }

    def _jev_state(self, obs: dict[str, Any], flight: dict[str, Any], event: str) -> dict[str, Any]:
        side = self._side_status(obs)
        lives = self._integer(obs.get("lives"), 0)
        paddle_motion = "still"
        if self.paddle_v_avg > 0.5:
            paddle_motion = "right"
        elif self.paddle_v_avg < -0.5:
            paddle_motion = "left"
        return {
            "game": "Atari 2600 Breakout, ordinary mode 0, difficulty 0",
            "observation": {
                "frame": self._frame(obs),
                "event": event,
                "score": self._integer(self._number(obs.get("score"), 0.0)),
                "lives": lives,
                "lives_are_low": lives <= 1,
                "serve_required": bool(obs.get("serve_required", False)),
                "flight_phase": flight.get("phase", "unknown"),
                "landing_urgency": flight.get("urgency", "unknown"),
                "landing_lane": flight.get("lane", "unknown"),
                "paddle_relation": flight.get("relation", "unknown"),
                "paddle_motion": paddle_motion,
                "speed_regime": self._speed(),
                "safety_margin": "wide" if flight.get("uncertainty", 99.0) <= 5.0 else "uncertain",
                "left_edge": side["left"],
                "right_edge": side["right"],
                "wall_index": side["wall"],
                "walls_cleared": side["walls_cleared"],
                "bricks_remaining": side["bricks_remaining"],
            },
            "current_code_mode": self.mode,
            "astra_proposal": self.astra_plan,
            "rules": {
                "legal_actions": ["NOOP", "FIRE", "RIGHT", "LEFT"],
                "priority": "catchability before brick shaping",
                "serve": "FIRE only while serve_required is true",
            },
        }

    def _launch_jev(self, obs: dict[str, Any], flight: dict[str, Any], event: str) -> None:
        if not self.s1_enabled or self.s1_future is not None:
            return
        frame = self._frame(obs)
        if frame < self.next_s1_frame or bool(obs.get("serve_required", False)):
            return
        if self._ball(obs) is None:
            return
        # Do not ask while the decoder has not established a velocity.
        if self._speed() == "unknown":
            return
        state = self._jev_state(obs, flight, event)
        questions = self._jev_questions(state)
        version = self.plan_version
        request_frame = frame
        self.s1_request_frame = frame
        self.s1_request_version = version
        self.s1_request_number += 1
        self.next_s1_frame = frame + (24 if event != "none" else 60)

        def call():
            return self.ctx.decide(state, questions, timeout_s=1.5)

        try:
            self.s1_future = self.ctx.spawn(call)
        except Exception as exc:
            self.s1_future = None
            self.ctx.log("jev_spawn_failed", error=type(exc).__name__)

    def _collect_jev(self, frame: int) -> None:
        future = self.s1_future
        if future is None or not future.done():
            return
        self.s1_future = None
        try:
            # A result judged before a life/speed/wall transition is not a
            # current tactical fact, even if the backend answered promptly.
            if (
                self.s1_request_version != self.plan_version
                or frame - self.s1_request_frame > 90
            ):
                return
            answers = future.result()
            if not isinstance(answers, dict):
                return
            a = answers.get("mode_a")
            b = answers.get("mode_b")
            av = self._answer_value(a)
            bv = self._answer_value(b)
            allowed = {"SAFE", "LEFT", "RIGHT"}
            if av not in allowed or av != bv:
                return
            probability = min(
                self._answer_probability(a, av), self._answer_probability(b, bv)
            )
            if probability < 0.50:
                return
            if av in ("LEFT", "RIGHT"):
                side = self._side_status(self.previous or {})
                if side.get(av.lower()) == "open":
                    av = "SAFE"
            self.mode = av
            self.mode_frame = self.s1_request_frame
            self.s1_answer = av
            self.s1_answer_frame = self.s1_request_frame
            self.s1_answer_version = self.s1_request_version
            self.ctx.log(
                "jev_mode_accepted",
                frame=frame,
                mode=av,
                probability=round(probability, 3),
            )
        except (TimeoutError, RuntimeError, BudgetExceeded, EpisodeOver) as exc:
            self.ctx.log("jev_unavailable", error=type(exc).__name__)
        except Exception as exc:
            # Arbitrary stand-in answers and malformed provider objects are
            # normal pre-flight inputs; they must not kill the game loop.
            self.ctx.log("jev_parse_ignored", error=type(exc).__name__)

    # ------------------------------------------------------------------
    # Astra interface: one asynchronous proposal, never direct control
    def _astra_prompt(self, obs: dict[str, Any], flight: dict[str, Any]) -> str:
        side = self._side_status(obs)
        return (
            "You are the slow strategic layer for ordinary Atari 2600 Breakout. "
            "The code controller predicts the paddle landing every native frame. "
            "Return exactly one token: SAFE, LEFT, or RIGHT. LEFT/RIGHT means a "
            "cautious paddle-section shaping attempt toward that edge only when "
            "the edge is not open, the ball is normal speed, and there are at least "
            "two lives. Choose SAFE if any fact is uncertain. Never give an action "
            "trace or arithmetic.\n"
            "Public state: "
            + repr(
                {
                    "frame": self._frame(obs),
                    "lives": self._integer(obs.get("lives"), 0),
                    "score": self._integer(self._number(obs.get("score"), 0.0)),
                    "flight": {
                        "phase": flight.get("phase"),
                        "urgency": flight.get("urgency"),
                        "lane": flight.get("lane"),
                    },
                    "speed": self._speed(),
                    "left_edge": side["left"],
                    "right_edge": side["right"],
                    "wall": side["wall"],
                    "bricks_remaining": side["bricks_remaining"],
                }
            )
        )

    def _launch_astra(self, obs: dict[str, Any], flight: dict[str, Any]) -> None:
        if self.astra_started or self.astra_future is not None:
            return
        if self._frame(obs) < 24 or self._ball(obs) is None:
            return
        try:
            remaining = self.ctx.budget.remaining()
            if int(remaining.get("system2_calls", 0)) <= 0:
                return
        except Exception:
            pass
        self.astra_started = True
        self.astra_calls += 1
        prompt = self._astra_prompt(obs, flight)

        def call():
            # The output is intentionally tiny, but the call remains fully
            # asynchronous because Astra can take longer than a rally.
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
            match = re.search(r"\b(SAFE|SURVIVE|LEFT|RIGHT)\b", text.upper())
            if match is None:
                return
            proposal = match.group(1)
            self.astra_plan = "SAFE" if proposal in ("SAFE", "SURVIVE") else proposal
            self.plan_version += 1
            self.mode = "SAFE"
            self.mode_frame = -100000
            self.ctx.log("astra_proposal_accepted", frame=frame, proposal=self.astra_plan)
        except (TimeoutError, RuntimeError, BudgetExceeded, EpisodeOver) as exc:
            self.ctx.log("astra_unavailable", error=type(exc).__name__)
        except Exception as exc:
            self.ctx.log("astra_parse_ignored", error=type(exc).__name__)

    # ------------------------------------------------------------------
    # Episode loop
    @staticmethod
    def _safe_observe(game: Any) -> dict[str, Any]:
        try:
            obs = game.observe()
        except Exception:
            return {}
        return obs if isinstance(obs, dict) else {}

    def _run_realtime(self, game: Any) -> None:
        ctx = self.ctx
        while not ctx.done and not bool(getattr(game, "done", False)):
            obs = self._safe_observe(game)
            frame = self._frame(obs)
            self._update_tracker(obs)
            event = self._event(obs)
            self._contact_seen(obs, frame)
            self._apply_event(obs, event, frame)
            flight = self._flight(obs)

            self._collect_jev(frame)
            self._collect_astra(frame)
            self._launch_astra(obs, flight)
            self._launch_jev(obs, flight, event)

            action = self._movement_action(obs, flight)
            if action not in ACTIONS:
                action = "NOOP"
            self.last_action = action
            self.last_frame = frame
            self.last_event = event
            try:
                ctx.act(action)
            except (BudgetExceeded, EpisodeOver):
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

        # The target is tier 2, but retaining a legal turn-game path makes
        # pre-flight and generic contract checks harmless.
        ctx = self.ctx
        while not ctx.done and not bool(getattr(game, "done", False)):
            obs = self._safe_observe(game)
            self._update_tracker(obs)
            flight = self._flight(obs)
            action = self._movement_action(obs, flight)
            if action not in ACTIONS:
                action = "NOOP"
            self.last_action = action
            try:
                ctx.act(action)
            except (BudgetExceeded, EpisodeOver):
                return
            self.previous = obs


def build(ctx: Any) -> Harness:
    return Harness(ctx)
