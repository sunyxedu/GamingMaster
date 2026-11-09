"""Event-locked corridor controller for Atari 2600 Breakout.

The emulator is never stopped for inference.  Code owns the 60 Hz loop, ball
reflection, a latched catch corridor, and a small motor state machine.  Astra
and Jev select a *contact objective* (preserve, or use a safe left/right edge
of the paddle); neither model emits a frame action.
"""
from __future__ import annotations

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
DIRECTIONS = (LEFT, RIGHT)


class Harness:
    def __init__(self, ctx: Any):
        self.ctx = ctx

        # The loop only reads completed futures.  A model is never on the
        # critical path and arbitrary pre-flight stand-ins are harmless.
        self.jev_enabled = getattr(ctx, "system1", None) is not None
        self.jev_future = None
        self.jev_frame = -100000
        self.jev_epoch = -1
        self.jev_serial = 0
        self.next_jev_frame = 0

        self.astra_future = None
        self.astra_started = False
        self.astra_calls = 0
        self.astra_plan = "AUTO"
        self.astra_epoch = 0

        # Observation history and measured actuator/ball motion.
        self.previous: dict[str, Any] | None = None
        self.last_frame = -1
        self.last_action = NOOP
        self.paddle_v = 0.0
        self.paddle_v_avg = 0.0
        self.noop_drift = 0.0
        self.ball_vx = 0.0
        self.ball_vy = 0.0
        self.last_speed = "unknown"

        # A return epoch changes only on a real event.  The corridor is latched
        # through small one-pixel RAM/velocity fluctuations instead of chasing
        # a new point every frame.
        self.return_epoch = 0
        self.corridor_valid = False
        self.corridor_low = 64.0
        self.corridor_high = 88.0
        self.corridor_center = 76.0
        self.corridor_landing = None
        self.corridor_eta = None
        self.corridor_vx = 0.0
        self.corridor_vy = 0.0
        self.corridor_frame = -1
        self.return_shape_side: str | None = None
        self.return_shape_trial = False

        # Contact objective.  AUTO means the deterministic candidate is used;
        # Jev can replace it with SURVIVE/LEFT/RIGHT at a boundary.
        self.intent = "AUTO"
        self.intent_epoch = 0
        self.rally_bounces = 0
        self.shape_attempts = 0
        self.shape_cooldown = 0
        self.shape_failures = 0
        self.endpoint_for = {"LEFT": "high", "RIGHT": "low"}
        self.last_shape_side: str | None = None
        self.last_shape_frame = -100000

        # Motor leases prevent a one-frame estimate from reversing the paddle
        # at the collision window.  A brake is rare and measured, not guessed.
        self.lease_action = NOOP
        self.lease_until = -1
        self.neutral_until = -1
        self.last_brake_frame = -100000

    # ------------------------- defensive primitives -------------------------
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

    # ----------------------- measured state and events ----------------------
    def _raw_velocity(self, obs: dict[str, Any] | None) -> tuple[float, float, bool]:
        if not isinstance(obs, dict):
            return 0.0, 0.0, False
        ball = self._ball(obs)
        if ball is None:
            return 0.0, 0.0, False
        dx = self._num(ball.get("dx"), 0.0)
        dy = self._num(ball.get("dy"), 0.0)
        valid = bool(obs.get("ball_velocity_valid", False))
        return dx, dy, valid

    def _velocity_between(
        self, old: dict[str, Any] | None, new: dict[str, Any]
    ) -> tuple[float, float, bool]:
        dx, dy, valid = self._raw_velocity(new)
        if self._ball(new) is None:
            return 0.0, 0.0, False
        old_ball = self._ball(old) if isinstance(old, dict) else None
        ball = self._ball(new)
        if old_ball is not None and ball is not None:
            inferred_x = self._clamp(
                self._num(ball.get("x"), 0.0) - self._num(old_ball.get("x"), 0.0),
                -8.0,
                8.0,
            )
            inferred_y = self._clamp(
                self._num(ball.get("y"), 0.0) - self._num(old_ball.get("y"), 0.0),
                -8.0,
                8.0,
            )
            if not valid and (abs(inferred_x) > 0.0 or abs(inferred_y) > 0.0):
                dx, dy, valid = inferred_x, inferred_y, True
        if not valid and (abs(dx) > 0.0 or abs(dy) > 0.0):
            valid = True
        return dx, dy, valid

    def _update_measurements(self, obs: dict[str, Any]) -> None:
        old = self.previous
        paddle = self._paddle(obs)
        px = self._num(paddle.get("x"), 99.0)
        if isinstance(old, dict):
            old_px = self._num(self._paddle(old).get("x"), px)
            gap = max(1, self._frame(obs) - self.last_frame)
            raw = self._clamp((px - old_px) / gap, -12.0, 12.0)
            self.paddle_v = raw
            self.paddle_v_avg = 0.62 * self.paddle_v_avg + 0.38 * raw
            if self.last_action == NOOP:
                moving = 1.0 if abs(raw) > 0.4 else 0.0
                self.noop_drift = 0.88 * self.noop_drift + 0.12 * moving

        dx, dy, valid = self._velocity_between(old, obs)
        if self._ball(obs) is None:
            self.ball_vx = 0.0
            self.ball_vy = 0.0
        elif valid:
            self.ball_vx, self.ball_vy = dx, dy
        else:
            # Retain the last good velocity for one ambiguous RAM frame.  The
            # return planner will refuse to make a corridor without dy > 0.
            if abs(self.ball_vx) + abs(self.ball_vy) < 0.5:
                self.ball_vx, self.ball_vy = dx, dy

    def _speed(self) -> str:
        total = abs(self.ball_vx) + abs(self.ball_vy)
        if total < 0.5:
            return "unknown"
        return "fast" if total >= 3.0 else "normal"

    def _event(self, obs: dict[str, Any]) -> str:
        old = self.previous
        if old is None:
            return "start"
        if self._int(obs.get("lives"), 0) < self._int(old.get("lives"), 0):
            return "life_loss"
        if self._int(obs.get("wall_index"), 0) != self._int(old.get("wall_index"), 0):
            return "wall_transition"

        old_ball = self._ball(old)
        ball = self._ball(obs)
        if old_ball is None and ball is not None:
            return "serve"
        if old_ball is not None and ball is None:
            return "ball_lost"

        if self._num(obs.get("score"), 0.0) > self._num(old.get("score"), 0.0):
            return "brick_hit"

        if old_ball is not None and ball is not None:
            odx, ody, _ = self._velocity_between(None, old)
            ndx, ndy, _ = self._velocity_between(old, obs)
            oy = self._num(old_ball.get("y"), 0.0)
            ny = self._num(ball.get("y"), oy)
            if ody > 0.0 and ndy < 0.0 and max(oy, ny) > 145.0:
                return "paddle_bounce"
            if odx * ndx < 0.0 and ny > 95.0:
                return "side_bounce"

        old_dx, old_dy, _ = self._velocity_between(None, old)
        old_speed = "unknown" if abs(old_dx) + abs(old_dy) < 0.5 else (
            "fast" if abs(old_dx) + abs(old_dy) >= 3.0 else "normal"
        )
        if self._speed() == "fast" and old_speed != "fast":
            return "speed_change"
        return "none"

    def _side_facts(self, obs: dict[str, Any]) -> dict[str, Any]:
        bitmap = obs.get("brick_bitmap") if isinstance(obs, dict) else None
        left = 999
        right = 999
        if isinstance(bitmap, list):
            left_n = 0
            right_n = 0
            rows = 0
            for row in bitmap:
                if not isinstance(row, list):
                    continue
                rows += 1
                if len(row) >= 2:
                    left_n += int(bool(row[0])) + int(bool(row[1]))
                if len(row) >= 18:
                    right_n += int(bool(row[16])) + int(bool(row[17]))
            if rows:
                left, right = left_n, right_n

        def bucket(n: int) -> str:
            if n <= 0:
                return "open"
            if n <= 3:
                return "partial"
            return "blocked"

        return {
            "left": bucket(left),
            "right": bucket(right),
            "left_count": min(12, max(0, left)),
            "right_count": min(12, max(0, right)),
            "wall": self._int(obs.get("wall_index"), 0),
            "walls_cleared": self._int(obs.get("walls_cleared"), 0),
            "bricks_remaining": self._int(obs.get("bricks_remaining"), 108),
        }

    def _fallback_side(self, obs: dict[str, Any]) -> str | None:
        side = self._side_facts(obs)
        if side["left"] == "open" and side["right"] == "open":
            return None
        if side["left"] == "open":
            return RIGHT
        if side["right"] == "open":
            return LEFT
        if self.ball_vx < -0.25:
            return LEFT
        if self.ball_vx > 0.25:
            return RIGHT
        # A stable tie-breaker is preferable to making Jev calculate one.
        return LEFT if (self._int(obs.get("wall_index"), 0) + self.rally_bounces) % 2 == 0 else RIGHT

    def _reset_corridor(self, reason: str, frame: int) -> None:
        self.corridor_valid = False
        self.corridor_landing = None
        self.corridor_eta = None
        self.corridor_frame = -1
        self.corridor_vx = 0.0
        self.corridor_vy = 0.0
        self.return_shape_side = None
        self.return_shape_trial = False
        self.return_epoch += 1
        self.lease_action = NOOP
        self.lease_until = frame
        self.neutral_until = frame + 1
        self._log("corridor_reset", frame=frame, reason=reason, epoch=self.return_epoch)

    def _contact_observed(self, obs: dict[str, Any], event: str, frame: int) -> None:
        if event != "paddle_bounce":
            return
        side = self.return_shape_side
        if side in (LEFT, RIGHT):
            outgoing = -1 if self.ball_vx < 0.0 else 1 if self.ball_vx > 0.0 else 0
            expected = -1 if side == LEFT else 1
            if outgoing and outgoing != expected:
                old_endpoint = self.endpoint_for[side]
                self.endpoint_for[side] = "low" if old_endpoint == "high" else "high"
                self.shape_failures += 1
                self._log(
                    "contact_mapping_flipped",
                    frame=frame,
                    side=side,
                    outgoing=outgoing,
                    endpoint=self.endpoint_for[side],
                )
            elif outgoing == expected:
                self._log("contact_shape_confirmed", frame=frame, side=side, outgoing=outgoing)
        self.last_shape_side = side
        self.last_shape_frame = frame
        self.return_shape_side = None
        self.return_shape_trial = False
        self.rally_bounces += 1
        if self.shape_cooldown > 0:
            self.shape_cooldown -= 1

    def _apply_event(self, obs: dict[str, Any], event: str, frame: int) -> None:
        if event in ("start", "serve", "life_loss", "ball_lost", "wall_transition", "paddle_bounce", "brick_hit", "side_bounce", "speed_change"):
            self._reset_corridor(event, frame)

        if event == "start":
            self.intent = "AUTO"
            self.rally_bounces = 0
            self.shape_attempts = 0
            self.shape_failures = 0
        elif event == "serve":
            # A new ball gets a clean return cycle, but learned endpoint
            # direction remains useful within this episode.
            self.rally_bounces = 0
            self.shape_attempts = 0
            self.shape_cooldown = 0
        elif event in ("life_loss", "ball_lost"):
            self.intent = "AUTO"
            self.intent_epoch += 1
            self.rally_bounces = 0
            self.shape_attempts = 0
            self.shape_cooldown = 1
            self.shape_failures = 0
        elif event == "wall_transition":
            self.intent = "AUTO"
            self.intent_epoch += 1
            self.rally_bounces = 0
            self.shape_attempts = 0
            self.shape_cooldown = 0
            self.astra_plan = "AUTO"
            self.astra_started = False
            self.astra_epoch += 1
        elif event == "paddle_bounce":
            # Leave at least one complete safe return between side trials.
            if self.last_shape_side in (LEFT, RIGHT):
                self.shape_cooldown = max(self.shape_cooldown, 1)
        elif event == "speed_change":
            self.intent = "SURVIVE"
            self.intent_epoch += 1

        self.last_speed = self._speed()

    # --------------------------- reflected geometry -------------------------
    @staticmethod
    def _reflect_x(x: float, dx: float, frames: float) -> float:
        left, right = 8.0, 150.0
        span = right - left
        if span <= 0.0:
            return x
        phase = (x - left + dx * max(0.0, frames)) % (2.0 * span)
        return left + phase if phase <= span else right - (phase - span)

    def _home(self, obs: dict[str, Any]) -> tuple[float, float, float]:
        paddle = self._paddle(obs)
        pw = self._num(paddle.get("w"), 16.0)
        # Keep a compact center home band so the next return begins with
        # lateral room on either side rather than an extreme paddle position.
        low, high = 61.0, 91.0
        center = 76.0
        pmax = max(8.0, 160.0 - pw)
        return self._clamp(low, 8.0, pmax), self._clamp(high, 8.0, pmax), center

    def _objective_side(self, obs: dict[str, Any], width: float) -> str | None:
        lives = self._int(obs.get("lives"), 0)
        if lives < 4 or self._speed() != "normal":
            return None
        if self.shape_cooldown > 0 or self.rally_bounces < 2:
            return None
        side = self._side_facts(obs)
        # SURVIVE is a high-level warning, not a permanent veto: the safe
        # corridor remains authoritative, and after the cooldown the code may
        # run a bounded endpoint trial if no explicit side contract exists.
        # This prevents a conservative Jev answer at every bounce from
        # silently disabling the study's tunnel hypothesis.
        candidate = self.astra_plan if self.astra_plan in (LEFT, RIGHT) else self._fallback_side(obs)
        if self.intent in (LEFT, RIGHT):
            candidate = self.intent
        if candidate not in (LEFT, RIGHT):
            return None
        if side[candidate.lower()] == "open":
            return None
        if width < 10.0:
            return None
        if self.shape_attempts >= 3:
            return None
        return candidate

    def _make_corridor(self, obs: dict[str, Any], frame: int) -> bool:
        ball = self._ball(obs)
        if ball is None or not bool(obs.get("ball_in_play", False)):
            return False
        vx, vy, valid = self._velocity_between(self.previous, obs)
        if not valid or vy <= 0.0:
            return False
        paddle = self._paddle(obs)
        py = self._num(paddle.get("y"), 189.0)
        pw = self._num(paddle.get("w"), 16.0)
        bx = self._num(ball.get("x"), 80.0)
        by = self._num(ball.get("y"), 100.0)
        bw = self._num(ball.get("w"), 2.0)
        bh = self._num(ball.get("h"), 4.0)
        if by < 94.0:
            return False
        eta = max(0.0, (py - bh - by) / max(0.25, vy))

        # Propagate a small, named uncertainty set in code.  Jev is not asked
        # to compare coordinates.  The intersection is the paddle-left set
        # that catches every nearby trajectory; the nominal interval is kept as
        # a fallback when quantization makes the intersection empty.
        candidates: list[float] = []
        for xoff in (-1.0, 0.0, 1.0):
            for dxoff in (-0.35, 0.0, 0.35):
                candidates.append(self._reflect_x(bx + xoff, vx + dxoff, eta))
        nominal_left = self._reflect_x(bx, vx, eta)
        margin = 1.0 if abs(vx) + abs(vy) < 4.0 else 0.5
        intervals = [(x + margin - pw, x + bw - margin) for x in candidates]
        low = max(item[0] for item in intervals)
        high = min(item[1] for item in intervals)
        nominal_low = nominal_left + margin - pw
        nominal_high = nominal_left + bw - margin
        pmin = 8.0
        pmax = max(pmin, 160.0 - pw)
        if high < low or high - low < 2.0:
            low, high = nominal_low, nominal_high
        low = self._clamp(low, pmin, pmax)
        high = self._clamp(high, pmin, pmax)
        if high < low:
            low = high = self._clamp(nominal_left + bw / 2.0 - pw / 2.0, pmin, pmax)

        landing = nominal_left + bw / 2.0
        if self.corridor_valid:
            changed = abs(landing - self._num(self.corridor_landing, landing)) > 3.0
            changed = changed or abs(vx - self.corridor_vx) > 1.0 or abs(vy - self.corridor_vy) > 1.0
            if not changed:
                return True
            self._log("corridor_replanned", frame=frame, reason="measured_change")

        self.corridor_valid = True
        self.corridor_low = low
        self.corridor_high = high
        self.corridor_center = (low + high) / 2.0
        self.corridor_landing = landing
        self.corridor_eta = eta
        self.corridor_vx = vx
        self.corridor_vy = vy
        self.corridor_frame = frame

        width = max(0.0, high - low)
        previous_shape = self.return_shape_side
        side = self._objective_side(obs, width)
        self.return_shape_side = side
        self.return_shape_trial = side in (LEFT, RIGHT)
        # A velocity replan is still the same contact trial.  Count the trial
        # once per return, not once per quantized x update.
        if self.return_shape_trial and previous_shape != side:
            self.shape_attempts += 1
            self._log("edge_contact_trial", frame=frame, side=side, width=round(width, 2))
        self._log(
            "catch_corridor",
            frame=frame,
            low=round(low, 2),
            high=round(high, 2),
            landing=round(landing, 2),
            eta=round(eta, 2),
            shape=side or "SURVIVE",
        )
        return True

    def _flight(self, obs: dict[str, Any], event: str, frame: int) -> dict[str, Any]:
        ball = self._ball(obs)
        if ball is None or not bool(obs.get("ball_in_play", False)):
            low, high, center = self._home(obs)
            return {"phase": "serve", "urgency": "serve", "eta": None, "low": low, "high": high, "center": center, "landing": None, "shape": None}

        vx, vy, valid = self._velocity_between(self.previous, obs)
        by = self._num(ball.get("y"), 100.0)
        paddle = self._paddle(obs)
        py = self._num(paddle.get("y"), 189.0)
        bh = self._num(ball.get("h"), 4.0)
        eta = None
        if valid and vy > 0.0:
            eta = max(0.0, (py - bh - by) / max(0.25, vy))

        if vy <= 0.0 or by < 94.0 or not valid:
            # A corridor made on the prior descent is not a valid home target:
            # brick/side contacts can change both x and speed before the next
            # descent.  Explicitly invalidate it while ascending so the next
            # clean return starts from fresh RAM velocity.
            if self.corridor_valid:
                self._reset_corridor("ascending" if vy < 0.0 else "collision_zone", frame)
            low, high, center = self._home(obs)
            return {
                "phase": "ascending" if vy < 0.0 else "uncertain",
                "urgency": "early",
                "eta": None if eta is None else round(eta, 2),
                "low": low,
                "high": high,
                "center": center,
                "landing": None,
                "shape": None,
            }

        # Revisit the latched corridor on each clean descending observation.
        # `_make_corridor` keeps it when the measured trajectory is stable, but
        # authorizes a replacement after a real x/velocity change (side wall,
        # brick, or speed event).  The previous version accidentally called
        # this only when invalid and could carry a stale landing through a
        # missed side-wall transition.
        self._make_corridor(obs, frame)
        if not self.corridor_valid:
            low, high, center = self._home(obs)
            return {"phase": "uncertain_descent", "urgency": "early", "eta": eta, "low": low, "high": high, "center": center, "landing": None, "shape": None}

        if eta <= 5.0:
            urgency = "imminent"
        elif eta <= 17.0:
            urgency = "near"
        else:
            urgency = "early"
        return {
            "phase": "descending",
            "urgency": urgency,
            "eta": round(eta, 2),
            "low": round(self.corridor_low, 2),
            "high": round(self.corridor_high, 2),
            "center": round(self.corridor_center, 2),
            "landing": round(self._num(self.corridor_landing, 0.0), 2),
            "shape": self.return_shape_side,
        }

    # ----------------------------- motor policy -----------------------------
    def _target(self, flight: dict[str, Any]) -> float:
        low = self._num(flight.get("low"), 48.0)
        high = self._num(flight.get("high"), 104.0)
        if flight.get("phase") == "descending" and flight.get("shape") in (LEFT, RIGHT):
            side = flight["shape"]
            # LEFT means ball should leave left: put it near the paddle's left
            # section, i.e. keep the paddle on the right side of its overlap.
            if self.endpoint_for.get(side) == "high":
                return high - min(3.0, max(1.0, (high - low) * 0.20))
            return low + min(3.0, max(1.0, (high - low) * 0.20))
        return self._num(flight.get("center"), (low + high) / 2.0)

    @staticmethod
    def _opposite(action: str) -> str:
        return LEFT if action == RIGHT else RIGHT

    def _motor(self, obs: dict[str, Any], flight: dict[str, Any], frame: int) -> str:
        if bool(obs.get("terminated", False)) or bool(obs.get("truncated", False)):
            return NOOP
        if bool(obs.get("serve_required", False)):
            return FIRE if self._int(obs.get("lives"), 0) > 0 else NOOP
        ball = self._ball(obs)
        if ball is None:
            return NOOP

        paddle = self._paddle(obs)
        px = self._num(paddle.get("x"), 99.0)
        low = self._num(flight.get("low"), 48.0)
        high = self._num(flight.get("high"), 104.0)
        eta_value = flight.get("eta")
        eta = self._num(eta_value, 10.0) if eta_value is not None else 10.0
        urgency = flight.get("urgency")
        target = self._target(flight)

        # Use the current measured velocity only over a short horizon.  It is a
        # brake estimate, not an assumed paddle dynamics table.
        horizon = self._clamp(eta, 1.0, 5.0)
        predicted = px + self.paddle_v_avg * horizon
        guard = 1.0 if urgency in ("near", "imminent") else 2.0
        inside = low - 1.0 <= px <= high + 1.0
        predicted_inside = low - guard <= predicted <= high + guard

        desired = NOOP
        if predicted < low - guard or px < low - guard:
            desired = RIGHT
        elif predicted > high + guard or px > high + guard:
            desired = LEFT
        elif not predicted_inside and self.paddle_v_avg > 0.45:
            desired = LEFT
        elif not predicted_inside and self.paddle_v_avg < -0.45:
            desired = RIGHT
        elif (
            flight.get("shape") in (LEFT, RIGHT)
            and urgency == "early"
            and eta > 8.0
            and predicted_inside
        ):
            # Once the paddle is safely inside the physical set, use a small
            # interior endpoint band to shape the outgoing angle.  The band is
            # never allowed to supersede the physical corridor or the final
            # impact window.
            endpoint_band = max(1.25, min(2.5, (high - low) * 0.18))
            if predicted < target - endpoint_band:
                desired = RIGHT
            elif predicted > target + endpoint_band:
                desired = LEFT

        # In the catch interval, release a lease immediately; do not keep
        # driving toward the contact endpoint once overlap is secured.  During
        # a safe early shape phase, the endpoint branch above is the exception.
        if inside and predicted_inside and not (
            flight.get("shape") in (LEFT, RIGHT) and urgency == "early" and eta > 8.0
        ):
            desired = NOOP

        if desired == NOOP:
            # One counter-pulse can cancel observed carry-through.  A cooldown
            # prevents the LEFT/RIGHT/LEFT death spiral seen in older traces.
            moving_out = (self.paddle_v > 0.45 and predicted > high + guard) or (
                self.paddle_v < -0.45 and predicted < low - guard
            )
            if moving_out and frame >= self.last_brake_frame + 4 and not (urgency == "imminent" and inside):
                action = self._opposite(RIGHT if self.paddle_v > 0.0 else LEFT)
                self.last_brake_frame = frame
                self.lease_action = NOOP
                self.lease_until = frame
                self.neutral_until = frame + 2
                self._log("measured_brake", frame=frame, action=action)
                return action
            self.lease_action = NOOP
            self.lease_until = frame + (1 if urgency in ("near", "imminent") else 2)
            return NOOP

        # Do not reverse a small error while a pulse is leased.  If the paddle
        # is genuinely outside the physical catch set, an opposite direction
        # is allowed immediately so a miss is not hidden by hysteresis.
        if self.lease_action in DIRECTIONS and frame < self.lease_until:
            if low - 2.5 <= px <= high + 2.5 and self.lease_action != desired:
                return self.lease_action
            if self.lease_action == desired:
                return desired

        if frame < self.neutral_until and low <= px <= high:
            return NOOP
        self.lease_action = desired
        self.lease_until = frame + (1 if urgency in ("near", "imminent") else 2)
        return desired

    # -------------------------- Jev contract bridge ------------------------
    @staticmethod
    def _answer_value(answer: Any) -> str | None:
        value = getattr(answer, "value", None)
        return value if isinstance(value, str) else None

    @staticmethod
    def _answer_probability(answer: Any, option: str) -> float:
        probs = getattr(answer, "probabilities", None)
        if isinstance(probs, dict):
            try:
                value = float(probs.get(option, 0.0))
                return value if math.isfinite(value) else 0.0
            except (TypeError, ValueError):
                return 0.0
        return 1.0 if Harness._answer_value(answer) == option else 0.0

    def _derived_state(self, obs: dict[str, Any], flight: dict[str, Any], event: str) -> dict[str, Any]:
        side = self._side_facts(obs)
        low = self._num(flight.get("low"), 48.0)
        high = self._num(flight.get("high"), 104.0)
        width = high - low
        fallback = self._fallback_side(obs)
        lives = self._int(obs.get("lives"), 0)
        shape_ready = (
            lives >= 4
            and self._speed() == "normal"
            and self.rally_bounces >= 2
            and self.shape_cooldown == 0
            and width >= 10.0
            and fallback in (LEFT, RIGHT)
            and side[fallback.lower()] != "open"
        )
        return {
            "frame": self._frame(obs),
            "event": event,
            "score": self._int(self._num(obs.get("score"), 0.0)),
            "lives": lives,
            "serve_required": bool(obs.get("serve_required", False)),
            "phase": flight.get("phase", "unknown"),
            "urgency": flight.get("urgency", "unknown"),
            "speed": self._speed(),
            "landing_lane": "left" if self._num(flight.get("landing"), 76.0) < 56.0 else "right" if self._num(flight.get("landing"), 76.0) > 104.0 else "center",
            "corridor": "wide" if width >= 10.0 else "narrow",
            "paddle_motion": "right" if self.paddle_v_avg > 0.45 else "left" if self.paddle_v_avg < -0.45 else "still",
            "eta_bucket": "imminent" if flight.get("urgency") == "imminent" else "near" if flight.get("urgency") == "near" else "early",
            "shape_ready": shape_ready,
            "candidate_side": fallback or "none",
            "current_intent": self.intent,
            "astra_plan": self.astra_plan,
            "left_edge": side["left"],
            "right_edge": side["right"],
            "bricks_remaining": side["bricks_remaining"],
            "wall_index": side["wall"],
            "rally_bounces": self.rally_bounces,
            "return_epoch": self.return_epoch,
        }

    def _questions(self, state: dict[str, Any]) -> dict[str, Choice]:
        options = {
            "SURVIVE": {
                "what": "Use the centered catch objective and do not deliberately choose a paddle edge.",
                "when": "Choose this for a fast, uncertain, imminent, narrow, or low-life return.",
            },
            "LEFT": {
                "what": "Use the code-computed safe overlap interval but bias its left-going paddle section to work on the left outer tunnel.",
                "when": "Only for shape_ready=true, an intact left edge, normal speed, plentiful lives, and an early return.",
            },
            "RIGHT": {
                "what": "Use the code-computed safe overlap interval but bias its right-going paddle section to work on the right outer tunnel.",
                "when": "Only for shape_ready=true, an intact right edge, normal speed, plentiful lives, and an early return.",
            },
            "NO_CHANGE": {
                "what": "Keep the current intent and let the deterministic corridor motor decide.",
                "when": "Use when the state is ambiguous or no change is needed.",
            },
        }
        keys = list(options)
        shift = self.jev_serial % len(keys)
        a = keys[shift:] + keys[:shift]
        b = list(reversed(a))
        instruction = (
            "Judge only the long-lived contact objective for this exact derived Breakout state. "
            "Do not calculate coordinates and do not emit NOOP/FIRE/LEFT/RIGHT frame actions. "
            "The corridor and motor are code-owned. If shape_ready is false, urgency is near or "
            "imminent, speed is fast/unknown, lives are below four, or the requested edge is open, "
            "select SURVIVE or NO_CHANGE. If shape_ready is true, the code candidate_side and the "
            "Astra proposal are evidence for a cautious tunnel attempt. Two answers are deliberately "
            "listed in different orders; only agreement is executable. State=" + repr(state)
        )
        return {
            "objective_a": Choice(instruction, {k: options[k] for k in a}),
            "objective_b": Choice(instruction, {k: options[k] for k in b}),
        }

    def _launch_jev(self, obs: dict[str, Any], flight: dict[str, Any], event: str) -> None:
        if not self.jev_enabled or self.jev_future is not None:
            return
        frame = self._frame(obs)
        boundary = event in ("start", "serve", "paddle_bounce", "life_loss", "wall_transition", "speed_change")
        if not boundary or frame < self.next_jev_frame or bool(obs.get("serve_required", False)):
            return
        if self._ball(obs) is None:
            return
        state = {
            "game": "Atari 2600 Breakout ordinary mode 0 difficulty 0 at native 60 Hz",
            "observation": self._derived_state(obs, flight, event),
            "rules": {
                "legal_frame_actions": list(ACTIONS),
                "FIRE": "only while serve_required is true",
                "priority": "preserve catchability before tunnel shaping",
                "bridge": "the selected objective changes only the next safe descending corridor; code posts the action",
            },
        }
        questions = self._questions(state)
        request_epoch = self.intent_epoch
        request_frame = frame
        self.jev_frame = frame
        self.jev_epoch = request_epoch
        self.jev_serial += 1
        self.next_jev_frame = frame + 42

        def call():
            return self.ctx.decide(state, questions, timeout_s=1.35)

        try:
            self.jev_future = self.ctx.spawn(call)
        except Exception as exc:
            self.jev_future = None
            self._log("jev_spawn_failed", error=type(exc).__name__)

    def _collect_jev(self, obs: dict[str, Any], frame: int) -> None:
        future = self.jev_future
        if future is None or not getattr(future, "done", lambda: False)():
            return
        self.jev_future = None
        try:
            answers = future.result()
            if not isinstance(answers, dict):
                return
            a = answers.get("objective_a")
            b = answers.get("objective_b")
            av = self._answer_value(a)
            bv = self._answer_value(b)
            allowed = {"SURVIVE", "LEFT", "RIGHT", "NO_CHANGE"}
            if av not in allowed or av != bv:
                return
            probability = min(self._answer_probability(a, av), self._answer_probability(b, bv))
            if probability < 0.55 or self.jev_epoch != self.intent_epoch or frame - self.jev_frame > 90:
                return
            lives = self._int(obs.get("lives"), 0)
            side = self._side_facts(obs)
            if av in (LEFT, RIGHT):
                if lives < 4 or self._speed() != "normal" or side[av.lower()] == "open":
                    av = "SURVIVE"
            if av == "NO_CHANGE":
                self._log("jev_objective_held", frame=frame, probability=round(probability, 3))
                return
            self.intent = av
            self.intent_epoch += 1
            self._log("jev_objective_accepted", frame=frame, objective=av, probability=round(probability, 3))
        except (TimeoutError, RuntimeError, BudgetExceeded, EpisodeOver) as exc:
            self._log("jev_unavailable", error=type(exc).__name__)
        except Exception as exc:
            self._log("jev_parse_ignored", error=type(exc).__name__)

    # -------------------------- Astra strategic prior -----------------------
    def _astra_prompt(self, obs: dict[str, Any], flight: dict[str, Any]) -> str:
        side = self._side_facts(obs)
        return (
            "You are the slow strategic planner for ordinary Atari 2600 Breakout mode 0, difficulty 0. "
            "A deterministic controller catches the ball every native frame. Return exactly one token: "
            "LEFT, RIGHT, or SURVIVE. LEFT/RIGHT is a cautious paddle-contact subgoal for an outer "
            "tunnel, never a raw frame action. Choose SURVIVE when lives are low, speed/flight is "
            "uncertain, a side is open, or preserving the ball is more important. Jev will validate "
            "your proposal at a paddle-bounce boundary and code will gate it by a catch corridor. "
            "State=" + repr({
                "frame": self._frame(obs),
                "lives": self._int(obs.get("lives"), 0),
                "score": self._int(self._num(obs.get("score"), 0.0)),
                "flight": {"phase": flight.get("phase"), "urgency": flight.get("urgency"), "landing_lane": flight.get("landing")},
                "speed": self._speed(),
                "left_edge": side["left"],
                "right_edge": side["right"],
                "wall_index": side["wall"],
                "bricks_remaining": side["bricks_remaining"],
                "rally_bounces": self.rally_bounces,
            })
        )

    def _launch_astra(self, obs: dict[str, Any], flight: dict[str, Any]) -> None:
        if self.astra_future is not None or self.astra_started:
            return
        if self._ball(obs) is None or self._frame(obs) < 8:
            return
        try:
            if int(self.ctx.budget.remaining().get("system2_calls", 0)) <= 0:
                return
        except Exception:
            pass
        self.astra_started = True
        self.astra_calls += 1
        prompt = self._astra_prompt(obs, flight)

        def call():
            # Short-token Astra answers commonly take tens of seconds; this is
            # intentionally asynchronous and may be cancelled at game end.
            return self.ctx.think(prompt, timeout_s=72.0)

        try:
            self.astra_future = self.ctx.spawn(call)
        except Exception as exc:
            self.astra_future = None
            self._log("astra_spawn_failed", error=type(exc).__name__)

    def _collect_astra(self, obs: dict[str, Any], frame: int) -> None:
        future = self.astra_future
        if future is None or not getattr(future, "done", lambda: False)():
            return
        self.astra_future = None
        try:
            result = future.result()
            text = getattr(result, "text", "")
            if not isinstance(text, str):
                return
            match = re.search(r"\b(LEFT|RIGHT|SURVIVE|SAFE|PRESERVE)\b", text.upper())
            if match is None:
                return
            token = match.group(1)
            self.astra_plan = "SURVIVE" if token in ("SAFE", "PRESERVE") else token
            self._log("astra_plan_received", frame=frame, plan=self.astra_plan)
        except (TimeoutError, RuntimeError, BudgetExceeded, EpisodeOver) as exc:
            self._log("astra_unavailable", error=type(exc).__name__)
        except Exception as exc:
            self._log("astra_parse_ignored", error=type(exc).__name__)

    # ------------------------------- runner ---------------------------------
    @staticmethod
    def _observe(game: Any) -> dict[str, Any]:
        try:
            value = game.observe()
        except Exception:
            return {}
        return value if isinstance(value, dict) else {}

    def _step(self, game: Any, realtime: bool) -> bool:
        obs = self._observe(game)
        frame = self._frame(obs)
        event = self._event(obs)
        self._update_measurements(obs)
        self._contact_observed(obs, event, frame)
        self._apply_event(obs, event, frame)
        flight = self._flight(obs, event, frame)

        self._collect_jev(obs, frame)
        self._collect_astra(obs, frame)
        self._launch_astra(obs, flight)
        self._launch_jev(obs, flight, event)

        action = self._motor(obs, flight, frame)
        if action not in ACTIONS:
            action = NOOP
        self.last_action = action
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
        while not self.ctx.done and not bool(getattr(game, "done", False)):
            if not self._step(game, realtime):
                return


def build(ctx: Any) -> Harness:
    return Harness(ctx)
