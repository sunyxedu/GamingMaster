"""Reservation-based Breakout controller (candidate 0021).

The fast path is deliberately not a language-model policy.  It estimates the
ball's reflected return and the paddle's queued-input response, reserves a
safe contact interval for one flight, and posts one legal action per native
frame.  Astra chooses a long-horizon contact objective and Jev adjudicates it;
neither is allowed to replace the reservation with a stale frame action.
"""
from __future__ import annotations

import json
import math
import re
from collections import deque
from statistics import median
from typing import Any

from metaharness.core.contract import BudgetExceeded, EpisodeOver
from metaharness.models.jev import Choice, Noul


NOOP = "NOOP"
FIRE = "FIRE"
RIGHT = "RIGHT"
LEFT = "LEFT"
ACTIONS = (NOOP, FIRE, RIGHT, LEFT)
STEERING = (LEFT, RIGHT)


class BreakoutController:
    """A native-frame controller with flight reservations and safe objectives."""

    def __init__(self, ctx: Any):
        self.ctx = ctx

        # Public observation history.  A missing or repeated observation never
        # becomes a fabricated ball at (0, 0).
        self.previous: dict[str, Any] | None = None
        self.last_frame = -1
        self.last_posted = NOOP
        self.previous_posted = NOOP
        self.paddle_x = 99.0
        self.paddle_v = 0.0
        self.last_raw_v = 0.0
        self.noop_motion = 0.0

        # A small online model of the cartridge's delayed paddle actuator.
        # v[t+1] ~= a*v[t] + b*u[t] + c*u[t-1].  It is deliberately bounded and
        # blended with a conservative prior; sticky actions are represented as
        # branches in the one-frame reservation search below.
        self.motor = [0.62, 0.30, 2.20]
        self.motor_prior = [0.62, 0.30, 2.20]
        self.motor_xx = [[30.0, 0.0, 0.0], [0.0, 80.0, 0.0], [0.0, 0.0, 80.0]]
        self.motor_xy = [18.6, 24.0, 176.0]
        self.motor_samples = 0
        self.motor_residual = 0.0
        self.motor_confident = False

        # Same-sign velocity windows handle the observed 1/2 and 3/3/4 RAM
        # cadences without treating a quantized zero as a reversal.
        self.ball_window: deque[tuple[int, float, float, float, float]] = deque(maxlen=6)
        self.ball_vx = 0.0
        self.ball_vy = 0.0
        self.ball_velocity_valid = False
        self.ball_velocity_warm = False
        self.velocity_frame = -1
        # Persist the observed slow/fast regime across a collision reset; the
        # next contact is planned while the ball is ascending.
        self.speed_regime = "unknown"
        # is recomputed from current public RAM, but the selected motor command
        # is hysteretic in the final contact window.
        self.flight_id = 0
        self.flight_valid = False
        self.flight_landing = None
        self.flight_eta = None
        self.flight_low = 61.0
        self.flight_high = 91.0
        self.flight_center = 76.0
        self.flight_vx = 0.0
        self.flight_vy = 0.0
        self.contact_plane = 186.5
        self.plane_samples: deque[float] = deque(maxlen=12)
        self.landing_bias = 0.0

        # Contact evidence is code-owned.  It decides how much of the reported
        # paddle rectangle is trusted, and also resolves endpoint experiments.
        self.contacts: deque[dict[str, Any]] = deque(maxlen=48)
        self.contact_samples: deque[dict[str, Any]] = deque(maxlen=32)
        self.pending_outgoing: dict[str, Any] | None = None
        self.stable_catches = 0
        self.stable_misses = 0
        self.safe_offsets: deque[float] = deque(maxlen=24)
        self.profile = "BROAD"
        self.last_progress_frame = 0
        self.last_score = 0.0

        # One objective and at most one shaping trial belong to one flight.
        self.contact_id = 0
        self.rally_catches = 0
        self.trials_this_life = 0
        self.trial_cooldown = 0
        self.objective = "AUTO"
        self.objective_contact = -1
        self.objective_version = 0
        self.trial: dict[str, Any] | None = None
        self.trial_considered = -1
        self.trial_failures = 0
        self.endpoint_stats = {"LEFT": [0, 0], "RIGHT": [0, 0]}
        self.endpoint_reliability = {"LEFT": 0, "RIGHT": 0}
        # Center-relative paddle offsets that have empirically produced the
        # requested outgoing direction.  A flip is allowed only after an
        # achieved, reliable contact disagrees twice; endpoint-not-reached
        # trials never teach this mapping.
        self.endpoint_offset = {"LEFT": -3.5, "RIGHT": 3.5}
        self.fast_endpoint_stats = {"LEFT": [0, 0], "RIGHT": [0, 0]}
        self.fast_speeds: deque[tuple[float, float]] = deque(maxlen=16)
        self.last_trial_outcome = "none"
        self.last_outgoing = "unknown"
        self.safety_epoch = 0
        self.wall_epoch = 0
        self.cleanup_mode = False

        # Final-window reservation.  This is the structural change over the
        # parents' per-frame action minimizers: a late direction cannot flip
        # merely because one quantized x sample crossed an interval edge.
        self.late_action = NOOP
        self.late_until = -1
        self.late_started = False
        self.late_flip_used = False
        self.last_late_flip = -10000
        self.emergency_used = False

        # Jev is advisory and asynchronous.  Astra revises only the objective
        # prior, never the geometry or the motor model.
        self.jev_enabled = getattr(ctx, "system1", None) is not None
        self.jev_future = None
        self.jev_stamp = None
        self.jev_request_frame = -100000
        self.jev_serial = 0
        self.next_jev_frame = 0

        self.astra_future = None
        self.astra_plan = "SURVIVE"
        self.astra_keys: set[tuple[int, str]] = set()
        self.astra_calls = 0
        self.astra_revision = 0
        self.astra_request_stamp = None

    # ----------------------------- safe utilities ----------------------------
    @staticmethod
    def _num(value: Any, default: float = 0.0) -> float:
        try:
            result = float(value)
            return result if math.isfinite(result) else default
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
        return obs.get("ball") if isinstance(obs, dict) and isinstance(obs.get("ball"), dict) else None

    @staticmethod
    def _paddle(obs: Any) -> dict[str, Any]:
        return obs.get("paddle") if isinstance(obs, dict) and isinstance(obs.get("paddle"), dict) else {}

    def _frame(self, obs: dict[str, Any]) -> int:
        return self._int(obs.get("frame"), self.last_frame + 1)

    def _log(self, event: str, **fields: Any) -> None:
        try:
            self.ctx.log(event, **fields)
        except Exception:
            # Diagnostics must never become a game error.
            pass

    @staticmethod
    def _u(action: str) -> float:
        return 1.0 if action == RIGHT else -1.0 if action == LEFT else 0.0

    @staticmethod
    def _action_from_u(value: float) -> str:
        return RIGHT if value > 0.4 else LEFT if value < -0.4 else NOOP

    # ----------------------------- event detection ---------------------------
    def _event(self, obs: dict[str, Any]) -> str:
        old = self.previous
        if old is None:
            return "start"
        if self._int(obs.get("lives"), 0) < self._int(old.get("lives"), 0):
            return "life_loss"
        if self._int(obs.get("wall_index"), 0) != self._int(old.get("wall_index"), 0):
            return "wall_transition"
        old_ball, ball = self._ball(old), self._ball(obs)
        if old_ball is None and ball is not None:
            return "serve"
        if old_ball is not None and ball is None:
            return "ball_lost"
        if self._num(obs.get("score"), 0.0) > self._num(old.get("score"), 0.0):
            return "brick_hit"
        if old_ball is not None and ball is not None:
            old_dy = self._num(old_ball.get("dy"))
            new_dy = self._num(ball.get("dy"))
            old_y = self._num(old_ball.get("y"))
            new_y = self._num(ball.get("y"), old_y)
            if old_dy > 0.0 and new_dy < 0.0 and max(old_y, new_y) >= 145.0:
                return "paddle_bounce"
            old_dx = self._num(old_ball.get("dx"))
            new_dx = self._num(ball.get("dx"), old_dx)
            if old_dx * new_dx < 0.0 and new_y > 94.0:
                return "side_bounce"
            # Treat a speed event as a regime transition, not as a repeated
            # threshold crossing.  Quantized dy can alternate around a boundary;
            # resetting the flight on every such sample destroys the reservation.
            old_fast = abs(old_dy) >= 2.8
            new_fast = abs(new_dy) >= 2.8
            if new_fast and not old_fast:
                return "speed_change"
        return "none"

    def _reset_reservation(self, reason: str, frame: int) -> None:
        self.flight_valid = False
        self.flight_landing = None
        self.flight_eta = None
        self.flight_vx = self.flight_vy = 0.0
        self.flight_id += 1
        self.late_action = NOOP
        self.late_until = frame
        self.late_started = False
        self.late_flip_used = False
        self.emergency_used = False
        self.contact_samples.clear()
        self.ball_window.clear()
        self.ball_velocity_valid = False
        self.ball_velocity_warm = False
        self.velocity_frame = -1
        self._log("flight_reset", frame=frame, reason=reason, flight=self.flight_id)

    def _apply_event(self, obs: dict[str, Any], event: str, frame: int) -> None:
        # Resolve the crossing before reset clears the descending samples.  A
        # life-loss observation is also a failed contact, even when the RAM
        # ball disappears in the same frame.
        if event == "paddle_bounce":
            self._record_contact(obs, frame, caught=True)
        elif event in ("ball_lost", "life_loss"):
            self._record_contact(obs, frame, caught=False)

        if event in ("start", "serve", "ball_lost", "life_loss", "wall_transition", "paddle_bounce", "brick_hit", "side_bounce", "speed_change"):
            self._reset_reservation(event, frame)
        if event in ("start", "serve"):
            self.rally_catches = 0
            self.trials_this_life = 0
            self.trial = None
            self.trial_cooldown = 0
            self.objective = "AUTO"
            self.objective_contact = -1
        elif event in ("ball_lost", "life_loss"):
            self.rally_catches = 0
            self.trials_this_life = 0
            self.trial = None
            self.trial_cooldown = 3
            self.objective = "AUTO"
            self.objective_contact = -1
            self.safety_epoch += 1
            self.profile = "BROAD"
            self.stable_catches = 0
        elif event == "wall_transition":
            self.wall_epoch += 1
            self.safety_epoch += 1
            self.contact_id += 1
            self.rally_catches = 0
            self.trials_this_life = 0
            self.trial = None
            self.trial_cooldown = 1
            self.objective = "AUTO"
            self.objective_contact = -1
            self.astra_plan = "SURVIVE"
            self.cleanup_mode = False
        elif event == "paddle_bounce":
            self.contact_id += 1
            self.rally_catches += 1
            if self.trial_cooldown > 0:
                self.trial_cooldown -= 1
            self.trial = None
            self.objective = "AUTO"
            self.objective_contact = -1
        elif event == "brick_hit":
            self.last_progress_frame = frame
            self.cleanup_mode = False
        elif event == "speed_change":
            # A speed transition invalidates a learned fast/slow ray for the
            # current descent, but not the contact offset evidence.
            self._log("speed_regime_changed", frame=frame)

    # --------------------------- motion identification -----------------------
    def _fit_motor(self, frame: int) -> None:
        matrix = [self.motor_xx[i][:] + [self.motor_xy[i]] for i in range(3)]
        for i in range(3):
            pivot = max(range(i, 3), key=lambda k: abs(matrix[k][i]))
            matrix[i], matrix[pivot] = matrix[pivot], matrix[i]
            divisor = matrix[i][i]
            if abs(divisor) < 1e-8:
                return
            matrix[i] = [x / divisor for x in matrix[i]]
            for j in range(3):
                if j == i:
                    continue
                factor = matrix[j][i]
                matrix[j] = [a - factor * b for a, b in zip(matrix[j], matrix[i])]
        candidate = [matrix[i][3] for i in range(3)]
        if not (0.30 < candidate[0] < 0.90 and -0.3 < candidate[1] < 1.2 and 0.8 < candidate[2] < 3.8):
            return
        # Slow adaptation avoids one sticky transition rewriting the actuator.
        self.motor = [0.94*a + 0.06*b for a, b in zip(self.motor, candidate)]
        self.motor_confident = self.motor_samples >= 30
        self._log("motor_fit", frame=frame, coefficients=[round(x, 3) for x in self.motor], samples=self.motor_samples)

    def _update_paddle(self, obs: dict[str, Any], frame: int) -> None:
        px = self._num(self._paddle(obs).get("x"), self.paddle_x)
        gap = max(1, frame - self.last_frame)
        raw = self._clamp((px - self.paddle_x) / gap, -9.0, 9.0) if self.last_frame >= 0 else 0.0
        old_v = self.last_raw_v
        self.paddle_v = 0.58 * self.paddle_v + 0.42 * raw
        if self.last_posted == NOOP:
            self.noop_motion = 0.92*self.noop_motion + 0.08*(1.0 if abs(raw) > 0.5 else 0.0)
        if gap == 1 and 10.0 < min(px, self.paddle_x) and max(px, self.paddle_x) < 142.0:
            features = [old_v, self._u(self.last_posted), self._u(self.previous_posted)]
            target = raw
            for i in range(3):
                self.motor_xy[i] = 0.995*self.motor_xy[i] + features[i]*target
                for j in range(3):
                    self.motor_xx[i][j] = 0.995*self.motor_xx[i][j] + features[i]*features[j]
            self.motor_samples += 1
            self.motor_residual = 0.97*self.motor_residual + 0.03*abs(
                target - sum(a*b for a, b in zip(self.motor, features))
            )
            if self.motor_samples % 80 == 0:
                self._fit_motor(frame)
        self.paddle_x = px
        self.last_raw_v = raw

    def _update_ball_velocity(self, obs: dict[str, Any], frame: int, event: str) -> None:
        ball = self._ball(obs)
        if ball is None or event in ("serve", "ball_lost", "life_loss", "paddle_bounce", "brick_hit", "side_bounce", "wall_transition"):
            self.ball_window.clear()
            self.ball_velocity_valid = False
            self.ball_velocity_warm = False
            self.ball_vx = self.ball_vy = 0.0
            self.velocity_frame = frame
            return
        if frame == self.velocity_frame:
            return
        dx = self._num(ball.get("dx"))
        dy = self._num(ball.get("dy"))
        valid = bool(obs.get("ball_velocity_valid")) or abs(dx)+abs(dy) > 0.0
        if not valid:
            self.ball_window.clear()
            self.ball_velocity_valid = False
            self.ball_velocity_warm = False
            self.velocity_frame = frame
            return
        if self.ball_window:
            prior_dx, prior_dy = self.ball_window[-1][3], self.ball_window[-1][4]
            # Preserve zero samples, but clear a true sign/collision change or
            # a discrete speed jump.  This covers both native RAM cadences.
            if (prior_dx and dx and prior_dx*dx < 0.0) or (prior_dy and dy and prior_dy*dy < 0.0):
                self.ball_window.clear()
            elif abs(dx-prior_dx) > 1.5 or abs(dy-prior_dy) > 1.5:
                self.ball_window.clear()
        self.ball_window.append((frame, self._num(ball.get("x")), self._num(ball.get("y")), dx, dy))
        xs = [item[3] for item in self.ball_window]
        ys = [item[4] for item in self.ball_window]
        self.ball_vx = sum(xs)/len(xs)
        self.ball_vy = sum(ys)/len(ys)
        self.ball_velocity_valid = valid
        self.ball_velocity_warm = len(self.ball_window) >= 2
        if abs(self.ball_vy) >= 2.8:
            self.speed_regime = "fast"
        elif abs(self.ball_vy) >= 0.5:
            self.speed_regime = "normal"
        self.velocity_frame = frame

    # --------------------------- contact evidence ----------------------------
    def _capture_sample(self, obs: dict[str, Any], frame: int) -> None:
        ball = self._ball(obs)
        if ball is None or not bool(obs.get("ball_in_play", False)):
            return
        dy = self._num(ball.get("dy"))
        y = self._num(ball.get("y"))
        if dy > 0.0 and y >= 154.0:
            paddle = self._paddle(obs)
            self.contact_samples.append({
                "frame": frame,
                "x": self._num(ball.get("x")),
                "y": y,
                "px": self._num(paddle.get("x"), self.paddle_x),
                "motion": self.last_raw_v,
                "dx": self._num(ball.get("dx")),
                "dy": dy,
                "landing": self.flight_landing,
                "action": self.last_posted,
            })

    def _stable_offset_band(self, pw: float) -> tuple[float, float]:
        # offset is ball center minus paddle-left.  A broad interval keeps the
        # survival controller from trusting an unverified decoder edge.
        if len(self.safe_offsets) < 4:
            return 2.0, max(8.0, pw - 2.0)
        values = sorted(self.safe_offsets)
        lo = max(1.5, values[max(0, len(values)//8)] - 1.5)
        hi = min(pw - 1.5, values[min(len(values)-1, (7*len(values))//8)] + 1.5)
        if hi - lo < 8.0:
            mid = (lo + hi)/2.0
            lo, hi = max(1.5, mid-4.0), min(pw-1.5, mid+4.0)
        return lo, hi

    def _record_contact(self, obs: dict[str, Any], frame: int, caught: bool) -> None:
        rows = list(self.contact_samples)
        crossing = min(rows, key=lambda r: abs(r["y"]-self.contact_plane)) if rows else None
        paddle = self._paddle(obs)
        pw = self._num(paddle.get("w"), 16.0)
        if crossing is None:
            self.pending_outgoing = None
            self.contact_samples.clear()
            return
        offset = crossing["x"] + 1.0 - crossing["px"]
        center_offset = crossing["x"] + 1.0 - (crossing["px"] + pw/2.0)
        drift = max((abs(row["motion"]) for row in rows[-5:]), default=0.0)
        predicted = crossing.get("landing")
        residual = None if predicted is None else abs((crossing["x"]+1.0) - self._num(predicted))
        stable = drift <= 1.5
        supported = residual is not None and residual <= 4.0
        target = None if self.trial is None else self.trial.get("desired_offset")
        achieved = target is not None and abs(center_offset-target) <= 2.2
        record = {
            "frame": frame, "caught": bool(caught), "offset": round(offset, 2),
            "center_offset": round(center_offset, 2), "y": crossing["y"],
            "drift": round(drift, 2), "stable": stable, "predicted": predicted,
            "residual": None if residual is None else round(residual, 2),
            "supported": supported, "trial": None if self.trial is None else dict(self.trial),
            "achieved": achieved, "wall": self._int(obs.get("wall_index")),
        }
        self.contacts.append(record)
        self._log("contact_evidence", **record, window=rows[-8:])
        if caught:
            self.stable_catches += int(stable and supported)
            self.rally_catches += 0  # rally count is incremented at the event boundary
            if stable and supported:
                self.safe_offsets.append(offset)
                if predicted is not None:
                    self.landing_bias = self._clamp(
                        0.85*self.landing_bias + 0.15*((crossing["x"]+1.0)-self._num(predicted)), -3.0, 3.0
                    )
                self.plane_samples.append(crossing["y"])
                if len(self.plane_samples) >= 4:
                    self.contact_plane = self._clamp(float(median(self.plane_samples)), 184.0, 188.0)
                # These are actual incoming fast-flight samples, not a generic
                # bounce-angle claim.  They support a sparse-wall ray even when
                # a deliberate endpoint trial was not reached.
                if abs(crossing.get("dx", 0.0)) + abs(crossing.get("dy", 0.0)) >= 4.0:
                    self.fast_speeds.append((abs(crossing.get("dx", 0.0)),
                                              abs(crossing.get("dy", 0.0))))
            if self.trial is not None:
                if achieved:
                    self.last_trial_outcome = "achieved"
                else:
                    self.last_trial_outcome = "endpoint_not_reached"
                    self.trial_failures += 1
                self._log("objective_contact", frame=frame, outcome=self.last_trial_outcome,
                          trial=self.trial, center_offset=round(center_offset, 2), outgoing="pending")
        else:
            self.stable_misses += int(stable and supported)
            if self.trial is not None:
                self.last_trial_outcome = "miss"
                self.trial_failures += 1
            # A miss is immediate evidence against an active objective.  The
            # next flight always starts with survival; no model is awaited.
            self.trial = None
            self.trial_cooldown = max(self.trial_cooldown, 3)
            self.objective = "AUTO"
            self.objective_contact = -1
        self.pending_outgoing = ({
            "frame": frame, "caught": True, "trial": record.get("trial"),
            "samples": [], "wall": self._int(obs.get("wall_index")),
            "reliable": bool(stable and supported), "achieved": bool(achieved),
        } if caught else None)
        self.contact_samples.clear()

    def _confirm_outgoing(self, obs: dict[str, Any], frame: int) -> None:
        pending = self.pending_outgoing
        ball = self._ball(obs)
        if pending is None or ball is None:
            return
        age = frame - pending["frame"]
        if 1 <= age <= 5 and self._num(ball.get("dy")) < 0.0:
            pending["samples"].append((self._num(ball.get("dx")), self._num(ball.get("dy"))))
        if age < 4:
            return
        values = pending["samples"]
        self.pending_outgoing = None
        if len(values) < 2:
            return
        vx = sum(x for x, _ in values)/len(values)
        vy = sum(y for _, y in values)/len(values)
        outgoing = LEFT if vx < -0.25 else RIGHT if vx > 0.25 else "unknown"
        self.last_outgoing = outgoing
        trial = pending.get("trial")
        if trial is not None and outgoing in STEERING and pending.get("caught"):
            side = trial.get("side")
            reliable = bool(pending.get("reliable") and pending.get("achieved"))
            if reliable:
                self.endpoint_reliability[side] += 1
                self.endpoint_stats[side][0 if outgoing == side else 1] += 1
                if abs(vy) >= 2.8 and abs(vx) >= 0.8:
                    self.fast_endpoint_stats[side][0 if outgoing == side else 1] += 1
                    self.fast_speeds.append((abs(vx), abs(vy)))
            self._log("outgoing_measured", frame=frame, side=side, outgoing=outgoing,
                      expected=side, reliable=reliable, samples=values)
            # Only a reliable achieved endpoint can teach the mapping.  A
            # miss/endpoint-not-reached must not flip LEFT into RIGHT.
            if reliable and outgoing != side:
                good, bad = self.endpoint_stats[side]
                if bad >= 2 and bad > good + 1:
                    self.endpoint_offset[side] = -self.endpoint_offset[side]
                    self.endpoint_stats[side] = [0, 0]
                    self.fast_endpoint_stats[side] = [0, 0]
                    self._log("endpoint_mapping_flipped", frame=frame, side=side,
                              desired_offset=self.endpoint_offset[side])

    # --------------------------- reflected geometry --------------------------
    @staticmethod
    def _reflect(x: float, dx: float, frames: float) -> float:
        left, right = 8.0, 150.0
        span = right-left
        if span <= 0:
            return x
        phase = (x-left + dx*max(0.0, frames)) % (2.0*span)
        return left+phase if phase <= span else right-(phase-span)

    def _speed_label(self) -> str:
        total = abs(self.ball_vx)+abs(self.ball_vy)
        if not self.ball_velocity_valid or total < 0.5:
            return self.speed_regime
        return "fast" if abs(self.ball_vy) >= 2.8 else "normal"

    def _project_landing(self, obs: dict[str, Any], frame: int) -> tuple[float, float, float] | None:
        ball = self._ball(obs)
        paddle = self._paddle(obs)
        if ball is None or not bool(obs.get("ball_in_play", False)):
            return None
        if not self.ball_velocity_valid or self.ball_vy <= 0.0:
            return None
        by = self._num(ball.get("y"))
        if by < 94.0:
            return None
        py = self._num(paddle.get("y"), 189.0)
        # A median of observed bounce planes avoids committing to an
        # unverified rectangle-edge convention.  The half-pixel plane is a
        # nominal center; the safety interval absorbs the remaining one-frame
        # ordering uncertainty.
        eta = max(0.0, (self.contact_plane-by)/max(0.25, self.ball_vy))
        bx = self._num(ball.get("x"))
        candidates = []
        for v in (self.ball_vx-0.30, self.ball_vx, self.ball_vx+0.30):
            for t in (max(0.0, eta-0.6), eta, eta+0.6):
                candidates.append(self._reflect(bx, v, t))
        landing = float(median(candidates)) + 1.0 + self.landing_bias
        uncertainty = max(1.0, (max(candidates)-min(candidates))/2.0)
        return landing, eta, uncertainty

    def _make_flight(self, obs: dict[str, Any], frame: int) -> dict[str, Any]:
        ball = self._ball(obs)
        paddle = self._paddle(obs)
        if ball is None or not bool(obs.get("ball_in_play", False)):
            low, high, center = self._home_interval(obs)
            return {"phase": "serve", "urgency": "serve", "eta": None,
                    "low": low, "high": high, "center": center,
                    "landing": None, "speed": "unknown", "goal": None}
        if self.ball_vy <= 0.0 or not self.ball_velocity_valid:
            self.flight_valid = False
            low, high, center = self._home_interval(obs)
            return {"phase": "ascending" if self.ball_vy < 0 else "uncertain",
                    "urgency": "early", "eta": None, "low": low, "high": high,
                    "center": center, "landing": None, "speed": self._speed_label(), "goal": None}
        projection = self._project_landing(obs, frame)
        if projection is None:
            low, high, center = self._home_interval(obs)
            return {"phase": "uncertain_descent", "urgency": "early", "eta": None,
                    "low": low, "high": high, "center": center, "landing": None,
                    "speed": self._speed_label(), "goal": None}
        landing, eta, uncertainty = projection
        pw = self._num(paddle.get("w"), 16.0)
        loff, hoff = self._stable_offset_band(pw)
        # Expand rather than shrink the certified set for projection noise.
        low = landing - hoff - uncertainty
        high = landing - loff + uncertainty
        pmin, pmax = 8.0, max(8.0, 160.0-pw)
        low, high = self._clamp(low, pmin, pmax), self._clamp(high, pmin, pmax)
        if high < low:
            low = high = self._clamp(landing-pw/2.0, pmin, pmax)
        # Hold the center through ordinary quantization.  A large change is a
        # real replan, usually after a side/brick/speed event.
        if self.flight_valid and self.flight_landing is not None and abs(landing-self.flight_landing) < 5.0:
            landing = 0.75*self.flight_landing + 0.25*landing
        self.flight_valid = True
        self.flight_landing = landing
        self.flight_eta = eta
        self.flight_low, self.flight_high = low, high
        self.flight_center = (low+high)/2.0
        self.flight_vx, self.flight_vy = self.ball_vx, self.ball_vy
        urgency = "imminent" if eta <= 5.0 else "near" if eta <= 12.0 else "early"
        return {"phase": "descending", "urgency": urgency, "eta": round(eta, 2),
                "low": round(low, 2), "high": round(high, 2),
                "center": round(self.flight_center, 2), "landing": round(landing, 2),
                "speed": self._speed_label(), "goal": None, "uncertainty": round(uncertainty, 2)}

    def _home_interval(self, obs: dict[str, Any]) -> tuple[float, float, float]:
        # The strategic objective moves only this broad early anchor.  It can
        # never replace a descending contact reservation.
        lives = self._int(obs.get("lives"), 0)
        center = 76.0
        if lives >= 4 and self.objective in STEERING:
            center = 68.0 if self.objective == LEFT else 84.0
        return self._clamp(center-15.0, 8.0, 144.0), self._clamp(center+15.0, 8.0, 144.0), center

    # ------------------------------ objective code ---------------------------
    def _side_facts(self, obs: dict[str, Any]) -> dict[str, Any]:
        bitmap = obs.get("brick_bitmap") if isinstance(obs, dict) else None
        left = right = 0
        if isinstance(bitmap, list):
            for row in bitmap:
                if not isinstance(row, list):
                    continue
                if len(row) >= 2:
                    left += int(bool(row[0]))+int(bool(row[1]))
                if len(row) >= 18:
                    right += int(bool(row[16]))+int(bool(row[17]))
        return {
            "left": "open" if left == 0 else "partial" if left <= 3 else "blocked",
            "right": "open" if right == 0 else "partial" if right <= 3 else "blocked",
            "left_count": min(12, left), "right_count": min(12, right),
            "wall": self._int(obs.get("wall_index")),
            "bricks": self._int(obs.get("bricks_remaining"), 108),
        }

    def _mapping_confidence(self, side: str) -> bool:
        good, bad = self.endpoint_stats[side]
        return good >= 2 and good >= bad+1

    def _cleanup_side(self, obs: dict[str, Any], flight: dict[str, Any]) -> str | None:
        if self._int(obs.get("bricks_remaining"), 108) > 12 or self._int(obs.get("lives"), 0) < 2:
            return None
        if flight.get("speed") != "fast":
            return None
        if self._frame(obs)-self.last_progress_frame < 420 or flight.get("phase") != "descending":
            return None
        # A bounded fallback ray is allowed only for the final few bricks; for
        # a larger sparse wall it still requires an observed endpoint mapping.
        if self.trials_this_life >= 4:
            return None
        if len(self.fast_speeds) < 3:
            return None
        bricks = obs.get("bricks")
        landing = flight.get("landing")
        if not isinstance(bricks, list) or landing is None:
            return None
        vx = sum(x for x, _ in self.fast_speeds)/len(self.fast_speeds)
        vy = sum(y for _, y in self.fast_speeds)/len(self.fast_speeds)
        if vy < 1.0:
            return None
        best_side, best_value = None, -1e9
        for side, sign in ((LEFT, -1.0), (RIGHT, 1.0)):
            if self._int(obs.get("bricks_remaining"), 108) > 8 and not self._mapping_confidence(side):
                continue
            score = -1e9
            for brick in bricks:
                if not isinstance(brick, dict):
                    continue
                top = self._num(brick.get("y"), 60.0)
                height = self._num(brick.get("h"), 6.0)
                t = max(0.0, (186.0-top-height)/vy)
                x = self._reflect(float(landing)-1.0, sign*abs(vx), t)+1.0
                bx, bw = self._num(brick.get("x")), self._num(brick.get("w"), 8.0)
                gap = max(bx-x, 0.0, x-(bx+bw))
                value = self._num(brick.get("value"), 1.0)
                score = max(score, (14.0 if gap <= 1.0 else 0.0)+value-gap)
            if score > best_value:
                best_side, best_value = side, score
        return best_side

    def _candidate_side(self, obs: dict[str, Any], flight: dict[str, Any]) -> tuple[str | None, str]:
        cleanup = self._cleanup_side(obs, flight)
        if cleanup is not None:
            self.cleanup_mode = True
            return cleanup, "cleanup"
        self.cleanup_mode = False
        side = self._side_facts(obs)
        regime = flight.get("speed") if flight.get("speed") in ("normal", "fast") else self.speed_regime
        if self._int(obs.get("lives"), 0) < 4 or regime != "normal":
            return None, "survival"
        # Jev's SURVIVE answer is a one-contact veto, not a permanent mode;
        # the next bounce/loss clears it.
        if self.objective == "SURVIVE" and self.objective_contact == self.contact_id:
            return None, "survival"
        if self.rally_catches < 2 or self.trial_cooldown > 0 or self.trials_this_life >= 3:
            return None, "survival"
        # Before the next descending segment there is no ETA yet, but the
        # contact objective can safely be reserved now.  The hard interval is
        # still created only by _shape_flight on descent.
        if flight.get("phase") == "descending":
            if self._num(flight.get("eta"), 0) <= 14.0:
                return None, "survival"
        elif flight.get("phase") not in ("ascending", "uncertain", "uncertain_descent"):
            return None, "survival"
        if self.trial_failures >= 2 and self.rally_catches < 5:
            return None, "survival"
        choice = self.objective if self.objective in STEERING else self.astra_plan if self.astra_plan in STEERING else None
        if choice not in STEERING:
            if side["left"] == "open" and side["right"] != "open":
                choice = RIGHT
            elif side["right"] == "open" and side["left"] != "open":
                choice = LEFT
            elif side["left_count"] < side["right_count"]:
                choice = LEFT
            elif side["right_count"] < side["left_count"]:
                choice = RIGHT
            else:
                choice = LEFT if self.ball_vx < 0 else RIGHT
        if side["left" if choice == LEFT else "right"] == "open":
            return None, "survival"
        return choice, "tunnel"

    def _shape_flight(self, obs: dict[str, Any], flight: dict[str, Any], frame: int) -> dict[str, Any]:
        if flight.get("phase") != "descending":
            return flight
        candidate, purpose = self._candidate_side(obs, flight)
        eligible = candidate in STEERING
        if self.trial is None and self.trial_considered != self.contact_id and eligible:
            # A soft goal is used, never a narrowed hard interval.  We only
            # start a trial when a short motor rollout can reach its target
            # without violating the broad reservation.
            desired_offset = self.endpoint_offset.get(candidate, -3.5 if candidate == LEFT else 3.5)
            if purpose == "cleanup":
                desired_offset = self.endpoint_offset.get(candidate, -3.5 if candidate == LEFT else 3.5)
            target = self._num(flight.get("landing"), 76.0) - (
                self._num(self._paddle(obs).get("w"), 16.0)/2.0 + desired_offset
            )
            if self._reachable_target(obs, flight, target):
                self.trial = {"side": candidate, "purpose": purpose, "contact": self.contact_id,
                              "frame": frame, "desired_offset": desired_offset, "target": target,
                              "source": "jev" if self.objective == candidate else "astra" if self.astra_plan == candidate else "code"}
                self.trials_this_life += 1
                self._log("objective_started", **self.trial)
            else:
                self._log("objective_deferred", frame=frame, side=candidate, reason="unreachable_target")
            self.trial_considered = self.contact_id
        result = dict(flight)
        if self.trial is not None and self.objective != "SURVIVE":
            # Keep the full physical interval and expose only a mild utility
            # target to the motor. This is safer than the parents' narrow
            # endpoint interval under queued/sticky motion.
            result["goal"] = self.trial.get("target")
            # A contact goal is a real but subordinate utility: it can move
            # within the broad reservation, while the large interval penalty
            # still dominates any unsafe endpoint.
            result["goal_weight"] = 0.9 if purpose == "tunnel" else 1.2
        return result

    def _reachable_target(self, obs: dict[str, Any], flight: dict[str, Any], target: float) -> bool:
        px = self._num(self._paddle(obs).get("x"), self.paddle_x)
        eta = max(1.0, self._num(flight.get("eta"), 1.0))
        possible = []
        for action in (NOOP, LEFT, RIGHT):
            for sticky in (False, True):
                x, v = self._simulate(px, self.last_raw_v, self.last_posted, action, min(8.0, eta), sticky)
                possible.append(x)
        return min(possible)-3.0 <= target <= max(possible)+3.0

    # -------------------------- queued motor reservation ---------------------
    def _simulate(self, x: float, v: float, prior: str, action: str,
                  steps: float, sticky: bool = False) -> tuple[float, float]:
        n = max(1, int(math.ceil(steps)))
        pmin, pmax = 8.0, 144.0
        current = self._u(prior)
        requested = self._u(action)
        first = current if sticky else requested
        a, b, c = self.motor
        for i in range(n):
            if i == 0:
                u = first
                old_u = self._u(prior)
            elif i == 1:
                u = requested
                old_u = first
            else:
                # Releasing a direction does not brake instantly; use the
                # fitted carry term for the next step, then let drag decay it.
                u = 0.0
                old_u = requested if i == 2 else 0.0
            v = self._clamp(a*v + b*u + c*old_u, -6.0, 6.0)
            next_x = self._clamp(x+v, pmin, pmax)
            if (next_x == pmin and v < 0) or (next_x == pmax and v > 0):
                v = 0.0
            x = next_x
        return x, v

    def _candidate_cost(self, obs: dict[str, Any], flight: dict[str, Any], action: str) -> tuple[float, float]:
        px = self._num(self._paddle(obs).get("x"), self.paddle_x)
        v = self.last_raw_v
        eta = max(1.0, self._num(flight.get("eta"), 1.0))
        low, high = self._num(flight.get("low"), 61.0), self._num(flight.get("high"), 91.0)
        goal = flight.get("goal")
        weight = self._num(flight.get("goal_weight"), 0.0)
        costs = []
        endpoints = []
        # 75% fresh command, 25% sticky previous command; also account for a
        # stale queued input in the first frame.  Worst-case distance keeps a
        # cheap average from choosing a lethal edge.
        for sticky in (False, True):
            x, tail = self._simulate(px, v, self.last_posted, action, eta, sticky)
            distance = max(low-x, 0.0, x-high)
            terminal = abs(tail)
            target_cost = 0.0 if goal is None else weight*(x-goal)*(x-goal)
            costs.append(26.0*distance*distance + 0.24*terminal*terminal + target_cost)
            endpoints.append(x)
        expected = 0.75*costs[0]+0.25*costs[1]
        worst = max(costs)
        # Directional commands are costly only when they create motion; NOOP
        # is not assumed to brake, so it is not given a free pass.
        return expected + 1.4*worst + (0.04 if action != NOOP else 0.0), float(median(endpoints))

    def _choose_first_action(self, obs: dict[str, Any], flight: dict[str, Any], frame: int) -> str:
        if bool(obs.get("terminated")) or bool(obs.get("truncated")):
            return NOOP
        if bool(obs.get("serve_required")):
            return FIRE if self._int(obs.get("lives"), 0) > 0 else NOOP
        if self._ball(obs) is None:
            return NOOP
        low, high = self._num(flight.get("low")), self._num(flight.get("high"))
        px = self._num(self._paddle(obs).get("x"), self.paddle_x)
        eta = self._num(flight.get("eta"), 20.0)
        phase = flight.get("phase")

        # Home/ascending flight uses the reservation's broad center. It is
        # intentionally quiet, but an unreachable return gets an early push.
        if phase != "descending":
            home_low, home_high, home_center = self._home_interval(obs)
            if px < home_low-3.0:
                return RIGHT
            if px > home_high+3.0:
                return LEFT
            return NOOP

        # Start a late reservation once per flight.  It is a small state
        # machine, not a point servo: commit for two frames, stop on entering
        # the safe set, and allow at most one emergency reversal.
        if not self.late_started and eta <= 10.0:
            self.late_started = True
            self.late_flip_used = False
            self.emergency_used = False
            self.late_until = frame
            self.late_action = NOOP
        costs = {action: self._candidate_cost(obs, flight, action) for action in (NOOP, LEFT, RIGHT)}
        best = min(costs, key=lambda key: costs[key][0])
        _, projected = costs[best]
        inside = low-1.0 <= px <= high+1.0
        clearly_left = projected < low-2.0
        clearly_right = projected > high+2.0

        if eta <= 10.0:
            # Maintain an existing directional reservation while it is moving
            # toward the interval. Do not mirror it on every alternating RAM
            # sample. Once inside, release rather than counter-steer unless the
            # no-op rollout is clearly lethal.
            if self.late_action in STEERING and frame < self.late_until:
                if low-2.0 <= px <= high+2.0:
                    self.late_action = NOOP
                    self.late_until = frame
                elif (self.late_action == RIGHT and not clearly_left) or (self.late_action == LEFT and not clearly_right):
                    return self.late_action
            if inside and not (clearly_left or clearly_right):
                self.late_action = NOOP
                self.late_until = frame+1
                return NOOP
            if best in STEERING:
                if self.late_action in STEERING and best != self.late_action and frame < self.late_until:
                    return self.late_action
                if self.late_action in STEERING and best != self.late_action:
                    if frame < self.last_late_flip+3 or self.late_flip_used:
                        return self.late_action
                    self.late_flip_used = True
                    self.last_late_flip = frame
                self.late_action = best
                self.late_until = frame+2 if eta > 5.0 else frame+1
                return best
            self.late_action = NOOP
            self.late_until = frame+1
            return NOOP

        # Early descent can use the full one-frame reservation search. A large
        # outside error is a reachability invariant: never let a learned model
        # settle on absorbing NOOP while the paddle is far away.
        if px < low-10.0:
            return RIGHT
        if px > high+10.0:
            return LEFT
        return best

    # ----------------------------- Jev interface ----------------------------
    @staticmethod
    def _answer_value(answer: Any) -> str | None:
        value = getattr(answer, "value", None)
        return value if isinstance(value, str) else None

    @staticmethod
    def _prob(answer: Any, key: str) -> float:
        probs = getattr(answer, "probabilities", None)
        if not isinstance(probs, dict):
            return 0.0
        try:
            p = float(probs.get(key, 0.0))
            return p if math.isfinite(p) else 0.0
        except (TypeError, ValueError):
            return 0.0

    def _model_state(self, obs: dict[str, Any], flight: dict[str, Any], event: str) -> dict[str, Any]:
        side = self._side_facts(obs)
        candidate, purpose = self._candidate_side(obs, flight)
        recent = list(self.contacts)[-5:]
        return {
            "game": "Atari 2600 Breakout ordinary cartridge mode 0 difficulty 0, native 60 Hz, sticky actions",
            "rules": {
                "frame_actions": list(ACTIONS), "fire": "only when serve_required",
                "priority": "preserve a catch before any tunnel or cleanup objective",
                "bridge": "choose a contact objective; code owns reflected geometry, motor and frame action",
                "life_loss": "a missing ball requests FIRE and does not reset the episode",
            },
            "observation": {
                "frame": self._frame(obs), "event": event, "lives": self._int(obs.get("lives")),
                "score": self._num(obs.get("score")), "wall": side["wall"], "bricks_remaining": side["bricks"],
                "phase": flight.get("phase"), "urgency": flight.get("urgency"), "speed": flight.get("speed"),
                "catch_interval": "wide" if self._num(flight.get("high"))-self._num(flight.get("low")) >= 8 else "narrow",
                "safe_for_shaping": candidate in STEERING,
                "candidate_side": candidate or "none", "objective_kind": purpose,
                "left_edge": side["left"], "right_edge": side["right"],
                "rally_catches": self.rally_catches, "trials_this_life": self.trials_this_life,
                "current_objective": self.objective, "astra_proposal": self.astra_plan,
                "trial_outcome": self.last_trial_outcome, "last_outgoing": self.last_outgoing,
                "endpoint_support": {k: self._mapping_confidence(k) for k in STEERING},
                "recent_contacts": recent,
                "contact_id": self.contact_id, "safety_epoch": self.safety_epoch,
                "astra_revision": self.astra_revision,
            },
        }

    def _questions(self) -> dict[str, Choice | Noul]:
        options = {
            "SURVIVE": {"what": "Use the broad code-computed catch reservation and do not bias a paddle contact.",
                        "when": "The return is uncertain, near, fast, narrow, lives are low, or no safe candidate exists."},
            "LEFT": {"what": "Approve one safe interior contact intended to send the ball left toward an occupied outer route.",
                     "when": "Only when safe_for_shaping is true and candidate_side or a measured cleanup route supports LEFT."},
            "RIGHT": {"what": "Approve one safe interior contact intended to send the ball right toward an occupied outer route.",
                      "when": "Only when safe_for_shaping is true and candidate_side or a measured cleanup route supports RIGHT."},
            "NO_CHANGE": {"what": "Leave the code-selected objective unchanged; do not invent a new side.",
                          "when": "The evidence is ambiguous or the current code fallback is already appropriate."},
        }
        keys = list(options)
        shift = self.jev_serial % len(keys)
        order = keys[shift:]+keys[:shift]
        instruction = (
            "Choose a long-lived objective for the NEXT paddle contact in this exact derived state. "
            "Do not emit a frame action and do not calculate coordinates. Code already computes the "
            "catch_interval, safe_for_shaping, candidate_side, endpoint_support and all arithmetic. "
            "Choose SURVIVE when safe_for_shaping is false, urgency is near/imminent, speed is fast or "
            "unknown, lives are below four, or the proposed side is open. For ordinary tunnel "
            "mode require endpoint_support; for cleanup trust candidate_side and code's "
            "remaining-brick ray, including the bounded final-eight-brick fallback. The answer is "
            "discarded if the two reordered judgments disagree or the contact changes. State="
            + json.dumps(self._jev_state_for_prompt, separators=(",", ":"))
        )
        return {
            "objective_a": Choice(instruction, {key: options[key] for key in order}),
            "objective_b": Choice(instruction, {key: options[key] for key in reversed(order)}),
            "safe_now": Noul("Is safe_for_shaping true for the stated next-contact objective?"),
        }

    def _launch_jev(self, obs: dict[str, Any], flight: dict[str, Any], event: str, frame: int) -> None:
        if not self.jev_enabled or self.jev_future is not None or frame < self.next_jev_frame:
            return
        if self._ball(obs) is None or bool(obs.get("serve_required")):
            return
        boundary = event in ("serve", "paddle_bounce", "life_loss", "wall_transition")
        candidate, _ = self._candidate_side(obs, flight)
        if not boundary and not (candidate in STEERING and self.trial_considered != self.contact_id):
            return
        try:
            if int(self.ctx.budget.remaining().get("system1_calls", 0)) <= 4:
                return
        except Exception:
            return
        state = self._model_state(obs, flight, event)
        self._jev_state_for_prompt = state
        questions = self._questions()
        self.jev_serial += 1
        self.jev_request_frame = frame
        self.jev_stamp = (self.safety_epoch, self.wall_epoch, self.contact_id, self.astra_revision)
        self.next_jev_frame = frame+45
        stamp = self.jev_stamp

        def call():
            return self.ctx.decide(state, questions, timeout_s=2.4)

        try:
            self.jev_future = self.ctx.spawn(call)
        except Exception as exc:
            self.jev_future = None
            self._log("jev_spawn_failed", error=type(exc).__name__)

    def _collect_jev(self, obs: dict[str, Any], flight: dict[str, Any], frame: int) -> None:
        future = self.jev_future
        if future is None or not getattr(future, "done", lambda: False)():
            return
        self.jev_future = None
        try:
            answers = future.result()
            if not isinstance(answers, dict):
                return
            a, b = answers.get("objective_a"), answers.get("objective_b")
            av, bv = self._answer_value(a), self._answer_value(b)
            allowed = {"SURVIVE", "LEFT", "RIGHT", "NO_CHANGE"}
            p = min(self._prob(a, av), self._prob(b, bv)) if av and bv else 0.0
            current = self.jev_stamp == (self.safety_epoch, self.wall_epoch, self.contact_id, self.astra_revision)
            fresh = frame-self.jev_request_frame <= 120
            if av not in allowed or av != bv or p < 0.58 or not current or not fresh:
                self._log("jev_ignored", frame=frame, reason="stale_disagree_or_low_probability")
                return
            candidate, purpose = self._candidate_side(obs, flight)
            if av in STEERING:
                if self._int(obs.get("lives"), 0) < 4 and purpose != "cleanup":
                    av = "SURVIVE"
                elif flight.get("speed") != "normal" and purpose != "cleanup":
                    av = "SURVIVE"
                elif candidate != av and purpose != "cleanup":
                    av = "SURVIVE"
                elif self._side_facts(obs)["left" if av == LEFT else "right"] == "open" and purpose != "cleanup":
                    av = "SURVIVE"
            self.objective = av
            self.objective_contact = self.contact_id
            self.objective_version += 1
            self._log("jev_objective_accepted", frame=frame, objective=av, probability=round(p, 3), contact=self.contact_id)
        except (TimeoutError, RuntimeError, BudgetExceeded, EpisodeOver) as exc:
            self._log("jev_unavailable", error=type(exc).__name__)
        except Exception as exc:
            self._log("jev_parse_ignored", error=type(exc).__name__)

    # ----------------------------- Astra interface ---------------------------
    def _astra_prompt(self, obs: dict[str, Any], flight: dict[str, Any], stage: str) -> str:
        side = self._side_facts(obs)
        state = {
            "game": "Atari 2600 Breakout mode 0 difficulty 0",
            "stage": stage, "wall": side["wall"], "lives": self._int(obs.get("lives")),
            "score": self._num(obs.get("score")), "bricks_remaining": side["bricks"],
            "left_edge": side["left"], "right_edge": side["right"],
            "phase": flight.get("phase"), "speed": flight.get("speed"),
            "rally_catches": self.rally_catches, "stable_catches": self.stable_catches,
            "stable_misses": self.stable_misses, "endpoint_support": {k: self._mapping_confidence(k) for k in STEERING},
            "last_trial_outcome": self.last_trial_outcome,
        }
        return (
            "You are the slow strategic layer for ordinary Atari 2600 Breakout. Code runs the paddle "
            "at 60 Hz and reserves every catch. Return exactly one token: SURVIVE, LEFT, or RIGHT. "
            "LEFT/RIGHT is only a future contact objective, never a frame action. Prefer SURVIVE for "
            "low lives, fast/uncertain flight, an open side, recent misses, or weak endpoint evidence. "
            "A side tunnel is worthwhile only when the corresponding occupied route is safe. In cleanup "
            "stage, prefer measured endpoint support; for the final eight bricks a bounded code ray "
            "may be used only when the remaining-brick evidence justifies it. "
            "Do not give coordinates, arithmetic, or explanations. State=" + json.dumps(state, separators=(",", ":"))
        )

    def _launch_astra(self, obs: dict[str, Any], flight: dict[str, Any], frame: int) -> None:
        if self.astra_future is not None or self.astra_calls >= 3 or self._ball(obs) is None:
            return
        side = self._side_facts(obs)
        stage = "cleanup" if self._cleanup_side(obs, flight) is not None else "tunnel"
        key = (side["wall"], stage)
        if key in self.astra_keys or frame < 30:
            return
        try:
            remain = self.ctx.budget.remaining()
            if int(remain.get("system2_calls", 0)) <= 1 or float(remain.get("wall_s", 0)) < 100.0:
                return
        except Exception:
            return
        prompt = self._astra_prompt(obs, flight, stage)
        self.astra_keys.add(key)
        self.astra_calls += 1
        self.astra_request_stamp = (self.safety_epoch, self.wall_epoch, side["wall"], stage)

        def call():
            return self.ctx.think(prompt, timeout_s=90.0)

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
            match = re.search(r"\b(SURVIVE|PRESERVE|SAFE|LEFT|RIGHT)\b", text.upper())
            if match is None:
                return
            token = match.group(1)
            token = "SURVIVE" if token in ("PRESERVE", "SAFE") else token
            side = self._side_facts(obs)
            if self.astra_request_stamp is None or self.astra_request_stamp[1] != self.wall_epoch:
                return
            if self._int(obs.get("lives"), 0) <= 2:
                token = "SURVIVE"
            if token == LEFT and side["left"] == "open":
                token = "SURVIVE"
            if token == RIGHT and side["right"] == "open":
                token = "SURVIVE"
            self.astra_plan = token
            self.astra_revision += 1
            self.objective = "AUTO"
            self.objective_contact = -1
            self._log("astra_plan_installed", frame=frame, plan=token, revision=self.astra_revision)
        except (TimeoutError, RuntimeError, BudgetExceeded, EpisodeOver) as exc:
            self._log("astra_unavailable", error=type(exc).__name__)
        except Exception as exc:
            self._log("astra_parse_ignored", error=type(exc).__name__)

    # ------------------------------- episode loop -----------------------------
    @staticmethod
    def _observe(game: Any) -> dict[str, Any]:
        try:
            value = game.observe()
        except Exception:
            return {}
        return value if isinstance(value, dict) else {}

    def _one(self, game: Any, realtime: bool) -> bool:
        obs = self._observe(game)
        if not isinstance(obs, dict):
            obs = {}
        frame = self._frame(obs)
        if frame == self.last_frame:
            if realtime:
                try:
                    self.ctx.wait_for_tick()
                except (BudgetExceeded, EpisodeOver):
                    return False
            return True
        event = self._event(obs)
        self._update_paddle(obs, frame)
        self._update_ball_velocity(obs, frame, event)
        self._confirm_outgoing(obs, frame)
        self._apply_event(obs, event, frame)
        self._capture_sample(obs, frame)
        flight = self._make_flight(obs, frame)
        self._collect_jev(obs, flight, frame)
        self._collect_astra(obs, frame)
        self._launch_astra(obs, flight, frame)
        self._launch_jev(obs, flight, event, frame)
        shaped = self._shape_flight(obs, flight, frame)
        action = self._choose_first_action(obs, shaped, frame)
        if action not in ACTIONS:
            action = NOOP
        self._log("control", frame=frame, action=action, phase=shaped.get("phase"),
                  eta=shaped.get("eta"), interval=[shaped.get("low"), shaped.get("high")],
                  landing=shaped.get("landing"), goal=shaped.get("goal"),
                  paddle=self._paddle(obs), paddle_v=round(self.paddle_v, 2),
                  flight=self.flight_id, profile=self.profile, objective=self.objective)
        try:
            self.ctx.act(action)
        except (BudgetExceeded, EpisodeOver):
            return False
        self.previous_posted = self.last_posted
        self.last_posted = action
        self.previous = obs
        self.last_frame = frame
        try:
            score = self._num(obs.get("score"), self.last_score)
            if score > self.last_score:
                self.last_progress_frame = frame
            self.last_score = score
        except Exception:
            pass
        if realtime:
            try:
                self.ctx.wait_for_tick()
            except (BudgetExceeded, EpisodeOver):
                return False
        return True

    def run_episode(self, game: Any) -> None:
        realtime = bool(getattr(game, "tick_s", 0.0))
        while not self.ctx.done and not bool(getattr(game, "done", False)):
            if not self._one(game, realtime):
                return


# This attribute is created lazily to keep the normal state strict JSON and to
# avoid duplicating a large prompt object between the two reordered questions.
BreakoutController._jev_state_for_prompt = {}


def build(ctx: Any) -> BreakoutController:
    return BreakoutController(ctx)
