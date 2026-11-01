"""Receding-horizon capture-set control for Atari 2600 Breakout.

The emulator-facing policy is code owned.  It predicts a *set* of paddle
positions that can catch the descending ball, then searches short LEFT/NOOP/
RIGHT pulse sequences through a measured paddle-response model.  Jev and
Astra may choose only a long-lived shaping preference; neither is on the
60-Hz critical path and neither can emit a raw action.
"""
from __future__ import annotations

from collections import deque
from itertools import product
import math
import re
from typing import Any

from metaharness.core.contract import BudgetExceeded, EpisodeOver
from metaharness.models.jev import Choice


ACTIONS = ("NOOP", "FIRE", "RIGHT", "LEFT")
MOVE_ACTIONS = ("LEFT", "NOOP", "RIGHT")
DIRECTIONS = {"LEFT", "RIGHT"}


def finite(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
        return result if math.isfinite(result) else default
    except (TypeError, ValueError, OverflowError):
        return default


def integer(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return default


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def direction(action: str) -> float:
    if action == "RIGHT":
        return 1.0
    if action == "LEFT":
        return -1.0
    return 0.0


class Harness:
    """A native-frame controller with a small adaptive action-sequence search."""

    def __init__(self, ctx: Any):
        self.ctx = ctx

        # Observation and event state.
        self.previous: dict[str, Any] | None = None
        self.last_frame = -1
        self.epoch = 0
        self.last_posted = "NOOP"
        self.last_effective = "NOOP"
        self.last_score = 0.0
        self.last_lives = 5
        self.losses = 0

        # Measured actuator state.  The defaults are intentionally modest; the
        # first few native frames replace them with the actual RAM displacement.
        self.paddle_v = 0.0
        self.paddle_v_ema = 0.0
        self.gain = {"LEFT": 3.0, "RIGHT": 3.0}
        self.gain_samples = {"LEFT": 0, "RIGHT": 0}
        self.noop_persistence = 0.55
        self.paddle_history = deque(maxlen=24)

        # Current ball velocity is taken from the adaptor's decoded fields,
        # with a positional fallback only for the serve transition.
        self.ball_vx = 0.0
        self.ball_vy = 0.0
        self.last_nonzero_vy = 0.0
        self.ball_history = deque(maxlen=8)

        # The action sequence search has a short memory only for diagnostics and
        # a mild anti-chatter tie break.  It is not a stale action mailbox.
        self.commit_action = "NOOP"
        self.commit_until = -1
        self.last_plan_cost = 0.0

        # Slow/fast model mailboxes.  They are advisory and may be absent in
        # pre-flight or in a tier-1 stand-in context.
        self.s1_enabled = getattr(ctx, "system1", object()) is not None
        self.s1_future = None
        self.s1_request_frame = -100000
        self.s1_request_epoch = -1
        self.s1_request_plan = -1
        self.s1_serial = 0
        self.s1_next_frame = 24
        self.s1_mode: str | None = None
        self.s1_mode_frame = -100000
        self.s1_mode_epoch = -1
        self.s1_mode_plan = -1

        self.astra_future = None
        self.astra_started = False
        self.astra_strategy = "SURVIVE"
        self.strategy_version = 0

        self.plan_version = 0
        self.current_event = "start"

    # ----------------------------- defensive plumbing ---------------------
    def _log(self, name: str, **fields: Any) -> None:
        try:
            self.ctx.log(name, **fields)
        except Exception:
            # Diagnostics must never delay a native-frame action.
            pass

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

    def _observe(self) -> dict[str, Any]:
        try:
            value = self.ctx.observe()
        except Exception:
            return {}
        return value if isinstance(value, dict) else {}

    def _frame(self, obs: dict[str, Any]) -> int:
        return integer(obs.get("frame"), self.last_frame + 1)

    def _remaining(self, key: str, default: float = 0.0) -> float:
        try:
            return finite(self.ctx.budget.remaining().get(key), default)
        except Exception:
            return default

    # ----------------------------- event/motion state ----------------------
    def _event(self, obs: dict[str, Any]) -> str:
        old = self.previous
        if old is None:
            return "start"
        old_lives = integer(old.get("lives"), self.last_lives)
        lives = integer(obs.get("lives"), old_lives)
        if lives < old_lives:
            return "life_loss"
        old_wall = integer(old.get("wall_index"), 0)
        wall = integer(obs.get("wall_index"), old_wall)
        if wall != old_wall:
            return "wall_transition"
        old_score = finite(old.get("score"), self.last_score)
        score = finite(obs.get("score"), old_score)
        if score > old_score:
            return "brick_hit"

        old_ball = self._ball(old)
        ball = self._ball(obs)
        if old_ball is None and ball is not None:
            return "serve"
        if old_ball is not None and ball is None:
            return "ball_lost"

        # This is a public, observable paddle-contact marker.  It invalidates
        # a landing plan even when the contact did not score a brick.
        if old_ball is not None and ball is not None:
            old_dy = finite(old_ball.get("dy"), 0.0)
            new_dy = finite(ball.get("dy"), self.ball_vy)
            old_y = finite(old_ball.get("y"), 0.0)
            new_y = finite(ball.get("y"), old_y)
            if old_dy > 0.0 and new_dy < 0.0 and max(old_y, new_y) >= 145.0:
                return "paddle_hit"
            old_dx = finite(old_ball.get("dx"), 0.0)
            new_dx = finite(ball.get("dx"), self.ball_vx)
            if old_dx * new_dx < -0.5 and max(old_y, new_y) >= 100.0:
                return "horizontal_bounce"
        return "none"

    def _reset_plan(self, reason: str, frame: int, major: bool = True) -> None:
        self.epoch += 1
        self.commit_action = "NOOP"
        self.commit_until = frame
        self.last_plan_cost = 0.0
        if major:
            self.plan_version += 1
            self.s1_mode = None
        self._log("capture_plan_reset", frame=frame, reason=reason, version=self.epoch)

    def _update_motion(self, obs: dict[str, Any]) -> None:
        old = self.previous
        paddle = self._paddle(obs)
        px = finite(paddle.get("x"), 99.0)
        old_paddle = self._paddle(old) if old is not None else {}
        old_px = finite(old_paddle.get("x"), px)
        dp = clamp(px - old_px, -12.0, 12.0)
        self.paddle_v = dp
        self.paddle_v_ema = 0.68 * self.paddle_v_ema + 0.32 * dp

        effective = obs.get("last_action") if isinstance(obs.get("last_action"), str) else self.last_posted
        if effective not in ACTIONS:
            effective = self.last_posted if self.last_posted in ACTIONS else "NOOP"
        self.last_effective = effective
        if old is not None:
            self.paddle_history.append((effective, dp))
            if effective in DIRECTIONS and dp * direction(effective) > 0.25:
                amount = abs(dp)
                self.gain[effective] = clamp(0.82 * self.gain[effective] + 0.18 * amount, 0.8, 6.0)
                self.gain_samples[effective] += 1
            elif effective == "NOOP" and abs(dp) > 0.2:
                # NOOP drift is the reason the action search simulates velocity
                # rather than assuming that NOOP is an instantaneous brake.
                self.noop_persistence = clamp(
                    0.88 * self.noop_persistence + 0.12 * min(0.95, abs(dp) / max(0.5, abs(self.paddle_v_ema) + 0.5)),
                    0.25,
                    0.90,
                )

        ball = self._ball(obs)
        old_ball = self._ball(old) if old is not None else None
        if ball is None:
            self.ball_vx = 0.0
            self.ball_vy = 0.0
            self.last_nonzero_vy = 0.0
            self.ball_history.clear()
            return

        raw_vx = finite(ball.get("dx"), 0.0)
        raw_vy = finite(ball.get("dy"), 0.0)
        inferred_vx = inferred_vy = 0.0
        if old_ball is not None:
            inferred_vx = clamp(
                finite(ball.get("x"), 0.0) - finite(old_ball.get("x"), 0.0), -8.0, 8.0
            )
            inferred_vy = clamp(
                finite(ball.get("y"), 0.0) - finite(old_ball.get("y"), 0.0), -8.0, 8.0
            )

        # The adaptor's decoded dx/dy is authoritative after a valid previous
        # position.  The zero/zero serve frame needs positional inference.
        if bool(obs.get("ball_velocity_valid", False)):
            vx, vy = raw_vx, raw_vy
        elif (abs(raw_vx) > 0.0 or abs(raw_vy) > 0.0) or old_ball is None:
            vx, vy = raw_vx, raw_vy
        elif abs(inferred_vx) > 0.0 or abs(inferred_vy) > 0.0:
            vx, vy = inferred_vx, inferred_vy
        else:
            vx, vy = self.ball_vx, self.ball_vy
        self.ball_vx = clamp(vx, -8.0, 8.0)
        self.ball_vy = clamp(vy, -8.0, 8.0)
        if abs(self.ball_vy) > 0.1:
            self.last_nonzero_vy = self.ball_vy
        self.ball_history.append((self.ball_vx, self.ball_vy))

    def _update_events(self, obs: dict[str, Any], event: str, frame: int) -> None:
        lives = integer(obs.get("lives"), self.last_lives)
        if event == "life_loss":
            self.losses += 1
            self._reset_plan("life_loss", frame, True)
            self.s1_next_frame = min(self.s1_next_frame, frame + 8)
        elif event in ("serve", "ball_lost", "wall_transition", "paddle_hit"):
            self._reset_plan(event, frame, True)
        elif event in ("brick_hit", "horizontal_bounce"):
            self._reset_plan(event, frame, False)

        self.last_lives = lives
        self.last_score = finite(obs.get("score"), self.last_score)
        self.current_event = event

    # ----------------------------- exact local geometry --------------------
    @staticmethod
    def _reflect_x(x: float, vx: float, frames: float) -> float:
        # Ball-left-coordinate bounds from the public Breakout RAM geometry.
        left, right = 8.0, 150.0
        span = right - left
        if span <= 0.0:
            return x
        phase = (x - left + vx * max(0.0, frames)) % (2.0 * span)
        return left + phase if phase <= span else right - (phase - span)

    @staticmethod
    def _home_interval() -> tuple[float, float]:
        # Paddle-left positions: a broad central parking corridor, not a
        # no-op instruction.  The sequence search actively reaches it.
        return 58.0, 94.0

    def _side_summary(self, obs: dict[str, Any]) -> dict[str, Any]:
        bitmap = obs.get("brick_bitmap") if isinstance(obs, dict) else None
        left = right = None
        if isinstance(bitmap, list):
            left_n = right_n = 0
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

        def bucket(value: int | None) -> str:
            if value is None:
                return "unknown"
            if value <= 0:
                return "open"
            if value <= 3:
                return "partial"
            return "blocked"

        return {
            "left": bucket(left),
            "right": bucket(right),
            "left_count": 0 if left is None else min(12, max(0, left)),
            "right_count": 0 if right is None else min(12, max(0, right)),
            "wall": integer(obs.get("wall_index"), 0),
            "walls_cleared": integer(obs.get("walls_cleared"), 0),
            "bricks_remaining": integer(obs.get("bricks_remaining"), 108),
        }

    def _speed_name(self) -> str:
        if abs(self.ball_vy) < 0.1 and abs(self.ball_vx) < 0.1:
            return "unknown"
        return "fast" if abs(self.ball_vy) >= 3.0 or abs(self.ball_vx) + abs(self.ball_vy) >= 5.0 else "normal"

    def _flight(self, obs: dict[str, Any], frame: int) -> dict[str, Any]:
        ball = self._ball(obs)
        paddle = self._paddle(obs)
        px = finite(paddle.get("x"), 99.0)
        pw = finite(paddle.get("w"), 16.0)
        if ball is None or not bool(obs.get("ball_in_play", False)):
            lo, hi = self._home_interval()
            return {
                "phase": "serve",
                "urgency": "serve",
                "eta": None,
                "safe_lo": lo,
                "safe_hi": hi,
                "target": (lo + hi) / 2.0,
                "landing": None,
                "lane": "unknown",
                "uncertainty": "unknown",
                "speed": "unknown",
                "shape_allowed": False,
            }

        bx = finite(ball.get("x"), 80.0)
        by = finite(ball.get("y"), 100.0)
        bw = finite(ball.get("w"), 2.0)
        bh = finite(ball.get("h"), 4.0)
        vx, vy = self.ball_vx, self.ball_vy

        if vy <= 0.0:
            lo, hi = self._home_interval()
            return {
                "phase": "ascending" if vy < 0.0 else "uncertain",
                "urgency": "early",
                "eta": None,
                "safe_lo": lo,
                "safe_hi": hi,
                "target": (lo + hi) / 2.0,
                "landing": None,
                "lane": "center",
                "uncertainty": "unknown" if vy == 0.0 else "wide",
                "speed": self._speed_name(),
                "shape_allowed": False,
            }

        py = finite(paddle.get("y"), 189.0)
        eta = max(0.0, (py - bh - by) / max(0.25, vy))
        speed = self._speed_name()

        # Before the ball has cleared the brick band, a point projection is not
        # trustworthy because another brick can change dx/dy.  We still steer
        # toward a broad home interval; unlike a parked NOOP this keeps the
        # paddle reachable for the later, clean descent.
        if by < 96.0:
            lo, hi = self._home_interval()
            urgency = "near" if eta <= 20.0 else "early"
            return {
                "phase": "collision_band",
                "urgency": urgency,
                "eta": round(eta, 2),
                "safe_lo": lo,
                "safe_hi": hi,
                "target": (lo + hi) / 2.0,
                "landing": None,
                "lane": "center",
                "uncertainty": "wide",
                "speed": speed,
                "shape_allowed": False,
            }

        # An ensemble is cheap and handles one-frame quantization, sticky
        # actions, and a just-observed speed transition without asking Jev to do
        # arithmetic.  It produces ball-center possibilities at the paddle.
        velocity_error = 0.85 if speed == "fast" else 0.45
        centers = []
        for xoff in (-1.0, 0.0, 1.0):
            for voff in (-velocity_error, 0.0, velocity_error):
                for eoff in (-0.75, 0.0, 0.75):
                    horizon = max(0.0, eta + eoff)
                    left_x = self._reflect_x(bx + xoff, vx + voff, horizon)
                    centers.append(left_x + bw / 2.0)
        center_lo, center_hi = min(centers), max(centers)
        nominal = self._reflect_x(bx, vx, eta) + bw / 2.0

        # For a ball center c, an interior paddle overlap is a range of paddle
        # left coordinates.  The interval is deliberately wider than a point
        # target and is what the action-sequence search tries to enter.
        margin = 1.5 if speed != "fast" else 2.5
        safe_lo = clamp(center_lo - pw + margin, 8.0, 144.0)
        safe_hi = clamp(center_hi - margin, 8.0, 144.0)
        if safe_hi < safe_lo:
            middle = clamp(nominal - pw / 2.0, 8.0, 144.0)
            safe_lo = safe_hi = middle
        spread = center_hi - center_lo
        urgency = "imminent" if eta <= 7.0 else "near" if eta <= 20.0 else "early"
        lane = "left" if nominal < 56.0 else "right" if nominal > 104.0 else "center"
        shape_allowed = (
            urgency == "early"
            and eta >= 20.0
            and speed == "normal"
            and integer(obs.get("lives"), 0) >= 3
            and (safe_hi - safe_lo) >= 8.0
        )
        return {
            "phase": "descending",
            "urgency": urgency,
            "eta": round(eta, 2),
            "safe_lo": round(safe_lo, 2),
            "safe_hi": round(safe_hi, 2),
            "target": round((safe_lo + safe_hi) / 2.0, 2),
            "landing": round(nominal, 2),
            "lane": lane,
            "uncertainty": "wide" if spread > 10.0 else "settled" if spread <= 4.0 else "moderate",
            "speed": speed,
            "shape_allowed": shape_allowed,
        }

    def _strategy_for_state(self, obs: dict[str, Any], flight: dict[str, Any], frame: int) -> str:
        """Return a fresh, currently legal strategic preference."""
        if (
            self.s1_mode in ("SHAPE_LEFT", "SHAPE_RIGHT")
            and self.s1_mode_epoch == self.epoch
            and self.s1_mode_plan == self.plan_version
            and frame - self.s1_mode_frame <= 90
        ):
            return self.s1_mode
        if self.astra_strategy in ("SHAPE_LEFT", "SHAPE_RIGHT"):
            return self.astra_strategy
        return "SURVIVE"

    def _goal_interval(self, obs: dict[str, Any], flight: dict[str, Any], frame: int) -> tuple[float, float]:
        lo = finite(flight.get("safe_lo"), 58.0)
        hi = finite(flight.get("safe_hi"), 94.0)
        if hi < lo:
            lo, hi = hi, lo
        strategy = self._strategy_for_state(obs, flight, frame)
        if not bool(flight.get("shape_allowed", False)) or strategy == "SURVIVE":
            return lo, hi

        # Shaping chooses an edge of an already-safe capture set, never a
        # coordinate outside it.  The mode expires quickly and is disabled by
        # the low-level flight/urgency gates above.
        width = hi - lo
        if strategy == "SHAPE_LEFT":
            center = lo + min(4.0, width * 0.25)
        else:
            center = hi - min(4.0, width * 0.25)
        return max(lo, center - 2.0), min(hi, center + 2.0)

    # ----------------------------- action sequence search ------------------
    def _step_paddle(self, position: float, velocity: float, effective: str) -> tuple[float, float]:
        """One-step local actuator model, refit from observed paddle motion."""
        if effective in DIRECTIONS:
            desired = direction(effective) * self.gain[effective]
            # A directional command changes velocity but does not teleport the
            # paddle.  This explicitly models braking/inertia and is refit every
            # frame by _update_motion.
            next_velocity = 0.35 * velocity + 0.65 * desired
        else:
            next_velocity = self.noop_persistence * velocity
        next_velocity = clamp(next_velocity, -6.0, 6.0)
        next_position = position + next_velocity
        if next_position < 8.0:
            return 8.0, 0.0 if next_velocity < 0.0 else next_velocity
        if next_position > 144.0:
            return 144.0, 0.0 if next_velocity > 0.0 else next_velocity
        return next_position, next_velocity

    @staticmethod
    def _gap(position: float, low: float, high: float) -> float:
        if position < low:
            return low - position
        if position > high:
            return position - high
        return 0.0

    def _simulate_sequence(
        self,
        start: float,
        velocity: float,
        sequence: tuple[str, ...],
        sticky: bool,
    ) -> tuple[float, float]:
        position, current_velocity = start, velocity
        previous_effective = self.last_effective if self.last_effective in ACTIONS else self.last_posted
        for selected in sequence:
            effective = selected
            if sticky and selected != previous_effective:
                effective = previous_effective
            position, current_velocity = self._step_paddle(position, current_velocity, effective)
            previous_effective = effective
        return position, current_velocity

    def _sequence_action(
        self,
        obs: dict[str, Any],
        flight: dict[str, Any],
        frame: int,
    ) -> str:
        if bool(obs.get("terminated", False)) or bool(obs.get("truncated", False)):
            return "NOOP"
        if bool(obs.get("serve_required", False)):
            self.commit_action = "FIRE"
            self.commit_until = frame + 1
            return "FIRE"
        ball = self._ball(obs)
        if ball is None or not bool(obs.get("ball_in_play", False)):
            return "NOOP"

        paddle = self._paddle(obs)
        start = finite(paddle.get("x"), 99.0)
        target_lo, target_hi = self._goal_interval(obs, flight, frame)
        eta = flight.get("eta")
        eta_n = finite(eta, 8.0) if eta is not None else 8.0
        # Receding horizon: exact contact when close, progressive movement when
        # the ball is still high.  Five frames is small enough for 60-Hz use.
        horizon = int(clamp(math.ceil(eta_n), 1.0, 5.0))
        velocity = clamp(0.65 * self.paddle_v + 0.35 * self.paddle_v_ema, -6.0, 6.0)

        best_action = "NOOP"
        best_cost = float("inf")
        previous = self.last_posted if self.last_posted in ACTIONS else "NOOP"
        for sequence in product(MOVE_ACTIONS, repeat=horizon):
            nominal_position, nominal_velocity = self._simulate_sequence(start, velocity, sequence, False)
            sticky_position, sticky_velocity = self._simulate_sequence(start, velocity, sequence, True)
            nominal_gap = self._gap(nominal_position, target_lo, target_hi)
            sticky_gap = self._gap(sticky_position, target_lo, target_hi)
            worst_gap = max(nominal_gap, sticky_gap)

            # Catchability dominates.  Once inside, residual velocity and a
            # direction reversal are costs, not reasons to issue a mirror pulse.
            cost = 6.0 * worst_gap + 2.0 * nominal_gap + 1.0 * sticky_gap
            if nominal_gap == 0.0:
                cost += 0.30 * abs(nominal_velocity)
            if sticky_gap == 0.0:
                cost += 0.20 * abs(sticky_velocity)
            switches = 0
            last_direction = direction(previous)
            for selected in sequence:
                selected_direction = direction(selected)
                if selected_direction and last_direction and selected_direction != last_direction:
                    switches += 1
                if selected_direction:
                    last_direction = selected_direction
            cost += 1.15 * switches
            if sequence[0] in DIRECTIONS and previous in DIRECTIONS and sequence[0] != previous:
                cost += 1.6
            cost += 0.025 * sum(1 for selected in sequence if selected in DIRECTIONS)

            # Prefer the same first command only on near ties.  It is a tie
            # break, not a hard latch: a clearly unreachable target can reverse.
            tie = 0.0 if sequence[0] == previous else 0.06
            ranked = cost + tie
            if ranked < best_cost:
                best_cost = ranked
                best_action = sequence[0]

        self.last_plan_cost = best_cost if math.isfinite(best_cost) else 0.0
        if best_action not in MOVE_ACTIONS:
            best_action = "NOOP"
        self.commit_action = best_action
        self.commit_until = frame + (1 if flight.get("urgency") in ("near", "imminent") else 2)
        return best_action

    # ----------------------------- Jev bridge ------------------------------
    @staticmethod
    def _answer_value(answer: Any) -> str | None:
        value = getattr(answer, "value", None)
        return value if isinstance(value, str) else None

    @staticmethod
    def _answer_probability(answer: Any, value: str) -> float:
        probs = getattr(answer, "probabilities", None)
        if isinstance(probs, dict) and probs:
            try:
                result = float(probs.get(value, 0.0))
                return result if math.isfinite(result) else 0.0
            except (TypeError, ValueError, OverflowError):
                return 0.0
        # Pre-flight stand-ins often expose only .value.
        return 1.0 if getattr(answer, "value", None) == value else 0.0

    def _jev_state(self, obs: dict[str, Any], flight: dict[str, Any], event: str, frame: int) -> dict[str, Any]:
        sides = self._side_summary(obs)
        if self.paddle_v_ema > 0.5:
            paddle_motion = "right"
        elif self.paddle_v_ema < -0.5:
            paddle_motion = "left"
        else:
            paddle_motion = "still"
        lives = integer(obs.get("lives"), 0)
        return {
            "game": "Atari 2600 Breakout ordinary cartridge mode 0 difficulty 0 at native 60 Hz",
            "observation": {
                "frame": frame,
                "event": event,
                "lives": lives,
                "low_lives": lives <= 2,
                "score": integer(finite(obs.get("score"), 0.0)),
                "flight_phase": flight.get("phase", "unknown"),
                "landing_urgency": flight.get("urgency", "unknown"),
                "landing_lane": flight.get("lane", "unknown"),
                "return_uncertainty": flight.get("uncertainty", "unknown"),
                "speed_regime": flight.get("speed", "unknown"),
                "paddle_motion": paddle_motion,
                "shape_is_code_eligible": bool(flight.get("shape_allowed", False)),
                "left_edge": sides["left"],
                "right_edge": sides["right"],
                "bricks_remaining": sides["bricks_remaining"],
                "wall_index": sides["wall"],
                "walls_cleared": sides["walls_cleared"],
            },
            "strategy_proposal": self.astra_strategy,
            "rules": {
                "legal_preferences": ["SURVIVE", "SHAPE_LEFT", "SHAPE_RIGHT"],
                "code_bridge": "The code searches short action sequences into a safe paddle capture set; this answer may only select a temporary safe shaping preference.",
                "safety": "Choose SURVIVE for low lives, fast or uncertain flight, near contact, an open requested edge, or ineligible code facts.",
            },
        }

    def _jev_questions(self, state: dict[str, Any]) -> dict[str, Choice]:
        descriptions = {
            "SURVIVE": "Use the complete code-computed capture set without deliberately choosing a contact edge.",
            "SHAPE_LEFT": "Only if code marks a settled, early, normal-speed, eligible return and the left edge is blocked, prefer the safe left part of that capture set.",
            "SHAPE_RIGHT": "Only if code marks a settled, early, normal-speed, eligible return and the right edge is blocked, prefer the safe right part of that capture set.",
        }
        keys = list(descriptions)
        shift = self.s1_serial % len(keys)
        first = keys[shift:] + keys[:shift]
        second = list(reversed(first))
        instruction = (
            "Classify the strategic preference for the exact derived Breakout state. "
            "Do not calculate coordinates, choose LEFT/RIGHT actions, or invent a new option. "
            "Choose SURVIVE whenever `observation.low_lives` is true, "
            "`observation.landing_urgency` is near/imminent, "
            "`observation.return_uncertainty` is wide/unknown, "
            "`observation.speed_regime` is fast/unknown, the requested edge is open, "
            "or `observation.shape_is_code_eligible` is false. "
            "The code rechecks all of these conditions before using a shaping answer. State: "
            + repr(state)
        )
        return {
            "preference_a": Choice(instruction, {key: descriptions[key] for key in first}),
            "preference_b": Choice(instruction, {key: descriptions[key] for key in second}),
        }

    def _launch_jev(self, obs: dict[str, Any], flight: dict[str, Any], event: str, frame: int) -> None:
        if not self.s1_enabled or self.s1_future is not None:
            return
        if frame < self.s1_next_frame:
            return
        if self._ball(obs) is None or bool(obs.get("serve_required", False)):
            return
        if flight.get("phase") not in ("descending", "collision_band"):
            return
        if self._remaining("system1_calls", 1.0) <= 2.0 or self._remaining("wall_s", 5.0) < 3.0:
            return
        state = self._jev_state(obs, flight, event, frame)
        questions = self._jev_questions(state)
        self.s1_request_frame = frame
        self.s1_request_epoch = self.epoch
        self.s1_request_plan = self.plan_version
        self.s1_serial += 1
        self.s1_next_frame = frame + (30 if event != "none" else 72)

        def call():
            return self.ctx.decide(state, questions, timeout_s=1.25)

        try:
            self.s1_future = self.ctx.spawn(call)
        except Exception as exc:
            self.s1_future = None
            self._log("jev_spawn_failed", error=type(exc).__name__)

    def _collect_jev(self, obs: dict[str, Any], frame: int) -> None:
        future = self.s1_future
        if future is None or not future.done():
            return
        self.s1_future = None
        try:
            answers = future.result()
            if not isinstance(answers, dict):
                return
            first = answers.get("preference_a")
            second = answers.get("preference_b")
            one = self._answer_value(first)
            two = self._answer_value(second)
            allowed = {"SURVIVE", "SHAPE_LEFT", "SHAPE_RIGHT"}
            if one not in allowed or one != two:
                self._log("jev_preference_rejected", frame=frame, reason="disagreement")
                return
            probability = min(self._answer_probability(first, one), self._answer_probability(second, two))
            if probability < 0.52:
                self._log("jev_preference_rejected", frame=frame, reason="low_probability")
                return
            if (
                frame - self.s1_request_frame > 90
                or self.s1_request_epoch != self.epoch
                or self.s1_request_plan != self.plan_version
            ):
                self._log("jev_preference_rejected", frame=frame, reason="stale")
                return
            # Revalidate cheap safety facts on the newest observation; Jev did
            # not receive a hidden reference to the game and may be wrong.
            current_flight = self._flight(obs, frame)
            sides = self._side_summary(obs)
            if one == "SHAPE_LEFT" and (
                not bool(current_flight.get("shape_allowed")) or sides["left"] == "open"
            ):
                one = "SURVIVE"
            if one == "SHAPE_RIGHT" and (
                not bool(current_flight.get("shape_allowed")) or sides["right"] == "open"
            ):
                one = "SURVIVE"
            self.s1_mode = one
            self.s1_mode_frame = frame
            self.s1_mode_epoch = self.epoch
            self.s1_mode_plan = self.plan_version
            self._log("jev_preference_accepted", frame=frame, preference=one, probability=round(probability, 3))
        except (TimeoutError, RuntimeError, BudgetExceeded, EpisodeOver) as exc:
            self._log("jev_unavailable", error=type(exc).__name__)
        except Exception as exc:
            # Arbitrary pre-flight answers and malformed provider objects are
            # advisory failures, never episode failures.
            self._log("jev_parse_ignored", error=type(exc).__name__)

    # ----------------------------- Astra bridge -----------------------------
    def _astra_prompt(self, obs: dict[str, Any], flight: dict[str, Any]) -> str:
        sides = self._side_summary(obs)
        evidence = {
            "frame": self._frame(obs),
            "lives": integer(obs.get("lives"), 0),
            "score": integer(finite(obs.get("score"), 0.0)),
            "flight_phase": flight.get("phase"),
            "landing_urgency": flight.get("urgency"),
            "return_uncertainty": flight.get("uncertainty"),
            "speed_regime": flight.get("speed"),
            "left_edge": sides["left"],
            "right_edge": sides["right"],
            "bricks_remaining": sides["bricks_remaining"],
            "wall_index": sides["wall"],
        }
        return (
            "You are the slow strategic advisor for ordinary Atari 2600 Breakout "
            "mode 0, difficulty 0. A code controller runs at 60 Hz: it estimates "
            "the descending paddle capture set and searches short LEFT/NOOP/RIGHT "
            "sequences. You must not give frame-by-frame controls or arithmetic. "
            "Choose one durable preference for the fast Jev interface. Return exactly "
            "SURVIVE, SHAPE_LEFT, or SHAPE_RIGHT. SHAPE_LEFT/RIGHT means only a "
            "safe edge-contact preference while code facts are settled, early, "
            "normal-speed, and the requested outer edge is blocked. Prefer SURVIVE "
            "with low lives, a fast/uncertain return, or no credible tunnel.\n"
            + repr(evidence)
        )

    def _launch_astra(self, obs: dict[str, Any], flight: dict[str, Any], frame: int) -> None:
        if self.astra_started or self.astra_future is not None:
            return
        if frame < 120 or self._ball(obs) is None or bool(obs.get("serve_required", False)):
            return
        if self._remaining("system2_calls", 0.0) <= 0.0 or self._remaining("wall_s", 20.0) < 10.0:
            return
        self.astra_started = True
        prompt = self._astra_prompt(obs, flight)

        def call():
            return self.ctx.think(prompt, timeout_s=55.0)

        try:
            self.astra_future = self.ctx.spawn(call)
        except Exception as exc:
            self.astra_future = None
            self._log("astra_spawn_failed", error=type(exc).__name__)

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
            match = re.match(r"^\s*(SURVIVE|SHAPE_LEFT|SHAPE_RIGHT|LEFT|RIGHT)\b", text.upper())
            if match is None:
                self._log("astra_parse_ignored", frame=frame)
                return
            token = match.group(1)
            strategy = token if token.startswith("SHAPE_") else ("SHAPE_" + token if token in ("LEFT", "RIGHT") else "SURVIVE")
            if strategy not in ("SURVIVE", "SHAPE_LEFT", "SHAPE_RIGHT"):
                strategy = "SURVIVE"
            self.astra_strategy = strategy
            self.strategy_version += 1
            self.plan_version += 1
            self.s1_mode = None
            self._log("astra_strategy_accepted", frame=frame, strategy=strategy)
        except (TimeoutError, RuntimeError, BudgetExceeded, EpisodeOver) as exc:
            self._log("astra_unavailable", error=type(exc).__name__)
        except Exception as exc:
            self._log("astra_parse_ignored", frame=frame, error=type(exc).__name__)

    # ----------------------------- real-time episode loop -------------------
    def _post(self, action: str) -> bool:
        if action not in ACTIONS:
            action = "NOOP"
        try:
            self.ctx.act(action)
            self.last_posted = action
            return True
        except (BudgetExceeded, EpisodeOver):
            return False
        except Exception as exc:
            self._log("action_post_failed", error=type(exc).__name__)
            return False

    def _run_realtime(self, game: Any) -> None:
        ctx = self.ctx
        while not ctx.done and not bool(getattr(game, "done", False)):
            obs = self._observe()
            if not obs:
                if not self._post("NOOP"):
                    return
                try:
                    ctx.wait_for_tick()
                except (BudgetExceeded, EpisodeOver):
                    return
                continue

            frame = self._frame(obs)
            if frame == self.last_frame:
                try:
                    ctx.wait_for_tick()
                except (BudgetExceeded, EpisodeOver):
                    return
                continue

            event = self._event(obs)
            self._update_motion(obs)
            self._update_events(obs, event, frame)
            flight = self._flight(obs, frame)

            # Consume completed answers before selecting this frame's action.
            self._collect_jev(obs, frame)
            self._collect_astra(frame)
            action = self._sequence_action(obs, flight, frame)
            if not self._post(action):
                return
            self.last_frame = frame
            self.previous = obs

            # Model work is launched only after the action is posted.  A slow
            # request therefore cannot steal the current native frame.
            self._launch_jev(obs, flight, event, frame)
            self._launch_astra(obs, flight, frame)
            try:
                ctx.wait_for_tick()
            except (BudgetExceeded, EpisodeOver):
                return

    def _run_turn_fallback(self, game: Any) -> None:
        # The target is tier 2, but a legal synchronous path keeps pre-flight
        # contract tests harmless if they instantiate a generic game shell.
        ctx = self.ctx
        while not ctx.done and not bool(getattr(game, "done", False)):
            obs = self._observe()
            if not obs:
                if not self._post("NOOP"):
                    return
                continue
            frame = self._frame(obs)
            event = self._event(obs)
            self._update_motion(obs)
            self._update_events(obs, event, frame)
            flight = self._flight(obs, frame)
            if not self._post(self._sequence_action(obs, flight, frame)):
                return
            self.last_frame = frame
            self.previous = obs

    def run_episode(self, game: Any) -> None:
        if bool(getattr(game, "tick_s", 0.0)):
            self._run_realtime(game)
        else:
            self._run_turn_fallback(game)


def build(ctx: Any) -> Harness:
    return Harness(ctx)
