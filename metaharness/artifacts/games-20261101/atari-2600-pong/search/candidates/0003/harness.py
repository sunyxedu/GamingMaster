"""Jev-only Robot Pong controller with a frame-rate local intercept loop.

The parent asked Jev for a raw action synchronously.  A Jev answer takes several
emulator frames, so this harness keeps Jev as a slow return-feature judge and
lets deterministic code own movement, prediction, and timing.
"""
from __future__ import annotations

import math

from metaharness.core.contract import BudgetExceeded, EpisodeOver, ModelUnavailable
from metaharness.models.jev import Choice


# These are screen coordinates from the public adaptor contract.  They are
# deliberately broad enough to tolerate a clipped sprite at a wall.
COURT_TOP = 34.0
COURT_BOTTOM = 193.0
COURT_CENTER = (COURT_TOP + COURT_BOTTOM) / 2.0
RIGHT_CONTACT_X = 140.0


class Harness:
    def __init__(self, ctx):
        self.ctx = ctx
        self._reset_controller()

    def _reset_controller(self):
        self._phase = None
        self._tick = 0
        self._last_ball = None       # (x, y, vx, vy, tick)
        self._vx = None
        self._vy = None
        self._last_paddle = None
        self._feature_hint = False
        self._feature_confidence = 0.0
        self._feature_used = False
        self._future = None
        self._next_query_tick = 0

    @staticmethod
    def _number(value, default=None):
        if isinstance(value, bool):
            return default
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            return float(value)
        return default

    @staticmethod
    def _sign(value):
        if isinstance(value, bool):
            return None
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            if value > 0:
                return 1
            if value < 0:
                return -1
            return 0
        return None

    @staticmethod
    def _reflect(value, lo=COURT_TOP, hi=COURT_BOTTOM):
        """Reflect a point in the horizontal court walls."""
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            return (lo + hi) / 2.0
        span = hi - lo
        if span <= 0:
            return lo
        z = (float(value) - lo) % (2.0 * span)
        if z <= span:
            return lo + z
        return hi - (z - span)

    @staticmethod
    def _clamp(value, lo, hi):
        return max(lo, min(hi, value))

    def _ingest(self, obs):
        """Update only code-owned observations; never infer from missing pixels."""
        if not isinstance(obs, dict):
            obs = {}
        tick = self._number(obs.get("ticks"), self._tick)
        self._tick = int(tick if tick is not None else self._tick)
        phase = obs.get("phase")
        if not isinstance(phase, str):
            phase = "unknown"

        # A point/serve can leave the last ball sprite on screen.  The phase,
        # not that stale sprite, invalidates the old rally trajectory.
        old_phase = self._phase
        if phase == "serve_delay" or phase == "finished":
            self._last_ball = None
            self._vx = None
            self._vy = None
            self._feature_used = False
            self._feature_hint = False
        elif old_phase not in ("rally", "rally_ball_occluded"):
            # First visible frame of a new rally must not derive velocity from
            # the previous serve or point.
            self._last_ball = None
            self._vx = None
            self._vy = None
            self._feature_used = False

        if phase in ("rally", "rally_ball_occluded"):
            x = self._number(obs.get("ball_x"))
            y = self._number(obs.get("ball_y"))
            visible = bool(obs.get("ball_visible")) and x is not None and y is not None
            if visible:
                vx = self._sign(obs.get("ball_vx_sign"))
                vy = self._sign(obs.get("ball_vy_sign"))
                # The adaptor intentionally reports unknown velocity at first
                # sight and after occlusion.  Position differences are a safe
                # fallback only within this rally.
                if vx is None and self._last_ball is not None:
                    dx = x - self._last_ball[0]
                    if abs(dx) <= 4.0 and abs(dx) > 0.1:
                        vx = 1 if dx > 0 else -1
                if vy is None and self._last_ball is not None:
                    dy = y - self._last_ball[1]
                    if abs(dy) <= 4.0 and abs(dy) > 0.1:
                        vy = 1 if dy > 0 else -1
                if vx is not None:
                    self._vx = vx
                if vy is not None:
                    self._vy = vy
                self._last_ball = (x, y, self._vx, self._vy, self._tick)

        paddle = self._number(obs.get("right_paddle_y"))
        if paddle is not None:
            self._last_paddle = paddle
        self._phase = phase

    def _ball_for_control(self, obs):
        """Return a current/shortly extrapolated ball tuple, or None."""
        if not isinstance(obs, dict):
            return None
        phase = obs.get("phase")
        if phase not in ("rally", "rally_ball_occluded"):
            return None

        x = self._number(obs.get("ball_x"))
        y = self._number(obs.get("ball_y"))
        visible = bool(obs.get("ball_visible")) and x is not None and y is not None
        vx = self._sign(obs.get("ball_vx_sign"))
        vy = self._sign(obs.get("ball_vy_sign"))
        if vx is None:
            vx = self._vx
        if vy is None:
            vy = self._vy

        if visible:
            return (x, y, vx, vy)

        # Occlusion is normally short.  Do not let an ancient trajectory drive
        # the paddle indefinitely; return to central coverage instead.
        if self._last_ball is None:
            return None
        age = self._tick - self._last_ball[4]
        if age < 0 or age > 14:
            return None
        lx, ly, lvx, lvy, _ = self._last_ball
        if vx is None:
            vx = lvx
        if vy is None:
            vy = lvy
        if vx is None or vy is None:
            return (lx, ly, vx, vy)
        return (lx + vx * age, self._reflect(ly + vy * age), vx, vy)

    def _target_y(self, obs):
        """Compute the right-paddle intercept; this is never delegated to Jev."""
        target = COURT_CENTER
        ball = self._ball_for_control(obs)
        if ball is None:
            return target, None, False
        x, y, vx, vy = ball
        approaching = vx is not None and vx > 0
        if not approaching:
            return target, None, False

        # The adaptor exposes velocity signs, and this ROM's observed ball
        # advances one horizontal pixel per frame.  A sign is therefore enough
        # for a reflected intercept estimate.
        distance = RIGHT_CONTACT_X - x
        time_to_contact = max(0.0, distance) if distance >= 0.0 else 0.0
        if vy is None:
            projected = y
        else:
            projected = self._reflect(y + vy * time_to_contact)
        projected = self._clamp(projected, COURT_TOP, COURT_BOTTOM)
        contact_window = time_to_contact <= 4.0
        return projected, time_to_contact, contact_window

    def _legal_names(self, game):
        spec = game.action_spec
        names = list(getattr(spec, "names", ()) or ())
        by_upper = {str(n).upper(): n for n in names}
        noop = getattr(spec, "noop", None)
        if noop not in names:
            noop = by_upper.get("NOOP", names[0] if names else "NOOP")
        return by_upper, noop

    def _movement_action(self, game, direction, fire=False):
        by_upper, noop = self._legal_names(game)
        if direction == "UP":
            base = by_upper.get("RIGHT", noop)
            fired = by_upper.get("RIGHTFIRE")
        elif direction == "DOWN":
            base = by_upper.get("LEFT", noop)
            fired = by_upper.get("LEFTFIRE")
        else:
            base = noop
            fired = by_upper.get("FIRE")
        if fire and fired is not None:
            return fired
        return base

    def _controller_action(self, game, obs):
        target, time_to_contact, contact_window = self._target_y(obs)
        current = self._number(obs.get("right_paddle_y"))
        if current is None:
            current = self._last_paddle
        if current is None:
            current = COURT_CENTER

        # A visible paddle is clipped at the wall.  Use nominal height for the
        # collision margin rather than a transient clipped height.
        height = self._number(obs.get("right_paddle_nominal_height_px"), 16.0)
        deadband = max(3.5, min(7.0, height * 0.36))
        delta = target - current
        if delta < -deadband:
            direction = "UP"
        elif delta > deadband:
            direction = "DOWN"
        else:
            direction = "HOLD"

        # FIRE is optional in Robot Pong and must be a one-frame contact pulse.
        # A model hint is never allowed to affect movement or fire away from a
        # contact, or when the paddle is visibly too far from the target.
        aligned_for_feature = abs(delta) <= max(deadband, height * 0.50)
        use_feature = (
            contact_window
            and time_to_contact is not None
            and aligned_for_feature
            and self._feature_hint
            and self._feature_confidence >= 0.60
            and not self._feature_used
            and obs.get("variation") in ("Robot Pong 1", "Robot Pong 2")
        )
        if use_feature:
            self._feature_used = True
        return self._movement_action(game, direction, fire=use_feature)

    def _question_state(self, obs):
        target, time_to_contact, contact_window = self._target_y(obs)
        current = self._number(obs.get("right_paddle_y"))
        if current is None:
            current = self._last_paddle
        if current is None:
            current = COURT_CENTER
        relation = "ALIGNED"
        if target < current - 6.0:
            relation = "ABOVE"
        elif target > current + 6.0:
            relation = "BELOW"
        if time_to_contact is None:
            contact_bucket = "UNKNOWN"
        elif time_to_contact <= 4:
            contact_bucket = "IMMINENT"
        elif time_to_contact <= 18:
            contact_bucket = "NEAR"
        else:
            contact_bucket = "FAR"
        ball = self._ball_for_control(obs)
        vx = ball[2] if ball is not None else None
        approach = "TOWARD_RIGHT" if vx == 1 else "AWAY_OR_UNKNOWN"
        height = self._number(obs.get("right_paddle_nominal_height_px"), 16.0)
        return {
            "game": "Atari 2600 Robot Pong; controlled paddle is the right paddle",
            "phase": obs.get("phase") if isinstance(obs.get("phase"), str) else "unknown",
            "variation": obs.get("variation") if isinstance(obs.get("variation"), str) else "unknown",
            "right_paddle": {
                "y": current,
                "nominal_height_px": height,
                "target_relation": relation,
            },
            "ball": {
                "visible": bool(obs.get("ball_visible")),
                "approach": approach,
                "time_to_right_contact_bucket": contact_bucket,
            },
            "controller": {
                "target_y": target,
                "contact_window": bool(contact_window),
                "feature_legal_in_variation": obs.get("variation") in ("Robot Pong 1", "Robot Pong 2"),
                "feature_already_used_this_rally": bool(self._feature_used),
            },
            "score": {
                "own_points": self._number(obs.get("own_points"), 0.0),
                "opponent_points": self._number(obs.get("opponent_points"), 0.0),
            },
        }

    @staticmethod
    def _answer_value(answer):
        if isinstance(answer, dict):
            value = answer.get("value", answer.get("choice"))
            probabilities = answer.get("probabilities", {})
        else:
            value = getattr(answer, "value", None)
            if value is None:
                value = getattr(answer, "choice", None)
            probabilities = getattr(answer, "probabilities", {})
        if not isinstance(value, str):
            return None, 0.0
        if not isinstance(probabilities, dict):
            return value, 0.0
        probability = probabilities.get(value, 0.0)
        if not isinstance(probability, (int, float)) or not math.isfinite(float(probability)):
            probability = 0.0
        return value, float(probability)

    def _ask_jev(self, state, questions):
        # This function is intentionally finite: ctx.spawn may cancel it when
        # the match ends, and it must never own a game loop.
        try:
            return self.ctx.decide(state, questions, timeout_s=1.0)
        except (TimeoutError, RuntimeError, BudgetExceeded, EpisodeOver, ModelUnavailable):
            return None
        except Exception:
            # Garbage stand-in answers or provider shape errors must not turn a
            # late advisory call into a game error.
            return None

    def _poll_jev(self):
        future = self._future
        if future is None:
            return
        try:
            if not future.done():
                return
            result = future.result()
        except Exception:
            result = None
        self._future = None
        if not isinstance(result, dict):
            return
        answer = result.get("feature")
        value, probability = self._answer_value(answer)
        if value == "USE_FEATURE" and probability >= 0.60:
            self._feature_hint = True
            self._feature_confidence = probability
        elif value == "NO_FEATURE":
            self._feature_hint = False
            self._feature_confidence = probability
        try:
            self.ctx.log(
                "jev_feature_judgment",
                value=value,
                probability=probability,
                tick=self._tick,
            )
        except Exception:
            pass

    def _maybe_start_jev(self, obs):
        if getattr(self.ctx, "system1", True) is None:
            return
        if self._future is not None or self.ctx.done:
            return
        if self._tick < self._next_query_tick:
            return
        state = self._question_state(obs)
        questions = {
            "feature": Choice(
                "Should the controller press FIRE for one contact pulse on the right paddle? "
                "Choose USE_FEATURE only when `controller.contact_window` is true, the ball "
                "is approaching the right paddle, the variation is Robot Pong 1 or 2, and "
                "the predicted paddle alignment is safe. Otherwise choose NO_FEATURE. "
                "FIRE is never needed to start an automatic serve.",
                {
                    "NO_FEATURE": "Do not press FIRE; use the ordinary movement or NOOP action.",
                    "USE_FEATURE": "Press FIRE exactly once at the imminent right-paddle contact; never use it during a serve or away from contact.",
                },
            )
        }
        try:
            self._future = self.ctx.spawn(
                lambda state=state, questions=questions: self._ask_jev(state, questions)
            )
        except Exception:
            self._future = None
        # One outstanding Jev request is enough.  It avoids a request storm and
        # leaves the frame-rate controller independent of model latency.
        self._next_query_tick = self._tick + 18

    def run_episode(self, game) -> None:
        self._reset_controller()
        ctx = self.ctx
        if ctx.done:
            return

        while not ctx.done and not getattr(game, "done", False):
            obs = game.observe()
            if not isinstance(obs, dict):
                obs = {}
            self._ingest(obs)
            self._poll_jev()
            self._maybe_start_jev(obs)
            action = self._controller_action(game, obs)
            try:
                # Posting before waiting makes this tick fresh.  The local
                # controller therefore continues while an advisory Jev call is
                # in flight.
                ctx.act(action)
                ctx.wait_for_tick()
            except (BudgetExceeded, EpisodeOver, ModelUnavailable):
                return


def build(ctx):
    return Harness(ctx)
