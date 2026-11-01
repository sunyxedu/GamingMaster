"""Predictive controller for ALE Robot Pong.

The frame-critical loop is deliberately local: it projects the visible ball to the
right paddle, reflects its y trajectory at the measured court edges, and posts one
legal action on every emulator frame.  Jev and Astra are advisory only.  Jev can
select a return style for a rally and Astra can revise the macro style after
repeated evidence of a bad policy; neither model is allowed to sit on the
collision loop.
"""
from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping

from metaharness.core.contract import BudgetExceeded, ModelUnavailable
from metaharness.models.jev import Choice


# The adaptor reports visible ball centres.  These are the stable extrema in the
# supplied observation contract, rather than the outside court scanlines.
BALL_MIN_Y = 37.5
BALL_MAX_Y = 191.5
RIGHT_CONTACT_X = 140.5


class Harness:
    def __init__(self, ctx):
        self.ctx = ctx
        self.game = None
        self.spec = None
        self.names = []
        self.noop = "NOOP"
        self.up = "RIGHT"       # explicit adaptor calibration: screen y decreases
        self.down = "LEFT"      # explicit adaptor calibration: screen y increases
        self.fire = "FIRE"
        self.up_fire = "RIGHTFIRE"
        self.down_fire = "LEFTFIRE"

        self.last_obs = None
        self.last_phase = None
        self.last_frame = None
        self.last_ball_x = None
        self.last_ball_y = None
        self.last_ball_frame = None
        self.last_vx = None
        self.last_vy = None
        self.last_paddle_y = None
        self.last_action = self.noop
        self.last_score = None

        self.point_count = 0
        self.miss_streak = 0
        self.rally_id = 0
        self.force_center_next = False
        self.edge_scale = 0.20
        self.edge_sign = 1
        self.policy_style = "CENTER"
        self.policy_feature = False
        self.target_y = 113.5
        self.feature_used = False

        self.astra_future = None
        self.astra_requested = False
        self.astra_mode = None       # None, CENTER, or EDGE
        self.astra_feature = None    # None, True, or False

        self.jev_future = None
        self.jev_meta = None

    # ---------- small defensive helpers ----------
    @staticmethod
    def _number(value):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
        value = float(value)
        return value if math.isfinite(value) else None

    @classmethod
    def _integer(cls, value, default=0):
        n = cls._number(value)
        return default if n is None else int(n)

    @staticmethod
    def _json_observation(value):
        if isinstance(value, Mapping):
            return dict(value)
        if isinstance(value, str):
            try:
                decoded = json.loads(value)
            except Exception:
                return {}
            return dict(decoded) if isinstance(decoded, Mapping) else {}
        return {}

    @staticmethod
    def _clamp(value, low, high):
        return max(low, min(high, value))

    def _has(self, name):
        return name in self.names

    def _choose_name(self, preferred, fallback):
        if self._has(preferred):
            return preferred
        if self._has(fallback):
            return fallback
        return self.noop

    def _combine_fire(self, direction):
        if direction == self.up and self._has(self.up_fire):
            return self.up_fire
        if direction == self.down and self._has(self.down_fire):
            return self.down_fire
        if self._has(self.fire):
            return self.fire
        return direction

    def _is_done(self):
        try:
            if bool(self.ctx.done):
                return True
        except Exception:
            return True
        try:
            return bool(getattr(self.game, "done", False))
        except Exception:
            return True

    def _observe(self):
        try:
            return self._json_observation(self.game.observe())
        except Exception:
            return {}

    # ---------- ROM geometry and local state ----------
    def _approach(self, obs):
        value = obs.get("ball_approaching_right")
        if isinstance(value, bool):
            return value
        sign = self._number(obs.get("ball_vx_sign"))
        if sign is None:
            raw = self._number(obs.get("ball_vx"))
            sign = 0 if raw is None else (1 if raw > 0 else -1 if raw < 0 else 0)
        if sign:
            return sign > 0
        return None

    def _velocity_sign(self, obs, axis):
        key = "ball_vx_sign" if axis == "x" else "ball_vy_sign"
        raw_key = "ball_vx" if axis == "x" else "ball_vy"
        sign = self._number(obs.get(key))
        if sign is None:
            raw = self._number(obs.get(raw_key))
            if raw is not None:
                sign = 1 if raw > 0 else -1 if raw < 0 else 0
        if sign is not None and sign != 0:
            return 1 if sign > 0 else -1
        return 0 if sign == 0 else None

    def _velocity_magnitude(self, obs, axis, default=1.0):
        raw_key = "ball_vx" if axis == "x" else "ball_vy"
        raw = self._number(obs.get(raw_key))
        if raw is not None and abs(raw) >= 0.25:
            return self._clamp(abs(raw), 0.5, 3.0)
        if axis == "x" and self.last_ball_x is not None:
            x = self._number(obs.get("ball_x"))
            frame = self._number(obs.get("frame"))
            if x is not None and frame is not None and self.last_ball_frame is not None:
                dt = frame - self.last_ball_frame
                if dt > 0:
                    measured = abs(x - self.last_ball_x) / dt
                    if measured >= 0.25:
                        return self._clamp(measured, 0.5, 3.0)
        return default

    @staticmethod
    def _reflect(value, low=BALL_MIN_Y, high=BALL_MAX_Y):
        span = high - low
        if span <= 0:
            return (low + high) / 2.0
        phase = (value - low) % (2.0 * span)
        if phase <= span:
            return low + phase
        return high - (phase - span)

    def _current_vy(self, obs):
        sign = self._velocity_sign(obs, "y")
        if sign is not None:
            return sign
        # A velocity is deliberately carried only through a short visible gap.
        # Long gaps and serve frames are handled by the centre fallback.
        frame = self._number(obs.get("frame"))
        if (frame is not None and self.last_ball_frame is not None and
                frame - self.last_ball_frame <= 5 and self.last_vy is not None):
            return self.last_vy
        return None

    def _base_intercept(self, obs):
        """Return (projected y, contact eta) or (None, None).

        This is arithmetic performed by the harness, not a model question.  The
        opponent's return angle is not extrapolated: after its left-side contact,
        the next visible vx/vy is used, which avoids baking in a wrong opponent
        model.
        """
        if str(obs.get("phase", "")) != "rally":
            return None, None
        if obs.get("ball_visible") is False:
            return None, None
        x = self._number(obs.get("ball_x"))
        y = self._number(obs.get("ball_y"))
        if x is None or y is None or self._approach(obs) is not True:
            return None, None
        vx = self._velocity_magnitude(obs, "x", 1.0)
        eta = max(0.0, (RIGHT_CONTACT_X - x) / max(vx, 0.5))
        vy_sign = self._current_vy(obs)
        if vy_sign is None:
            return self._clamp(y, BALL_MIN_Y, BALL_MAX_Y), eta
        vy = self._velocity_magnitude(obs, "y", 1.0)
        projected = self._reflect(y + vy_sign * vy * eta)
        return projected, eta

    def _feature_legal(self, obs):
        game_number = self._integer(obs.get("cartridge_game"), -1)
        variation = str(obs.get("variation", ""))
        return game_number in (1, 2) or "Robot Pong" in variation

    def _style_offset(self, obs):
        style = self.policy_style
        if self.astra_mode == "CENTER":
            style = "CENTER"
        if style == "CENTER":
            return 0.0
        height = self._number(obs.get("right_paddle_nominal_height_px"))
        if height is None:
            height = self._number(obs.get("right_paddle_height_px")) or 16.0
        # Small edge bias preserves overlap on half-size paddles while changing
        # the return point enough for Speed/WHAMMY or the robot's lag to matter.
        magnitude = self._clamp(height * self.edge_scale, 1.0, 4.5)
        if style == "EDGE_DOWN":
            return magnitude
        return -magnitude

    def _target_for(self, obs):
        phase = str(obs.get("phase", ""))
        if phase == "rally":
            base, _ = self._base_intercept(obs)
            if base is not None:
                target = base + self._style_offset(obs)
                self.target_y = self._clamp(target, BALL_MIN_Y, BALL_MAX_Y)
                return self.target_y
            # On the outbound leg, cover the middle.  Once the opponent returns
            # the ball, vx becomes positive and the exact intercept takes over.
            if obs.get("ball_visible") is not False:
                self.target_y = 113.5
                return self.target_y
        elif phase == "rally_ball_occluded":
            # Keep the most recently computed target briefly; changing to the
            # centre during a collision occlusion is worse than holding coverage.
            return self.target_y

        # phase=serve_delay is authoritative even if a stale ball sprite remains.
        self.target_y = 113.5
        return self.target_y

    def _paddle_action(self, obs, target):
        paddle = self._number(obs.get("right_paddle_y"))
        if paddle is None:
            # Never let an occlusion extend a one-frame FIRE feature or an old
            # movement command; NOOP is the safe hardware-filtered fallback.
            return self.noop

        measured_dy = 0.0
        frame = self._number(obs.get("frame"))
        if self.last_paddle_y is not None and self.last_frame is not None and frame is not None:
            dt = max(1.0, frame - self.last_frame)
            measured_dy = self._clamp((paddle - self.last_paddle_y) / dt, -8.0, 8.0)

        # Predict one settling frame.  This is a small brake against the observed
        # hardware filter: simply using target-paddle oscillates at the contact.
        predicted = paddle + measured_dy
        error = target - predicted
        deadband = 2.0 if str(obs.get("phase", "")) == "rally" else 3.0
        if error > deadband:
            return self.down
        if error < -deadband:
            return self.up
        return self.noop

    def _contact_window(self, obs, eta):
        if eta is None or self._approach(obs) is not True:
            return False
        x = self._number(obs.get("ball_x"))
        if x is None:
            return False
        # Use both x and projected time so a transient velocity estimate cannot
        # fire on a ball that is already travelling away.
        return eta <= 4.0 and RIGHT_CONTACT_X - 5.0 <= x <= RIGHT_CONTACT_X + 4.0

    def _control_action(self, obs):
        target = self._target_for(obs)
        direction = self._paddle_action(obs, target)
        action = direction

        base, eta = self._base_intercept(obs)
        feature_ok = self._feature_legal(obs)
        if (feature_ok and self.policy_feature and not self.feature_used and
                self.astra_feature is not False and self._contact_window(obs, eta)):
            action = self._combine_fire(direction)
            self.feature_used = True
        # A point/side change resets the one-frame button latch.  Do it after the
        # decision so the feature is never held throughout a serve.
        x = self._number(obs.get("ball_x"))
        if (self._approach(obs) is False and x is not None and
                x < RIGHT_CONTACT_X - 8.0):
            self.feature_used = False
        return action

    def _update_ball_history(self, obs):
        phase = str(obs.get("phase", ""))
        x = self._number(obs.get("ball_x"))
        y = self._number(obs.get("ball_y"))
        frame = self._number(obs.get("frame"))
        if phase not in ("rally", "rally_ball_occluded") or x is None or y is None:
            return
        vx = self._velocity_sign(obs, "x")
        vy = self._velocity_sign(obs, "y")
        if vx is not None:
            self.last_vx = vx
        if vy is not None:
            self.last_vy = vy
        self.last_ball_x = x
        self.last_ball_y = y
        if frame is not None:
            self.last_ball_frame = frame

    # ---------- macro policy and model bridges ----------
    def _set_default_rally_policy(self):
        if self.force_center_next:
            self.policy_style = "CENTER"
            self.force_center_next = False
        elif self.point_count == 0:
            self.policy_style = "CENTER"
        elif self.astra_mode == "CENTER":
            self.policy_style = "CENTER"
        else:
            self.policy_style = "EDGE_UP" if self.edge_sign < 0 else "EDGE_DOWN"
        # Feature use is opt-in: the local fallback never presses FIRE merely
        # because a rally happens to use an edge style.  Jev or an explicit
        # Astra YES can authorize the one-frame contact press.
        self.policy_feature = (
            self.policy_style != "CENTER" and
            self._feature_legal(self.last_obs or {}) and
            self.astra_feature is True
        )
        self.feature_used = False

    def _handle_score_and_phase(self, obs):
        own = self._integer(obs.get("own_points"), 0)
        opponent = self._integer(obs.get("opponent_points"), 0)
        score = (own, opponent)
        if self.last_score is None:
            self.last_score = score
        elif score != self.last_score:
            old_own, old_opponent = self.last_score
            own_delta = own - old_own
            opponent_delta = opponent - old_opponent
            self.point_count += max(0, own_delta) + max(0, opponent_delta)
            if opponent_delta > 0:
                self.miss_streak += opponent_delta
                self.edge_scale = max(0.10, self.edge_scale - 0.025)
                # One conservative rally after an edge-related miss, but do not
                # permanently retreat to the idle policy.
                self.force_center_next = True
            if own_delta > 0:
                self.miss_streak = 0
                self.edge_scale = min(0.27, self.edge_scale + 0.02)
                self.edge_sign *= -1
            self.feature_used = False
            try:
                self.ctx.log("point", own_points=own, opponent_points=opponent,
                             edge_scale=round(self.edge_scale, 3))
            except Exception:
                pass
            self.last_score = score

        phase = str(obs.get("phase", ""))
        if phase == "rally" and self.last_phase != "rally":
            self.rally_id += 1
            self._set_default_rally_policy()
            self._start_jev(obs)
        self.last_phase = phase

        # Astra is an escalation, not a startup stall.  It is requested only
        # after three point outcomes provide evidence that the local policy needs
        # a macro change; the frame loop continues while it reasons.  Do not
        # create a new advisory task from a terminal observation.
        if (opponent >= 3 and not self.astra_requested and
                str(obs.get("phase", "")) != "finished" and not self._is_done()):
            self._start_astra(obs)

    @staticmethod
    def _choice_probability(answer):
        value = getattr(answer, "value", None)
        probabilities = getattr(answer, "probabilities", None)
        if not isinstance(probabilities, Mapping):
            return str(value), 0.0
        try:
            probability = float(probabilities.get(value, 0.0))
        except Exception:
            probability = 0.0
        return str(value), probability

    def _start_jev(self, obs):
        if self.jev_future is not None or getattr(self.ctx, "system1", None) is None:
            return
        approach = self._approach(obs)
        base, eta = self._base_intercept(obs)
        paddle = self._number(obs.get("right_paddle_y"))
        if base is None:
            band = "unknown"
            relation = "unknown"
        else:
            band = "upper" if base < 88 else "lower" if base > 145 else "middle"
            if paddle is None:
                relation = "unknown"
            elif paddle < base - 5:
                relation = "above_target"
            elif paddle > base + 5:
                relation = "below_target"
            else:
                relation = "aligned"
        eta_bucket = "unknown" if eta is None else (
            "contact" if eta <= 8 else "near" if eta <= 30 else "far")
        facts = {
            "phase": str(obs.get("phase", "unknown")),
            "approach": ("toward_right" if approach is True else
                         "away_from_right" if approach is False else "unknown"),
            "contact_eta": eta_bucket,
            "projected_contact_band": band,
            "paddle_relation": relation,
            "right_paddle_size": str(obs.get("right_paddle_size", "unknown")),
            "feature_legal": bool(self._feature_legal(obs)),
            "score_state": ("ahead" if self._integer(obs.get("own_points")) >
                             self._integer(obs.get("opponent_points")) else
                             "behind" if self._integer(obs.get("own_points")) <
                             self._integer(obs.get("opponent_points")) else "tied"),
            "local_default": self.policy_style,
        }
        state = {
            "facts": facts,
            "controller_rule": (
                "Code has already projected the ball and will post a legal one-frame "
                "action. Choose only a rally-level return style; do not calculate "
                "coordinates, scores, or a frame action."
            ),
            "candidate_meanings": {
                "CENTER": "aim the paddle centre at the projected contact; safest overlap",
                "EDGE_UP": "aim the paddle centre slightly above the ball (smaller screen y); keep overlap",
                "EDGE_DOWN": "aim the paddle centre slightly below the ball (larger screen y); keep overlap",
            },
            "rally_id": int(self.rally_id),
        }
        questions = {
            "style_a": Choice(
                "Using only `facts`, choose the return style for this rally. Prefer CENTER "
                "when the contact or velocity is uncertain; use an edge only when the "
                "facts support a measured attacking return. Do not do arithmetic.",
                {
                    "EDGE_UP": "Move the paddle centre a few pixels above the projected ball (smaller screen y); preserve overlap.",
                    "CENTER": "A central defensive contact with maximum margin for error.",
                    "EDGE_DOWN": "Move the paddle centre a few pixels below the projected ball (larger screen y); preserve overlap.",
                },
            ),
            "style_b": Choice(
                "Independently choose one named style for this rally from `facts`. This is "
                "a recognition judgment, not a calculation; uncertain evidence means CENTER.",
                {
                    "EDGE_DOWN": "A small lower-edge attacking contact is supported.",
                    "CENTER": "A central defensive contact is the appropriate fallback.",
                    "EDGE_UP": "A small upper-edge attacking contact is supported.",
                },
            ),
            "feature": Choice(
                "Should the controller press the variation's button for one frame at a "
                "verified right-paddle contact? Choose NO_FEATURE if it is illegal, the "
                "phase is not rally, or the contact is uncertain.",
                {
                    "USE_AT_CONTACT": "Use only in a confirmed Robot Pong rally contact window.",
                    "NO_FEATURE": "Do not press the button; automatic serves do not require FIRE.",
                },
            ),
        }
        meta = (self.rally_id, int(self._integer(obs.get("frame"), 0)))

        def ask():
            return self.ctx.decide(state, questions, timeout_s=0.9)

        try:
            self.jev_future = self.ctx.spawn(ask)
            self.jev_meta = meta
        except Exception:
            self.jev_future = None
            self.jev_meta = None

    def _poll_jev(self, obs):
        future = self.jev_future
        if future is None or not future.done():
            return
        self.jev_future = None
        meta = self.jev_meta
        self.jev_meta = None
        try:
            answers = future.result()
        except (TimeoutError, RuntimeError, BudgetExceeded, ModelUnavailable, Exception):
            return
        if not isinstance(answers, Mapping) or meta is None:
            return
        if meta[0] != self.rally_id or str(obs.get("phase", "")) != "rally":
            return

        a = answers.get("style_a")
        b = answers.get("style_b")
        va, pa = self._choice_probability(a) if a is not None else ("", 0.0)
        vb, pb = self._choice_probability(b) if b is not None else ("", 0.0)
        allowed = {"CENTER", "EDGE_UP", "EDGE_DOWN"}
        if va == vb and va in allowed and pa >= 0.58 and pb >= 0.58:
            if not (self.astra_mode == "CENTER" and va != "CENTER"):
                self.policy_style = va
                self.policy_feature = va != "CENTER" and self._feature_legal(obs)
                try:
                    self.ctx.log("jev_style", style=va, rally_id=self.rally_id,
                                 probability=round(min(pa, pb), 3))
                except Exception:
                    pass

        feature_answer = answers.get("feature")
        vf, pf = self._choice_probability(feature_answer) if feature_answer is not None else ("", 0.0)
        if vf == "USE_AT_CONTACT" and pf >= 0.72 and self._feature_legal(obs):
            self.policy_feature = self.policy_style != "CENTER" and self.astra_feature is not False
        elif vf == "NO_FEATURE" and pf >= 0.72:
            self.policy_feature = False

    def _start_astra(self, obs):
        if self.astra_future is not None or self.astra_requested:
            return
        if getattr(self.ctx, "system2", None) is None:
            self.astra_requested = True
            return
        try:
            remaining = self.ctx.budget.remaining()
            if int(remaining.get("system2_calls", 0)) <= 0:
                self.astra_requested = True
                return
        except Exception:
            self.astra_requested = True
            return
        self.astra_requested = True
        try:
            game_description = str(self.game.describe())
        except Exception:
            game_description = "Atari 2600 Robot Pong; right paddle is controlled."
        prompt = (
            "You are a slow macro adviser, not a frame controller. The local controller "
            "already computes ball reflection, paddle timing, score bookkeeping, and "
            "legal actions. Diagnose whether its rally policy should be conservative or "
            "attacking from the public game rules and this evidence. Reply with exactly "
            "four lines and no other text:\n"
            "MODE: CENTER or EDGE\n"
            "EDGE: UP or DOWN\n"
            "FEATURE: YES or NO\n"
            "MARGIN: SMALL or MEDIUM\n\n"
            f"Game:\n{game_description}\n"
            f"Current verified observation:\n{json.dumps(obs, sort_keys=True, default=str)}\n"
            f"Local evidence: own_points={self._integer(obs.get('own_points'))}, "
            f"opponent_points={self._integer(obs.get('opponent_points'))}, "
            f"miss_streak={self.miss_streak}, edge_scale={self.edge_scale:.3f}.\n"
            "FIRE is never a serve requirement; FEATURE refers only to a one-frame "
            "variation feature at a verified paddle contact."
        )

        def ask():
            return self.ctx.think(prompt, timeout_s=55.0)

        try:
            self.astra_future = self.ctx.spawn(ask)
        except Exception:
            self.astra_future = None

    def _poll_astra(self):
        future = self.astra_future
        if future is None or not future.done():
            return
        self.astra_future = None
        try:
            result = future.result()
            text = str(getattr(result, "text", ""))
        except (TimeoutError, RuntimeError, BudgetExceeded, ModelUnavailable, Exception):
            return
        mode = re.search(r"\bMODE\s*:\s*(CENTER|EDGE)\b", text.upper())
        edge = re.search(r"\bEDGE\s*:\s*(UP|DOWN)\b", text.upper())
        feature = re.search(r"\bFEATURE\s*:\s*(YES|NO)\b", text.upper())
        margin = re.search(r"\bMARGIN\s*:\s*(SMALL|MEDIUM)\b", text.upper())
        if mode:
            self.astra_mode = mode.group(1)
            if self.astra_mode == "CENTER":
                self.policy_style = "CENTER"
                self.policy_feature = False
        if edge:
            self.edge_sign = -1 if edge.group(1) == "UP" else 1
        if feature:
            self.astra_feature = feature.group(1) == "YES"
            self.policy_feature = (
                self.policy_style != "CENTER" and self.astra_feature
            )
        if margin and margin.group(1) == "SMALL":
            self.edge_scale = min(self.edge_scale, 0.15)
        if mode or edge or feature or margin:
            try:
                self.ctx.log("astra_policy", mode=self.astra_mode,
                             edge_sign=self.edge_sign, feature=self.astra_feature,
                             edge_scale=round(self.edge_scale, 3))
            except Exception:
                pass

    # ---------- episode loop ----------
    def _reset_local_state(self, game):
        self.__init__(self.ctx)
        self.game = game
        self.spec = getattr(game, "action_spec", None)
        raw_names = getattr(self.spec, "names", []) if self.spec is not None else []
        if isinstance(raw_names, Mapping):
            self.names = [str(k) for k in raw_names.keys()]
        else:
            try:
                self.names = list(raw_names)
            except Exception:
                self.names = []
        if not self.names:
            self.names = ["NOOP", "FIRE", "RIGHT", "LEFT", "RIGHTFIRE", "LEFTFIRE"]
        declared_noop = getattr(self.spec, "noop", None) if self.spec is not None else None
        self.noop = declared_noop if declared_noop in self.names else self._choose_name("NOOP", self.names[0])
        self.up = self._choose_name("RIGHT", self.noop)
        self.down = self._choose_name("LEFT", self.noop)
        self.fire = self._choose_name("FIRE", self.noop)
        self.up_fire = self._choose_name("RIGHTFIRE", self.up)
        self.down_fire = self._choose_name("LEFTFIRE", self.down)
        self.last_action = self.noop

    def run_episode(self, game) -> None:
        self._reset_local_state(game)
        if not getattr(game, "tick_s", None):
            return

        obs = self._observe()
        self.last_obs = obs
        self.last_phase = None
        self._handle_score_and_phase(obs)
        self._update_ball_history(obs)
        action = self._control_action(obs)
        try:
            if not self._is_done():
                self.ctx.act(action)
                self.last_action = action
        except (BudgetExceeded, ModelUnavailable):
            return

        while not self._is_done():
            try:
                self.ctx.wait_for_tick()
            except (BudgetExceeded, ModelUnavailable):
                return
            if self._is_done():
                return
            self._poll_astra()
            self._poll_jev(obs)
            obs = self._observe()
            self._handle_score_and_phase(obs)
            self._update_ball_history(obs)
            self.last_obs = obs
            action = self._control_action(obs)
            try:
                if not self._is_done():
                    self.ctx.act(action)
                    self.last_action = action
            except (BudgetExceeded, ModelUnavailable):
                return
            self.last_frame = self._number(obs.get("frame"))
            self.last_paddle_y = self._number(obs.get("right_paddle_y"))

        # Consume completed advisory futures without waiting for a slow model.
        self._poll_astra()
        self._poll_jev(obs)


def build(ctx):
    return Harness(ctx)
