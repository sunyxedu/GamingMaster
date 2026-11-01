"""Candidate 0005: predictive frame controller with asynchronous supervision.

The paddle loop never waits for a model.  It projects the visible ball to the
right paddle, reflects the projection at the court walls, and posts one action
per emulator frame.  Astra and Jev are supervisors: their results can change a
safe policy setting or resolve an uncertain frame, but a late/garbage result
cannot block or invalidate the local controller.
"""
from __future__ import annotations

import json
import math
import re
from collections import deque

from metaharness.core.contract import BudgetExceeded, EpisodeOver, ModelUnavailable
from metaharness.models.jev import Choice


# The adaptor's public geometry.  These are visible-pixel centers, not RAM
# addresses, and are deliberately kept here rather than guessed by a model.
BALL_TOP = 37.5
BALL_BOTTOM = 191.5
PADDLE_TOP = 37.5
PADDLE_BOTTOM = 191.5
COURT_CENTER = (PADDLE_TOP + PADDLE_BOTTOM) / 2.0
RIGHT_CONTACT_X = 140.5


def _number(value, default=None):
    """Return a finite float without allowing malformed observations to escape."""
    if isinstance(value, bool):
        return default
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return default
    return result if math.isfinite(result) else default


def _integer(value, default=None):
    number = _number(value, None)
    if number is None:
        return default
    return int(number)


def _sign(value):
    number = _number(value, None)
    if number is None or number == 0:
        return 0
    return 1 if number > 0 else -1


def _bucket(value, cuts, labels):
    number = _number(value, None)
    if number is None:
        return labels[-1]
    for cut, label in zip(cuts, labels):
        if number < cut:
            return label
    return labels[-1]


def _clamp(value, low, high):
    return max(low, min(high, value))


def _reflect_y(y, vy, frames):
    """Project a y coordinate with mirror reflections at the visible walls."""
    y = _number(y, COURT_CENTER)
    vy = _number(vy, 0.0)
    frames = max(0.0, _number(frames, 0.0))
    span = BALL_BOTTOM - BALL_TOP
    if span <= 0:
        return y
    phase = (y - BALL_TOP + vy * frames) % (2.0 * span)
    if phase <= span:
        return BALL_TOP + phase
    return BALL_BOTTOM - (phase - span)


def _json_value(value):
    """Make a compact state safe for ctx.decide's strict JSON contract."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    number = _number(value, None)
    return number


def _safe_text(value):
    try:
        return str(value)
    except Exception:
        return ""


def _extract_json(text):
    """Parse an optional JSON object from Astra's otherwise unconstrained text."""
    text = _safe_text(text).strip()
    candidates = [text]
    fenced = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.S | re.I)
    candidates.extend(fenced)
    first = text.find("{")
    last = text.rfind("}")
    if first >= 0 and last > first:
        candidates.append(text[first:last + 1])
    for candidate in candidates:
        try:
            value = json.loads(candidate)
        except (TypeError, ValueError, json.JSONDecodeError):
            continue
        if isinstance(value, dict):
            return value
    return {}


def _answer_value(answer, allowed):
    """Read a typed Jev answer defensively, including stand-in answers."""
    value = getattr(answer, "value", None)
    if value not in allowed:
        return None, 0.0
    probabilities = getattr(answer, "probabilities", None)
    if not isinstance(probabilities, dict):
        probabilities = {}
    probability = _number(probabilities.get(value), 0.0) or 0.0
    confidence = _number(getattr(answer, "confidence", None), None)
    # Choices normally have confidence.  A stand-in may not; the probability
    # gate is still useful and keeps the fallback executable.
    if confidence is not None and confidence < 0.52:
        return value, min(probability, confidence)
    return value, probability


class Harness:
    def __init__(self, ctx):
        self.ctx = ctx
        self._reset_episode_state()

    def _reset_episode_state(self):
        self.last_frame = None
        self.last_ball_frame = None
        self.last_ball_x = None
        self.last_ball_y = None
        self.ball_vx = None
        self.ball_vy = None
        self.last_paddle_frame = None
        self.last_paddle_y = None
        self.paddle_velocity = 0.0
        self.last_own_points = 0
        self.last_opponent_points = 0
        self.recent_points = deque(maxlen=8)
        self.last_target = COURT_CENTER
        self.last_control_action = "NOOP"
        self.plan = {
            "style": "safe",
            "use_feature": False,
            "target_bias_px": 0.0,
            "deadband_px": 4.0,
            "version": 0,
        }
        self.astra_future = None
        self.astra_session = None
        self.astra_started = False
        self.astra_replan_started = False
        self.jev_future = None
        self.next_jev_frame = 0
        self.jev_hint = None
        self.jev_feature_vote = False
        self.feature_used_for_point = False
        self.last_model_log_frame = -100000

    # ---------- public-model supervision ----------

    def _compact_observation(self, obs):
        if not isinstance(obs, dict):
            obs = {}
        fields = (
            "seed", "scenario_index", "mode", "difficulty", "cartridge_game",
            "variation", "left_difficulty_switch", "right_difficulty_switch",
            "left_paddle_size", "right_paddle_size", "frame", "ticks",
            "phase", "serve_waiting", "serve_countdown_frames",
            "ball_visible", "ball_x", "ball_y", "ball_vx", "ball_vy",
            "ball_approaching_right", "right_paddle_y", "right_paddle_height_px",
            "left_paddle_y", "left_paddle_height_px", "own_points",
            "opponent_points", "last_reward", "return", "done",
        )
        return {name: _json_value(obs.get(name)) for name in fields}

    def _start_astra(self, game, obs):
        if self.astra_started or getattr(self.ctx, "system2", None) is None:
            return
        self.astra_started = True
        try:
            description = game.describe()
        except Exception:
            description = "Atari 2600 Robot Pong; right paddle is controlled by this harness."
        state = self._compact_observation(obs)
        prompt = (
            "You are the slow supervisory planner for a real-time Atari 2600 Robot Pong "
            "controller. The right paddle must be controlled every 1/60-second frame. "
            "The local program, not you, performs arithmetic, wall reflection, score "
            "bookkeeping, timing, and one-frame actions. RIGHT moves screen-y upward; "
            "LEFT moves it downward. Both Robot Pong variations serve automatically, so "
            "never recommend FIRE to start a serve. The safe strategy is to project the "
            "ball to the right paddle and cover its reflected y; use a central position "
            "while the ball travels away. A button feature is optional and must be gated "
            "at contact.\n\n"
            "Public game description:\n" + _safe_text(description)[:5000] +
            "\n\nInitial compact observation:\n" + json.dumps(state, sort_keys=True) +
            "\n\nReturn a short JSON object only, with these optional fields: "
            "style ('safe' or 'attack'), use_feature (boolean), target_bias_px "
            "(number from -6 to 6), deadband_px (number from 2 to 8), and a brief "
            "diagnosis. Default to safe, no feature, zero bias. Do not prescribe a "
            "single-frame action. The code will reject unsafe or malformed values."
        )

        def call():
            return self.ctx.think(prompt, timeout_s=60.0)

        try:
            self.astra_future = self.ctx.spawn(call)
        except (BudgetExceeded, EpisodeOver, ModelUnavailable, TimeoutError, RuntimeError):
            self.astra_future = None
        except Exception:
            # Pre-flight stand-ins and optional System-2 implementations may not
            # expose a spawnable backend. The local controller is sufficient.
            self.astra_future = None

    def _start_astra_replan(self, game, obs):
        # One optional diagnostic, only after several observed point outcomes and
        # only after the initial planner has returned. It cannot interrupt play.
        if self.astra_replan_started or self.astra_future is not None:
            return
        if getattr(self.ctx, "system2", None) is None:
            return
        if len(self.recent_points) < 4:
            return
        self.astra_replan_started = True
        evidence = list(self.recent_points)
        prompt = (
            "Supervise an already-running Atari 2600 Robot Pong controller. The "
            "frame-level controller uses verified visible ball coordinates, the "
            "right-paddle y, and reflected interception. Diagnose only systematic "
            "prediction drift from this evidence and return JSON with optional "
            "style, use_feature, target_bias_px (-6..6), deadband_px (2..8). "
            "Keep safe/no-feature defaults when evidence is insufficient. Do not "
            "give a one-frame action and do not invent RAM addresses.\n"
            "Current observation: " + json.dumps(self._compact_observation(obs), sort_keys=True) +
            "\nRecent score events: " + json.dumps(evidence, sort_keys=True)
        )

        def call():
            return self.ctx.think(prompt, session_id=self.astra_session, timeout_s=60.0)

        try:
            self.astra_future = self.ctx.spawn(call)
        except (BudgetExceeded, EpisodeOver, ModelUnavailable, TimeoutError, RuntimeError):
            self.astra_future = None
        except Exception:
            self.astra_future = None

    def _accept_astra(self, result):
        if result is None:
            return
        text = getattr(result, "text", result)
        parsed = _extract_json(text)
        style = parsed.get("style")
        if isinstance(style, str) and style.lower() in ("safe", "attack"):
            self.plan["style"] = style.lower()
        use_feature = parsed.get("use_feature")
        if isinstance(use_feature, bool):
            # Even an attacking plan cannot enable FIRE by itself: Jev must
            # recognize the contact window and the current variation below.
            self.plan["use_feature"] = use_feature
        bias = _number(parsed.get("target_bias_px"), None)
        if bias is not None:
            self.plan["target_bias_px"] = _clamp(bias, -6.0, 6.0)
        deadband = _number(parsed.get("deadband_px"), None)
        if deadband is not None:
            self.plan["deadband_px"] = _clamp(deadband, 2.0, 8.0)
        self.plan["version"] = int(self.plan.get("version", 0)) + 1
        session = getattr(result, "session_id", None)
        if session:
            self.astra_session = session
        try:
            self.ctx.log("astra_plan", plan=dict(self.plan), diagnosis=_safe_text(parsed.get("diagnosis", ""))[:300])
        except Exception:
            pass

    def _poll_astra(self):
        future = self.astra_future
        if future is None:
            return
        try:
            if not future.done():
                return
            self.astra_future = None
            self._accept_astra(future.result())
        except (BudgetExceeded, EpisodeOver, ModelUnavailable, TimeoutError, RuntimeError):
            self.astra_future = None
        except Exception:
            self.astra_future = None

    def _jev_questions(self):
        # The first and second direction choices deliberately reorder options.
        # A direction hint is used only as a tie-breaker when code's ball state
        # is uncertain; clear arithmetic remains authoritative.
        direction_up = "UP (RIGHT action): the controlled paddle should move toward smaller screen y."
        direction_down = "DOWN (LEFT action): the controlled paddle should move toward larger screen y."
        direction_hold = "HOLD (NOOP): the paddle already covers the predicted contact band or the state is serve/unknown."
        common = {
            "UP": direction_up,
            "DOWN": direction_down,
            "HOLD": direction_hold,
        }
        return {
            "approach": Choice(
                "Classify the ball's horizontal state from `ball.approach`. Do not infer it from the score.",
                {
                    "RIGHT": "The ball is reliably moving toward the controlled right paddle.",
                    "AWAY": "The ball is moving toward the left/opponent side.",
                    "UNKNOWN": "The ball is hidden, serving, or its horizontal velocity is unknown.",
                },
            ),
            "alignment": Choice(
                "Classify the relation in `control.target_relation`, which was computed by code from the projected contact y and the current right paddle y.",
                {
                    "ABOVE": "The predicted contact is above the right paddle; an upward move may be needed.",
                    "ALIGNED": "The paddle overlaps or is very close to the predicted contact band.",
                    "BELOW": "The predicted contact is below the right paddle; a downward move may be needed.",
                    "UNKNOWN": "There is no reliable target because the ball is serving/occluded or the fields are absent.",
                },
            ),
            "contact": Choice(
                "Classify `control.contact_window`: is the ball within the supplied small frame bucket of the right-paddle x?",
                {
                    "NOW": "Yes: the predicted contact is at most 3 frames away and the phase is rally.",
                    "SOON": "No: it is 4 to 12 frames away.",
                    "LATER": "No: it is more than 12 frames away or is moving away.",
                    "UNKNOWN": "The phase or ball trajectory is not reliable.",
                },
            ),
            "direction_a": Choice(
                "Choose the safe immediate direction using only `control.target_relation`, `control.code_action`, and the phase. Do not calculate a new trajectory. If uncertain choose HOLD.",
                {"UP": common["UP"], "HOLD": common["HOLD"], "DOWN": common["DOWN"]},
            ),
            "direction_b": Choice(
                "Independently choose the safe immediate direction using the same supplied control facts. If uncertain choose HOLD.",
                {"DOWN": common["DOWN"], "HOLD": common["HOLD"], "UP": common["UP"]},
            ),
            "feature": Choice(
                "Decide whether the red-button feature should be used on this frame. Never use it during serve_delay or when `plan.use_feature` is false.",
                {
                    "USE": "Use only when plan.use_feature is true, the variation has a legal feature, contact is NOW, and the movement direction is already safe.",
                    "NO": "Do not press FIRE; safe center contact or uncertain timing is preferable.",
                },
            ),
        }

    def _launch_jev(self, state, frame):
        if self.jev_future is not None:
            return
        if getattr(self.ctx, "system1", None) is None:
            return
        questions = self._jev_questions()

        def call():
            return self.ctx.decide(state, questions, timeout_s=1.0)

        try:
            self.jev_future = self.ctx.spawn(call)
            self.next_jev_frame = frame + 18
        except (BudgetExceeded, EpisodeOver, ModelUnavailable, TimeoutError, RuntimeError):
            self.jev_future = None
            self.next_jev_frame = frame + 30
        except Exception:
            self.jev_future = None
            self.next_jev_frame = frame + 30

    def _poll_jev(self, current_frame):
        future = self.jev_future
        if future is None:
            return
        try:
            if not future.done():
                return
            self.jev_future = None
            answers = future.result()
            if not isinstance(answers, dict):
                return
            a = answers.get("direction_a")
            b = answers.get("direction_b")
            av, ap = _answer_value(a, {"UP", "DOWN", "HOLD"}) if a is not None else (None, 0.0)
            bv, bp = _answer_value(b, {"UP", "DOWN", "HOLD"}) if b is not None else (None, 0.0)
            # Two differently ordered choices must agree before a stale model
            # hint is allowed to affect an uncertain local state.
            if av is not None and av == bv and min(ap, bp) >= 0.58:
                self.jev_hint = {"value": av, "expires": current_frame + 5, "strength": min(ap, bp)}
            feature_answer = answers.get("feature")
            fv, fp = _answer_value(feature_answer, {"USE", "NO"}) if feature_answer is not None else (None, 0.0)
            self.jev_feature_vote = fv == "USE" and fp >= 0.68
            try:
                self.ctx.log(
                    "jev_supervision",
                    direction=av if av == bv else "disagree",
                    direction_probability=min(ap, bp),
                    feature=fv,
                    feature_probability=fp,
                )
            except Exception:
                pass
        except (BudgetExceeded, EpisodeOver, ModelUnavailable, TimeoutError, RuntimeError):
            self.jev_future = None
        except Exception:
            self.jev_future = None

    # ---------- verified local state and controller ----------

    def _update_tracking(self, obs):
        if not isinstance(obs, dict):
            obs = {}
        frame = _integer(obs.get("frame"), None)
        if frame is None:
            frame = _integer(obs.get("ticks"), 0) or 0
        phase = obs.get("phase") if isinstance(obs.get("phase"), str) else "unknown"

        own = _integer(obs.get("own_points"), self.last_own_points)
        opponent = _integer(obs.get("opponent_points"), self.last_opponent_points)
        if own != self.last_own_points or opponent != self.last_opponent_points:
            reward = _integer(obs.get("last_reward"), 0) or 0
            self.recent_points.append({
                "frame": frame,
                "reward": reward,
                "own_points": own,
                "opponent_points": opponent,
                "right_y": _number(obs.get("right_paddle_y"), self.last_paddle_y),
                "target_y": self.last_target,
            })
            self.feature_used_for_point = False
        self.last_own_points = own
        self.last_opponent_points = opponent

        # A point/serve is a hard phase boundary. Do not carry an old velocity
        # through the stale ball sprite shown on a score frame.
        if phase != "rally":
            if phase == "serve_delay" or phase == "finished":
                self.ball_vx = None
                self.ball_vy = None
                self.last_ball_frame = None
                self.last_ball_x = None
                self.last_ball_y = None

        paddle_y = _number(obs.get("right_paddle_y"), None)
        if paddle_y is not None:
            if self.last_paddle_y is not None and self.last_paddle_frame is not None:
                gap = max(1, frame - self.last_paddle_frame)
                self.paddle_velocity = (paddle_y - self.last_paddle_y) / float(gap)
            self.last_paddle_y = paddle_y
            self.last_paddle_frame = frame

        x = _number(obs.get("ball_x"), None)
        y = _number(obs.get("ball_y"), None)
        visible = bool(obs.get("ball_visible")) and x is not None and y is not None
        if visible and phase == "rally":
            if self.last_ball_x is not None and self.last_ball_y is not None and self.last_ball_frame is not None:
                gap = frame - self.last_ball_frame
                if 0 < gap <= 8:
                    dx = (x - self.last_ball_x) / float(gap)
                    dy = (y - self.last_ball_y) / float(gap)
                    if abs(dx) <= 2.5 and abs(dx) >= 0.25:
                        self.ball_vx = dx
                    if abs(dy) <= 4.0 and abs(dy) >= 0.25:
                        self.ball_vy = dy
            field_vx = _number(obs.get("ball_vx"), None)
            field_vy = _number(obs.get("ball_vy"), None)
            if field_vx is not None and abs(field_vx) >= 0.25:
                self.ball_vx = field_vx
            if field_vy is not None and abs(field_vy) >= 0.25:
                self.ball_vy = field_vy
            self.last_ball_frame = frame
            self.last_ball_x = x
            self.last_ball_y = y
        elif visible and phase not in ("serve_delay", "finished"):
            # Keep a visible ball as a last-known point during a transient
            # adaptor phase, but do not call it a rally velocity yet.
            self.last_ball_frame = frame
            self.last_ball_x = x
            self.last_ball_y = y

        if phase == "serve_delay":
            self.feature_used_for_point = False
        self.last_frame = frame
        return frame, phase, x, y, visible, obs

    def _trajectory(self, frame, phase, x, y, visible, obs):
        """Return target y and named facts used by both code and Jev."""
        paddle_y = self.last_paddle_y if self.last_paddle_y is not None else COURT_CENTER
        if self.last_ball_frame is not None and self.last_ball_x is not None and self.last_ball_y is not None:
            gap = max(0, frame - self.last_ball_frame)
            bx = x if x is not None else self.last_ball_x + (self.ball_vx or 0.0) * gap
            by = y if y is not None else _reflect_y(self.last_ball_y, self.ball_vy or 0.0, gap)
        else:
            gap = 0
            bx = x
            by = y

        # Prefer the adaptor's explicitly inferred approach, but recover from
        # consecutive visible positions when it is null.
        approach_field = obs.get("ball_approaching_right") if isinstance(obs, dict) else None
        if approach_field is True:
            approach = "RIGHT"
        elif approach_field is False:
            approach = "AWAY"
        elif self.ball_vx is not None and self.ball_vx > 0:
            approach = "RIGHT"
        elif self.ball_vx is not None and self.ball_vx < 0:
            approach = "AWAY"
        else:
            approach = "UNKNOWN"

        rally_like = phase in ("rally", "rally_ball_occluded")
        reliable = (
            rally_like
            and bx is not None
            and by is not None
            and self.ball_vx is not None
            and abs(self.ball_vx) >= 0.25
            and self.ball_vy is not None
            and abs(self.ball_vy) >= 0.25
            and gap <= 8
        )
        if approach == "RIGHT" and reliable:
            speed_x = max(0.5, abs(self.ball_vx))
            frames_to_contact = max(0.0, (RIGHT_CONTACT_X - bx) / speed_x)
            target = _reflect_y(by, self.ball_vy, frames_to_contact)
            target += _number(self.plan.get("target_bias_px"), 0.0) or 0.0
            target = _clamp(target, PADDLE_TOP, PADDLE_BOTTOM)
            contact_bucket = _bucket(frames_to_contact, (3.5, 12.5), ("NOW", "SOON", "LATER"))
        else:
            frames_to_contact = None
            target = COURT_CENTER
            contact_bucket = "UNKNOWN" if phase != "rally" or approach == "UNKNOWN" else "LATER"

        # When the ball is travelling away, central coverage is deliberately
        # preferred to chasing its current y. This is the key difference from
        # the parent that asked Jev to choose a raw action on each stale state.
        target_relation = "UNKNOWN"
        if phase == "rally" and self.last_paddle_y is not None and reliable:
            diff = target - paddle_y
            margin = max(3.0, min(8.0, _number(self.plan.get("deadband_px"), 4.0) + 1.0))
            if diff < -margin:
                target_relation = "ABOVE"
            elif diff > margin:
                target_relation = "BELOW"
            else:
                target_relation = "ALIGNED"

        if not rally_like:
            target = COURT_CENTER
        error = target - paddle_y
        deadband = _number(self.plan.get("deadband_px"), 4.0) or 4.0
        # Full paddles can tolerate a little more centering slack; half paddles
        # need earlier movement and a narrower final deadband.
        nominal = _number(obs.get("right_paddle_nominal_height_px"), 16.0) or 16.0
        deadband = _clamp(deadband * (0.85 if nominal <= 8 else 1.0), 2.5, 8.0)
        if phase != "rally" or not reliable:
            deadband = max(deadband, 6.0)

        if abs(error) <= deadband:
            code_action = "HOLD"
        elif error < 0:
            code_action = "UP"
        else:
            code_action = "DOWN"

        contact_now = (
            rally_like
            and approach == "RIGHT"
            and frames_to_contact is not None
            and frames_to_contact <= 3.0
        )
        return {
            "ball": {
                "visible": visible,
                "x": bx,
                "y": by,
                "vx": self.ball_vx,
                "vy": self.ball_vy,
                "approach": approach,
                "reliable": reliable,
                "visibility_gap": gap,
            },
            "paddle": {"y": paddle_y, "velocity": self.paddle_velocity},
            "target_y": target,
            "frames_to_contact": frames_to_contact,
            "frames_to_contact_bucket": contact_bucket,
            "target_relation": target_relation,
            "contact_window": contact_now,
            "code_action": code_action,
            "phase": phase,
        }

    def _state_for_jev(self, obs, control):
        compact = self._compact_observation(obs)
        own = _integer(compact.get("own_points"), self.last_own_points) or 0
        opponent = _integer(compact.get("opponent_points"), self.last_opponent_points) or 0
        return _json_value({
            "game": "Atari 2600 Robot Pong, controlled right paddle",
            "rules": {
                "right_action": "RIGHT means screen-y up; LEFT means screen-y down; NOOP holds",
                "serve": "automatic; FIRE is never needed to start",
                "goal": "cover the reflected ball at the right paddle; safe center contact",
            },
            "scenario": {
                "mode": compact.get("mode"),
                "difficulty": compact.get("difficulty"),
                "variation": compact.get("variation"),
                "right_paddle_size": compact.get("right_paddle_size"),
            },
            "score": {"own": own, "opponent": opponent, "last_reward": compact.get("last_reward")},
            "observation": {
                "frame": compact.get("frame"),
                "phase": compact.get("phase"),
                "serve_waiting": compact.get("serve_waiting"),
                "countdown": compact.get("serve_countdown_frames"),
            },
            "ball": control["ball"],
            "right_paddle": control["paddle"],
            "control": {
                "predicted_contact_y": control["target_y"],
                "target_relation": control["target_relation"],
                "contact_window": control["contact_window"],
                "contact_bucket": control["frames_to_contact_bucket"],
                "code_action": control["code_action"],
                "uncertain": not control["ball"]["reliable"],
            },
            "plan": {
                "style": self.plan.get("style", "safe"),
                "use_feature": bool(self.plan.get("use_feature", False)),
                "target_bias_px": self.plan.get("target_bias_px", 0.0),
                "deadband_px": self.plan.get("deadband_px", 4.0),
            },
            "recent_score_events": list(self.recent_points)[-4:],
        })

    def _apply_model_hints(self, control, frame):
        # A late answer is never allowed to override a reliable projection. It
        # is useful only when the ball/velocity parser is temporarily uncertain.
        hint = self.jev_hint
        if hint is not None and frame > hint.get("expires", -1):
            self.jev_hint = None
            hint = None
        action = control["code_action"]
        if hint is not None and not control["ball"]["reliable"] and control["phase"] in ("rally", "rally_ball_occluded"):
            if hint.get("value") in ("UP", "DOWN"):
                action = hint["value"]
            elif hint.get("value") == "HOLD":
                action = "HOLD"
        return action

    def _feature_allowed_now(self, obs, control, action):
        if not self.plan.get("use_feature") or not self.jev_feature_vote:
            return False
        if self.feature_used_for_point or not control.get("contact_window"):
            return False
        if control.get("phase") not in ("rally", "rally_ball_occluded") or control["ball"].get("approach") != "RIGHT":
            return False
        variation = _safe_text(obs.get("variation", ""))
        if "Robot Pong" not in variation:
            return False
        # A feature is never used on a HOLD frame; it is combined only with
        # the already selected movement action at the verified contact window.
        if action not in ("UP", "DOWN"):
            return False
        return True

    def _action_name(self, spec, semantic):
        names = set(getattr(spec, "names", ()) or ())
        mapping = {"UP": "RIGHT", "DOWN": "LEFT", "HOLD": "NOOP"}
        candidate = mapping.get(semantic, "NOOP")
        if candidate in names:
            return candidate
        noop = getattr(spec, "noop", None)
        if noop in names:
            return noop
        return next(iter(names), "NOOP")

    def _choose_action(self, game, obs, control, frame):
        semantic = self._apply_model_hints(control, frame)
        spec = game.action_spec
        action = self._action_name(spec, semantic)
        if self._feature_allowed_now(obs, control, semantic):
            feature_name = "RIGHTFIRE" if semantic == "UP" else "LEFTFIRE"
            if feature_name in set(getattr(spec, "names", ()) or ()):
                action = feature_name
                self.feature_used_for_point = True
        self.last_target = control["target_y"]
        self.last_control_action = action
        return action

    def _maybe_start_jev(self, obs, control, frame):
        if self.jev_future is not None or frame < self.next_jev_frame:
            return
        # The model is most useful near an approaching contact or when the
        # parser is uncertain. A periodic request also refreshes feature gating.
        approach = control["ball"].get("approach")
        near = control.get("frames_to_contact") is not None and control.get("frames_to_contact") <= 55.0
        uncertain = not control["ball"].get("reliable", False)
        point_recent = bool(self.recent_points and frame - (self.recent_points[-1].get("frame") or frame) < 20)
        if not (near or uncertain or point_recent or frame == 0):
            self.next_jev_frame = frame + 6
            return
        self._launch_jev(self._state_for_jev(obs, control), frame)

    def run_episode(self, game) -> None:
        ctx = self.ctx
        # The no-model path still runs the exact same real-time controller.
        if not getattr(game, "tick_s", None):
            return

        first_obs = game.observe()
        self._start_astra(game, first_obs)

        while not ctx.done and not getattr(game, "done", False):
            obs = game.observe()
            frame, phase, _x, _y, visible, obs = self._update_tracking(obs)
            self._poll_astra()
            self._poll_jev(frame)

            if len(self.recent_points) >= 4 and not self.astra_replan_started:
                self._start_astra_replan(game, obs)

            control = self._trajectory(frame, phase, _x, _y, visible, obs)
            self._maybe_start_jev(obs, control, frame)
            action = self._choose_action(game, obs, control, frame)

            try:
                ctx.act(action)
            except (BudgetExceeded, EpisodeOver):
                break
            except ModelUnavailable:
                raise
            # Align the next local decision to the next emulator frame. The
            # action has already been posted, so a model call can never stall it.
            try:
                ctx.wait_for_tick()
            except (BudgetExceeded, EpisodeOver):
                break


def build(ctx):
    return Harness(ctx)
