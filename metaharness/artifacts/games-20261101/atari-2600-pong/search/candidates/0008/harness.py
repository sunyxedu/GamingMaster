"""Predictive Robot Pong control with contact-timed feature authorization.

The local loop owns every frame, geometric projection, and legal action.  Astra
chooses a high-level return style asynchronously.  Jev only authorizes a pulse
for the next contact; it never supplies a stale frame action.
"""
from __future__ import annotations

import json
import math
import re
from collections import deque

from metaharness.core.contract import BudgetExceeded, EpisodeOver, ModelUnavailable
from metaharness.models.jev import Choice


# Visible sprite-center geometry from the public adaptor contract.
BALL_TOP = 37.5
BALL_BOTTOM = 191.5
PADDLE_TOP = 37.5
PADDLE_BOTTOM = 191.5
COURT_CENTER = (PADDLE_TOP + PADDLE_BOTTOM) / 2.0
RIGHT_CONTACT_X = 140.5


def number(value, default=None):
    if isinstance(value, bool):
        return default
    try:
        value = float(value)
    except (TypeError, ValueError, OverflowError):
        return default
    return value if math.isfinite(value) else default


def integer(value, default=None):
    value = number(value, None)
    return default if value is None else int(value)


def clamp(value, low, high):
    return max(low, min(high, value))


def sign(value):
    value = number(value, None)
    if value is None or value == 0:
        return None
    return 1 if value > 0 else -1


def json_safe(value):
    """Convert observations and derived facts to strict JSON values."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    converted = number(value, None)
    return converted


def reflect_y(y, velocity, frames):
    """Mirror-project a ball center between the visible court walls."""
    y = number(y, COURT_CENTER)
    velocity = number(velocity, 0.0)
    frames = max(0.0, number(frames, 0.0))
    span = BALL_BOTTOM - BALL_TOP
    if span <= 0:
        return y
    phase = (y - BALL_TOP + velocity * frames) % (2.0 * span)
    if phase <= span:
        return BALL_TOP + phase
    return BALL_BOTTOM - (phase - span)


def safe_text(value):
    try:
        return str(value)
    except Exception:
        return ""


def extract_json(text):
    """Accept short JSON, fenced JSON, or a JSON object embedded in prose."""
    text = safe_text(text).strip()
    candidates = [text]
    candidates.extend(re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S | re.I))
    first, last = text.find("{"), text.rfind("}")
    if first >= 0 and last > first:
        candidates.append(text[first : last + 1])
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except (TypeError, ValueError, json.JSONDecodeError):
            continue
        if isinstance(parsed, dict):
            return parsed
    return {}


def answer_value(answer, allowed):
    """Read real or stand-in typed answers without trusting their shape."""
    if isinstance(answer, dict):
        value = answer.get("value", answer.get("choice"))
        probabilities = answer.get("probabilities", {})
        confidence = answer.get("confidence")
    else:
        value = getattr(answer, "value", None)
        if value is None:
            value = getattr(answer, "choice", None)
        probabilities = getattr(answer, "probabilities", {})
        confidence = getattr(answer, "confidence", None)
    if value not in allowed:
        return None, 0.0
    if not isinstance(probabilities, dict):
        probabilities = {}
    probability = number(probabilities.get(value), 0.0) or 0.0
    confidence = number(confidence, None)
    if confidence is not None:
        probability = min(probability, confidence)
    return value, probability


class Harness:
    def __init__(self, ctx):
        self.ctx = ctx
        self._reset_episode()

    def _reset_episode(self):
        # Tracking is deliberately episode-local.  A raw ball sprite on a point
        # frame never survives a serve boundary into the next trajectory.
        self.last_frame = 0
        self.previous_phase = None
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
        self.last_semantic_action = "HOLD"

        # This is intentionally an attacking fallback.  The parent already
        # demonstrated that safe central returns can produce an endless draw.
        # Astra may switch it after seeing evidence, but model failure does not
        # silently turn this experiment back into the old candidate.
        self.plan = {
            "style": "attack",
            "use_feature": True,
            "attack_strength_px": 3.5,
            "deadband_px": 2.5,
            "version": 0,
        }
        self.rally_generation = 0
        self.attack_sign = 1
        self.feature_used_this_rally = False
        self.jev_feature_vote = False
        self.feature_vote_expires = -1
        self.pending_feature_expires = -1
        self.pending_rally_generation = -1

        self.astra_future = None
        self.astra_session = None
        self.astra_started = False
        self.astra_replan_started = False
        self.jev_future = None
        self.next_jev_frame = 0

    # ---------- model supervision ----------

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
        return {field: json_safe(obs.get(field)) for field in fields}

    def _start_astra(self, game, obs):
        if self.astra_started or getattr(self.ctx, "system2", None) is None:
            return
        self.astra_started = True
        try:
            description = game.describe()
        except Exception:
            description = "Atari 2600 Robot Pong; this harness controls the right paddle."
        prompt = (
            "You are the slow strategic supervisor for an already frame-rate-safe "
            "Atari 2600 Robot Pong controller. The prior safe controller can sustain a "
            "scoreless rally, so this candidate tests deliberate off-center returns and "
            "the Robot Pong Speed/WHAMMY button at contact. Start with attack and a legal "
            "feature unless evidence says that is causing misses. The local code owns all "
            "arithmetic, wall reflection, score bookkeeping, timing, and one-frame actions. "
            "RIGHT moves the right paddle upward (smaller screen y); LEFT moves it downward. "
            "Both Robot Pong variations serve automatically: FIRE is never a serve command. "
            "A button pulse is separately gated by Jev and by the current variation/contact.\n\n"
            "Return a short JSON object only with optional fields: style ('attack' or 'safe'), "
            "use_feature (boolean), attack_strength_px (0..6), deadband_px (1.5..6), and "
            "diagnosis. With no evidence, return attack, use_feature true, strength 3.5, "
            "deadband 2.5. Never return a one-frame action.\n\n"
            "Public game description:\n" + safe_text(description)[:5000] +
            "\n\nInitial observation:\n" +
            json.dumps(self._compact_observation(obs), sort_keys=True)
        )

        def call():
            return self.ctx.think(prompt, timeout_s=60.0)

        try:
            self.astra_future = self.ctx.spawn(call)
        except Exception:
            self.astra_future = None

    def _start_astra_replan(self, game, obs):
        if self.astra_replan_started or self.astra_future is not None:
            return
        if getattr(self.ctx, "system2", None) is None or len(self.recent_points) < 2:
            return
        self.astra_replan_started = True
        prompt = (
            "Diagnose an Atari 2600 Robot Pong attack controller from the supplied public "
            "evidence. The code projects the ball and posts every frame; do not prescribe "
            "an action or recalculate a trajectory. Decide whether the opponent is being "
            "beaten by edge/feature returns or whether misses require a safer center return. "
            "Return only JSON fields style ('attack'/'safe'), use_feature (boolean), "
            "attack_strength_px (0..6), deadband_px (1.5..6), diagnosis. Keep attack and "
            "feature enabled when the evidence is a scoreless safe rally.\n\nCurrent:\n" +
            json.dumps(self._compact_observation(obs), sort_keys=True) +
            "\nRecent point evidence:\n" + json.dumps(list(self.recent_points), sort_keys=True)
        )

        def call():
            return self.ctx.think(prompt, session_id=self.astra_session, timeout_s=60.0)

        try:
            self.astra_future = self.ctx.spawn(call)
        except Exception:
            self.astra_future = None

    def _accept_astra(self, result):
        if result is None:
            return
        parsed = extract_json(getattr(result, "text", result))
        style = parsed.get("style")
        if isinstance(style, str) and style.lower() in ("attack", "safe"):
            self.plan["style"] = style.lower()
        use_feature = parsed.get("use_feature")
        if isinstance(use_feature, bool):
            self.plan["use_feature"] = use_feature
        strength = number(parsed.get("attack_strength_px"), None)
        if strength is not None:
            self.plan["attack_strength_px"] = clamp(abs(strength), 0.0, 6.0)
        deadband = number(parsed.get("deadband_px"), None)
        if deadband is not None:
            self.plan["deadband_px"] = clamp(deadband, 1.5, 6.0)
        self.plan["version"] = int(self.plan.get("version", 0)) + 1
        session = getattr(result, "session_id", None)
        if session:
            self.astra_session = session
        try:
            self.ctx.log(
                "astra_plan",
                plan=dict(self.plan),
                diagnosis=safe_text(parsed.get("diagnosis", ""))[:300],
            )
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
        except Exception:
            self.astra_future = None

    def _jev_questions(self):
        # Reversed option order cancels the provider's documented first-option
        # preference.  The answer is an intention for the upcoming contact, not a
        # permission to press FIRE on the snapshot frame.
        first = {
            "USE": "Authorize one pulse for the upcoming right-paddle return when the supplied feature_horizon is NEXT_CONTACT_WITHIN_55 and the variation is Robot Pong 1 or 2.",
            "NO": "Do not authorize a pulse because the horizon/phase/variation facts do not support it.",
        }
        reversed_order = {
            "NO": first["NO"],
            "USE": first["USE"],
        }
        return {
            "feature_a": Choice(
                "Should one mode-legal button pulse be authorized for the NEXT right-paddle contact? Read the explicit code-derived fields; do not invent timing.",
                first,
            ),
            "feature_b": Choice(
                "Independently judge the same upcoming-contact authorization from the explicit state. This is not a current-frame action.",
                reversed_order,
            ),
        }

    def _launch_jev(self, state, frame):
        if self.jev_future is not None or getattr(self.ctx, "system1", None) is None:
            return
        questions = self._jev_questions()
        control = state.get("control", {}) if isinstance(state, dict) else {}
        eta = number(control.get("frames_to_contact"), None) if isinstance(control, dict) else None
        horizon = control.get("feature_horizon") if isinstance(control, dict) else None
        if horizon == "NEXT_CONTACT_WITHIN_55" and eta is not None:
            self.pending_feature_expires = frame + int(max(0.0, eta)) + 6
        else:
            self.pending_feature_expires = -1
        self.pending_rally_generation = self.rally_generation

        def call():
            try:
                return self.ctx.decide(state, questions, timeout_s=1.0)
            except Exception:
                return None

        try:
            self.jev_future = self.ctx.spawn(call)
            self.next_jev_frame = frame + 24
        except Exception:
            self.jev_future = None
            self.next_jev_frame = frame + 36
            self.pending_feature_expires = -1

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
            av, ap = answer_value(answers.get("feature_a"), {"USE", "NO"})
            bv, bp = answer_value(answers.get("feature_b"), {"USE", "NO"})
            agree = av is not None and av == bv and min(ap, bp) >= 0.60
            same_rally = self.pending_rally_generation == self.rally_generation
            timely = self.pending_feature_expires >= current_frame
            if agree and same_rally and timely and av == "USE":
                self.jev_feature_vote = True
                self.feature_vote_expires = self.pending_feature_expires
            elif agree and av == "NO" and same_rally:
                self.jev_feature_vote = False
                self.feature_vote_expires = -1
            try:
                self.ctx.log(
                    "jev_feature_authorization",
                    value=av if av == bv else "DISAGREE",
                    probability=min(ap, bp),
                    timely=timely,
                    same_rally=same_rally,
                    expires=self.feature_vote_expires,
                    frame=current_frame,
                )
            except Exception:
                pass
        except Exception:
            self.jev_future = None

    # ---------- observation tracking and deterministic geometry ----------

    def _update_tracking(self, obs):
        if not isinstance(obs, dict):
            obs = {}
        frame = integer(obs.get("frame"), None)
        if frame is None:
            frame = integer(obs.get("ticks"), self.last_frame) or self.last_frame
        phase = obs.get("phase") if isinstance(obs.get("phase"), str) else "unknown"
        rally_like = phase in ("rally", "rally_ball_occluded")
        old_rally_like = self.previous_phase in ("rally", "rally_ball_occluded")

        own = integer(obs.get("own_points"), self.last_own_points)
        opponent = integer(obs.get("opponent_points"), self.last_opponent_points)
        if own != self.last_own_points or opponent != self.last_opponent_points:
            reward = integer(obs.get("last_reward"), 0) or 0
            self.recent_points.append({
                "frame": frame,
                "reward": reward,
                "own_points": own,
                "opponent_points": opponent,
                "paddle_y": number(obs.get("right_paddle_y"), self.last_paddle_y),
                "last_target": self.last_target,
            })
            # The rally counter advances at the next rally phase boundary;
            # keeping it here would flip twice (score event plus serve exit).
            self.feature_used_this_rally = False
            self.jev_feature_vote = False
            self.feature_vote_expires = -1
        self.last_own_points = own
        self.last_opponent_points = opponent

        # Explicit serve/finished phases are hard boundaries.  A stale point
        # sprite is never used as a velocity sample.
        if phase in ("serve_delay", "finished") and self.previous_phase not in ("serve_delay", "finished"):
            self.ball_vx = None
            self.ball_vy = None
            self.last_ball_frame = None
            self.last_ball_x = None
            self.last_ball_y = None
            self.feature_used_this_rally = False
            self.jev_feature_vote = False
            self.feature_vote_expires = -1
        if rally_like and not old_rally_like:
            self.rally_generation += 1
            self.attack_sign = 1 if self.rally_generation % 2 else -1
            self.ball_vx = None
            self.ball_vy = None
            self.last_ball_frame = None
            self.last_ball_x = None
            self.last_ball_y = None
            self.feature_used_this_rally = False

        paddle_y = number(obs.get("right_paddle_y"), None)
        if paddle_y is not None:
            if self.last_paddle_y is not None and self.last_paddle_frame is not None:
                gap = max(1, frame - self.last_paddle_frame)
                self.paddle_velocity = (paddle_y - self.last_paddle_y) / float(gap)
            self.last_paddle_y = paddle_y
            self.last_paddle_frame = frame

        x = number(obs.get("ball_x"), None)
        y = number(obs.get("ball_y"), None)
        visible = bool(obs.get("ball_visible")) and x is not None and y is not None
        if visible and rally_like:
            if self.last_ball_x is not None and self.last_ball_y is not None and self.last_ball_frame is not None:
                gap = frame - self.last_ball_frame
                if 0 < gap <= 8:
                    dx = (x - self.last_ball_x) / float(gap)
                    dy = (y - self.last_ball_y) / float(gap)
                    if 0.25 <= abs(dx) <= 2.5:
                        self.ball_vx = dx
                    if 0.25 <= abs(dy) <= 4.0:
                        self.ball_vy = dy
            field_vx = number(obs.get("ball_vx"), None)
            field_vy = number(obs.get("ball_vy"), None)
            if field_vx is None:
                field_vx = number(obs.get("ball_vx_sign"), None)
            if field_vy is None:
                field_vy = number(obs.get("ball_vy_sign"), None)
            if field_vx is not None and abs(field_vx) >= 0.25:
                self.ball_vx = field_vx
            if field_vy is not None and abs(field_vy) >= 0.25:
                self.ball_vy = field_vy
            self.last_ball_frame = frame
            self.last_ball_x = x
            self.last_ball_y = y
        self.last_frame = frame
        self.previous_phase = phase
        return frame, phase, x, y, visible, obs

    def _ball_for_control(self, frame, phase, x, y, visible):
        if phase not in ("rally", "rally_ball_occluded"):
            return None
        if visible and x is not None and y is not None:
            return x, y, self.ball_vx, self.ball_vy, 0
        if self.last_ball_frame is None or self.last_ball_x is None or self.last_ball_y is None:
            return None
        age = frame - self.last_ball_frame
        if age < 0 or age > 14:
            return None
        vx = self.ball_vx
        vy = self.ball_vy
        bx = self.last_ball_x + (vx or 0.0) * age
        by = reflect_y(self.last_ball_y, vy or 0.0, age)
        return bx, by, vx, vy, age

    def _trajectory(self, frame, phase, x, y, visible, obs):
        ball = self._ball_for_control(frame, phase, x, y, visible)
        if ball is None:
            approach = "UNKNOWN"
            bx = by = vx = vy = None
            gap = None
        else:
            bx, by, vx, vy, gap = ball
            field_approach = obs.get("ball_approaching_right")
            if field_approach is True:
                approach = "RIGHT"
            elif field_approach is False:
                approach = "AWAY"
            elif vx is not None and vx > 0:
                approach = "RIGHT"
            elif vx is not None and vx < 0:
                approach = "AWAY"
            else:
                approach = "UNKNOWN"

        ttc = None
        safe_target = COURT_CENTER
        target_valid = (
            phase in ("rally", "rally_ball_occluded")
            and approach == "RIGHT"
            and bx is not None
            and by is not None
            and vx is not None
            and vx > 0
        )
        if target_valid:
            ttc = max(0.0, (RIGHT_CONTACT_X - bx) / max(0.5, abs(vx)))
            safe_target = by if vy is None else reflect_y(by, vy, ttc)
            safe_target = clamp(safe_target, PADDLE_TOP, PADDLE_BOTTOM)

        feature_legal = "Robot Pong" in safe_text(obs.get("variation", ""))
        attack_enabled = self.plan.get("style") == "attack"
        nominal = number(obs.get("right_paddle_nominal_height_px"), 16.0) or 16.0
        if attack_enabled and target_valid:
            strength = number(self.plan.get("attack_strength_px"), 3.5) or 0.0
            # A half paddle has less safe overlap.  Keep the same structural
            # edge-return experiment while shrinking the offset in that case.
            if nominal <= 8.0:
                strength = min(strength, 1.75)
            offset = self.attack_sign * abs(strength)
            target = clamp(safe_target + offset, PADDLE_TOP, PADDLE_BOTTOM)
        else:
            target = safe_target if target_valid else COURT_CENTER

        paddle_y = self.last_paddle_y if self.last_paddle_y is not None else COURT_CENTER
        base_deadband = number(self.plan.get("deadband_px"), 2.5) or 2.5
        deadband = clamp(base_deadband, 1.5, 6.0)
        if nominal <= 8.0:
            deadband = max(1.5, deadband * 0.75)
        if not target_valid:
            deadband = max(deadband, 5.5)
        error = target - paddle_y
        if error < -deadband:
            semantic = "UP"
        elif error > deadband:
            semantic = "DOWN"
        else:
            semantic = "HOLD"

        reliable = (
            target_valid and vx is not None and vy is not None and gap is not None and gap <= 14
        )
        if ttc is None:
            bucket = "UNKNOWN"
        elif ttc <= 4:
            bucket = "NOW"
        elif ttc <= 16:
            bucket = "SOON"
        elif ttc <= 55:
            bucket = "UPCOMING"
        else:
            bucket = "LATER"
        relation = "ALIGNED"
        relation_margin = max(3.0, deadband + 1.0)
        if target < paddle_y - relation_margin:
            relation = "ABOVE"
        elif target > paddle_y + relation_margin:
            relation = "BELOW"
        if not target_valid and phase in ("rally", "rally_ball_occluded"):
            relation = "UNKNOWN"

        contact_window = bool(target_valid and ttc is not None and ttc <= 4.0)
        horizon = bool(target_valid and ttc is not None and ttc <= 55.0)
        return {
            "ball": {
                "visible": bool(visible),
                "x": bx,
                "y": by,
                "vx": vx,
                "vy": vy,
                "approach": approach,
                "reliable": reliable,
                "visibility_gap": gap,
            },
            "paddle": {"y": paddle_y, "velocity": self.paddle_velocity, "nominal_height": nominal},
            "target_y": target,
            "safe_target_y": safe_target,
            "attack_offset": target - safe_target if target_valid else 0.0,
            "frames_to_contact": ttc,
            "frames_to_contact_bucket": bucket,
            "target_relation": relation,
            "contact_window": contact_window,
            "feature_horizon": "NEXT_CONTACT_WITHIN_55" if horizon else "NONE",
            "feature_legal": feature_legal,
            "code_action": semantic,
            "phase": phase,
        }

    def _state_for_jev(self, obs, control):
        compact = self._compact_observation(obs)
        return json_safe({
            "game": "Atari 2600 Robot Pong; controlled right paddle",
            "rules": {
                "movement": "RIGHT means screen-y up; LEFT means screen-y down; NOOP holds",
                "serve": "automatic; FIRE is never needed to start",
                "feature": "Jev authorizes at most one contact pulse; code presses it only at live contact",
            },
            "scenario": {
                "mode": compact.get("mode"),
                "difficulty": compact.get("difficulty"),
                "variation": compact.get("variation"),
                "feature_kind": "Speed for Robot Pong 1; WHAMMY for Robot Pong 2",
                "right_paddle_size": compact.get("right_paddle_size"),
            },
            "score": {
                "own": integer(compact.get("own_points"), self.last_own_points),
                "opponent": integer(compact.get("opponent_points"), self.last_opponent_points),
                "last_reward": compact.get("last_reward"),
            },
            "observation": {
                "frame": compact.get("frame"),
                "phase": compact.get("phase"),
                "serve_waiting": compact.get("serve_waiting"),
                "serve_countdown_frames": compact.get("serve_countdown_frames"),
            },
            "ball": control["ball"],
            "right_paddle": control["paddle"],
            "control": {
                "safe_predicted_contact_y": control["safe_target_y"],
                "attack_predicted_contact_y": control["target_y"],
                "attack_offset_px": control["attack_offset"],
                "target_relation": control["target_relation"],
                "code_action": control["code_action"],
                "frames_to_contact": control["frames_to_contact"],
                "contact_bucket": control["frames_to_contact_bucket"],
                "contact_window": control["contact_window"],
                "feature_horizon": control["feature_horizon"],
                "feature_legal": control["feature_legal"],
                "feature_already_used_this_rally": self.feature_used_this_rally,
            },
            "plan": {
                "style": self.plan.get("style"),
                "use_feature": bool(self.plan.get("use_feature")),
                "attack_strength_px": self.plan.get("attack_strength_px"),
                "deadband_px": self.plan.get("deadband_px"),
            },
            "recent_score_events": list(self.recent_points)[-4:],
        })

    def _maybe_start_jev(self, obs, control, frame):
        if self.jev_future is not None or frame < self.next_jev_frame:
            return
        ttc = control.get("frames_to_contact")
        near = ttc is not None and ttc <= 55.0
        uncertain = not control.get("ball", {}).get("reliable", False)
        point_recent = bool(
            self.recent_points and frame - (self.recent_points[-1].get("frame") or frame) < 25
        )
        if not (near or uncertain or point_recent):
            self.next_jev_frame = frame + 8
            return
        self._launch_jev(self._state_for_jev(obs, control), frame)

    # ---------- action bridge ----------

    def _feature_allowed_now(self, obs, control, semantic):
        if not self.plan.get("use_feature") or not self.jev_feature_vote:
            return False
        if self.feature_used_this_rally or self.feature_vote_expires < self.last_frame:
            self.jev_feature_vote = False
            return False
        if not control.get("contact_window") or control.get("phase") not in ("rally", "rally_ball_occluded"):
            return False
        if not control.get("feature_legal") or control.get("ball", {}).get("approach") != "RIGHT":
            return False
        return True

    def _action_name(self, game, semantic, pulse=False):
        try:
            spec = game.action_spec
            names = set(getattr(spec, "names", ()) or ())
            mapping = {"UP": "RIGHT", "DOWN": "LEFT", "HOLD": "NOOP"}
            base = mapping.get(semantic, "NOOP")
            if pulse:
                base = {"UP": "RIGHTFIRE", "DOWN": "LEFTFIRE", "HOLD": "FIRE"}.get(semantic, "FIRE")
            if base in names:
                return base
            noop = getattr(spec, "noop", None)
            if noop in names:
                return noop
            return next(iter(names), "NOOP")
        except Exception:
            return "NOOP"

    def _choose_action(self, game, obs, control):
        semantic = control.get("code_action", "HOLD")
        pulse = self._feature_allowed_now(obs, control, semantic)
        if pulse:
            self.feature_used_this_rally = True
            self.jev_feature_vote = False
            try:
                self.ctx.log(
                    "contact_feature_pulse",
                    frame=self.last_frame,
                    variation=obs.get("variation"),
                    semantic=semantic,
                    attack_offset=control.get("attack_offset"),
                )
            except Exception:
                pass
        self.last_target = control.get("target_y", COURT_CENTER)
        self.last_semantic_action = semantic
        return self._action_name(game, semantic, pulse=pulse)

    def run_episode(self, game) -> None:
        ctx = self.ctx
        if not getattr(game, "tick_s", None):
            return
        try:
            self._start_astra(game, game.observe())
        except Exception:
            pass

        while not ctx.done and not getattr(game, "done", False):
            try:
                obs = game.observe()
            except Exception:
                obs = {}
            frame, phase, x, y, visible, obs = self._update_tracking(obs)
            self._poll_astra()
            self._poll_jev(frame)
            if len(self.recent_points) >= 2:
                self._start_astra_replan(game, obs)
            control = self._trajectory(frame, phase, x, y, visible, obs)
            self._maybe_start_jev(obs, control, frame)
            action = self._choose_action(game, obs, control)
            try:
                ctx.act(action)
                ctx.wait_for_tick()
            except (BudgetExceeded, EpisodeOver, ModelUnavailable):
                return


def build(ctx):
    return Harness(ctx)
