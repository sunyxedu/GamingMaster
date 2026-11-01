"""Outcome-aware predictive controller for Atari 2600 Robot Pong.

The game loop is deliberately local and frame-rate owned.  Astra chooses a
high-level return style and Jev supplies typed, short-lived supervision; neither
model is allowed to delay a paddle action.  Unlike the earlier safe-only
controllers, this harness treats a long scoreless rally as evidence that a
legal contact feature or a small off-centre return should be tried.
"""
from __future__ import annotations

import json
import math
import re
from collections import deque

from metaharness.core.contract import BudgetExceeded, EpisodeOver, ModelUnavailable
from metaharness.models.jev import Choice


# Visible-pixel geometry from the public adaptor contract.  The ball has been
# observed at 34.5..191.5; the paddle can visibly clip at the lower wall.
BALL_TOP = 34.0
BALL_BOTTOM = 193.0
PADDLE_TOP = 34.0
PADDLE_BOTTOM = 193.0
COURT_CENTER = (PADDLE_TOP + PADDLE_BOTTOM) / 2.0
RIGHT_CONTACT_X = 140.5


def _number(value, default=None):
    if isinstance(value, bool):
        return default
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return default
    return result if math.isfinite(result) else default


def _integer(value, default=None):
    number = _number(value, None)
    return default if number is None else int(number)


def _clamp(value, low, high):
    return max(low, min(high, value))


def _json_safe(value):
    """Convert a derived state to strict JSON-compatible finite values."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    number = _number(value, None)
    return number


def _safe_text(value):
    try:
        return str(value)
    except Exception:
        return ""


def _extract_json(text):
    """Accept one JSON object from arbitrary Astra prose or stand-ins."""
    text = _safe_text(text).strip()
    candidates = [text]
    candidates.extend(re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S | re.I))
    # Astra occasionally emits two adjacent flat JSON objects.  Try each
    # object before attempting the whole span, so the first valid plan wins.
    candidates.extend(re.findall(r"\{[^{}]*\}", text, re.S))
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


def _reflect_y(y, vy, frames, top=BALL_TOP, bottom=BALL_BOTTOM):
    y = _number(y, COURT_CENTER)
    vy = _number(vy, 0.0)
    frames = max(0.0, _number(frames, 0.0))
    span = bottom - top
    if span <= 0:
        return top
    phase = (y - top + vy * frames) % (2.0 * span)
    if phase <= span:
        return top + phase
    return bottom - (phase - span)


def _answer(answer, allowed):
    """Read real or pre-flight stand-in Jev answers without trusting shape."""
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
    if not isinstance(value, str) or value not in allowed:
        return None, 0.0
    if not isinstance(probabilities, dict):
        probabilities = {}
    probability = _number(probabilities.get(value), 0.0) or 0.0
    confidence = _number(confidence, None)
    if confidence is not None:
        probability = min(probability, confidence)
    return value, probability


class Harness:
    def __init__(self, ctx):
        self.ctx = ctx
        self._reset_episode()

    def _reset_episode(self):
        # Perception and short-term physics state.
        self.phase = None
        self.frame = 0
        self.last_ball_frame = None
        self.last_ball_x = None
        self.last_ball_y = None
        self.ball_vx = None
        self.ball_vy = None
        self.last_paddle_frame = None
        self.last_paddle_y = None
        self.paddle_velocity = 0.0
        self.paddle_velocity_ema = 0.0
        self.last_approach = None
        self.wall_top = BALL_TOP
        self.wall_bottom = BALL_BOTTOM

        # Reward/rally evidence.  A rally is bounded by serve_delay, not by
        # visible pixels, since the adaptor may leave a stale point sprite.
        self.own_points = 0
        self.opponent_points = 0
        self.recent_points = deque(maxlen=8)
        self.rally_number = -1
        self.rally_epoch = 0
        self.right_contacts = 0
        self.last_contact_frame = -100000
        self.feature_used = False
        self.feature_start_frame = None
        self.feature_cooldown = 0
        self.edge_sign = -1
        self.edge_choice = None

        # The policy starts with a deliberately modest attack trial.  Astra may
        # select safe mode, but malformed/late plans leave a playable policy.
        self.plan = {
            "style": "attack",
            "use_feature": True,
            "edge": "alternate",
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
        self.jev_feature_epoch = -1
        self.jev_edge_vote = None
        self.jev_edge_epoch = -1
        self.jev_request_near = False
        self.last_target = COURT_CENTER
        self.last_semantic = "HOLD"

    # ---------- model supervision ----------

    def _compact_observation(self, obs):
        if not isinstance(obs, dict):
            obs = {}
        fields = (
            "seed", "scenario_index", "mode", "difficulty", "cartridge_game",
            "variation", "left_difficulty_switch", "right_difficulty_switch",
            "left_paddle_size", "right_paddle_size", "frame", "ticks", "phase",
            "serve_waiting", "serve_countdown_frames", "ball_visible", "ball_x",
            "ball_y", "ball_vx", "ball_vy", "ball_approaching_right",
            "right_paddle_y", "right_paddle_height_px",
            "right_paddle_nominal_height_px", "left_paddle_y",
            "left_paddle_height_px", "left_paddle_nominal_height_px", "own_points",
            "opponent_points", "last_reward", "return", "done",
        )
        return {name: _json_safe(obs.get(name)) for name in fields}

    def _start_astra(self, game, obs):
        if self.astra_started or getattr(self.ctx, "system2", None) is None:
            return
        self.astra_started = True
        try:
            description = game.describe()
        except Exception:
            description = "Atari 2600 Robot Pong; this harness controls the right paddle."
        prompt = (
            "You are a slow supervisory planner for a real-time Atari 2600 Robot Pong "
            "controller. The local controller posts one legal action every 1/60-second "
            "frame and owns timing, score bookkeeping, velocity arithmetic, wall "
            "reflection, and action legality. RIGHT moves the controlled paddle upward "
            "(smaller screen y), LEFT downward. Both Robot Pong variations serve "
            "automatically; FIRE is a contact feature, never a serve action.\n\n"
            "Important diagnosis: a safe reflected-intercept controller can survive a "
            "long rally without scoring. The controller therefore begins a controlled "
            "attack trial: after alignment it may use one legal feature pulse at right "
            "contact and a small paddle-edge offset. Prefer this trial when the variation "
            "is Robot Pong 1 (Speed) or Robot Pong 2 (WHAMMY); retreat to safe/no-feature "
            "only after negative evidence. Do not prescribe a single-frame action.\n\n"
            "Return exactly one short JSON object with optional fields: style ('attack' "
            "or 'safe'), use_feature (boolean), edge ('upper', 'lower', 'alternate', "
            "or 'center'), target_bias_px (number -4..4), deadband_px (number 2..8), "
            "and diagnosis. Choose an attack plan explicitly when a legal feature exists; "
            "the code will gate it again at the current contact.\n\n"
            "Public game description:\n" + _safe_text(description)[:5000] +
            "\n\nInitial observation:\n" +
            json.dumps(self._compact_observation(obs), sort_keys=True)
        )

        def call():
            return self.ctx.think(prompt, timeout_s=60.0)

        try:
            self.astra_future = self.ctx.spawn(call)
        except Exception:
            self.astra_future = None

    def _start_astra_replan(self, game, obs, reason):
        if self.astra_replan_started or self.astra_future is not None:
            return
        if getattr(self.ctx, "system2", None) is None or getattr(self.ctx, "done", False):
            return
        self.astra_replan_started = True
        evidence = list(self.recent_points)[-6:]
        prompt = (
            "Revise the supervisory policy of an already-running Atari 2600 Robot Pong "
            "controller. The frame controller is fresh every tick and its reflected "
            "intercept arithmetic is code-owned. Decide only whether to keep the "
            "controlled attack trial (one Speed/WHAMMY pulse and a small edge return) or "
            "retreat to safe center returns. A scoreless long rally is evidence against "
            "safe-only play; a negative reward immediately after a feature/edge contact "
            "is evidence for a temporary retreat. Return one JSON object only with style, "
            "use_feature, edge, target_bias_px (-4..4), deadband_px (2..8), diagnosis. "
            "Never return a one-frame action.\nReason: " + _safe_text(reason)[:200] +
            "\nCurrent observation:\n" +
            json.dumps(self._compact_observation(obs), sort_keys=True) +
            "\nRecent point/contact evidence:\n" + json.dumps(_json_safe(evidence), sort_keys=True) +
            "\nCurrent local policy:\n" + json.dumps(self.plan, sort_keys=True)
        )

        def call():
            return self.ctx.think(prompt, session_id=self.astra_session, timeout_s=60.0)

        try:
            self.astra_future = self.ctx.spawn(call)
        except Exception:
            self.astra_future = None

    def _accept_astra(self, result):
        text = getattr(result, "text", result)
        parsed = _extract_json(text)
        # The initial call is asked to design the attack experiment.  Do not let
        # a generic safety-preferring first answer silently repeat the parents'
        # untested safe-only policy; only the evidence-triggered replan may
        # retreat.  Numeric tuning remains available immediately.
        initial = not self.astra_replan_started
        style = parsed.get("style")
        if isinstance(style, str) and style.lower() in ("safe", "attack"):
            if not initial or style.lower() == "attack":
                self.plan["style"] = style.lower()
        use_feature = parsed.get("use_feature")
        if isinstance(use_feature, bool):
            if not initial or use_feature:
                self.plan["use_feature"] = use_feature
        edge = parsed.get("edge")
        if isinstance(edge, str) and edge.lower() in ("upper", "lower", "alternate", "center"):
            if not initial or edge.lower() != "center":
                self.plan["edge"] = edge.lower()
        bias = _number(parsed.get("target_bias_px"), None)
        if bias is not None:
            self.plan["target_bias_px"] = _clamp(bias, -4.0, 4.0)
        deadband = _number(parsed.get("deadband_px"), None)
        if deadband is not None:
            self.plan["deadband_px"] = _clamp(deadband, 2.0, 8.0)
        self.plan["version"] = int(self.plan.get("version", 0)) + 1
        session = getattr(result, "session_id", None)
        if session:
            self.astra_session = session
        try:
            self.ctx.log(
                "astra_plan",
                plan=dict(self.plan),
                diagnosis=_safe_text(parsed.get("diagnosis", ""))[:300],
            )
        except Exception:
            pass

    def _poll_astra(self):
        future = self.astra_future
        if future is None or not future.done():
            return
        self.astra_future = None
        try:
            self._accept_astra(future.result())
        except Exception:
            # Astra is advisory; arbitrary stand-in prose must not stop play.
            pass

    @staticmethod
    def _direction_questions():
        up = "UP (the RIGHT action): move the controlled paddle toward smaller screen y."
        hold = "HOLD (the NOOP action): keep/brake because the supplied target is covered or state is uncertain."
        down = "DOWN (the LEFT action): move the controlled paddle toward larger screen y."
        return {
            "direction_a": Choice(
                "Choose the safe immediate direction from `control.code_action` and "
                "`control.target_relation`; do not calculate a trajectory. If uncertain, HOLD.",
                {"UP": up, "HOLD": hold, "DOWN": down},
            ),
            "direction_b": Choice(
                "Independently choose the same safe immediate direction from the supplied "
                "facts; do not calculate a trajectory. If uncertain, HOLD.",
                {"DOWN": down, "HOLD": hold, "UP": up},
            ),
            "feature_a": Choice(
                "Should the controller reserve one red-button pulse for the next predicted "
                "right-paddle contact in this rally? `control.future_contact_window` means "
                "the contact is within the supplied early window, not that contact is this "
                "instant. Choose USE_FEATURE only if the variation has a legal contact "
                "feature, the plan permits it, and the predicted paddle overlap is safe. "
                "FIRE is never for an automatic serve.",
                {
                    "USE_FEATURE": "Reserve/use one pulse at the imminent contact; this is a legal Speed or WHAMMY trial.",
                    "NO_FEATURE": "Do not use the button; make an ordinary safe return.",
                },
            ),
            "feature_b": Choice(
                "Independently judge the supplied future contact authorization. Do not wait "
                "for a current NOW label: the code will gate the actual pulse again.",
                {
                    "NO_FEATURE": "Do not use the button on this rally.",
                    "USE_FEATURE": "Reserve one legal button pulse for a safe right-paddle contact.",
                },
            ),
            "edge_a": Choice(
                "Choose the return shape for the next contact from the explicit `control` "
                "facts. Use a small off-centre edge only when alignment remains safe; do "
                "not invent numeric offsets.",
                {
                    "UPPER": "Favor the upper portion of the controlled paddle for a steeper upward return.",
                    "CENTER": "Use a centered defensive return.",
                    "LOWER": "Favor the lower portion of the controlled paddle for a steeper downward return.",
                },
            ),
            "edge_b": Choice(
                "Independently choose a safe/attacking return shape using only the supplied "
                "candidate descriptions and plan.",
                {
                    "LOWER": "Favor the lower portion for a downward return.",
                    "CENTER": "Use the center for a defensive return.",
                    "UPPER": "Favor the upper portion for an upward return.",
                },
            ),
        }

    def _jev_state(self, obs, control):
        compact = self._compact_observation(obs)
        return _json_safe({
            "game": "Atari 2600 Robot Pong; this harness controls the right paddle",
            "rules": {
                "right_action": "RIGHT is screen-y up; LEFT is screen-y down; NOOP brakes/holds",
                "serve": "automatic; FIRE is never needed to start",
                "feature": "Robot Pong 1 offers Speed and Robot Pong 2 offers WHAMMY at contact",
                "goal": "return the ball, then test a safe attack if a rally is scoreless",
            },
            "scenario": {
                "mode": compact.get("mode"),
                "difficulty": compact.get("difficulty"),
                "variation": compact.get("variation"),
                "right_paddle_size": compact.get("right_paddle_size"),
                "right_nominal_height_px": compact.get("right_paddle_nominal_height_px"),
            },
            "observation": {
                "frame": compact.get("frame"),
                "phase": compact.get("phase"),
                "serve_waiting": compact.get("serve_waiting"),
                "countdown": compact.get("serve_countdown_frames"),
            },
            "score": {
                "own": compact.get("own_points"),
                "opponent": compact.get("opponent_points"),
                "last_reward": compact.get("last_reward"),
            },
            "ball": control["ball"],
            "right_paddle": control["paddle"],
            "left_paddle": {
                "y": compact.get("left_paddle_y"),
                "size": compact.get("left_paddle_size"),
            },
            "control": {
                "predicted_contact_y": control["predicted_y"],
                "target_relation": control["target_relation"],
                "code_action": control["code_action"],
                "future_contact_window": bool(
                    control["frames_to_contact"] is not None
                    and control["frames_to_contact"] <= 70.0
                ),
                "contact_now": bool(control["contact_window"]),
                "feature_legal": control["feature_legal"],
                "feature_already_used": self.feature_used,
                "feature_cooldown": self.feature_cooldown,
                "attack_stage": "edge_trial" if self.right_contacts >= 1 else "safe_first_return",
                "planned_edge": self.plan.get("edge", "alternate"),
                "edge_vote": self.jev_edge_vote,
                "uncertain": not control["ball"]["reliable"],
            },
            "plan": {
                "style": self.plan.get("style"),
                "use_feature": bool(self.plan.get("use_feature")),
                "edge": self.plan.get("edge"),
                "deadband_px": self.plan.get("deadband_px"),
            },
            "recent_points": list(self.recent_points)[-4:],
        })

    def _ask_jev(self, state, questions):
        try:
            return self.ctx.decide(state, questions, timeout_s=1.0)
        except (TimeoutError, RuntimeError, BudgetExceeded, EpisodeOver, ModelUnavailable):
            return None
        except Exception:
            return None

    def _launch_jev(self, state, frame, epoch):
        if self.jev_future is not None:
            return
        if getattr(self.ctx, "system1", None) is None:
            return
        questions = self._direction_questions()
        try:
            self.jev_future = self.ctx.spawn(
                lambda state=state, questions=questions: self._ask_jev(state, questions)
            )
            self.next_jev_frame = frame + 24
            self.jev_request_epoch = epoch
            self.jev_request_frame = frame
            self.jev_request_contact = self.right_contacts
            self.jev_request_plan = self.plan["version"]
            self.jev_request_near = bool(
                state.get("control", {}).get("feature_legal")
                and state.get("control", {}).get("future_contact_window")
                and state.get("control", {}).get("attack_stage") == "edge_trial"
            )
        except Exception:
            self.jev_future = None
            self.next_jev_frame = frame + 36

    def _poll_jev(self, current_frame):
        future = self.jev_future
        if future is None or not future.done():
            return
        self.jev_future = None
        if (getattr(self, "jev_request_epoch", -1) != self.rally_epoch
                or getattr(self, "jev_request_contact", -1) != self.right_contacts
                or current_frame - getattr(self, "jev_request_frame", -10000) > 60
                or getattr(self, "jev_request_plan", -1) != self.plan["version"]):
            return
        try:
            answers = future.result()
        except Exception:
            return
        if not isinstance(answers, dict):
            return

        av, ap = _answer(answers.get("direction_a"), {"UP", "DOWN", "HOLD"})
        bv, bp = _answer(answers.get("direction_b"), {"UP", "DOWN", "HOLD"})
        if av is not None and av == bv and min(ap, bp) >= 0.58:
            self.jev_hint = {
                "value": av,
                "expires": getattr(self, "jev_request_frame", current_frame) + 15,
                "epoch": getattr(self, "jev_request_epoch", self.rally_epoch),
            }

        fa, fap = _answer(
            answers.get("feature_a"), {"USE_FEATURE", "NO_FEATURE"}
        )
        fb, fbp = _answer(
            answers.get("feature_b"), {"USE_FEATURE", "NO_FEATURE"}
        )
        if fa is not None and fa == fb and min(fap, fbp) >= 0.58:
            # Early serve/trajectory judgments are usually NO by definition;
            # they must not veto the later contact experiment.  A USE answer
            # can authorize a future pulse, while NO is a veto only when the
            # request actually described a relevant edge-contact opportunity.
            if fa == "USE_FEATURE" or self.jev_request_near:
                self.jev_feature_vote = fa == "USE_FEATURE"
                self.jev_feature_epoch = getattr(self, "jev_request_epoch", self.rally_epoch)

        ea, eap = _answer(answers.get("edge_a"), {"UPPER", "CENTER", "LOWER"})
        eb, ebp = _answer(answers.get("edge_b"), {"UPPER", "CENTER", "LOWER"})
        if ea is not None and ea == eb and min(eap, ebp) >= 0.55:
            self.jev_edge_vote = ea
            self.jev_edge_epoch = getattr(self, "jev_request_epoch", self.rally_epoch)

        try:
            self.ctx.log(
                "jev_supervision",
                direction=av if av == bv else "disagree",
                direction_probability=min(ap, bp),
                feature=fa if fa == fb else "disagree",
                feature_probability=min(fap, fbp),
                edge=ea if ea == eb else "disagree",
                edge_probability=min(eap, ebp),
            )
        except Exception:
            pass

    # ---------- local perception and predictive control ----------

    def _new_rally(self):
        self.rally_number += 1
        self.rally_epoch += 1
        self.right_contacts = 0
        self.feature_used = False
        self.jev_feature_vote = False
        self.feature_start_frame = None
        self.jev_feature_epoch = self.rally_epoch
        self.jev_edge_vote = None
        self.jev_edge_epoch = self.rally_epoch
        self.edge_sign = -1 if self.rally_number % 2 == 0 else 1
        self.last_approach = None
        self.last_ball_frame = None
        self.last_ball_x = None
        self.last_ball_y = None
        self.ball_vx = None
        self.ball_vy = None

    def _ingest(self, obs):
        if not isinstance(obs, dict):
            obs = {}
        frame = _integer(obs.get("frame"), None)
        if frame is None:
            frame = _integer(obs.get("ticks"), self.frame) or self.frame
        phase = obs.get("phase") if isinstance(obs.get("phase"), str) else "unknown"

        # Process reward counters before resetting rally-local feature state.
        own = _integer(obs.get("own_points"), self.own_points)
        opponent = _integer(obs.get("opponent_points"), self.opponent_points)
        if own != self.own_points or opponent != self.opponent_points:
            reward = _integer(obs.get("last_reward"), 0) or 0
            self.recent_points.append({
                "frame": frame,
                "reward": reward,
                "own_points": own,
                "opponent_points": opponent,
                "feature_used": bool(self.feature_used),
                "right_contacts": self.right_contacts,
                "right_y": _number(obs.get("right_paddle_y"), self.last_paddle_y),
                "target_y": self.last_target,
            })
            # A feature immediately preceding a loss gets one conservative
            # rally; a successful or scoreless trial is not punished.
            if reward < 0 and self.feature_used:
                self.feature_cooldown = 1
            else:
                self.feature_cooldown = max(0, self.feature_cooldown - 1)
        self.own_points, self.opponent_points = own, opponent

        if phase == "serve_delay" and self.phase != "serve_delay":
            self._new_rally()
        elif phase == "finished":
            self.last_ball_frame = None
            self.last_ball_x = None
            self.last_ball_y = None
            self.ball_vx = None
            self.ball_vy = None
            self.last_approach = None

        paddle_y = _number(obs.get("right_paddle_y"), None)
        if paddle_y is not None:
            if self.last_paddle_y is not None and self.last_paddle_frame is not None:
                gap = max(1, frame - self.last_paddle_frame)
                velocity = (paddle_y - self.last_paddle_y) / float(gap)
                velocity = _clamp(velocity, -6.0, 6.0)
                self.paddle_velocity = velocity
                self.paddle_velocity_ema = 0.55 * self.paddle_velocity_ema + 0.45 * velocity
            self.last_paddle_y = paddle_y
            self.last_paddle_frame = frame

        rally_like = phase in ("rally", "rally_ball_occluded")
        x = _number(obs.get("ball_x"), None)
        y = _number(obs.get("ball_y"), None)
        visible = bool(obs.get("ball_visible")) and x is not None and y is not None
        if visible and rally_like:
            self.wall_top = min(self.wall_top, y)
            self.wall_bottom = max(self.wall_bottom, y)
            if self.last_ball_x is not None and self.last_ball_frame is not None:
                gap = frame - self.last_ball_frame
                if 0 < gap <= 10:
                    dx = (x - self.last_ball_x) / float(gap)
                    dy = (y - self.last_ball_y) / float(gap)
                    if abs(dx) >= 0.25 and abs(dx) <= 3.0:
                        self.ball_vx = dx
                    if abs(dy) <= 4.0:
                        self.ball_vy = dy
            field_vx = _number(obs.get("ball_vx"), None)
            field_vy = _number(obs.get("ball_vy"), None)
            if field_vx is not None and abs(field_vx) >= 0.25:
                self.ball_vx = field_vx
            if field_vy is not None and abs(field_vy) <= 4.0:
                self.ball_vy = field_vy
            self.last_ball_frame = frame
            self.last_ball_x = x
            self.last_ball_y = y

        # A horizontal sign change is a verified contact only when the adaptor
        # explicitly says the ball is no longer approaching right.  Unknown
        # occlusion does not manufacture a contact.
        approach_field = obs.get("ball_approaching_right")
        approach = None
        if approach_field is True:
            approach = True
        elif approach_field is False:
            approach = False
        elif self.ball_vx is not None:
            approach = self.ball_vx > 0
        if rally_like and self.last_approach is True and approach is False:
            if frame - self.last_contact_frame > 4:
                self.right_contacts += 1
                self.last_contact_frame = frame
                self.jev_feature_vote = False
                self.jev_edge_vote = None
                self.edge_sign = -self.edge_sign
        if phase == "serve_delay" or phase == "finished":
            approach = None
        if approach is not None:
            self.last_approach = approach

        self.phase = phase
        self.frame = frame

    def _trajectory(self, obs):
        if not isinstance(obs, dict):
            obs = {}
        phase = obs.get("phase") if isinstance(obs.get("phase"), str) else "unknown"
        rally_like = phase in ("rally", "rally_ball_occluded")
        paddle_y = _number(obs.get("right_paddle_y"), self.last_paddle_y)
        if paddle_y is None:
            paddle_y = COURT_CENTER

        x = _number(obs.get("ball_x"), None)
        y = _number(obs.get("ball_y"), None)
        visible = bool(obs.get("ball_visible")) and x is not None and y is not None
        gap = 100
        if visible:
            bx, by, gap = x, y, 0
        elif self.last_ball_frame is not None and rally_like:
            gap = max(0, self.frame - self.last_ball_frame)
            if gap <= 14:
                bx = self.last_ball_x + (self.ball_vx or 0.0) * gap
                by = _reflect_y(
                    self.last_ball_y,
                    self.ball_vy or 0.0,
                    gap,
                    self.wall_top,
                    self.wall_bottom,
                )
            else:
                bx = by = None
        else:
            bx = by = None

        field_approach = obs.get("ball_approaching_right")
        if field_approach is True:
            approach = "RIGHT"
        elif field_approach is False:
            approach = "AWAY"
        elif self.ball_vx is not None and self.ball_vx > 0:
            approach = "RIGHT"
        elif self.ball_vx is not None and self.ball_vx < 0:
            approach = "AWAY"
        else:
            approach = "UNKNOWN"

        vx = self.ball_vx
        vy = self.ball_vy
        reliable = (
            rally_like and bx is not None and by is not None and approach == "RIGHT"
            and vx is not None and abs(vx) >= 0.25
            and vy is not None and gap <= 14
        )
        frames_to_contact = None
        predicted_center = COURT_CENTER
        if reliable:
            speed_x = max(0.5, abs(vx))
            frames_to_contact = max(0.0, (RIGHT_CONTACT_X - bx) / speed_x)
            # Correct the outward sign of a wall sample before reflection.
            projected_vy = vy
            if by <= self.wall_top + 2.0 and projected_vy < 0:
                projected_vy = abs(projected_vy)
            if by >= self.wall_bottom - 2.0 and projected_vy > 0:
                projected_vy = -abs(projected_vy)
            predicted_center = _reflect_y(
                by, projected_vy, frames_to_contact, self.wall_top, self.wall_bottom
            )
            predicted_center += _number(self.plan.get("target_bias_px"), 0.0) or 0.0
            predicted_center = _clamp(predicted_center, PADDLE_TOP, PADDLE_BOTTOM)

        feature_legal = _safe_text(obs.get("variation", "")) in (
            "Robot Pong 1", "Robot Pong 2"
        )
        # Edge targeting is intentionally small.  It changes the return angle
        # without asking a half-size paddle to miss its own collision.
        edge_offset = 0.0
        edge_name = self._edge_name()
        if (
            reliable and self.plan.get("style") == "attack"
            and self.right_contacts >= 1 and edge_name != "CENTER"
        ):
            nominal = _number(obs.get("right_paddle_nominal_height_px"), 16.0) or 16.0
            edge_offset = min(4.0 if nominal > 8 else 1.5, max(0.0, nominal / 2.0 - 2.0))
            if edge_name == "UPPER":
                predicted_center += edge_offset
            elif edge_name == "LOWER":
                predicted_center -= edge_offset
            predicted_center = _clamp(predicted_center, PADDLE_TOP, PADDLE_BOTTOM)

        nominal = _number(obs.get("right_paddle_nominal_height_px"), 16.0) or 16.0
        deadband = _number(self.plan.get("deadband_px"), 4.0) or 4.0
        deadband = deadband * (0.72 if nominal <= 8 else 1.0)
        deadband = _clamp(deadband, 2.0, 8.0)
        if not reliable:
            predicted_center = COURT_CENTER
            deadband = max(deadband, 6.0)

        error = predicted_center - paddle_y
        velocity = self.paddle_velocity_ema
        # NOOP is a brake in this ROM: use measured paddle motion to stop before
        # a target crossing instead of reversing every time a sample jitters.
        stopping_margin = max(deadband, abs(velocity) * 2.0 + 1.5)
        if abs(error) <= deadband:
            semantic = "HOLD"
        elif error < 0:
            semantic = "HOLD" if velocity < -0.35 and abs(error) <= stopping_margin else "UP"
        else:
            semantic = "HOLD" if velocity > 0.35 and abs(error) <= stopping_margin else "DOWN"

        contact_window = bool(
            reliable and frames_to_contact is not None and frames_to_contact <= 4.0
        )
        relation = "UNKNOWN"
        if reliable:
            if error < -deadband:
                relation = "ABOVE"
            elif error > deadband:
                relation = "BELOW"
            else:
                relation = "ALIGNED"

        return {
            "ball": {
                "visible": visible,
                "x": bx,
                "y": by,
                "vx": vx,
                "vy": vy,
                "approach": approach,
                "reliable": reliable,
                "visibility_gap": gap,
            },
            "paddle": {"y": paddle_y, "velocity": velocity},
            "predicted_y": predicted_center,
            "frames_to_contact": frames_to_contact,
            "contact_window": contact_window,
            "target_relation": relation,
            "code_action": semantic,
            "feature_legal": feature_legal,
            "edge_name": edge_name,
            "edge_offset": edge_offset,
            "phase": phase,
        }

    def _edge_name(self):
        vote = self.jev_edge_vote if self.jev_edge_epoch == self.rally_epoch else None
        if vote in ("UPPER", "LOWER", "CENTER"):
            return vote
        planned = _safe_text(self.plan.get("edge", "alternate")).upper()
        if planned == "UPPER":
            return "UPPER"
        if planned == "LOWER":
            return "LOWER"
        if planned == "CENTER":
            return "CENTER"
        return "UPPER" if self.edge_sign < 0 else "LOWER"

    def _apply_hint(self, control):
        hint = self.jev_hint
        if hint is not None:
            if self.frame > hint.get("expires", -1) or hint.get("epoch") != self.rally_epoch:
                self.jev_hint = None
                hint = None
        semantic = control["code_action"]
        # A typed hint is only a fallback for uncertain perception.  A reliable
        # reflected projection is more exact than a 0.2 s old classification.
        if hint is not None and not control["ball"]["reliable"] and control["phase"] in (
            "rally", "rally_ball_occluded"
        ):
            if hint.get("value") in ("UP", "DOWN", "HOLD"):
                semantic = hint["value"]
        return semantic

    def _feature_allowed(self, obs, control, semantic):
        if not self.plan.get("use_feature") or not self.jev_feature_vote:
            return False
        if self.feature_cooldown > 0:
            return False
        if self.feature_used:
            return False
        # Establish one ordinary return before experimenting with the button.
        # This keeps a first-rally perception/calibration miss from being
        # mistaken for evidence against the feature.
        if self.right_contacts < 1:
            return False
        if not control["feature_legal"] or not control["contact_window"]:
            return False
        if control["phase"] not in ("rally", "rally_ball_occluded"):
            return False
        if control["ball"].get("approach") != "RIGHT":
            return False
        bx = control["ball"].get("x")
        if bx is None or not 137.5 <= bx <= 144.5:
            return False
        paddle_y = control["paddle"]["y"]
        ball_y = control["ball"].get("y")
        nominal = _number(obs.get("right_paddle_nominal_height_px"), 16.0) or 16.0
        # Recheck actual overlap at the contact window; the model may have
        # judged an older snapshot and must not authorize an obvious miss.
        if ball_y is None or abs(paddle_y - ball_y) > nominal / 2.0 + 2.0:
            return False
        return True

    def _action_name(self, game, semantic, fire=False):
        try:
            spec = game.action_spec
            names = list(getattr(spec, "names", ()) or ())
        except Exception:
            names = []
            spec = None
        by_upper = {str(name).upper(): name for name in names}
        noop = getattr(spec, "noop", None) if spec is not None else None
        if noop not in names:
            noop = by_upper.get("NOOP", names[0] if names else "NOOP")
        base_name = {"UP": "RIGHT", "DOWN": "LEFT", "HOLD": "NOOP"}.get(
            semantic, "NOOP"
        )
        if fire:
            fire_name = {"UP": "RIGHTFIRE", "DOWN": "LEFTFIRE", "HOLD": "FIRE"}.get(
                semantic, "FIRE"
            )
            if fire_name in by_upper:
                return by_upper[fire_name]
        return by_upper.get(base_name.upper(), noop)

    def _choose_action(self, game, obs, control):
        semantic = self._apply_hint(control)
        fire = self._feature_allowed(obs, control, semantic)
        if fire and not self.feature_used:
            self.feature_used = True
            self.feature_start_frame = self.frame
            try:
                self.ctx.log("contact_feature_window", frame=self.frame,
                             rally=self.rally_epoch, contact=self.right_contacts,
                             target=control["predicted_y"], ball=control["ball"],
                             edge=control["edge_name"])
            except Exception:
                pass
        self.last_target = control["predicted_y"]
        self.last_semantic = semantic
        return self._action_name(game, semantic, fire=fire)

    def _maybe_start_jev(self, obs, control):
        if self.jev_future is not None or self.frame < self.next_jev_frame:
            return
        if getattr(self.ctx, "system1", None) is None:
            return
        frames = control.get("frames_to_contact")
        near = frames is not None and 18.0 <= frames <= 70.0
        if not near:
            self.next_jev_frame = self.frame + 8
            return
        self._launch_jev(self._jev_state(obs, control), self.frame, self.rally_epoch)

    def _maybe_replan(self, game, obs):
        # One slow diagnosis after enough outcomes, or after a prolonged
        # scoreless rally where the attack trial has actually been exercised.
        if self.astra_replan_started:
            return
        if len(self.recent_points) >= 3:
            self._start_astra_replan(game, obs, "three point outcomes observed")
        elif self.right_contacts >= 6 and not self.recent_points:
            self._start_astra_replan(game, obs, "six right returns with no score event")

    def run_episode(self, game) -> None:
        self._reset_episode()
        ctx = self.ctx
        if getattr(ctx, "done", False):
            return
        try:
            first = game.observe()
        except Exception:
            first = {}
        self._start_astra(game, first)

        while not getattr(ctx, "done", False) and not getattr(game, "done", False):
            obs = game.observe()
            if not isinstance(obs, dict):
                obs = {}
            self._ingest(obs)
            self._poll_astra()
            self._poll_jev(self.frame)
            control = self._trajectory(obs)
            self._maybe_replan(game, obs)
            self._maybe_start_jev(obs, control)
            action = self._choose_action(game, obs, control)
            try:
                # Post before waiting: model latency cannot make this tick stale.
                ctx.act(action)
            except (BudgetExceeded, EpisodeOver, ModelUnavailable):
                return
            try:
                ctx.wait_for_tick()
            except (BudgetExceeded, EpisodeOver, ModelUnavailable):
                return


def build(ctx):
    return Harness(ctx)
