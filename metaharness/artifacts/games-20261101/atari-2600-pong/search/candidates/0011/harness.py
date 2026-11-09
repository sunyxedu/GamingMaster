"""Adaptive trajectory/return controller for Atari 2600 Robot Pong.

This candidate keeps the collision loop deterministic, but gives the two model
systems a different job from the parent: Astra chooses a *rally policy and
experiment schedule*, while Jev recognizes whether that policy is safe for the
current contact.  Code owns the reflected trajectory, paddle-band geometry,
actuator settling, score bookkeeping, and every frame-level action.
"""
from __future__ import annotations

import json
import math
import re
from collections import deque

from metaharness.core.contract import BudgetExceeded, EpisodeOver, ModelUnavailable
from metaharness.models.jev import Choice


# Visible sprite-center geometry from the public adaptor contract.
COURT_TOP = 34.0
COURT_BOTTOM = 193.0
BALL_TOP = 37.5
BALL_BOTTOM = 191.5
COURT_CENTER = (COURT_TOP + COURT_BOTTOM) / 2.0
RIGHT_CONTACT_X = 140.5


def _num(value, default=None):
    if isinstance(value, bool):
        return default
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return default
    return result if math.isfinite(result) else default


def _int(value, default=0):
    n = _num(value, None)
    return default if n is None else int(n)


def _sign(value):
    n = _num(value, None)
    if n is None or n == 0:
        return None
    return 1 if n > 0 else -1


def _clamp(value, low, high):
    return max(low, min(high, value))


def _json_safe(value):
    """Convert a state to strict JSON without allowing model garbage to escape."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    n = _num(value, None)
    return n


def _text(value):
    try:
        return str(value)
    except Exception:
        return ""


def _reflect(y, velocity, frames):
    """Mirror a ball y trajectory between the visible ball extrema."""
    y = _num(y, COURT_CENTER)
    velocity = _num(velocity, 0.0)
    frames = max(0.0, _num(frames, 0.0))
    span = BALL_BOTTOM - BALL_TOP
    if span <= 0:
        return COURT_CENTER
    phase = (y - BALL_TOP + velocity * frames) % (2.0 * span)
    if phase <= span:
        return BALL_TOP + phase
    return BALL_BOTTOM - (phase - span)


def _extract_json(text):
    """Best-effort extraction for Astra's short JSON, including fenced output."""
    raw = _text(text).strip()
    if not raw:
        return {}
    candidates = [raw]
    candidates.extend(re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.I | re.S))
    # Astra can return the same short object more than once in one response.
    # Try each flat object before the whole span; concatenated JSON is not one
    # valid document, but its first object is still a usable policy.
    candidates.extend(re.findall(r"\{[^{}]*\}", raw, re.S))
    first, last = raw.find("{"), raw.rfind("}")
    if first >= 0 and last > first:
        candidates.append(raw[first:last + 1])
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except Exception:
            continue
        if isinstance(parsed, dict):
            return parsed
    return {}


def _answer(answer, allowed):
    """Read a typed answer defensively, also tolerating pre-flight stand-ins."""
    value = getattr(answer, "value", None)
    if value not in allowed and isinstance(answer, dict):
        value = answer.get("value", answer.get("choice"))
    if value not in allowed:
        return None, 0.0
    probabilities = getattr(answer, "probabilities", None)
    if probabilities is None and isinstance(answer, dict):
        probabilities = answer.get("probabilities")
    if not isinstance(probabilities, dict):
        probabilities = {}
    p = _num(probabilities.get(value), 0.0) or 0.0
    confidence_value = getattr(answer, "confidence", None)
    if confidence_value is None and isinstance(answer, dict):
        confidence_value = answer.get("confidence")
    confidence = _num(confidence_value, None)
    if confidence is not None:
        p = min(p, confidence)
    return value, max(0.0, min(1.0, p))


class Harness:
    """One episode controller; all mutable state is intentionally per episode."""

    def __init__(self, ctx):
        self.ctx = ctx
        self.game = None
        self.names = []
        self.noop = "NOOP"
        self.up = "RIGHT"          # calibrated public mapping: screen-y up
        self.down = "LEFT"         # calibrated public mapping: screen-y down
        self.up_fire = "RIGHTFIRE"
        self.down_fire = "LEFTFIRE"
        self.fire = "FIRE"
        self._reset_state()

    def _reset_state(self):
        # Observation/trajectory memory.
        self.last_frame = None
        self.phase = None
        self.last_ball = None       # x, y, frame
        self.vx = None
        self.vy = None
        self.last_approach = None
        self.last_paddle_y = None
        self.last_paddle_frame = None
        self.paddle_velocity = 0.0
        self.last_target = COURT_CENTER

        # Point/rally evidence.  A point is a hard trajectory boundary.
        self.own_points = 0
        self.opponent_points = 0
        self.last_score = None
        self.rally_id = 0
        self.completed_right_contacts = 0
        self.contact_latched = False
        self.contact_was_attack = False
        self.post_point_center = 2
        self.last_point_frame = 0
        self.evidence_contacts = 0
        self.evidence_points = 0
        self.recent_evidence = deque(maxlen=12)

        # Low-level servo latch.
        self.last_direction = "HOLD"
        self.direction_age = 0
        self.last_action = self.noop

        # Astra's macro policy.  This fallback deliberately tests a mixed return
        # schedule rather than repeating the parent's permanently safe center.
        self.plan = {
            "mode": "MIXED",             # CENTER, MIXED, EDGE
            "edge_sign": "ALTERNATE",   # UP, DOWN, ALTERNATE
            "attack_every": 3,
            "bias_fraction": 0.28,
            "use_feature": False,
            "brake_frames": 2,
            "warmup_contacts": 2,
            "force_after_contacts": 8,
            "version": 0,
        }
        self.attack_sign = 1
        self.feature_hint = False
        self.feature_used = False

        # Model task state.
        self.astra_future = None
        self.astra_session = None
        self.astra_started = False
        self.astra_replans = 0
        self.astra_replan_pending = False
        self.jev_future = None
        self.jev_meta = None
        self.next_jev_frame = 0
        self.jev_style = None
        self.jev_style_rally = None
        self.jev_style_contact = None
        self.jev_feature = False
        self.jev_feature_rally = None
        self.jev_feature_contact = None

    # ---------- action-space bridge ----------

    def _prepare_actions(self, game):
        spec = getattr(game, "action_spec", None)
        raw = getattr(spec, "names", ()) if spec is not None else ()
        try:
            names = list(raw.keys()) if isinstance(raw, dict) else list(raw)
        except Exception:
            names = []
        self.names = [str(n) for n in names] or [
            "NOOP", "FIRE", "RIGHT", "LEFT", "RIGHTFIRE", "LEFTFIRE"
        ]
        declared = getattr(spec, "noop", None) if spec is not None else None
        self.noop = declared if declared in self.names else self._name("NOOP", self.names[0])
        self.up = self._name("RIGHT", self.noop)
        self.down = self._name("LEFT", self.noop)
        self.fire = self._name("FIRE", self.noop)
        self.up_fire = self._name("RIGHTFIRE", self.up)
        self.down_fire = self._name("LEFTFIRE", self.down)
        self.last_action = self.noop

    def _name(self, name, fallback):
        return name if name in self.names else fallback

    def _fix_action_names(self):
        # Keep this separate so unusual action_spec implementations cannot make
        # pre-flight fail while retaining the public six-name mapping.
        self.noop = self._name("NOOP", self.noop)
        self.up = self._name("RIGHT", self.noop)
        self.down = self._name("LEFT", self.noop)
        self.fire = self._name("FIRE", self.noop)
        self.up_fire = self._name("RIGHTFIRE", self.up)
        self.down_fire = self._name("LEFTFIRE", self.down)

    # ---------- observations and exact local bookkeeping ----------

    def _frame(self, obs):
        n = _num(obs.get("frame"), None)
        if n is None:
            n = _num(obs.get("ticks"), self.last_frame if self.last_frame is not None else 0)
        return int(n or 0)

    def _approach(self, obs):
        value = obs.get("ball_approaching_right")
        if isinstance(value, bool):
            return value
        sign = _sign(obs.get("ball_vx_sign"))
        if sign is None:
            sign = _sign(obs.get("ball_vx"))
        if sign is None:
            sign = self.vx
        return None if sign is None else sign > 0

    def _reset_trajectory(self):
        self.last_ball = None
        self.vx = None
        self.vy = None
        self.last_approach = None
        self.contact_latched = False

    def _handle_score(self, obs, frame):
        own = _int(obs.get("own_points"), self.own_points)
        opp = _int(obs.get("opponent_points"), self.opponent_points)
        if self.last_score is None:
            self.last_score = (own, opp)
        if (own, opp) != self.last_score:
            old_own, old_opp = self.last_score
            d_own = max(0, own - old_own)
            d_opp = max(0, opp - old_opp)
            reward = _int(obs.get("last_reward"), 0)
            event = {
                "frame": frame,
                "reward": reward,
                "own": own,
                "opponent": opp,
                "contacts": self.completed_right_contacts,
            }
            self.recent_evidence.append(event)
            self.evidence_points += d_own + d_opp
            self.post_point_center = 1 if d_own else 2
            self.feature_used = False
            self.completed_right_contacts = 0
            self.contact_latched = False
            self.contact_was_attack = False
            self.evidence_contacts = 0
            # A successful return earns permission to continue an attack; a miss
            # buys a short center reset before the mixed schedule resumes.
            if d_own:
                self.attack_sign *= -1
            self.last_point_frame = frame
            try:
                self.ctx.log(
                    "point_boundary",
                    own_points=own,
                    opponent_points=opp,
                    reward=reward,
                    contacts=event["contacts"],
                )
            except Exception:
                pass
            self.last_score = (own, opp)
        self.own_points, self.opponent_points = own, opp

    def _ingest(self, obs):
        if not isinstance(obs, dict):
            obs = {}
        frame = self._frame(obs)
        phase = obs.get("phase") if isinstance(obs.get("phase"), str) else "unknown"
        self._handle_score(obs, frame)

        if phase in ("serve_delay", "finished"):
            self._reset_trajectory()
            self.feature_used = False
        elif self.phase not in ("rally", "rally_ball_occluded"):
            # A stale ball on the first point frame cannot seed velocity.
            self._reset_trajectory()
            self.rally_id += 1
            self.completed_right_contacts = 0
            self.contact_latched = False
            self.contact_was_attack = False
            self.post_point_center = max(self.post_point_center, 1)

        x = _num(obs.get("ball_x"), None)
        y = _num(obs.get("ball_y"), None)
        visible = bool(obs.get("ball_visible")) and x is not None and y is not None
        if phase in ("rally", "rally_ball_occluded") and visible:
            prior = self.last_ball
            vx = _sign(obs.get("ball_vx_sign"))
            vy = _sign(obs.get("ball_vy_sign"))
            raw_vx = _num(obs.get("ball_vx"), None)
            raw_vy = _num(obs.get("ball_vy"), None)
            if vx is None and prior is not None:
                dt = frame - prior[2]
                if 0 < dt <= 8 and abs(x - prior[0]) >= 0.25:
                    vx = 1 if x > prior[0] else -1
            if vy is None and prior is not None:
                dt = frame - prior[2]
                if 0 < dt <= 8 and abs(y - prior[1]) >= 0.25:
                    vy = 1 if y > prior[1] else -1
            if vx is not None:
                self.vx = vx
            elif raw_vx is not None and abs(raw_vx) >= 0.25:
                self.vx = raw_vx
            if vy is not None:
                self.vy = vy
            elif raw_vy is not None and abs(raw_vy) >= 0.25:
                self.vy = raw_vy

            approach = self._approach(obs)
            # A contact is counted only after the ball has visibly left the
            # right-side window.  This prevents one paddle hit from being counted
            # repeatedly while the sprite is sheared/occluded.
            if self.contact_latched and (approach is False or x < RIGHT_CONTACT_X - 8.0):
                self.completed_right_contacts += 1
                self.evidence_contacts += 1
                if self.post_point_center > 0:
                    self.post_point_center -= 1
                if self.contact_was_attack and self.plan.get("edge_sign") == "ALTERNATE":
                    self.attack_sign *= -1
                self.contact_latched = False
                self.contact_was_attack = False
            if approach is True and x >= RIGHT_CONTACT_X - 5.0 and not self.contact_latched:
                self.contact_latched = True
                self.contact_was_attack = abs(self._attack_bias(obs)) >= 0.5
            self.last_approach = approach
            self.last_ball = (x, y, frame)
        elif phase == "rally_ball_occluded" and self.last_ball is not None:
            # Do not invent a new velocity; the controller may use the last
            # trajectory for a short gap, then retreats to center.
            pass

        paddle = _num(obs.get("right_paddle_y"), None)
        if paddle is not None:
            if self.last_paddle_y is not None and self.last_paddle_frame is not None:
                dt = max(1, frame - self.last_paddle_frame)
                self.paddle_velocity = _clamp(
                    (paddle - self.last_paddle_y) / float(dt), -4.0, 4.0
                )
            self.last_paddle_y = paddle
            self.last_paddle_frame = frame
        self.phase = phase
        self.last_frame = frame
        return obs, frame, phase, x, y, visible

    def _ball_for_control(self, obs, frame, x, y, visible):
        vx = self.vx
        vy = self.vy
        if visible:
            return x, y, vx, vy, 0
        if self.last_ball is None:
            return None
        age = frame - self.last_ball[2]
        if age < 0 or age > 12:
            return None
        lx, ly, _, = self.last_ball
        return lx + (vx or 0.0) * age, _reflect(ly, vy or 0.0, age), vx, vy, age

    def _attack_bias(self, obs):
        """Select an edge return only at the strategic contact, not every frame."""
        if self.post_point_center > 0:
            return 0.0
        mode = self.plan.get("mode", "MIXED")
        count = self.completed_right_contacts
        warmup = max(0, _int(self.plan.get("warmup_contacts"), 2))
        if count < warmup:
            return 0.0
        every = max(1, _int(self.plan.get("attack_every"), 3))
        force_after = max(4, _int(self.plan.get("force_after_contacts"), 8))
        forced = count >= force_after
        if mode == "CENTER" and not forced:
            return 0.0
        if mode == "MIXED" and not forced and ((count + 1) % every != 0):
            return 0.0
        nominal = _num(obs.get("right_paddle_nominal_height_px"), None)
        if nominal is None:
            nominal = _num(obs.get("right_paddle_height_px"), 16.0) or 16.0
        fraction = _clamp(_num(self.plan.get("bias_fraction"), 0.28) or 0.28, 0.10, 0.42)
        magnitude = _clamp(nominal * fraction, 1.5, 5.0)
        sign_mode = self.plan.get("edge_sign", "ALTERNATE")
        if sign_mode == "UP":
            sign = -1
        elif sign_mode == "DOWN":
            sign = 1
        else:
            sign = self.attack_sign
        # A fresh, confident Jev edge recognition may choose the upper/lower
        # member of the already code-approved attack.  A CENTER answer never
        # disables the safe code fallback, so option-order bias cannot restore
        # the parent's endless center policy.
        if self.jev_style_rally == self.rally_id and self.jev_style_contact == count:
            if self.jev_style == "EDGE_UP":
                sign = -1
            elif self.jev_style == "EDGE_DOWN":
                sign = 1
        return sign * magnitude

    def _trajectory(self, obs, frame, phase, x, y, visible):
        ball = self._ball_for_control(obs, frame, x, y, visible)
        target = COURT_CENTER
        eta = None
        approach = self._approach(obs)
        reliable = False
        if ball is not None:
            bx, by, vx, vy, gap = ball
            if approach is None and vx is not None:
                approach = vx > 0
            if approach is True and bx is not None and by is not None:
                speed_x = abs(_num(obs.get("ball_vx"), None) or 0.0)
                if speed_x < 0.5:
                    speed_x = abs(_num(vx, None) or 1.0)
                speed_x = max(0.5, speed_x)
                eta = max(0.0, (RIGHT_CONTACT_X - bx) / speed_x)
                vertical = _num(obs.get("ball_vy"), None)
                if vertical is None:
                    vertical = vy if vy is not None else 0.0
                target = _reflect(by, vertical, eta) + self._attack_bias(obs)
                target = _clamp(target, BALL_TOP, BALL_BOTTOM)
                reliable = vy is not None and gap <= 12
        if phase not in ("rally", "rally_ball_occluded") or approach is not True:
            target = COURT_CENTER
            eta = None
        elif not visible and ball is None:
            target = COURT_CENTER
            eta = None
            reliable = False

        paddle = self.last_paddle_y if self.last_paddle_y is not None else COURT_CENTER
        nominal = _num(obs.get("right_paddle_nominal_height_px"), 16.0) or 16.0
        visible_h = _num(obs.get("right_paddle_height_px"), nominal) or nominal
        # The visible sprite center is biased when clipped.  The visible band is
        # a safer collision fact than treating that center as a perfect actuator
        # coordinate.
        half_band = max(2.0, visible_h / 2.0)
        predicted_paddle = paddle + self.paddle_velocity * _clamp(
            _num(self.plan.get("brake_frames"), 2.0) or 2.0, 0.0, 4.0
        )
        predicted_paddle = _clamp(predicted_paddle, BALL_TOP, BALL_BOTTOM)
        top = predicted_paddle - half_band
        bottom = predicted_paddle + half_band
        contact = eta is not None and eta <= 4.0 and approach is True
        margin = 0.4 if contact else 1.0
        if target < top - margin:
            direction = "UP"
        elif target > bottom + margin:
            direction = "DOWN"
        else:
            direction = "HOLD"

        # Direction reversal is expensive because the paddle input settles for a
        # few frames.  Only reverse when the target is genuinely outside the
        # current collision band; otherwise NOOP is a braking action.
        if direction != "HOLD" and self.last_direction in ("UP", "DOWN"):
            opposite = (direction == "UP" and self.last_direction == "DOWN") or (
                direction == "DOWN" and self.last_direction == "UP"
            )
            if opposite and target >= (paddle - half_band - 2.0) and target <= (paddle + half_band + 2.0):
                direction = "HOLD"
        if direction == self.last_direction:
            self.direction_age += 1
        else:
            self.direction_age = 0
        self.last_direction = direction
        self.last_target = target
        return {
            "phase": phase,
            "target_y": target,
            "eta": eta,
            "contact_window": bool(contact),
            "approach": "RIGHT" if approach is True else "AWAY" if approach is False else "UNKNOWN",
            "reliable": bool(reliable),
            "paddle_y": paddle,
            "paddle_height": visible_h,
            "predicted_paddle_y": predicted_paddle,
            "band_top": top,
            "band_bottom": bottom,
            "direction": direction,
            "contact_count": self.completed_right_contacts,
            "bias": self._attack_bias(obs),
        }

    # ---------- Astra: slow policy author and re-planner ----------

    def _compact(self, obs):
        if not isinstance(obs, dict):
            return {}
        keys = (
            "seed", "scenario_index", "mode", "difficulty", "cartridge_game", "variation",
            "left_paddle_y", "left_paddle_height_px", "right_paddle_y",
            "right_paddle_height_px", "right_paddle_nominal_height_px", "phase",
            "serve_waiting", "serve_countdown_frames", "ball_visible", "ball_x",
            "ball_y", "ball_vx", "ball_vy", "ball_approaching_right", "frame", "ticks",
            "own_points", "opponent_points", "last_reward", "return", "done",
        )
        return _json_safe({key: obs.get(key) for key in keys})

    def _astra_prompt(self, obs, reason):
        try:
            description = _text(self.game.describe())[:4500]
        except Exception:
            description = "Atari 2600 Robot Pong; right paddle is controlled."
        policy = {k: self.plan.get(k) for k in (
            "mode", "edge_sign", "attack_every", "bias_fraction", "use_feature",
            "brake_frames", "warmup_contacts", "force_after_contacts"
        )}
        return (
            "You are the slow System-2 policy author for an already-running real-time "
            "Atari 2600 Robot Pong controller. Do not give a frame action. Code performs "
            "all score bookkeeping, ball reflection, collision timing, paddle-band tests, "
            "and action legality. It posts an action every 1/60-second frame even while "
            "you think. Both Robot Pong serves are automatic; FIRE is never a serve input. "
            "RIGHT moves screen y up and LEFT moves it down.\n\n"
            "Your job is to choose a rally-level return experiment. A CENTER return is the "
            "safest way to survive but can produce an endless 0-0 rally against the robot. "
            "A small EDGE_UP or EDGE_DOWN contact bias changes the outgoing angle; code "
            "keeps the ball inside the paddle band. Use MIXED when evidence is insufficient: "
            "mostly center with an occasional edge contact. If the evidence shows many "
            "returns without a point, actively recommend EDGE or a more frequent MIXED "
            "schedule rather than defaulting forever to CENTER. The red button is legal "
            "only as a one-frame contact feature in Robot Pong 1/2; recommend it only when "
            "the attack schedule has a safe confirmed contact.\n\n"
            "Return one short JSON object (no prose) with optional fields: "
            "mode ('CENTER','MIXED','EDGE'), edge_sign ('UP','DOWN','ALTERNATE'), "
            "attack_every (integer 1..6), bias_fraction (number .10.. .42), "
            "use_feature (boolean), brake_frames (integer 0..4), "
            "warmup_contacts (integer 0..4), force_after_contacts (integer 4..12), "
            "diagnosis (short string). Keep omitted fields at their current values.\n\n"
            f"Reason for this call: {reason}\n"
            f"Public game description:\n{description}\n"
            f"Current observation:\n{json.dumps(self._compact(obs), sort_keys=True)}\n"
            f"Current local policy:\n{json.dumps(policy, sort_keys=True)}\n"
            f"Recent verified evidence:\n{json.dumps(list(self.recent_evidence), sort_keys=True)}\n"
            f"Current point contact count: {self.completed_right_contacts}; "
            f"score: {self.own_points}:{self.opponent_points}."
        )

    def _call_astra(self, prompt, session_id=None):
        try:
            return self.ctx.think(prompt, session_id=session_id, timeout_s=58.0)
        except (TimeoutError, RuntimeError, BudgetExceeded, EpisodeOver, ModelUnavailable):
            return None
        except Exception:
            return None

    def _start_astra(self, obs, reason="initial policy"):
        if self.astra_future is not None or self.astra_replans >= 3:
            return
        if getattr(self.ctx, "system2", None) is None:
            self.astra_replans = 3
            return
        self.astra_replans += 1
        prompt = self._astra_prompt(obs, reason)
        session = self.astra_session
        try:
            self.astra_future = self.ctx.spawn(
                lambda p=prompt, s=session: self._call_astra(p, s)
            )
            self.astra_replan_pending = True
        except Exception:
            self.astra_future = None
            self.astra_replan_pending = False

    def _apply_plan(self, result):
        if result is None:
            return
        parsed = _extract_json(getattr(result, "text", result))
        mode = parsed.get("mode")
        if isinstance(mode, str) and mode.upper() in {"CENTER", "MIXED", "EDGE"}:
            self.plan["mode"] = mode.upper()
        sign = parsed.get("edge_sign")
        if isinstance(sign, str) and sign.upper() in {"UP", "DOWN", "ALTERNATE"}:
            self.plan["edge_sign"] = sign.upper()
        for key, low, high in (
            ("attack_every", 1, 6),
            ("brake_frames", 0, 4),
            ("warmup_contacts", 0, 4),
            ("force_after_contacts", 4, 12),
        ):
            value = _num(parsed.get(key), None)
            if value is not None:
                self.plan[key] = int(_clamp(value, low, high))
        fraction = _num(parsed.get("bias_fraction"), None)
        if fraction is not None:
            self.plan["bias_fraction"] = _clamp(fraction, 0.10, 0.42)
        if isinstance(parsed.get("use_feature"), bool):
            self.plan["use_feature"] = parsed["use_feature"]
        self.plan["version"] += 1
        session = getattr(result, "session_id", None)
        if session:
            self.astra_session = session
        try:
            self.ctx.log(
                "astra_policy",
                plan={k: self.plan[k] for k in self.plan if k != "version"},
                diagnosis=_text(parsed.get("diagnosis", ""))[:240],
            )
        except Exception:
            pass

    def _poll_astra(self):
        future = self.astra_future
        if future is None or not future.done():
            return
        self.astra_future = None
        self.astra_replan_pending = False
        try:
            self._apply_plan(future.result())
        except Exception:
            pass

    def _maybe_replan(self, obs):
        if self.astra_future is not None or self.astra_replans >= 3:
            return
        # The first replan is deliberately evidence-triggered by stagnation, not
        # only by opponent points (a safe controller can otherwise be 0-0 forever).
        stagnant = self.evidence_contacts >= 8 and self.evidence_points == 0
        score_evidence = self.evidence_points >= 4
        if stagnant or score_evidence:
            reason = "stagnation: repeated right contacts without a point" if stagnant else "four point outcomes"
            self._start_astra(obs, reason)
            self.evidence_contacts = 0
            self.evidence_points = 0

    # ---------- Jev: fast recognition/gating at a rally snapshot ----------

    def _jev_questions(self):
        style_a = {
            "CENTER": "Keep the paddle centered on the projected contact; use maximum overlap and no edge attack.",
            "EDGE_UP": "Use the supplied small upper-edge bias (smaller screen y) for this contact; it must remain inside the paddle band.",
            "EDGE_DOWN": "Use the supplied small lower-edge bias (larger screen y) for this contact; it must remain inside the paddle band.",
            "HOLD_CENTER": "The phase/trajectory is uncertain or this is a post-point warmup; choose a safe center return.",
        }
        style_b = {
            "EDGE_DOWN": style_a["EDGE_DOWN"],
            "CENTER": style_a["CENTER"],
            "EDGE_UP": style_a["EDGE_UP"],
            "HOLD_CENTER": style_a["HOLD_CENTER"],
        }
        return {
            "style_a": Choice(
                "Choose the named return style from `control.requested_style`. "
                "This is recognition of the supplied facts, not arithmetic. If "
                "`control.uncertain` is true, use HOLD_CENTER.",
                style_a,
            ),
            "style_b": Choice(
                "Independently classify whether the supplied requested contact is a "
                "safe center or safe edge return. Do not calculate coordinates; "
                "uncertainty means HOLD_CENTER.",
                style_b,
            ),
            "feature": Choice(
                "Should FIRE be pressed on this exact frame? Use USE only when "
                "`control.contact_window` is true, `feature_preapproved` is true, "
                "the variation is Robot Pong 1 or 2, and the paddle action is safe. "
                "Automatic serves never need FIRE.",
                {
                    "NO": "Do not press FIRE; center/uncertain/non-contact/serve state.",
                    "USE": "Press one FIRE pulse at the verified safe right-paddle contact.",
                },
            ),
        }

    def _jev_state(self, obs, control):
        variation = _text(obs.get("variation", "unknown"))
        requested = "CENTER"
        bias = control.get("bias", 0.0)
        if bias < -0.5:
            requested = "EDGE_UP"
        elif bias > 0.5:
            requested = "EDGE_DOWN"
        if control.get("phase") not in ("rally", "rally_ball_occluded"):
            requested = "HOLD_CENTER"
        return _json_safe({
            "game": "Atari 2600 Robot Pong; the controlled paddle is on the right",
            "rules": {
                "RIGHT": "move right paddle up (smaller screen y)",
                "LEFT": "move right paddle down (larger screen y)",
                "serve": "automatic; FIRE is not a serve action",
            },
            "scenario": {
                "variation": variation,
                "difficulty": obs.get("difficulty"),
                "right_paddle_size": obs.get("right_paddle_size"),
            },
            "score": {"own": self.own_points, "opponent": self.opponent_points},
            "observation": {
                "frame": self._frame(obs),
                "phase": obs.get("phase"),
                "serve_waiting": obs.get("serve_waiting"),
                "contact_count_this_point": self.completed_right_contacts,
            },
            "ball": {
                "approach": control.get("approach"),
                "reliable": control.get("reliable"),
                "eta_bucket": (
                    "NOW" if control.get("eta") is not None and control["eta"] <= 4 else
                    "SOON" if control.get("eta") is not None and control["eta"] <= 15 else
                    "LATER" if control.get("eta") is not None else "UNKNOWN"
                ),
            },
            "paddle": {
                "y": control.get("paddle_y"),
                "visible_band": [control.get("band_top"), control.get("band_bottom")],
                "predicted_y_after_settle": control.get("predicted_paddle_y"),
                "code_direction": control.get("direction"),
            },
            "control": {
                "requested_style": requested,
                "target_relation": (
                    "ABOVE" if control.get("target_y", COURT_CENTER) < control.get("paddle_y", COURT_CENTER) - 5 else
                    "BELOW" if control.get("target_y", COURT_CENTER) > control.get("paddle_y", COURT_CENTER) + 5 else
                    "ALIGNED"
                ),
                "contact_window": bool(control.get("contact_window")),
                "uncertain": not bool(control.get("reliable")),
                "feature_preapproved": bool(self.plan.get("use_feature")) and requested != "CENTER",
            },
        })

    def _start_jev(self, obs, control, frame):
        if self.jev_future is not None or frame < self.next_jev_frame:
            return
        if getattr(self.ctx, "system1", None) is None:
            return
        # One judgment at the start of a point and then again after enough
        # contacts to make the next edge decision meaningful.  It is advisory;
        # code still has a complete policy if it is late or absent.
        if not (self.phase in ("serve_delay", "rally", "rally_ball_occluded")):
            return
        state = self._jev_state(obs, control)
        questions = self._jev_questions()
        meta = (self.rally_id, self.completed_right_contacts, frame)

        def ask():
            try:
                return self.ctx.decide(state, questions, timeout_s=0.9)
            except (TimeoutError, RuntimeError, BudgetExceeded, EpisodeOver, ModelUnavailable):
                return None
            except Exception:
                return None

        try:
            self.jev_future = self.ctx.spawn(ask)
            self.jev_meta = meta
            self.next_jev_frame = frame + 36
        except Exception:
            self.jev_future = None
            self.jev_meta = None
            self.next_jev_frame = frame + 60

    def _poll_jev(self, obs, control, frame):
        future = self.jev_future
        if future is None or not future.done():
            return
        self.jev_future = None
        meta = self.jev_meta
        self.jev_meta = None
        # Even a timed-out completed future must not immediately create a
        # retry storm; one request per 36-frame advisory window is sufficient.
        self.next_jev_frame = max(self.next_jev_frame, frame + 18)
        try:
            answers = future.result()
        except Exception:
            return
        if not isinstance(answers, dict) or meta is None:
            return
        # A plan answer from an old point is not allowed to affect a new serve.
        if meta[0] != self.rally_id or frame - meta[2] > 80:
            return
        a = answers.get("style_a")
        b = answers.get("style_b")
        va, pa = _answer(a, {"CENTER", "EDGE_UP", "EDGE_DOWN", "HOLD_CENTER"}) if a is not None else (None, 0.0)
        vb, pb = _answer(b, {"CENTER", "EDGE_UP", "EDGE_DOWN", "HOLD_CENTER"}) if b is not None else (None, 0.0)
        if va == vb and va is not None and min(pa, pb) >= 0.56:
            self.jev_style = va
            self.jev_style_rally = meta[0]
            self.jev_style_contact = meta[1]
        feature, fp = _answer(answers.get("feature"), {"USE", "NO"}) if answers.get("feature") is not None else (None, 0.0)
        self.jev_feature = feature == "USE" and fp >= 0.62
        self.jev_feature_rally = meta[0] if self.jev_feature else None
        self.jev_feature_contact = meta[1] if self.jev_feature else None
        try:
            self.ctx.log(
                "jev_rally_judgment",
                style=va if va == vb else "DISAGREE",
                style_probability=round(min(pa, pb), 3),
                feature=feature,
                feature_probability=round(fp, 3),
                rally_id=self.rally_id,
            )
        except Exception:
            pass

    # ---------- frame controller ----------

    def _effective_direction(self, control):
        # Jev can veto an uncertain edge or authorize a center/edge recognition,
        # but it never calculates or replaces the servo direction.
        style = self.jev_style
        if style == "HOLD_CENTER" and not control.get("reliable"):
            return "HOLD"
        return control.get("direction", "HOLD")

    def _feature_now(self, obs, control, direction):
        if self.feature_used:
            return False
        if not self.plan.get("use_feature") or not self.jev_feature:
            return False
        if self.jev_feature_rally != self.rally_id or self.jev_feature_contact != self.completed_right_contacts:
            return False
        if not control.get("contact_window"):
            return False
        if direction not in ("UP", "DOWN", "HOLD"):
            return False
        if control.get("approach") != "RIGHT":
            return False
        if "Robot Pong" not in _text(obs.get("variation", "")):
            return False
        # A FIRE pulse is an attack experiment, never an emergency correction.
        if abs(_num(control.get("bias"), 0.0) or 0.0) < 0.5:
            return False
        self.feature_used = True
        return True

    def _action(self, obs, control):
        direction = self._effective_direction(control)
        use_feature = self._feature_now(obs, control, direction)
        if direction == "UP":
            action = self.up_fire if use_feature else self.up
        elif direction == "DOWN":
            action = self.down_fire if use_feature else self.down
        else:
            action = self.fire if use_feature else self.noop
        self.last_action = action
        return action

    def run_episode(self, game) -> None:
        self.game = game
        self._reset_state()
        self._prepare_actions(game)
        self._fix_action_names()
        if not getattr(game, "tick_s", None):
            return
        try:
            first = game.observe()
        except Exception:
            first = {}
        if not isinstance(first, dict):
            first = {}
        self._start_astra(first, "initial policy and attack schedule")

        while True:
            if getattr(self.ctx, "done", False) or getattr(game, "done", False):
                break
            try:
                raw = game.observe()
            except Exception:
                raw = {}
            if not isinstance(raw, dict):
                raw = {}
            obs, frame, phase, x, y, visible = self._ingest(raw)
            self._poll_astra()
            self._maybe_replan(obs)
            control = self._trajectory(obs, frame, phase, x, y, visible)
            # Jev is asked with a complete derived state; it is never awaited.
            self._poll_jev(obs, control, frame)
            if self.jev_future is None and (
                frame == 0 or control.get("contact_window") or
                self.evidence_contacts > 0 or not control.get("reliable")
            ):
                self._start_jev(obs, control, frame)
            action = self._action(obs, control)
            try:
                self.ctx.act(action)
                self.ctx.wait_for_tick()
            except (BudgetExceeded, EpisodeOver):
                return


def build(ctx):
    return Harness(ctx)
