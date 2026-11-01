"""Adaptive native-frame Breakout controller.

The fast path is deliberately a controller, not a model call: it predicts a
paddle crossing, estimates paddle motion from successive RAM observations, and
counter-steers when a NOOP would carry the paddle past the safe interval.  Jev
only selects a short-lived tactical mode and Astra only supplies a late tunnel
preference.  Both are optional advice and can never block or replace the
interceptor.
"""
from __future__ import annotations

import re
from typing import Any

from metaharness.core.contract import BudgetExceeded, EpisodeOver
from metaharness.models.jev import Choice


NOOP = "NOOP"
FIRE = "FIRE"
RIGHT = "RIGHT"
LEFT = "LEFT"
LEGAL = {NOOP, FIRE, RIGHT, LEFT}


class Harness:
    def __init__(self, ctx: Any):
        self.ctx = ctx

        # Per-episode controller state.
        self.previous: dict[str, Any] | None = None
        self.last_frame = -1
        self.last_action = NOOP
        self.previous_paddle_x: float | None = None
        self.paddle_velocity = 0.0
        self.step_estimate = 1.0
        self.plan_side: str | None = None
        self.astra_preference: str | None = None
        self.plan_version = 0
        self.cooldown_until = 0
        self.contact_offsets: dict[str, list[float]] = {LEFT: [], RIGHT: []}

        # There is at most one outstanding request of each kind.  In particular,
        # a slow Astra call cannot create a thread per tick.
        self.jev_future = None
        self.jev_request_frame = -10_000
        self.jev_request_version = -1
        self.jev_answer: str | None = None
        self.jev_answer_frame = -10_000
        self.jev_answer_version = -1
        self.jev_request_number = 0
        self.jev_enabled = getattr(ctx, "system1", None) is not None

        self.astra_future = None
        self.astra_started = False

    # ---------------- defensive primitives ----------------

    @staticmethod
    def _num(value: Any, default: float = 0.0) -> float:
        try:
            x = float(value)
            if x != x or x in (float("inf"), float("-inf")):
                return default
            return x
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _int(value: Any, default: int = 0) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    @classmethod
    def _rect(cls, obs: Any, name: str) -> dict[str, Any]:
        value = obs.get(name) if isinstance(obs, dict) else None
        return value if isinstance(value, dict) else {}

    @classmethod
    def _ball(cls, obs: Any) -> dict[str, Any] | None:
        value = obs.get("ball") if isinstance(obs, dict) else None
        return value if isinstance(value, dict) else None

    @classmethod
    def _clamp(cls, value: float, low: float, high: float) -> float:
        return max(low, min(high, value))

    def _log(self, event: str, **fields: Any) -> None:
        try:
            self.ctx.log(event, **fields)
        except Exception:
            # Logging is diagnostic and must not turn malformed stand-ins into a
            # game error during pre-flight.
            pass

    # ---------------- observed dynamics and events ----------------

    def _update_paddle_motion(self, obs: dict[str, Any]) -> None:
        paddle = self._rect(obs, "paddle")
        px = self._num(paddle.get("x"), 99.0)
        frame = self._int(obs.get("frame"), self.last_frame + 1)
        if self.previous_paddle_x is not None:
            elapsed = max(1, frame - self.last_frame)
            delta = (px - self.previous_paddle_x) / float(elapsed)
            # A frame jump can happen in a virtual pre-flight.  Ignore an
            # impossible jump rather than poisoning the braking estimate.
            if abs(delta) <= 8.0:
                self.paddle_velocity = 0.65 * self.paddle_velocity + 0.35 * delta
                if abs(delta) > 0.05:
                    self.step_estimate = self._clamp(
                        0.8 * self.step_estimate + 0.2 * abs(delta), 0.7, 3.0
                    )
        self.previous_paddle_x = px

    def _event(self, obs: dict[str, Any]) -> str:
        if self.previous is None:
            return "start"
        old = self.previous
        old_lives = self._int(old.get("lives"), 0)
        new_lives = self._int(obs.get("lives"), old_lives)
        if new_lives < old_lives:
            return "life_loss"
        old_wall = self._int(old.get("wall_index"), 0)
        new_wall = self._int(obs.get("wall_index"), old_wall)
        if new_wall != old_wall:
            return "wall_transition"
        old_score = self._num(old.get("score"), 0.0)
        new_score = self._num(obs.get("score"), old_score)
        if new_score > old_score:
            return "brick_hit"
        old_ball = self._ball(old)
        new_ball = self._ball(obs)
        if old_ball is None and new_ball is not None:
            return "serve"
        if old_ball is not None and new_ball is None:
            return "ball_lost"
        old_dy = self._num(old_ball.get("dy"), 0.0) if old_ball else 0.0
        new_dy = self._num(new_ball.get("dy"), 0.0) if new_ball else 0.0
        # A change from descending to ascending is the observable paddle
        # contact.  Learn the actual ROM's section-to-angle relation instead of
        # importing a generic Breakout bounce table.
        if old_ball is not None and new_ball is not None and old_dy > 0 and new_dy < 0:
            return "paddle_bounce"
        return "none"

    def _learn_contact(self, obs: dict[str, Any]) -> None:
        if self.previous is None:
            return
        old_ball = self._ball(self.previous)
        new_ball = self._ball(obs)
        if old_ball is None or new_ball is None:
            return
        old_dy = self._num(old_ball.get("dy"), 0.0)
        new_dy = self._num(new_ball.get("dy"), 0.0)
        if not (old_dy > 0 and new_dy < 0):
            return
        dx = self._num(new_ball.get("dx"), 0.0)
        direction = RIGHT if dx > 0 else LEFT if dx < 0 else None
        if direction is None:
            return
        paddle = self._rect(obs, "paddle")
        ball_x = self._num(new_ball.get("x"), 80.0)
        ball_w = self._num(new_ball.get("w"), 2.0)
        paddle_x = self._num(paddle.get("x"), 76.0)
        paddle_w = self._num(paddle.get("w"), 16.0)
        offset = (ball_x + ball_w / 2.0) - (paddle_x + paddle_w / 2.0)
        if abs(offset) <= paddle_w / 2.0 + 2.0:
            samples = self.contact_offsets[direction]
            samples.append(offset)
            if len(samples) > 12:
                del samples[:-12]
            self._log(
                "contact_calibrated",
                frame=self._int(obs.get("frame"), 0),
                outgoing=direction,
                offset=round(offset, 2),
            )

    def _side_facts(self, obs: dict[str, Any]) -> dict[str, Any]:
        bitmap = obs.get("brick_bitmap") if isinstance(obs, dict) else None
        rows = bitmap if isinstance(bitmap, list) else []

        def occupied(col: int) -> int:
            total = 0
            for row in rows:
                if isinstance(row, list) and 0 <= col < len(row) and bool(row[col]):
                    total += 1
            return total

        def clear_columns(columns: tuple[int, ...]) -> int:
            count = 0
            for col in columns:
                if occupied(col) == 0:
                    count += 1
            return count

        left_occ = sum(occupied(c) for c in (0, 1, 2))
        right_occ = sum(occupied(c) for c in (15, 16, 17))
        left_clear = clear_columns((0, 1, 2))
        right_clear = clear_columns((15, 16, 17))
        left_open = left_clear > 0
        right_open = right_clear > 0
        if left_clear > right_clear:
            default_side = "LEFT"
        elif right_clear > left_clear:
            default_side = "RIGHT"
        elif left_occ < right_occ:
            default_side = "LEFT"
        elif right_occ < left_occ:
            default_side = "RIGHT"
        else:
            default_side = None
        return {
            "left_open": left_open,
            "right_open": right_open,
            "left_clear_columns": left_clear,
            "right_clear_columns": right_clear,
            "left_occupied_outer": left_occ,
            "right_occupied_outer": right_occ,
            "default_side": default_side,
            "remaining": self._int(obs.get("bricks_remaining"), 108),
            "wall": self._int(obs.get("wall_index"), 0),
        }

    def _refresh_plan(self, obs: dict[str, Any], event: str) -> None:
        frame = self._int(obs.get("frame"), self.last_frame + 1)
        lives = self._int(obs.get("lives"), 0)
        facts = self._side_facts(obs)
        if event == "wall_transition":
            self.plan_side = None
            self.astra_preference = None
            self.cooldown_until = frame + 90
            self.plan_version += 1
            self.jev_answer = None
        if event == "life_loss":
            # The next rally begins conservatively.  Learned contact geometry
            # remains useful, but do not force a tunnel immediately after a miss.
            self.plan_side = None
            self.paddle_velocity = 0.0
            self.cooldown_until = frame + 75
            self.plan_version += 1
            self.jev_answer = None

        if lives <= 1 or frame < self.cooldown_until:
            desired = None
        elif facts["left_open"] or facts["right_open"]:
            preferred = self.astra_preference
            if preferred == "SURVIVE":
                desired = None
            else:
                if preferred not in ("LEFT", "RIGHT"):
                    preferred = facts["default_side"]
                if preferred == "LEFT" and facts["left_open"]:
                    desired = "LEFT"
                elif preferred == "RIGHT" and facts["right_open"]:
                    desired = "RIGHT"
                elif facts["left_open"]:
                    desired = "LEFT"
                elif facts["right_open"]:
                    desired = "RIGHT"
                else:
                    desired = None
        else:
            desired = None

        if desired != self.plan_side:
            self.plan_side = desired
            self.plan_version += 1
            self.jev_answer = None
            if desired is not None:
                self._log("tunnel_plan", frame=frame, side=desired, facts=facts)

    # ---------------- geometry ----------------

    @staticmethod
    def _reflect_x(x: float, dx: float, frames: float) -> float:
        left, right = 8.0, 150.0
        span = right - left
        if span <= 0:
            return x
        phase = (x - left + dx * max(0.0, frames)) % (2.0 * span)
        return left + phase if phase <= span else right - (phase - span)

    def _path_hits_brick(
        self, obs: dict[str, Any], bx: float, by: float, bw: float, bh: float, dx: float, dy: float
    ) -> bool:
        """Conservatively detect a brick before a direct paddle return.

        A descending ball inside the wall may bounce upward before reaching the
        paddle.  Treating that uncertain return as a direct intercept is worse
        than holding a broad anchor for a few frames.  This is only a safety
        classifier; it does not pretend to emulate collision ordering.
        """
        if dy <= 0 or by > 96:
            return False
        bitmap = obs.get("brick_bitmap")
        if not isinstance(bitmap, list):
            return False
        for t in range(1, 31):
            x = self._reflect_x(bx, dx, float(t))
            y = by + dy * t
            if y > 96:
                break
            for row in range(min(6, len(bitmap))):
                line = bitmap[row]
                if not isinstance(line, list):
                    continue
                brick_y = 57.0 + 6.0 * row
                for col in range(min(18, len(line))):
                    if not bool(line[col]):
                        continue
                    brick_x = 8.0 + 8.0 * col
                    if (
                        x + bw > brick_x
                        and x < brick_x + 8.0
                        and y + bh > brick_y
                        and y < brick_y + 6.0
                    ):
                        return True
        return False

    def _median_offset(self, direction: str, paddle_w: float) -> float:
        samples = self.contact_offsets.get(direction, [])
        if samples:
            ordered = sorted(samples)
            value = ordered[len(ordered) // 2]
            return self._clamp(value, -paddle_w / 2.0 + 2.0, paddle_w / 2.0 - 2.0)
        # This is only a low-risk bootstrap.  It is replaced by the measured
        # ROM-specific value as soon as a paddle bounce is observed.
        return 3.0 if direction == RIGHT else -3.0

    def _projection(self, obs: dict[str, Any], strategy_mode: str | None) -> dict[str, Any]:
        ball = self._ball(obs)
        paddle = self._rect(obs, "paddle")
        px = self._num(paddle.get("x"), 99.0)
        py = self._num(paddle.get("y"), 189.0)
        pw = self._num(paddle.get("w"), 16.0)
        pmin, pmax = 8.0, max(8.0, 160.0 - pw)
        anchor = self._clamp(76.0, pmin, pmax)
        if ball is None or not bool(obs.get("ball_in_play", False)):
            return {
                "phase": "serve",
                "urgency": "serve",
                "eta": None,
                "landing_x": None,
                "target_left": anchor,
                "lane": "unknown",
                "relation": "unknown",
                "speed": "unknown",
                "uncertain": True,
            }

        bx = self._num(ball.get("x"), 80.0)
        by = self._num(ball.get("y"), 100.0)
        bw = self._num(ball.get("w"), 2.0)
        bh = self._num(ball.get("h"), 4.0)
        vx = self._num(ball.get("dx"), 0.0)
        vy = self._num(ball.get("dy"), 0.0)
        valid = bool(obs.get("ball_velocity_valid", False)) or abs(vx) + abs(vy) > 0
        if not valid:
            return {
                "phase": "calibrating",
                "urgency": "early",
                "eta": None,
                "landing_x": None,
                "target_left": anchor,
                "lane": "center",
                "relation": "centered" if abs(px - anchor) <= 8 else ("left" if px < anchor else "right"),
                "speed": "unknown",
                "uncertain": True,
            }

        fast = abs(vx) >= 2.0 or abs(vy) >= 3.0
        speed = "fast" if fast else "normal"
        eta: float | None = None
        landing_x: float | None = None
        uncertain = False
        phase = "ascending"
        if vy > 0:
            phase = "descending"
            target_y = py - bh
            eta = max(0.0, (target_y - by) / vy)
            # Above the lower edge of the wall, a brick can reverse the ball;
            # only call this a reliable paddle intercept through an actual gap.
            uncertain = self._path_hits_brick(obs, bx, by, bw, bh, vx, vy)
            if not uncertain:
                landing_x = self._reflect_x(bx, vx, eta)
        if landing_x is None:
            target_left = anchor
            if strategy_mode in ("LEFT", "RIGHT") and phase == "ascending":
                # Keep a little mobility toward a selected side, but never
                # abandon the center merely because a model said so.
                target_left = pmin + (pmax - pmin) * (0.30 if strategy_mode == "LEFT" else 0.70)
        else:
            landing_center = landing_x + bw / 2.0
            offset = 0.0
            if strategy_mode in ("LEFT", "RIGHT") and eta is not None and eta > 10.0:
                offset = self._median_offset(LEFT if strategy_mode == "LEFT" else RIGHT, pw)
            target_left = landing_center - offset - pw / 2.0
            target_left = self._clamp(target_left, pmin, pmax)

        if eta is None or uncertain or landing_x is None:
            urgency = "uncertain" if uncertain else "early"
        elif eta <= 7:
            urgency = "imminent"
        elif eta <= 18:
            urgency = "near"
        else:
            urgency = "early"

        if landing_x is None:
            lane = "center"
        elif landing_x + bw / 2.0 < 56:
            lane = "left"
        elif landing_x + bw / 2.0 > 104:
            lane = "right"
        else:
            lane = "center"
        wanted_gap = target_left - px
        if abs(wanted_gap) <= max(3.0, pw / 4.0):
            relation = "overlap"
        elif wanted_gap < 0:
            relation = "left_of_target"
        else:
            relation = "right_of_target"
        return {
            "phase": phase,
            "urgency": urgency,
            "eta": None if eta is None else round(eta, 1),
            "landing_x": None if landing_x is None else round(landing_x, 1),
            "target_left": round(self._clamp(target_left, pmin, pmax), 1),
            "lane": lane,
            "relation": relation,
            "speed": speed,
            "uncertain": uncertain,
        }

    def _steer(self, obs: dict[str, Any], projection: dict[str, Any], force_brake: bool = False) -> str:
        paddle = self._rect(obs, "paddle")
        px = self._num(paddle.get("x"), 99.0)
        pw = self._num(paddle.get("w"), 16.0)
        pmin, pmax = 8.0, max(8.0, 160.0 - pw)
        target = self._clamp(self._num(projection.get("target_left"), 76.0), pmin, pmax)
        urgency = projection.get("urgency")
        eta = self._num(projection.get("eta"), 12.0)
        deadband = 2.0 if urgency == "imminent" else max(3.0, pw / 4.0)
        error = target - px

        # Predict where the paddle drifts over the next few frames.  A NOOP is
        # not a brake in this cartridge, and sticky actions can repeat the last
        # direction, so an opposite one-frame pulse is the safe brake.
        look = 1.0 if urgency == "imminent" else min(4.0, max(1.0, eta if eta > 0 else 1.0))
        forecast = px + self.paddle_velocity * look
        if force_brake and abs(self.paddle_velocity) > 0.2:
            if self.paddle_velocity > 0 and forecast >= target - 1.0:
                return LEFT
            if self.paddle_velocity < 0 and forecast <= target + 1.0:
                return RIGHT
        if error > deadband:
            if self.paddle_velocity > 0 and forecast > target + 1.0:
                return LEFT
            return RIGHT
        if error < -deadband:
            if self.paddle_velocity < 0 and forecast < target - 1.0:
                return RIGHT
            return LEFT

        # Within the catch interval, cancel momentum only when it predicts an
        # overshoot.  Otherwise NOOP preserves a broad overlap instead of
        # oscillating every frame.
        if forecast > target + deadband:
            return LEFT
        if forecast < target - deadband:
            return RIGHT
        return NOOP

    # ---------------- model bridges ----------------

    def _derived(self, obs: dict[str, Any], event: str, projection: dict[str, Any]) -> dict[str, Any]:
        side = self._side_facts(obs)
        frame = self._int(obs.get("frame"), self.last_frame + 1)
        paddle = self._rect(obs, "paddle")
        motion = "right" if self.paddle_velocity > 0.25 else "left" if self.paddle_velocity < -0.25 else "still"
        return {
            "frame": frame,
            "lives": self._int(obs.get("lives"), 0),
            "score": int(self._num(obs.get("score"), 0.0)),
            "event": event,
            "serve_required": bool(obs.get("serve_required", False)),
            "flight_phase": projection["phase"],
            "return_certainty": "uncertain" if projection["uncertain"] else "direct",
            "landing_lane": projection["lane"],
            "landing_urgency": projection["urgency"],
            "paddle_relation": projection["relation"],
            "paddle_motion": motion,
            "speed_regime": projection["speed"],
            "eta": projection["eta"],
            "fallback_action": self._steer(obs, projection),
            "tunnel_side": self.plan_side or "SURVIVE",
            "left_open": side["left_open"],
            "right_open": side["right_open"],
            "left_clear_columns": side["left_clear_columns"],
            "right_clear_columns": side["right_clear_columns"],
            "bricks_remaining": side["remaining"],
            "wall_index": side["wall"],
            "contact_samples_left": len(self.contact_offsets[LEFT]),
            "contact_samples_right": len(self.contact_offsets[RIGHT]),
            "plan_version": self.plan_version,
            "paddle_x": round(self._num(paddle.get("x"), 99.0), 1),
        }

    def _make_questions(self, derived: dict[str, Any]) -> dict[str, Choice]:
        descriptions = {
            "TRACK": "Use the code-computed direct landing target and preserve overlap. This is the safe default.",
            "BRAKE": "Prefer a one-frame counter-steer if the named paddle motion predicts overshoot; never leave a direct catch.",
            "CENTER": "Keep a conservative center anchor while the ball return is uncertain or still far away.",
            "SHAPE_LEFT": "Only when the tunnel side is LEFT and the return is direct and early, use the measured left-contact shaping target.",
            "SHAPE_RIGHT": "Only when the tunnel side is RIGHT and the return is direct and early, use the measured right-contact shaping target.",
        }
        keys = list(descriptions)
        shift = self.jev_request_number % len(keys)
        order_a = keys[shift:] + keys[:shift]
        order_b = list(reversed(order_a))
        common = (
            "Choose a tactical mode for this exact derived state. Geometry, arithmetic, "
            "legality, and the final action are handled by code. Do not calculate a "
            "landing; use the named fields. A near or imminent direct return has priority "
            "over shaping. Facts: " + str(derived)
        )
        return {
            "mode_a": Choice(common, {k: descriptions[k] for k in order_a}),
            "mode_b": Choice(common, {k: descriptions[k] for k in order_b}),
        }

    def _launch_jev(self, obs: dict[str, Any], derived: dict[str, Any]) -> None:
        if not self.jev_enabled or self.jev_future is not None:
            return
        frame = derived["frame"]
        if derived["serve_required"] or frame - self.jev_request_frame < 24:
            return
        if derived["landing_urgency"] == "imminent":
            return
        state = {
            "game": "Atari 2600 Breakout, ordinary cartridge mode 0, difficulty 0, native 60 Hz",
            "observation": derived,
            "rules": {
                "legal_actions": [NOOP, FIRE, RIGHT, LEFT],
                "FIRE": "serve only when serve_required is true",
                "priority": "catchability first; tunnel shaping is optional",
                "bridge": "Your mode is advisory. Code recomputes the action on the newest observation.",
            },
        }
        questions = self._make_questions(derived)
        self.jev_request_number += 1
        self.jev_request_frame = frame
        self.jev_request_version = self.plan_version

        def call():
            return self.ctx.decide(state, questions, timeout_s=1.5)

        try:
            self.jev_future = self.ctx.spawn(call)
        except Exception as exc:
            self.jev_future = None
            self._log("jev_spawn_failed", error=type(exc).__name__)

    @staticmethod
    def _answer_value(answer: Any) -> str | None:
        value = getattr(answer, "value", None)
        return value if isinstance(value, str) else None

    @staticmethod
    def _answer_probability(answer: Any, selected: str) -> float:
        probs = getattr(answer, "probabilities", None)
        if isinstance(probs, dict) and probs:
            try:
                return float(probs.get(selected, 0.0))
            except (TypeError, ValueError):
                return 0.0
        return 1.0 if Harness._answer_value(answer) == selected else 0.0

    def _collect_jev(self, frame: int) -> None:
        future = self.jev_future
        if future is None or not getattr(future, "done", lambda: False)():
            return
        self.jev_future = None
        try:
            answers = future.result()
            a = answers.get("mode_a") if isinstance(answers, dict) else None
            b = answers.get("mode_b") if isinstance(answers, dict) else None
            av, bv = self._answer_value(a), self._answer_value(b)
            allowed = {"TRACK", "BRAKE", "CENTER", "SHAPE_LEFT", "SHAPE_RIGHT"}
            if av in allowed and av == bv and self.jev_request_version == self.plan_version:
                probability = min(self._answer_probability(a, av), self._answer_probability(b, bv))
                if probability >= 0.50:
                    self.jev_answer = av
                    self.jev_answer_frame = self.jev_request_frame
                    self.jev_answer_version = self.jev_request_version
                    self._log("jev_mode_accepted", frame=frame, mode=av, probability=round(probability, 3))
        except (TimeoutError, RuntimeError, BudgetExceeded, EpisodeOver) as exc:
            self._log("jev_unavailable", error=type(exc).__name__)
        except Exception as exc:
            # Stand-in models intentionally return arbitrary/malformed answers.
            self._log("jev_parse_ignored", error=type(exc).__name__)

    def _astra_prompt(self, obs: dict[str, Any], derived: dict[str, Any]) -> str:
        side = self._side_facts(obs)
        return (
            "You are a slow strategic advisor for ordinary Atari 2600 Breakout mode 0. "
            "A deterministic native-frame controller catches the ball; you must not give "
            "frame-by-frame actions. Decide whether an already-observed side tunnel is a "
            "reasonable optional shaping target. Return exactly one token: LEFT, RIGHT, "
            "or SURVIVE. Choose SURVIVE if the ball is fast, lives are scarce, or neither "
            "outer corridor is open. The controller will verify your choice against the "
            "newest RAM bitmap and can ignore it.\nDerived state: " + str(derived) +
            "\nSide facts: " + str(side)
        )

    def _launch_astra(self, obs: dict[str, Any], derived: dict[str, Any]) -> None:
        if self.astra_started or self.astra_future is not None:
            return
        if derived["frame"] < 120 or derived["serve_required"]:
            return
        try:
            remaining = self.ctx.budget.remaining()
            if int(remaining.get("system2_calls", 0)) <= 0:
                return
        except Exception:
            pass
        self.astra_started = True
        prompt = self._astra_prompt(obs, derived)

        def call():
            # Keep this advisory call bounded.  A long Astra response must never
            # hold the native-frame loop hostage.
            return self.ctx.think(prompt, timeout_s=24.0)

        try:
            self.astra_future = self.ctx.spawn(call)
        except Exception as exc:
            self.astra_future = None
            self._log("astra_spawn_failed", error=type(exc).__name__)

    def _collect_astra(self, frame: int) -> None:
        future = self.astra_future
        if future is None or not getattr(future, "done", lambda: False)():
            return
        self.astra_future = None
        try:
            result = future.result()
            text = getattr(result, "text", "")
            if not isinstance(text, str):
                return
            match = re.search(r"\b(LEFT|RIGHT|SURVIVE|SURVIVAL)\b", text.upper())
            if match is None:
                return
            token = match.group(1)
            self.astra_preference = "SURVIVE" if token.startswith("SURV") else token
            self.plan_version += 1
            self.jev_answer = None
            self._log("astra_preference", frame=frame, preference=self.astra_preference)
        except (TimeoutError, RuntimeError, BudgetExceeded, EpisodeOver) as exc:
            self._log("astra_unavailable", error=type(exc).__name__)
        except Exception as exc:
            self._log("astra_parse_ignored", error=type(exc).__name__)

    # ---------------- final action bridge ----------------

    def _advice(self, derived: dict[str, Any]) -> str | None:
        if self.jev_answer is None:
            return None
        if self.jev_answer_version != self.plan_version:
            return None
        if derived["frame"] - self.jev_answer_frame > 42:
            return None
        return self.jev_answer

    def _action(self, obs: dict[str, Any], derived: dict[str, Any], projection: dict[str, Any]) -> str:
        if bool(obs.get("terminated", False)) or bool(obs.get("truncated", False)):
            return NOOP
        if bool(obs.get("serve_required", False)):
            return FIRE if self._int(obs.get("lives"), 0) > 0 else NOOP
        if self._ball(obs) is None:
            return NOOP

        advice = self._advice(derived)
        mode = advice or "TRACK"
        # Jev is not allowed to turn an uncertain or imminent state into a
        # shaping maneuver.  These gates are code-owned safety invariants.
        if projection["uncertain"] or projection["urgency"] in ("near", "imminent"):
            mode = "BRAKE" if advice == "BRAKE" else "TRACK"
        if mode in ("SHAPE_LEFT", "SHAPE_RIGHT"):
            wanted = "LEFT" if mode.endswith("LEFT") else "RIGHT"
            if wanted != self.plan_side or projection["uncertain"]:
                mode = "TRACK"
        if mode == "CENTER" and projection["urgency"] not in ("early", "uncertain"):
            mode = "TRACK"

        # The projection already includes the measured side-contact offset when
        # the strategic plan is active.  CENTER is the sole advice that changes
        # the target; BRAKE only changes the final steering pulse.
        if mode == "CENTER":
            adjusted = dict(projection)
            paddle = self._rect(obs, "paddle")
            pw = self._num(paddle.get("w"), 16.0)
            pmin, pmax = 8.0, max(8.0, 160.0 - pw)
            adjusted["target_left"] = self._clamp(76.0, pmin, pmax)
        else:
            adjusted = projection
        return self._steer(obs, adjusted, force_brake=(mode == "BRAKE"))

    def _observe(self, game: Any) -> dict[str, Any]:
        try:
            value = game.observe()
        except Exception:
            return {}
        return value if isinstance(value, dict) else {}

    def run_episode(self, game: Any) -> None:
        ctx = self.ctx
        realtime = bool(getattr(game, "tick_s", 0.0))

        if not realtime:
            # The target is tier 2, but retaining a nonblocking legal fallback
            # makes the module harmless in generic pre-flight wrappers.
            while not ctx.done and not bool(getattr(game, "done", False)):
                obs = self._observe(game)
                event = self._event(obs)
                self._update_paddle_motion(obs)
                self._refresh_plan(obs, event)
                projection = self._projection(obs, self.plan_side)
                derived = self._derived(obs, event, projection)
                action = self._action(obs, derived, projection)
                if action not in LEGAL:
                    action = NOOP
                ctx.act(action)
                self.previous = obs
                self.last_frame = derived["frame"]
                self.last_action = action
            return

        while not ctx.done and not bool(getattr(game, "done", False)):
            obs = self._observe(game)
            event = self._event(obs)
            self._update_paddle_motion(obs)
            self._learn_contact(obs)
            self._refresh_plan(obs, event)
            projection = self._projection(obs, self.plan_side)
            derived = self._derived(obs, event, projection)

            self._collect_jev(derived["frame"])
            self._collect_astra(derived["frame"])
            self._launch_astra(obs, derived)
            self._launch_jev(obs, derived)

            action = self._action(obs, derived, projection)
            if action not in LEGAL:
                action = NOOP
            try:
                ctx.act(action)
            except (BudgetExceeded, EpisodeOver):
                return

            self.previous = obs
            self.last_frame = derived["frame"]
            self.last_action = action
            try:
                ctx.wait_for_tick()
            except (BudgetExceeded, EpisodeOver):
                return


def build(ctx: Any) -> Harness:
    return Harness(ctx)
