"""Asynchronous strategic advice over a deterministic Breakout interceptor.

The controller never waits for a model between real-time ticks.  Code predicts
where the ball will meet the paddle and posts a legal action on every tick;
Jev can provide a short-lived shaping bias and Astra can select a side tunnel,
but either answer is advisory and is discarded when it is stale.
"""
from __future__ import annotations

import re
from typing import Any

from metaharness.core.contract import BudgetExceeded, EpisodeOver
from metaharness.models.jev import Choice


ACTIONS = ("NOOP", "FIRE", "RIGHT", "LEFT")


class Harness:
    def __init__(self, ctx: Any):
        self.ctx = ctx
        self.s1_future = None
        self.s1_request_frame = -10_000
        self.s1_request_version = 0
        self.s1_answer = None
        self.s1_answer_frame = -10_000
        self.s1_answer_version = -1
        self.s1_enabled = getattr(ctx, "system1", object()) is not None
        self.astra_future = None
        self.astra_calls = 0
        self.astra_started_frame = -10_000
        self.strategy = "SURVIVE"
        self.plan_version = 0
        self.previous = None
        self.last_posted = "NOOP"
        self.last_frame = -1
        self.request_number = 0
        self.last_major_event_frame = -10_000

        # The parent recomputed a point target every frame.  These few fields
        # form the one experimental actuator guard: a requested reversal must
        # clear a wider overshoot band before it is posted.
        self.paddle_velocity = 0.0
        self.last_direction = None
        self.last_direction_frame = -10_000
        self.last_gate_frame = -10_000
        self.turn_gate_until = -10_000
        self.turn_gate_pair = None
        self.control_frame = -1

    # ---------- small, defensive observation helpers ----------

    @staticmethod
    def _number(value: Any, default: float = 0.0) -> float:
        try:
            result = float(value)
            if result != result or result in (float("inf"), float("-inf")):
                return default
            return result
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _integer(value: Any, default: int = 0) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    @classmethod
    def _clamp(cls, value: float, low: float, high: float) -> float:
        return max(low, min(high, value))

    @classmethod
    def _field(cls, obs: Any, name: str, default: Any = None) -> Any:
        return obs.get(name, default) if isinstance(obs, dict) else default

    @classmethod
    def _ball(cls, obs: dict[str, Any]) -> dict[str, Any] | None:
        ball = obs.get("ball")
        return ball if isinstance(ball, dict) else None

    @classmethod
    def _paddle(cls, obs: dict[str, Any]) -> dict[str, Any]:
        paddle = obs.get("paddle")
        return paddle if isinstance(paddle, dict) else {}

    # ---------- exact, local geometry ----------

    def _reflected_x(self, x: float, dx: float, frames: float) -> float:
        """Project a ball x coordinate with side-wall reflection.

        The RAM decoder reports the ball's left coordinate.  The ordinary
        Breakout playfield has stable paddle bounds (8..144) and a two-pixel
        ball, so using the corresponding ball bounds avoids chasing the ball's
        center outside the legal paddle range.
        """
        left = 8.0
        right = 150.0
        span = right - left
        if span <= 0:
            return x
        period = 2.0 * span
        phase = (x - left + dx * max(0.0, frames)) % period
        if phase <= span:
            return left + phase
        return right - (phase - span)

    def _projection(self, obs: dict[str, Any]) -> dict[str, Any]:
        """Return named, code-computed flight facts for control and Jev."""
        ball = self._ball(obs)
        paddle = self._paddle(obs)
        px = self._number(paddle.get("x"), 99.0)
        py = self._number(paddle.get("y"), 189.0)
        pw = self._number(paddle.get("w"), 16.0)
        if ball is None or not bool(obs.get("ball_in_play", False)):
            return {
                "phase": "serve",
                "eta": None,
                "landing_x": None,
                "landing_lane": "unknown",
                "urgency": "serve",
                "paddle_relation": "unknown",
                "speed": "unknown",
                "target_left": self._clamp(px, 8.0, 144.0),
            }

        bx = self._number(ball.get("x"), 80.0)
        by = self._number(ball.get("y"), 100.0)
        bw = self._number(ball.get("w"), 2.0)
        bh = self._number(ball.get("h"), 4.0)
        dx = self._number(ball.get("dx"), 0.0)
        dy = self._number(ball.get("dy"), 0.0)
        speed_name = "fast" if abs(dx) + abs(dy) >= 3.0 else "normal"
        target_y = py - bh
        eta = None
        landing_x = None
        if dy > 0.0:
            eta = max(0.0, (target_y - by) / dy)
            landing_x = self._reflected_x(bx, dx, eta)
            if eta <= 7.0:
                phase = "imminent"
                urgency = "imminent"
            elif eta <= 18.0:
                phase = "descending"
                urgency = "near"
            else:
                phase = "descending"
                urgency = "early"
        else:
            phase = "ascending"
            urgency = "early"

        if landing_x is None:
            lane = "unknown"
            target_left = 76.0 if self.strategy == "SURVIVE" else (
                24.0 if self.strategy == "LEFT" else 120.0
            )
        else:
            center = landing_x + bw / 2.0
            if center < 56.0:
                lane = "left"
            elif center > 104.0:
                lane = "right"
            else:
                lane = "center"
            target_left = center - pw / 2.0

        paddle_center = px + pw / 2.0
        if landing_x is None:
            relation = "centered" if abs(paddle_center - 80.0) <= 20.0 else (
                "left" if paddle_center < 80.0 else "right"
            )
        else:
            wanted = landing_x + bw / 2.0
            gap = wanted - paddle_center
            if abs(gap) <= max(4.0, pw / 3.0):
                relation = "overlap"
            elif gap < 0:
                relation = "left_of_landing"
            else:
                relation = "right_of_landing"

        return {
            "phase": phase,
            "eta": None if eta is None else round(eta, 1),
            "landing_x": None if landing_x is None else round(landing_x, 1),
            "landing_lane": lane,
            "urgency": urgency,
            "paddle_relation": relation,
            "speed": speed_name,
            "target_left": self._clamp(target_left, 8.0, 144.0),
        }

    def _side_status(self, obs: dict[str, Any]) -> dict[str, Any]:
        bitmap = obs.get("brick_bitmap")
        columns = []
        if isinstance(bitmap, list):
            for col in (0, 1, 16, 17):
                occupied = False
                for row in bitmap:
                    if isinstance(row, list) and len(row) > col and bool(row[col]):
                        occupied = True
                        break
                columns.append(occupied)
        while len(columns) < 4:
            columns.append(True)
        return {
            "left_open": not (columns[0] or columns[1]),
            "right_open": not (columns[2] or columns[3]),
            "remaining": self._integer(obs.get("bricks_remaining"), 108),
            "wall": self._integer(obs.get("wall_index"), 0),
            "walls_cleared": self._integer(obs.get("walls_cleared"), 0),
        }

    def _event(self, obs: dict[str, Any]) -> str:
        if self.previous is None:
            return "start"
        old = self.previous
        old_lives = self._integer(old.get("lives"), 0)
        new_lives = self._integer(obs.get("lives"), old_lives)
        old_score = self._number(old.get("score"), 0.0)
        new_score = self._number(obs.get("score"), old_score)
        old_wall = self._integer(old.get("wall_index"), 0)
        new_wall = self._integer(obs.get("wall_index"), old_wall)
        old_ball = self._ball(old)
        new_ball = self._ball(obs)
        if new_lives < old_lives:
            return "life_loss"
        if new_wall != old_wall:
            return "wall_transition"
        if new_score > old_score:
            return "brick_hit"
        if old_ball is None and new_ball is not None:
            return "serve"
        if old_ball is not None and new_ball is None:
            return "ball_lost"
        return "none"

    def _derived(self, obs: dict[str, Any]) -> dict[str, Any]:
        flight = self._projection(obs)
        side = self._side_status(obs)
        event = self._event(obs)
        frame = self._integer(obs.get("frame"), self.last_frame + 1)
        if event in ("life_loss", "wall_transition", "brick_hit"):
            self.s1_answer = None
            self.plan_version += 1
            self.last_major_event_frame = frame
        return {
            "frame": frame,
            "lives": self._integer(obs.get("lives"), 0),
            "score": self._integer(self._number(obs.get("score"), 0.0)),
            "event": event,
            "serve_required": bool(obs.get("serve_required", False)),
            "flight_phase": flight["phase"],
            "landing_lane": flight["landing_lane"],
            "landing_urgency": flight["urgency"],
            "paddle_relation": flight["paddle_relation"],
            "speed_regime": flight["speed"],
            "landing_x": flight["landing_x"],
            "eta": flight["eta"],
            "fallback_action": self._movement_action(obs, flight, None, commit=False),
            "strategy": self.strategy,
            "left_open": side["left_open"],
            "right_open": side["right_open"],
            "bricks_remaining": side["remaining"],
            "wall_index": side["wall"],
            "walls_cleared": side["walls_cleared"],
            "plan_version": self.plan_version,
        }

    def _update_paddle_motion(self, obs: dict[str, Any]) -> None:
        """Estimate actual paddle motion from observations, not commands."""
        paddle = self._paddle(obs)
        current = self._number(paddle.get("x"), 99.0)
        if isinstance(self.previous, dict):
            old = self._paddle(self.previous)
            old_x = self._number(old.get("x"), current)
            delta = self._clamp(current - old_x, -8.0, 8.0)
            # This is only used to decide whether a one-frame neutral turn is
            # safer; it is not a replacement for the parent's projection.
            self.paddle_velocity = 0.65 * self.paddle_velocity + 0.35 * delta
        else:
            self.paddle_velocity = 0.0

    def _movement_action(
        self,
        obs: dict[str, Any],
        flight: dict[str, Any],
        steering_bias: str | None,
        *,
        commit: bool = True,
    ) -> str:
        """Translate the parent's target through a narrow reversal gate.

        The original point servo is retained outside a turn.  Near a target,
        however, an opposite command is withheld until the paddle is clearly
        past the target.  This is intentionally a small hysteresis, not a
        multi-frame landing plan.
        """
        if bool(obs.get("serve_required", False)):
            if commit:
                self.last_direction = None
                self.paddle_velocity = 0.0
            return "FIRE"
        if bool(obs.get("terminated", False)) or bool(obs.get("truncated", False)):
            return "NOOP"
        ball = self._ball(obs)
        if ball is None:
            return "NOOP"

        paddle = self._paddle(obs)
        px = self._number(paddle.get("x"), 99.0)
        pw = self._number(paddle.get("w"), 16.0)
        target = self._number(flight.get("target_left"), 76.0)

        # Preserve the parent's advisory bridge.  It cannot alter an imminent
        # catch, and the turn gate is applied after this translation.
        if flight.get("urgency") not in ("near", "imminent"):
            if steering_bias == "BIAS_LEFT":
                target -= 7.0
            elif steering_bias == "BIAS_RIGHT":
                target += 7.0
            elif steering_bias == "CENTER":
                target = 76.0
        target = self._clamp(target, 8.0, 144.0)
        gap = target - px
        deadband = 2.0 if flight.get("urgency") == "imminent" else 4.0
        if gap > deadband:
            wanted = "RIGHT"
        elif gap < -deadband:
            wanted = "LEFT"
        else:
            wanted = "NOOP"

        if not commit or wanted not in ("LEFT", "RIGHT"):
            return wanted

        prior = self.last_direction
        gate_eligible = (
            flight.get("urgency") == "imminent"
            and flight.get("paddle_relation") == "overlap"
        )
        if gate_eligible and prior in ("LEFT", "RIGHT") and wanted != prior:
            # A reversal inside this wider band is exactly the trace pattern
            # under test.  Permit one neutral frame, then let the original
            # servo reverse if the target still requests it.  The pair latch
            # prevents the gate from becoming the permanent NOOP bug seen in
            # the first try of this candidate.
            turn_band = 8.0 if flight.get("urgency") in ("near", "imminent") else 7.0
            clearly_crossed = gap < -turn_band if prior == "RIGHT" else gap > turn_band
            pair = (prior, wanted)
            now = self.control_frame if self.control_frame >= 0 else self.last_frame
            if not clearly_crossed:
                if self.turn_gate_pair != pair:
                    self.turn_gate_pair = pair
                    self.turn_gate_until = now + 1
                    if self.last_gate_frame != now:
                        try:
                            self.ctx.log(
                                "turn_gate",
                                frame=now,
                                prior=prior,
                                requested=wanted,
                                gap=round(gap, 1),
                                velocity=round(self.paddle_velocity, 2),
                            )
                        except Exception:
                            pass
                        self.last_gate_frame = now
                    return "NOOP"
                if now < self.turn_gate_until:
                    return "NOOP"
            self.turn_gate_pair = None
            self.turn_gate_until = -10_000
        else:
            self.turn_gate_pair = None
            self.turn_gate_until = -10_000

        if wanted != prior:
            self.last_direction_frame = self.control_frame if self.control_frame >= 0 else self.last_frame
        self.last_direction = wanted
        return wanted

    # ---------- model interfaces ----------

    def _questions(self, derived: dict[str, Any]) -> dict[str, Choice]:
        descriptions = {
            "TRACK": (
                "Follow the code-computed projected landing. Preserve paddle overlap; "
                "do not chase the ball's current x."
            ),
            "CENTER": (
                "Use a conservative center-paddle bias while the ball is far from "
                "the paddle; never override an imminent catch."
            ),
            "BIAS_LEFT": (
                "Use a small left shaping bias only if the code says the landing is "
                "not near or imminent and the catch remains safe."
            ),
            "BIAS_RIGHT": (
                "Use a small right shaping bias only if the code says the landing is "
                "not near or imminent and the catch remains safe."
            ),
        }
        keys = list(descriptions)
        shift = self.request_number % len(keys)
        order_a = keys[shift:] + keys[:shift]
        order_b = list(reversed(order_a))
        criteria_a = {key: descriptions[key] for key in order_a}
        criteria_b = {key: descriptions[key] for key in order_b}
        common = (
            "Choose one steering bias for this exact state. Code already computed "
            "the geometry and fallback action; do not calculate distances. "
            "If the landing urgency is near or imminent, TRACK is safest. "
            "Facts: " + repr(derived)
        )
        return {
            "steering_a": Choice(common, criteria_a),
            "steering_b": Choice(common, criteria_b),
        }

    def _launch_jev(self, obs: dict[str, Any], derived: dict[str, Any]) -> None:
        if not self.s1_enabled or self.s1_future is not None:
            return
        frame = derived["frame"]
        if derived["serve_required"]:
            return
        if frame - self.s1_request_frame < 18:
            return
        # A fresh, compact state is captured before spawning.  The answer is
        # associated with both this frame and the current strategic version.
        state = {
            "game": "Atari 2600 Breakout ordinary mode 0 difficulty 0",
            "observation": {
                "frame": frame,
                "lives": derived["lives"],
                "score": derived["score"],
                "event": derived["event"],
                "serve_required": derived["serve_required"],
                "flight_phase": derived["flight_phase"],
                "landing_lane": derived["landing_lane"],
                "landing_urgency": derived["landing_urgency"],
                "paddle_relation": derived["paddle_relation"],
                "speed_regime": derived["speed_regime"],
                "landing_x": derived["landing_x"],
                "eta": derived["eta"],
                "fallback_action": derived["fallback_action"],
                "strategy": derived["strategy"],
                "left_open": derived["left_open"],
                "right_open": derived["right_open"],
                "bricks_remaining": derived["bricks_remaining"],
                "wall_index": derived["wall_index"],
                "walls_cleared": derived["walls_cleared"],
            },
            "rules": {
                "actions": ["NOOP", "FIRE", "RIGHT", "LEFT"],
                "serve": "FIRE is needed only when serve_required is true",
                "priority": "keep the ball catchable before attempting side shaping",
            },
        }
        questions = self._questions(derived)
        version = self.plan_version
        self.request_number += 1
        self.s1_request_frame = frame
        self.s1_request_version = version

        def call():
            return self.ctx.decide(state, questions, timeout_s=1.5)

        try:
            self.s1_future = self.ctx.spawn(call)
        except Exception as exc:
            self.s1_future = None
            self.ctx.log("jev_spawn_failed", error=type(exc).__name__)

    @staticmethod
    def _answer_value(answer: Any) -> str | None:
        value = getattr(answer, "value", None)
        return value if isinstance(value, str) else None

    @staticmethod
    def _answer_probability(answer: Any, value: str) -> float:
        probabilities = getattr(answer, "probabilities", None)
        if isinstance(probabilities, dict) and probabilities:
            try:
                return float(probabilities.get(value, 0.0))
            except (TypeError, ValueError):
                return 0.0
        # Some pre-flight stand-ins provide only a value.  Treat it as usable;
        # freshness and the two-question agreement still gate its effect.
        return 1.0 if isinstance(getattr(answer, "value", None), str) else 0.0

    def _collect_jev(self, current_frame: int) -> None:
        future = self.s1_future
        if future is None or not future.done():
            return
        self.s1_future = None
        try:
            answers = future.result()
            a = answers.get("steering_a") if isinstance(answers, dict) else None
            b = answers.get("steering_b") if isinstance(answers, dict) else None
            av = self._answer_value(a)
            bv = self._answer_value(b)
            allowed = {"TRACK", "CENTER", "BIAS_LEFT", "BIAS_RIGHT"}
            if av in allowed and av == bv:
                confidence = min(self._answer_probability(a, av), self._answer_probability(b, bv))
                if confidence >= 0.45:
                    self.s1_answer = av
                    self.s1_answer_frame = self.s1_request_frame
                    self.s1_answer_version = self.s1_request_version
                    self.ctx.log(
                        "jev_bias_accepted",
                        frame=current_frame,
                        bias=av,
                        confidence=round(confidence, 3),
                    )
        except (TimeoutError, RuntimeError, BudgetExceeded, EpisodeOver) as exc:
            self.ctx.log("jev_unavailable", error=type(exc).__name__)
        except Exception as exc:
            # Garbage stand-in answers must not terminate a game.
            self.ctx.log("jev_parse_ignored", error=type(exc).__name__)

    def _astra_prompt(self, obs: dict[str, Any], derived: dict[str, Any]) -> str:
        return (
            "You are the slow strategic layer for Atari 2600 Breakout mode 0. "
            "The fast controller already predicts paddle interception every native "
            "frame. Decide only whether a safe long-horizon side shaping attempt is "
            "worthwhile. Return exactly one token: LEFT, RIGHT, or SURVIVE. Choose "
            "SURVIVE when the ball is fast, a side gap is not reachable, or the "
            "current life should be protected. Never provide a frame-by-frame trace.\n"
            "Current derived state:\n" + repr(derived) + "\n"
            "The complete observation has serve, ball, paddle, score, lives, and "
            "brick bitmap fields:\n" + repr({
                "frame": obs.get("frame"),
                "ball": obs.get("ball"),
                "paddle": obs.get("paddle"),
                "brick_bitmap": obs.get("brick_bitmap"),
                "bricks_remaining": obs.get("bricks_remaining"),
                "wall_index": obs.get("wall_index"),
            })
        )

    def _launch_astra(self, obs: dict[str, Any], derived: dict[str, Any]) -> None:
        if self.astra_future is not None or self.astra_calls >= 2:
            return
        if derived["serve_required"] or self._ball(obs) is None:
            return
        if self.astra_calls > 0 and derived["frame"] - self.astra_started_frame < 180:
            return
        try:
            remaining = self.ctx.budget.remaining()
            if int(remaining.get("system2_calls", 0)) <= 0:
                return
        except Exception:
            pass
        prompt = self._astra_prompt(obs, derived)
        self.astra_calls += 1
        self.astra_started_frame = derived["frame"]

        def call():
            return self.ctx.think(prompt, timeout_s=60.0)

        try:
            self.astra_future = self.ctx.spawn(call)
        except Exception as exc:
            self.astra_future = None
            self.ctx.log("astra_spawn_failed", error=type(exc).__name__)

    def _collect_astra(self, current_frame: int) -> None:
        future = self.astra_future
        if future is None or not future.done():
            return
        self.astra_future = None
        try:
            result = future.result()
            text = getattr(result, "text", "")
            if not isinstance(text, str):
                return
            stripped = text.strip().upper()
            match = re.match(r"^(LEFT|RIGHT|SURVIVE|SURVIVAL)\b", stripped)
            if match is None:
                return
            token = match.group(1)
            new_strategy = "SURVIVE" if token.startswith("SURV") else token
            if new_strategy != self.strategy:
                self.strategy = new_strategy
                self.plan_version += 1
                self.s1_answer = None
                self.ctx.log("astra_strategy_accepted", frame=current_frame, strategy=new_strategy)
        except (TimeoutError, RuntimeError, BudgetExceeded, EpisodeOver) as exc:
            self.ctx.log("astra_unavailable", error=type(exc).__name__)
        except Exception as exc:
            self.ctx.log("astra_parse_ignored", error=type(exc).__name__)

    # ---------- episode loop ----------

    def _safe_observe(self, game: Any) -> dict[str, Any]:
        try:
            observation = game.observe()
        except Exception:
            return {}
        return observation if isinstance(observation, dict) else {}

    def _current_bias(self, derived: dict[str, Any]) -> str | None:
        if self.s1_answer is None:
            return None
        if self.s1_answer_version != self.plan_version:
            return None
        if derived["frame"] - self.s1_answer_frame > 36:
            return None
        if derived["landing_urgency"] in ("near", "imminent"):
            return None
        return self.s1_answer

    def run_episode(self, game: Any) -> None:
        ctx = self.ctx
        realtime = bool(getattr(game, "tick_s", 0.0))
        if not realtime:
            # Tier-1 compatibility is intentionally simple; the target is tier 2.
            while not ctx.done and not bool(getattr(game, "done", False)):
                obs = self._safe_observe(game)
                self._update_paddle_motion(obs)
                frame = self._integer(obs.get("frame"), self.last_frame + 1)
                self.control_frame = frame
                self.last_frame = frame
                flight = self._projection(obs)
                action = self._movement_action(obs, flight, None)
                ctx.act(action)
            return

        while not ctx.done and not bool(getattr(game, "done", False)):
            obs = self._safe_observe(game)
            self._update_paddle_motion(obs)
            derived = self._derived(obs)
            frame = derived["frame"]
            self.control_frame = frame
            self.last_frame = frame
            self._collect_jev(frame)
            self._collect_astra(frame)

            if derived["event"] in ("life_loss", "wall_transition"):
                self._launch_astra(obs, derived)
            elif self.astra_calls == 0:
                self._launch_astra(obs, derived)

            self._launch_jev(obs, derived)
            bias = self._current_bias(derived)
            flight = self._projection(obs)
            action = self._movement_action(obs, flight, bias)
            if action not in ACTIONS:
                action = "NOOP"
            self.last_posted = action
            self.last_frame = frame
            ctx.act(action)
            self.previous = obs
            try:
                ctx.wait_for_tick()
            except (BudgetExceeded, EpisodeOver):
                return


def build(ctx: Any) -> Harness:
    return Harness(ctx)
