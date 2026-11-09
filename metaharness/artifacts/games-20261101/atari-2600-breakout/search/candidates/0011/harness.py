"""Breakout controller with an impact-window actuator guard.

The strategic/model plumbing is deliberately advisory.  The change tested by
this candidate is in the fast controller: it filters and commits a landing
lane as the ball descends, then uses observed paddle motion to counter-steer
before the paddle overshoots that lane.  This is important because NOOP is not
an instantaneous brake in the Atari cartridge.
"""
from __future__ import annotations

import re
from typing import Any

from metaharness.core.contract import BudgetExceeded, EpisodeOver
from metaharness.models.jev import Choice


ACTIONS = ("NOOP", "FIRE", "RIGHT", "LEFT")
DIRECTIONS = {"RIGHT", "LEFT"}


class Harness:
    def __init__(self, ctx: Any):
        self.ctx = ctx
        self.previous: dict[str, Any] | None = None
        self.last_frame = -1
        self.strategy = "SURVIVE"
        self.plan_version = 0

        # Jev and Astra never sit in the real-time path.
        self.jev_future = None
        self.jev_request_frame = -10_000
        self.jev_request_version = -1
        self.jev_answer: str | None = None
        self.jev_answer_frame = -10_000
        self.jev_answer_version = -1
        self.request_number = 0
        self.jev_enabled = getattr(ctx, "system1", object()) is not None

        self.astra_future = None
        self.astra_started = -10_000
        self.astra_calls = 0

        # Measured actuator state.  The previous candidates treated NOOP as a
        # stop; this state makes the braking decision explicit.
        self.prev_paddle_x: float | None = None
        self.paddle_velocity = 0.0
        self.filtered_target: float | None = None
        self.committed_target: float | None = None
        self.brake_ticks = 0
        self.brake_action: str | None = None
        self.last_direction: str | None = None
        self.last_direction_frame = -10_000
        self.last_posted = "NOOP"

    # ---------- defensive primitive helpers ----------

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

    @staticmethod
    def _ball(obs: Any) -> dict[str, Any] | None:
        ball = obs.get("ball") if isinstance(obs, dict) else None
        return ball if isinstance(ball, dict) else None

    @staticmethod
    def _paddle(obs: Any) -> dict[str, Any]:
        paddle = obs.get("paddle") if isinstance(obs, dict) else None
        return paddle if isinstance(paddle, dict) else {}

    def _frame(self, obs: dict[str, Any]) -> int:
        fallback = self.last_frame + 1
        return self._integer(obs.get("frame"), fallback)

    # ---------- event and actuator bookkeeping ----------

    def _event(self, obs: dict[str, Any]) -> str:
        old = self.previous
        if old is None:
            return "start"
        old_lives = self._integer(old.get("lives"), 0)
        new_lives = self._integer(obs.get("lives"), old_lives)
        if new_lives < old_lives:
            return "life_loss"
        old_wall = self._integer(old.get("wall_index"), 0)
        new_wall = self._integer(obs.get("wall_index"), old_wall)
        if new_wall != old_wall:
            return "wall_transition"
        old_score = self._number(old.get("score"), 0.0)
        new_score = self._number(obs.get("score"), old_score)
        if new_score > old_score:
            return "brick_hit"
        old_ball = self._ball(old)
        new_ball = self._ball(obs)
        if old_ball is None and new_ball is not None:
            return "serve"
        if old_ball is not None and new_ball is None:
            return "ball_lost"
        return "none"

    def _reset_targeting(self) -> None:
        self.filtered_target = None
        self.committed_target = None
        self.brake_ticks = 0
        self.brake_action = None

    def _prepare_actuator(self, obs: dict[str, Any], event: str) -> None:
        """Update velocity and invalidate target memory at actual collisions."""
        paddle = self._paddle(obs)
        x = self._number(paddle.get("x"), 99.0)
        if self.prev_paddle_x is None:
            self.paddle_velocity = 0.0
        else:
            observed_delta = self._clamp(x - self.prev_paddle_x, -8.0, 8.0)
            # The EMA suppresses one sticky-action artifact without hiding a
            # sustained move.  It is measured from RAM, not inferred from the
            # requested action.
            self.paddle_velocity = 0.60 * self.paddle_velocity + 0.40 * observed_delta
        self.prev_paddle_x = x

        old_ball = self._ball(self.previous) if self.previous is not None else None
        ball = self._ball(obs)
        old_dx = self._number(old_ball.get("dx"), 0.0) if old_ball else 0.0
        new_dx = self._number(ball.get("dx"), 0.0) if ball else 0.0
        old_dy = self._number(old_ball.get("dy"), 0.0) if old_ball else 0.0
        new_dy = self._number(ball.get("dy"), 0.0) if ball else 0.0
        velocity_flip = (
            old_ball is not None
            and ball is not None
            and old_dx != 0.0
            and new_dx != 0.0
            and old_dx * new_dx < 0.0
        )
        vertical_bounce = (
            old_ball is not None
            and ball is not None
            and old_dy != 0.0
            and new_dy != 0.0
            and old_dy * new_dy < 0.0
        )
        if event in ("start", "serve", "life_loss", "ball_lost", "brick_hit", "wall_transition"):
            self._reset_targeting()
        elif velocity_flip or vertical_bounce:
            # A wall or brick reflection changes the landing problem.  Do not
            # carry the old committed lane through that discontinuity.
            self._reset_targeting()

    # ---------- exact reflected-coordinate geometry ----------

    @staticmethod
    def _reflect(center_x: float, dx: float, frames: float) -> float:
        # The decoder's ball x is its left edge.  Work with the ball centre and
        # reflect it between the inside edges of the 160-pixel playfield.
        low, high = 9.0, 151.0
        span = high - low
        if span <= 0.0:
            return center_x
        phase = (center_x - low + dx * max(0.0, frames)) % (2.0 * span)
        return low + phase if phase <= span else high - (phase - span)

    def _lane_name(self, landing_center: float | None) -> str:
        if landing_center is None:
            return "unknown"
        if landing_center < 56.0:
            return "left"
        if landing_center > 104.0:
            return "right"
        return "center"

    def _projection(self, obs: dict[str, Any], event: str = "none") -> dict[str, Any]:
        ball = self._ball(obs)
        paddle = self._paddle(obs)
        px = self._number(paddle.get("x"), 99.0)
        py = self._number(paddle.get("y"), 189.0)
        pw = self._number(paddle.get("w"), 16.0)
        if ball is None or not bool(obs.get("ball_in_play", False)):
            return {
                "phase": "serve",
                "urgency": "serve",
                "eta": None,
                "landing_center": None,
                "target_left": self._clamp(px, 8.0, 144.0),
                "lane": "unknown",
                "relation": "unknown",
                "speed": "unknown",
                "capture": False,
                "control_mode": "serve",
            }

        bx = self._number(ball.get("x"), 80.0)
        by = self._number(ball.get("y"), 100.0)
        bw = self._number(ball.get("w"), 2.0)
        bh = self._number(ball.get("h"), 4.0)
        dx = self._number(ball.get("dx"), 0.0)
        dy = self._number(ball.get("dy"), 0.0)
        speed = "fast" if abs(dx) + abs(dy) >= 3.0 else "normal"
        eta: float | None = None
        raw_center: float | None = None
        if dy > 0.0:
            eta = max(0.0, (py - bh - by) / dy)
            raw_center = self._reflect(bx + bw / 2.0, dx, eta)
            # Smooth only the continuous approach.  Collision events and
            # direction flips clear this value in _prepare_actuator.
            raw_left = raw_center - pw / 2.0
            if self.filtered_target is None:
                self.filtered_target = raw_left
            else:
                alpha = 0.30 if eta > 14.0 else 0.20
                self.filtered_target = (1.0 - alpha) * self.filtered_target + alpha * raw_left
            if eta <= 14.0 and self.committed_target is None:
                # Commit the lane before the impact window.  This prevents a
                # one-frame prediction wobble from causing a direction flip.
                self.committed_target = self.filtered_target
            target_left = self.committed_target if self.committed_target is not None else self.filtered_target
            if target_left is None:
                target_left = raw_left
            target_left = self._clamp(target_left, 8.0, 144.0)
            landing_center = target_left + pw / 2.0
            if eta <= 6.0:
                phase, urgency = "imminent", "imminent"
            elif eta <= 16.0:
                phase, urgency = "descending", "near"
            else:
                phase, urgency = "descending", "early"
        else:
            # While ascending, pre-position gently but do not chase a lane that
            # will be replaced by the next brick/wall reflection.
            self.committed_target = None
            if self.strategy == "LEFT":
                target_left = 22.0
            elif self.strategy == "RIGHT":
                target_left = 122.0
            else:
                target_left = 76.0
            landing_center = target_left + pw / 2.0
            phase, urgency = "ascending", "early"

        paddle_center = px + pw / 2.0
        safe_radius = (pw + bw) / 2.0 - 1.0
        gap = landing_center - paddle_center
        capture = abs(gap) <= max(1.0, safe_radius)
        if capture:
            relation = "overlap"
        elif gap < 0.0:
            relation = "left_of_landing"
        else:
            relation = "right_of_landing"
        if eta is None:
            control_mode = "preposition"
        elif self.brake_ticks > 0:
            control_mode = "brake"
        else:
            control_mode = "track"
        return {
            "phase": phase,
            "urgency": urgency,
            "eta": None if eta is None else round(eta, 1),
            "landing_center": round(landing_center, 1),
            "target_left": round(target_left, 1),
            "lane": self._lane_name(raw_center if raw_center is not None else landing_center),
            "relation": relation,
            "speed": speed,
            "capture": bool(capture),
            "control_mode": control_mode,
        }

    # ---------- code-owned control ----------

    def _base_action(self, obs: dict[str, Any], flight: dict[str, Any], bias: str | None) -> str:
        if bool(obs.get("serve_required", False)):
            return "FIRE"
        if bool(obs.get("terminated", False)) or bool(obs.get("truncated", False)):
            return "NOOP"
        if self._ball(obs) is None:
            return "NOOP"

        paddle = self._paddle(obs)
        px = self._number(paddle.get("x"), 99.0)
        pw = self._number(paddle.get("w"), 16.0)
        target = self._number(flight.get("target_left"), 76.0)
        urgency = flight.get("urgency")
        if urgency not in ("near", "imminent"):
            if bias == "BIAS_LEFT":
                target -= 6.0
            elif bias == "BIAS_RIGHT":
                target += 6.0
            elif bias == "CENTER":
                target = 76.0
        target = self._clamp(target, 8.0, 144.0)
        gap = target - px
        deadband = 2.0 if urgency == "imminent" else 3.5
        if gap > deadband:
            return "RIGHT"
        if gap < -deadband:
            return "LEFT"
        return "NOOP"

    def _guarded_action(self, obs: dict[str, Any], flight: dict[str, Any], base: str, frame: int) -> str:
        """Apply the one experimental change: commit-and-brake near impact."""
        if base in ("FIRE", "NOOP") or flight.get("eta") is None:
            return base
        if flight.get("urgency") not in ("near", "imminent"):
            return base

        paddle = self._paddle(obs)
        px = self._number(paddle.get("x"), 99.0)
        pw = self._number(paddle.get("w"), 16.0)
        target = self._number(flight.get("target_left"), px)
        gap = target - px
        v = self.paddle_velocity
        eta = self._number(flight.get("eta"), 0.0)
        safe = max(2.0, pw * 0.42)

        # Finish an already-started counter-steer instead of alternating on
        # successive frames.  It is released if the ball is no longer close.
        if self.brake_ticks > 0 and self.brake_action in DIRECTIONS:
            if abs(gap) <= safe + 5.0 or eta <= 4.0:
                self.brake_ticks -= 1
                return self.brake_action
            self.brake_ticks = 0
            self.brake_action = None

        moving_toward_target = gap * v > 0.35
        overshoot_distance = abs(v) * max(2.0, min(eta, 4.0)) + 2.5
        inside_capture = abs(gap) <= safe
        if abs(v) > 0.35 and (inside_capture or (moving_toward_target and abs(gap) <= overshoot_distance)):
            # NOOP is not a brake.  A short opposite pulse is safer than
            # allowing paddle inertia to carry the edge past the ball.
            brake = "LEFT" if v > 0.0 else "RIGHT"
            self.brake_action = brake
            self.brake_ticks = 2 if eta > 4.0 else 1
            return brake

        # Do not chase a newly flipped target in the last few frames unless it
        # is clearly outside the paddle.  The committed target normally makes
        # this branch rare, but it protects against a malformed/late velocity.
        if (
            base in DIRECTIONS
            and self.last_direction in DIRECTIONS
            and base != self.last_direction
            and frame - self.last_direction_frame <= 2
            and abs(gap) <= safe + 4.0
        ):
            return "NOOP" if abs(v) <= 0.35 else ("LEFT" if v > 0.0 else "RIGHT")
        return base

    def _choose_action(self, obs: dict[str, Any], flight: dict[str, Any], bias: str | None, frame: int) -> str:
        action = self._base_action(obs, flight, bias)
        action = self._guarded_action(obs, flight, action, frame)
        if action not in ACTIONS:
            action = "NOOP"
        if action in DIRECTIONS:
            if action != self.last_direction:
                self.last_direction_frame = frame
            self.last_direction = action
        self.last_posted = action
        return action

    # ---------- derived Jev state ----------

    def _side_status(self, obs: dict[str, Any]) -> tuple[bool, bool, int, int, int]:
        bitmap = obs.get("brick_bitmap")
        side = []
        if isinstance(bitmap, list):
            for col in (0, 1, 16, 17):
                occupied = any(
                    isinstance(row, list) and len(row) > col and bool(row[col])
                    for row in bitmap
                )
                side.append(occupied)
        while len(side) < 4:
            side.append(True)
        return (
            not (side[0] or side[1]),
            not (side[2] or side[3]),
            self._integer(obs.get("bricks_remaining"), 108),
            self._integer(obs.get("wall_index"), 0),
            self._integer(obs.get("walls_cleared"), 0),
        )

    def _derived(self, obs: dict[str, Any], event: str, frame: int) -> dict[str, Any]:
        flight = self._projection(obs, event)
        left_open, right_open, remaining, wall, cleared = self._side_status(obs)
        return {
            "frame": frame,
            "event": event,
            "lives": self._integer(obs.get("lives"), 0),
            "score": self._integer(self._number(obs.get("score"), 0.0)),
            "serve_required": bool(obs.get("serve_required", False)),
            "flight_phase": flight["phase"],
            "landing_lane": flight["lane"],
            "landing_urgency": flight["urgency"],
            "paddle_relation": flight["relation"],
            "speed_regime": flight["speed"],
            "eta": flight["eta"],
            "landing_center": flight["landing_center"],
            "capture_corridor": flight["capture"],
            "control_mode": flight["control_mode"],
            "paddle_motion": (
                "moving_right" if self.paddle_velocity > 0.35 else
                "moving_left" if self.paddle_velocity < -0.35 else "still"
            ),
            "fallback_action": self._base_action(obs, flight, None),
            "strategy": self.strategy,
            "left_open": left_open,
            "right_open": right_open,
            "bricks_remaining": remaining,
            "wall_index": wall,
            "walls_cleared": cleared,
            "plan_version": self.plan_version,
        }

    # ---------- Jev advisory layer ----------

    @staticmethod
    def _answer_value(answer: Any) -> str | None:
        value = getattr(answer, "value", None)
        return value if isinstance(value, str) else None

    @staticmethod
    def _prob(answer: Any, option: str) -> float:
        p = getattr(answer, "probabilities", None)
        if isinstance(p, dict) and p:
            try:
                return float(p.get(option, 0.0))
            except (TypeError, ValueError):
                return 0.0
        return 1.0 if isinstance(getattr(answer, "value", None), str) else 0.0

    def _questions(self, facts: dict[str, Any]) -> dict[str, Choice]:
        options = {
            "TRACK": "Follow the committed code-computed landing lane and preserve overlap.",
            "CENTER": "Use a small center bias only while the landing is early; never override the code brake or catch.",
            "BIAS_LEFT": "Shape a safe, early return slightly left; do not use this near or at impact.",
            "BIAS_RIGHT": "Shape a safe, early return slightly right; do not use this near or at impact.",
        }
        keys = list(options)
        shift = self.request_number % len(keys)
        a = keys[shift:] + keys[:shift]
        b = list(reversed(a))
        instructions = (
            "Choose the advisory steering mode for the supplied Breakout facts. "
            "The code has already done geometry and actuator arithmetic. "
            "When `landing_urgency` is near or imminent, choose TRACK. "
            "Never invent an action or a new candidate. Facts: " + repr(facts)
        )
        return {
            "mode_a": Choice(instructions, {k: options[k] for k in a}),
            "mode_b": Choice(instructions, {k: options[k] for k in b}),
        }

    def _launch_jev(self, obs: dict[str, Any], facts: dict[str, Any]) -> None:
        if not self.jev_enabled or self.jev_future is not None:
            return
        frame = facts["frame"]
        if facts["serve_required"] or frame - self.jev_request_frame < 24:
            return
        state = {
            "game": "Atari 2600 Breakout, ordinary mode 0, difficulty 0",
            "observation": facts,
            "rules": {
                "legal_actions": ["NOOP", "FIRE", "RIGHT", "LEFT"],
                "FIRE": "only for serve_required",
                "priority": "catchability before shaping",
                "control": "code commits a descending landing lane and counter-steers measured paddle motion",
            },
        }
        questions = self._questions(facts)
        self.request_number += 1
        self.jev_request_frame = frame
        self.jev_request_version = self.plan_version

        def call():
            return self.ctx.decide(state, questions, timeout_s=1.4)

        try:
            self.jev_future = self.ctx.spawn(call)
        except Exception as exc:
            self.jev_future = None
            self.ctx.log("jev_spawn_failed", error=type(exc).__name__)

    def _collect_jev(self, frame: int) -> None:
        future = self.jev_future
        if future is None or not future.done():
            return
        self.jev_future = None
        try:
            answers = future.result()
            a = answers.get("mode_a") if isinstance(answers, dict) else None
            b = answers.get("mode_b") if isinstance(answers, dict) else None
            av, bv = self._answer_value(a), self._answer_value(b)
            allowed = {"TRACK", "CENTER", "BIAS_LEFT", "BIAS_RIGHT"}
            if av in allowed and av == bv:
                confidence = min(self._prob(a, av), self._prob(b, bv))
                if confidence >= 0.45:
                    self.jev_answer = av
                    self.jev_answer_frame = self.jev_request_frame
                    self.jev_answer_version = self.jev_request_version
                    self.ctx.log("jev_mode_accepted", frame=frame, mode=av, confidence=round(confidence, 3))
        except (TimeoutError, RuntimeError, BudgetExceeded, EpisodeOver) as exc:
            self.ctx.log("jev_unavailable", error=type(exc).__name__)
        except Exception as exc:
            self.ctx.log("jev_parse_ignored", error=type(exc).__name__)

    def _bias(self, facts: dict[str, Any]) -> str | None:
        if self.jev_answer is None:
            return None
        if self.jev_answer_version != self.plan_version:
            return None
        if facts["frame"] - self.jev_answer_frame > 42:
            return None
        if facts["landing_urgency"] in ("near", "imminent"):
            return None
        return self.jev_answer

    # ---------- Astra strategic layer ----------

    def _astra_prompt(self, obs: dict[str, Any], facts: dict[str, Any]) -> str:
        return (
            "You are the slow strategic adviser for ordinary Atari 2600 Breakout. "
            "The code controller predicts and catches the ball every native frame; "
            "you may only choose LEFT, RIGHT, or SURVIVE as a long-lived shaping "
            "preference. Prefer SURVIVE when the return is fast or a side tunnel is "
            "not evidenced. Return one token first and no frame-by-frame actions.\n"
            "Derived facts:\n" + repr(facts) + "\n"
            "Relevant observation:\n" + repr({
                "frame": obs.get("frame"),
                "ball": obs.get("ball"),
                "paddle": obs.get("paddle"),
                "brick_bitmap": obs.get("brick_bitmap"),
                "bricks_remaining": obs.get("bricks_remaining"),
                "wall_index": obs.get("wall_index"),
            })
        )

    def _launch_astra(self, obs: dict[str, Any], facts: dict[str, Any]) -> None:
        if self.astra_future is not None or self.astra_calls >= 1:
            return
        if facts["serve_required"] or self._ball(obs) is None:
            return
        try:
            if int(self.ctx.budget.remaining().get("system2_calls", 0)) <= 0:
                return
        except Exception:
            pass
        self.astra_calls += 1
        self.astra_started = facts["frame"]
        prompt = self._astra_prompt(obs, facts)

        def call():
            return self.ctx.think(prompt, timeout_s=60.0)

        try:
            self.astra_future = self.ctx.spawn(call)
        except Exception as exc:
            self.astra_future = None
            self.ctx.log("astra_spawn_failed", error=type(exc).__name__)

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
            match = re.match(r"^\s*(LEFT|RIGHT|SURVIVE|SURVIVAL)\b", text.upper())
            if match is None:
                return
            new = "SURVIVE" if match.group(1).startswith("SURV") else match.group(1)
            if new != self.strategy:
                self.strategy = new
                self.plan_version += 1
                self.jev_answer = None
                self._reset_targeting()
                self.ctx.log("astra_strategy_accepted", frame=frame, strategy=new)
        except (TimeoutError, RuntimeError, BudgetExceeded, EpisodeOver) as exc:
            self.ctx.log("astra_unavailable", error=type(exc).__name__)
        except Exception as exc:
            self.ctx.log("astra_parse_ignored", error=type(exc).__name__)

    # ---------- episode loop ----------

    @staticmethod
    def _observe(game: Any) -> dict[str, Any]:
        try:
            value = game.observe()
        except Exception:
            return {}
        return value if isinstance(value, dict) else {}

    def run_episode(self, game: Any) -> None:
        ctx = self.ctx
        realtime = bool(getattr(game, "tick_s", 0.0))
        if not realtime:
            # The target is realtime; retaining this small path keeps preflight
            # and generic contract checks harmless.
            while not ctx.done and not bool(getattr(game, "done", False)):
                obs = self._observe(game)
                event = self._event(obs)
                self._prepare_actuator(obs, event)
                frame = self._frame(obs)
                flight = self._projection(obs, event)
                ctx.act(self._choose_action(obs, flight, None, frame))
                self.previous = obs
                self.last_frame = frame
            return

        while not ctx.done and not bool(getattr(game, "done", False)):
            obs = self._observe(game)
            event = self._event(obs)
            frame = self._frame(obs)
            self._prepare_actuator(obs, event)
            self._collect_jev(frame)
            self._collect_astra(frame)

            facts = self._derived(obs, event, frame)
            if event in ("life_loss", "wall_transition") or self.astra_calls == 0:
                self._launch_astra(obs, facts)
            self._launch_jev(obs, facts)

            flight = self._projection(obs, event)
            action = self._choose_action(obs, flight, self._bias(facts), frame)
            try:
                ctx.act(action)
            except (BudgetExceeded, EpisodeOver):
                return
            self.previous = obs
            self.last_frame = frame
            try:
                ctx.wait_for_tick()
            except (BudgetExceeded, EpisodeOver):
                return


def build(ctx: Any) -> Harness:
    return Harness(ctx)
