"""Candidate 0009: full-rate traverses with feedback-calibrated boundaries.

The controller deliberately keeps the useful part of candidate 0006--a broad,
committed lateral traverse with FIRE held on every tick--while importing the
closed-loop observations from candidate 0008.  The latter are used to learn the
actual travel envelope, detect a stalled command, remember shot/life events,
and choose a target-ordering hint.  They never turn the actuator into a slow,
selective, model-controlled loop.

Only the public symbolic observation and the HarnessContext interface are used.
"""
from __future__ import annotations

import json
import math
import statistics
from collections import deque
from concurrent.futures import CancelledError

from metaharness.models.jev import Choice


ACTIONS = ("noop", "fire", "right", "left", "right_fire", "left_fire")
POLICIES = ("WIDE", "LOWEST", "LEFT_EDGE", "RIGHT_EDGE", "KEEP")


def number(value, default=0.0):
    try:
        result = float(value)
        return result if math.isfinite(result) else default
    except (TypeError, ValueError, OverflowError):
        return default


def integer(value, default=0):
    try:
        return int(number(value, default))
    except (TypeError, ValueError, OverflowError):
        return default


def clamp(value, low, high):
    return max(low, min(high, value))


def as_dict(value):
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            decoded = json.loads(value)
            return decoded if isinstance(decoded, dict) else {}
        except (TypeError, ValueError, json.JSONDecodeError):
            return {}
    return {}


def dict_items(value):
    return [item for item in value if isinstance(item, dict)] if isinstance(value, list) else []


def is_alive(value):
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "alive"}
    return bool(value)


def direction_of(action):
    text = str(action or "")
    if "left" in text:
        return -1
    if "right" in text:
        return 1
    return 0


def answer_value(answer):
    if isinstance(answer, dict):
        return answer.get("value", answer.get("choice"))
    return getattr(answer, "value", None)


def answer_probabilities(answer):
    if isinstance(answer, dict):
        value = answer.get("probabilities", {})
    else:
        value = getattr(answer, "probabilities", {})
    return value if isinstance(value, dict) else {}


def probability_for(answer, choice):
    wanted = str(choice).upper()
    for key, value in answer_probabilities(answer).items():
        if str(key).upper() == wanted:
            return number(value, 0.0)
    return 0.0


def answer_confidence(answer):
    value = answer.get("confidence") if isinstance(answer, dict) else getattr(answer, "confidence", None)
    if value is None:
        return None
    parsed = number(value, -1.0)
    return parsed if parsed >= 0.0 else None


def make_questions(frame):
    options = {
        "WIDE": (
            "Use a broad left-to-right traverse across the living formation, "
            "keeping FIRE held. Prefer this when visibility is uncertain or many "
            "columns remain; it is the safe throughput default."
        ),
        "LOWEST": (
            "Bias the next broad traverse toward the column containing the lowest "
            "surviving invader. Code still keeps a wide moving lane and holds FIRE."
        ),
        "LEFT_EDGE": (
            "Begin the next broad traverse toward the living left outer flank. "
            "This is an ordering hint, not permission to park at a boundary."
        ),
        "RIGHT_EDGE": (
            "Begin the next broad traverse toward the living right outer flank. "
            "This is an ordering hint, not permission to park at a boundary."
        ),
        "KEEP": (
            "Keep the current ordering hint for the next traverse if it remains "
            "valid; otherwise the code falls back to WIDE."
        ),
    }
    ordered = list(options.items())
    if (integer(frame) // 48) % 2:
        ordered.reverse()
    first = dict(ordered)
    second = dict(reversed(ordered))
    instruction = (
        "Choose a policy for the NEXT committed spatial traverse, not an immediate "
        "joystick action. Code computes coordinates, learns boundaries, predicts "
        "visible bomb risk, enforces legal actions, and holds FIRE on every tick. "
        "An empty bomb list is not evidence of safety. Do not count, calculate "
        "distances, invent a target, or request a selective fire/movement action. "
        "Use only the categorical evidence in state."
    )
    return {
        "policy_a": Choice(instruction, first),
        "policy_b": Choice(instruction, second),
    }


class JevMailbox:
    """At most one short System-1 request is in flight.

    A late answer is merely advice for a future route.  The game loop never
    waits for it, and every malformed/failed answer is converted to a marker.
    """

    def __init__(self, ctx):
        self.ctx = ctx
        self.enabled = getattr(ctx, "system1", True) is not None
        self.future = None
        self.tag = None

    def _log(self, *args, **fields):
        try:
            self.ctx.log(*args, **fields)
        except Exception:
            pass

    def submit(self, state, frame, lives, wave, version):
        if not self.enabled or self.future is not None or getattr(self.ctx, "done", False):
            return False
        try:
            if self.ctx.budget.remaining().get("system1_calls", 0) <= 2:
                self.enabled = False
                return False
        except Exception:
            pass
        tag = {
            "frame": integer(frame),
            "lives": integer(lives),
            "wave": integer(wave),
            "version": integer(version),
        }
        questions = make_questions(frame)

        def request():
            try:
                answers = self.ctx.decide(state, questions, timeout_s=0.65)
                return {"answers": answers, "tag": tag}
            except Exception as exc:
                # Timeout, malformed stand-in answers, and ordinary backend
                # errors are safe fallback events.  A persistent unavailable
                # backend also makes ctx.done true, so the main loop exits.
                return {"error": type(exc).__name__, "tag": tag}

        try:
            self.future = self.ctx.spawn(request)
            self.tag = tag
            return True
        except Exception as exc:
            self.future = None
            self.tag = None
            self._log("jev_spawn_failed", error=type(exc).__name__)
            return False

    def take(self):
        future = self.future
        if future is None or not future.done():
            return None
        self.future = None
        tag = self.tag or {}
        self.tag = None
        try:
            result = future.result()
        except CancelledError:
            return {"error": "cancelled", "tag": tag}
        except Exception as exc:
            return {"error": type(exc).__name__, "tag": tag}
        if not isinstance(result, dict):
            return {"error": "non_dict", "tag": tag}
        return result

    def stop(self):
        # A still-running ctx.decide belongs to the evaluator and is cancelled
        # when the episode closes.  There is no persistent Python worker loop.
        self.future = None
        self.tag = None


class Controller:
    """Code-owned full-rate sweep controller."""

    def __init__(self, ctx):
        self.ctx = ctx
        self.frame = 0
        self.lives = 3
        self.wave = 1
        self.score = 0.0
        self.px = 90.0
        self.py = 190.0
        self.last_obs = None
        self.prev_action = "noop"
        self.prev_shot_active = False
        self.alive_keys = set()
        self.alive = []
        self.columns = {}
        self.formation = deque(maxlen=30)
        self.vx = 0.0
        self.shot_speed = 2.0
        self.move_speed = 0.65
        self.move_samples = deque(maxlen=20)
        self.last_moved_frame = 0
        self.stalled = False

        # These are deliberately permissive probe limits.  A committed opening
        # route can expose the ROM's actual limit; repeated no-motion, rather
        # than one sticky tick, tightens the usable envelope.
        self.left_limit = 30.0
        self.right_limit = 149.0
        self.left_known = False
        self.right_known = False
        self.left_stall = 0
        self.right_stall = 0

        self.route = None
        self.route_id = 0
        self.route_count = 0
        self.direction = 1
        self.policy = "WIDE"
        self.advice = None
        self.policy_version = 0
        self.last_outcome = "NONE"
        self.life_event = False
        self.wave_event = False
        self.recovery_until = -1
        self.last_progress_frame = 0
        self.last_score = 0.0
        self.death_lanes = deque(maxlen=8)
        self.unseen_deaths = 0

        self.tracks = []
        self.bombs = []
        self.last_bomb_frame = -100000
        self.dodge_until = -1
        self.dodge_direction = 0
        self.actions = 0
        self.accepted = 0
        self.effective = 0
        self.safety_overrides = 0
        self.last_state_query = -100000

    def log(self, *args, **fields):
        try:
            self.ctx.log(*args, **fields)
        except Exception:
            pass

    def _player(self, obs):
        value = obs.get("player")
        if not isinstance(value, dict):
            return self.px, self.py
        return (
            clamp(number(value.get("x"), self.px), 0.0, 159.0),
            clamp(number(value.get("y"), self.py), 0.0, 209.0),
        )

    def _shot(self, obs):
        value = obs.get("player_shot")
        return value if isinstance(value, dict) else {}

    def _living(self, obs):
        result = []
        for item in dict_items(obs.get("invaders")):
            if not is_alive(item.get("alive")):
                continue
            if not isinstance(item.get("x"), (int, float)):
                continue
            if not isinstance(item.get("y"), (int, float)):
                continue
            col = integer(item.get("col"), -1)
            if 0 <= col <= 5:
                result.append(item)
        return result

    def _living_keys(self, obs):
        return {
            (integer(item.get("row"), -1), integer(item.get("col"), -1))
            for item in dict_items(obs.get("invaders"))
            if is_alive(item.get("alive"))
        }

    def _update_boundaries(self, px, dx, previous_direction):
        near_left = px <= self.left_limit + 10.0
        near_right = px >= self.right_limit - 10.0
        if previous_direction < 0 and near_left and abs(dx) < 0.75:
            self.left_stall += 1
        elif previous_direction >= 0:
            self.left_stall = 0
        if previous_direction > 0 and near_right and abs(dx) < 0.75:
            self.right_stall += 1
        elif previous_direction <= 0:
            self.right_stall = 0

        if self.left_stall >= 3:
            old = self.left_limit
            self.left_limit = max(self.left_limit, px + 2.0)
            self.left_known = True
            self.left_stall = 0
            if self.left_limit > old + 0.5:
                self.log(
                    "left_boundary_inferred",
                    observed_x=round(px, 1),
                    limit=round(self.left_limit, 1),
                )
        if self.right_stall >= 3:
            old = self.right_limit
            self.right_limit = min(self.right_limit, px - 2.0)
            self.right_known = True
            self.right_stall = 0
            if self.right_limit < old - 0.5:
                self.log(
                    "right_boundary_inferred",
                    observed_x=round(px, 1),
                    limit=round(self.right_limit, 1),
                )

    def safe_left(self):
        if self.left_known:
            return clamp(self.left_limit + 4.0, 39.0, 65.0)
        return clamp(self.left_limit + 2.0, 32.0, 60.0)

    def safe_right(self):
        if self.right_known:
            return clamp(self.right_limit - 4.0, 105.0, 145.0)
        return clamp(self.right_limit - 2.0, 105.0, 147.0)

    def _target_for_column(self, values, col):
        entries = values.get(col) or []
        if not entries:
            return None
        return max(entries, key=lambda item: number(item.get("y"), -999.0))

    def _intercept(self, alien):
        if not alien:
            return self.px
        flight = max(0.0, self.py - 6.0 - number(alien.get("y"), 0.0)) / max(0.5, self.shot_speed)
        return clamp(number(alien.get("x"), self.px) + clamp(self.vx * flight, -14.0, 14.0), 12.0, 150.0)

    def _update_tracks(self, obs, frame):
        seen = []
        for bomb in dict_items(obs.get("bombs"))[:16]:
            if not isinstance(bomb.get("x"), (int, float)) or not isinstance(bomb.get("y"), (int, float)):
                continue
            bx, by = number(bomb.get("x")), number(bomb.get("y"))
            velocity = 1.2
            matches = [
                old for old in self.tracks
                if abs(old["x"] - bx) <= 4.0
                and 0 < frame - old["frame"] <= 20
                and by >= old["y"]
            ]
            if matches:
                old = min(matches, key=lambda item: abs(by - item["y"]))
                velocity = clamp((by - old["y"]) / max(1, frame - old["frame"]), 0.4, 3.0)
            seen.append({"x": bx, "y": by, "v": velocity, "frame": frame})
        if seen:
            self.last_bomb_frame = frame
        # A short decoder gap is bridged, but old threats are discarded quickly.
        for old in self.tracks:
            if frame - old["frame"] <= 16 and not any(abs(old["x"] - item["x"]) <= 4.0 for item in seen):
                seen.append(old)
        self.tracks = seen[:20]
        self.bombs = [
            {
                "x": item["x"],
                "y": item["y"] + item["v"] * max(0, frame - item["frame"]),
                "v": item["v"],
            }
            for item in self.tracks
        ]

    def observe(self, raw):
        obs = as_dict(raw)
        frame = integer(obs.get("frame"), self.frame + 4)
        if frame < self.frame:
            frame = self.frame
        df = max(1, frame - self.frame) if self.last_obs is not None else 4
        old_lives = self.lives
        old_wave = self.wave
        old_keys = self.alive_keys
        old_shot_active = self.prev_shot_active
        old_px = self.px
        old_obs = self.last_obs

        px, py = self._player(obs)
        previous_direction = direction_of(self.prev_action)
        dx = px - old_px
        if old_obs is not None and previous_direction and abs(dx) >= 0.5:
            if 0.1 <= dx * previous_direction / df <= 1.6:
                self.move_samples.append(abs(dx) / df)
                self.move_speed = clamp(statistics.median(self.move_samples), 0.35, 1.2)
                self.last_moved_frame = frame
        self._update_boundaries(px, dx, previous_direction)
        if old_obs is None and px != old_px:
            self.last_moved_frame = frame

        lives = integer(obs.get("lives"), self.lives)
        wave = integer(obs.get("wave"), self.wave)
        self.life_event = lives < old_lives
        self.wave_event = wave != old_wave
        self.frame = frame
        self.px, self.py = px, py
        self.lives, self.wave = lives, wave
        self.score = number(obs.get("score"), self.score)
        if self.score > self.last_score:
            self.last_progress_frame = frame
        self.last_score = self.score

        self.alive = self._living(obs)
        self.alive_keys = self._living_keys(obs)
        self.columns = {}
        for alien in self.alive:
            col = integer(alien.get("col"), -1)
            self.columns.setdefault(col, []).append(alien)

        if old_obs is not None:
            killed = old_keys - self.alive_keys
            current_shot = self._shot(obs)
            current_active = bool(current_shot.get("active"))
            if killed:
                self.last_outcome = "HIT"
            elif old_shot_active and not current_active:
                self.last_outcome = "MISS"
            else:
                self.last_outcome = "NONE"
        else:
            self.last_outcome = "NONE"

        if self.alive:
            try:
                lattice = statistics.median(
                    number(item.get("x")) - 16.0 * number(item.get("col"))
                    for item in self.alive
                )
                self.formation.append((frame, lattice))
                while self.formation and frame - self.formation[0][0] > 96:
                    self.formation.popleft()
                if len(self.formation) >= 2:
                    f0, x0 = self.formation[0]
                    self.vx = clamp((lattice - x0) / max(1, frame - f0), -0.8, 0.8)
            except (TypeError, ValueError, statistics.StatisticsError):
                pass

        self._update_tracks(obs, frame)
        self.stalled = (
            bool(previous_direction)
            and frame - self.last_moved_frame >= 20
            and (px <= self.safe_left() + 8.0 or px >= self.safe_right() - 8.0)
        )

        if self.life_event:
            blind = frame - self.last_bomb_frame > 90
            self.unseen_deaths += int(blind)
            self.death_lanes.append({"x": old_px, "frame": frame})
            self.route = None
            self.advice = None
            self.tracks = []
            self.bombs = []
            self.recovery_until = frame + 20
            self.direction = -self.direction if self.direction else 1
            self.log(
                "life_loss_evidence",
                frame=frame,
                previous_x=round(old_px, 1),
                unseen_bomb=bool(blind),
                score=round(self.score, 1),
            )
        if self.wave_event:
            self.route = None
            self.advice = None
            self.formation.clear()
            self.tracks = []
            self.bombs = []
            self.recovery_until = frame + 16
            self.last_progress_frame = frame

        current_shot = self._shot(obs)
        self.prev_shot_active = bool(current_shot.get("active"))
        if old_shot_active and self.prev_shot_active and old_obs is not None:
            old_shot = self._shot(old_obs)
            if abs(number(old_shot.get("x"), 0.0) - number(current_shot.get("x"), 0.0)) <= 2.0:
                speed = (number(old_shot.get("y"), 0.0) - number(current_shot.get("y"), 0.0)) / df
                if 0.5 <= speed <= 4.0:
                    self.shot_speed = 0.8 * self.shot_speed + 0.2 * speed

        self.last_obs = obs
        return obs

    def _policy_target_xs(self):
        if not self.columns:
            return []
        lowest = [self._target_for_column(self.columns, col) for col in sorted(self.columns)]
        lowest = [item for item in lowest if item is not None]
        if not lowest:
            return []
        if self.policy == "LOWEST":
            bottom = max(number(item.get("y"), 0.0) for item in lowest)
            chosen = [item for item in lowest if number(item.get("y"), 0.0) >= bottom - 12.0]
            return [self._intercept(item) for item in chosen]
        # Edge policies alter the first endpoint preference, but retain all
        # lanes in the span.  This preserves 0006's coverage rather than
        # reducing the route to a narrow target lock.
        return [self._intercept(item) for item in lowest]

    def _route_bounds(self):
        left, right = self.safe_left(), self.safe_right()
        if right - left < 48:
            left, right = 35.0, 145.0
        xs = self._policy_target_xs()
        if not xs:
            lo, hi = left, right
        else:
            center = (min(xs) + max(xs)) / 2.0
            half = max(42.0, (max(xs) - min(xs)) / 2.0 + 8.0)
            lo = max(left, center - half)
            hi = min(right, center + half)
            if hi - lo < 78.0:
                middle = (lo + hi) / 2.0
                lo = max(left, middle - 39.0)
                hi = min(right, middle + 39.0)
            # Even when only one outer column survives, do not oscillate in a
            # small lane.  The full traverse is also a bomb-avoidance pattern.
            if hi - lo < 60.0:
                lo, hi = left, right
        return clamp(lo, left, right), clamp(hi, left, right)

    def _preferred_direction(self, lo, hi):
        if self.policy == "LEFT_EDGE":
            return -1
        if self.policy == "RIGHT_EDGE":
            return 1
        if self.policy == "LOWEST" and self.columns:
            target = max(
                (self._target_for_column(self.columns, col) for col in self.columns),
                key=lambda item: number(item.get("y"), -999.0),
            )
            target_x = self._intercept(target)
            if abs(target_x - self.px) > 8.0:
                return 1 if target_x > self.px else -1
        if self.px <= (lo + hi) / 2.0:
            return 1
        return -1

    def _commit_route(self):
        lo, hi = self._route_bounds()
        if hi <= lo + 4.0:
            lo, hi = self.safe_left(), self.safe_right()
        d = self._preferred_direction(lo, hi)
        destination = hi if d > 0 else lo
        duration = int(abs(destination - self.px) / max(0.35, self.move_speed) + 96)
        self.route_id += 1
        self.route_count += 1
        self.route = {
            "id": self.route_id,
            "lo": lo,
            "hi": hi,
            "destination": destination,
            "direction": d,
            "start": self.frame,
            "expires": self.frame + max(100, duration),
            "policy": self.policy,
            "start_score": self.score,
        }
        self.direction = d
        self.log(
            "sweep_committed",
            route_id=self.route_id,
            policy=self.policy,
            lo=round(lo, 1),
            hi=round(hi, 1),
            direction=d,
        )

    def _route_needs_refresh(self):
        if self.route is None:
            return True
        route = self.route
        d = integer(route.get("direction"), self.direction)
        reached = (number(route.get("destination"), self.px) - self.px) * d <= max(2.0, self.move_speed * 5.0)
        expired = self.frame >= integer(route.get("expires"), self.frame + 1)
        blocked = self.stalled and self.frame - integer(route.get("start"), self.frame) >= 20
        if reached or expired or blocked:
            self.log(
                "sweep_finished",
                route_id=route.get("id"),
                reason="reached" if reached else "blocked" if blocked else "expired",
                frames=self.frame - integer(route.get("start"), self.frame),
                score_gain=round(self.score - number(route.get("start_score"), self.score), 1),
            )
            self.route = None
            return True
        return False

    def _risk(self, direction):
        if not self.bombs:
            return 0.0
        direction = -1 if direction < 0 else 1 if direction > 0 else 0
        worst = 0.0
        for bomb in self.bombs:
            bx, by, velocity = number(bomb.get("x"), self.px), number(bomb.get("y"), -999.0), max(0.4, number(bomb.get("v"), 1.2))
            for delay in (0, 4):
                for dt in range(0, 65, 4):
                    projected_y = by + velocity * dt
                    if not self.py - 14.0 <= projected_y <= self.py + 8.0:
                        continue
                    projected_x = clamp(
                        self.px + direction * self.move_speed * max(0, dt - delay),
                        self.safe_left() - 2.0,
                        self.safe_right() + 2.0,
                    )
                    gap = abs(projected_x - bx)
                    if gap < 11.0:
                        worst = max(worst, 100.0 + 11.0 - gap + (64.0 - dt) / 64.0)
                    elif gap < 19.0:
                        worst = max(worst, (19.0 - gap) / 8.0)
        return worst

    def _escape_direction(self, route_direction):
        if not self.bombs:
            return None
        urgent = []
        for bomb in self.bombs:
            bx = number(bomb.get("x"), self.px)
            by = number(bomb.get("y"), -999.0)
            velocity = max(0.4, number(bomb.get("v"), 1.2))
            eta = (self.py - by) / velocity
            if -4.0 <= eta <= 70.0 and abs(bx - self.px) <= 34.0:
                urgent.append((max(0.0, eta) + abs(bx - self.px) * 0.25, bx))
        if not urgent:
            if self._risk(route_direction) < 100.0:
                return None
        if urgent:
            _, bx = min(urgent)
            wanted = 1 if bx <= self.px else -1
        else:
            wanted = route_direction
        if self.px <= self.safe_left() + 2.0:
            wanted = 1
        elif self.px >= self.safe_right() - 2.0:
            wanted = -1
        candidates = {wanted: self._risk(wanted), -wanted: self._risk(-wanted)}
        chosen = min(candidates, key=lambda item: (candidates[item], item != wanted))
        if urgent or candidates[wanted] >= 100.0:
            self.dodge_until = max(self.dodge_until, self.frame + 24)
            self.dodge_direction = chosen
            self.safety_overrides += 1
            self.log("visible_bomb_override", direction=chosen, risk=round(candidates[chosen], 2))
            return chosen
        return None

    def action(self):
        self._route_needs_refresh()
        if self.route is None:
            self._commit_route()
        route = self.route
        d = integer(route.get("direction"), self.direction)
        if self.px <= self.safe_left() + 1.0 and d < 0:
            d = 1
        elif self.px >= self.safe_right() - 1.0 and d > 0:
            d = -1
        route["direction"] = d
        self.direction = d

        escape = self._escape_direction(d)
        if escape is not None:
            d = escape
        elif self.frame < self.dodge_until and self.dodge_direction:
            d = self.dodge_direction
        elif self._risk(d) >= 100.0:
            alternative = -d if d else 1
            if self._risk(alternative) < self._risk(d):
                d = alternative
                self.safety_overrides += 1

        # FIRE is intentionally unconditional.  The Atari ROM ignores a fire
        # press while its one laser is active; suppressing FIRE during that
        # period was the major throughput regression in candidate 0008.
        if d > 0:
            return "right_fire", "sweep" if escape is None else "bomb_escape"
        if d < 0:
            return "left_fire", "sweep" if escape is None else "bomb_escape"
        return "fire", "fire_only"

    def accept_advice(self, result):
        if not isinstance(result, dict) or result.get("error"):
            return
        tag = result.get("tag")
        if not isinstance(tag, dict):
            return
        if (
            integer(tag.get("lives"), -1) != self.lives
            or integer(tag.get("wave"), -1) != self.wave
            or integer(tag.get("version"), -1) != self.policy_version
            or not 0 <= self.frame - integer(tag.get("frame"), -100000) <= 64
        ):
            self.log("jev_discarded", reason="stale_or_epoch")
            return
        answers = result.get("answers")
        if not isinstance(answers, dict):
            return
        first, second = answers.get("policy_a"), answers.get("policy_b")
        value_a, value_b = answer_value(first), answer_value(second)
        if not isinstance(value_a, str) or not isinstance(value_b, str):
            return
        choice = value_a.upper()
        if value_b.upper() != choice or choice not in POLICIES:
            return
        for answer in (first, second):
            if probability_for(answer, choice) < 0.52:
                return
            confidence = answer_confidence(answer)
            if confidence is not None and confidence < 0.20:
                return
        if choice == "KEEP":
            choice = self.policy if self.policy in POLICIES[:-1] else "WIDE"
        previous = self.policy
        self.advice = {"policy": choice, **tag}
        self.accepted += 1
        self.log(
            "jev_policy_accepted",
            policy=choice,
            previous=previous,
            age_frames=self.frame - integer(tag.get("frame"), self.frame),
        )

    def apply_advice_at_boundary(self):
        advice = self.advice
        if not isinstance(advice, dict):
            return
        if (
            integer(advice.get("lives"), -1) != self.lives
            or integer(advice.get("wave"), -1) != self.wave
            or integer(advice.get("version"), -1) != self.policy_version
            or self.frame - integer(advice.get("frame"), -100000) > 128
        ):
            return
        chosen = advice.get("policy")
        if chosen not in POLICIES[:-1]:
            return
        if chosen != self.policy:
            self.policy = chosen
            self.effective += 1
            self.log("jev_policy_applied", policy=chosen, frame=self.frame)
        self.advice = None

    def compact_state(self):
        columns = []
        for col in sorted(self.columns):
            low = self._target_for_column(self.columns, col)
            y = number(low.get("y"), 0.0) if low else 0.0
            side = "LEFT" if col <= 1 else "RIGHT" if col >= 4 else "CENTER"
            band = "BOTTOM" if y >= 115 else "MIDDLE" if y >= 75 else "TOP"
            columns.append({"name": "C" + str(col), "side": side, "lowest_band": band})
        if self.px <= self.safe_left() + 2.0:
            lane = "LEFT_MARGIN"
        elif self.px >= self.safe_right() - 2.0:
            lane = "RIGHT_MARGIN"
        else:
            lane = "INTERIOR"
        if any(self._risk(-1 if self.px > number(b.get("x"), self.px) else 1) >= 100.0 for b in self.bombs):
            bomb_state = "NEAR_CANNON"
        elif self.bombs:
            bomb_state = "TRACKED_APPROACH"
        else:
            bomb_state = "UNSEEN_OR_EMPTY"
        route = self.route or {}
        return {
            "schema": "space_invaders_full_sweep_feedback_v1",
            "rules": (
                "One player laser is active at a time. FIRE may be held while moving; "
                "the ROM ignores busy presses. Empty bombs are not proof of safety. "
                "Code owns coordinates, arithmetic, boundaries, and legal actions."
            ),
            "frame": integer(self.frame),
            "lives": integer(self.lives),
            "wave": integer(self.wave),
            "phase": "LATE" if len(self.alive) <= 12 else "MID" if len(self.alive) <= 24 else "OPENING",
            "player_lane": lane,
            "motion": "STALLED_AT_MARGIN" if self.stalled else "COMMITTED_TRAVERSE",
            "shot": "ACTIVE" if self.prev_shot_active else "READY_OR_DECODER_GAP",
            "last_shot_outcome": self.last_outcome,
            "bomb_evidence": bomb_state,
            "boundary": {
                "left": "OBSERVED" if self.left_known else "PROBING",
                "right": "OBSERVED" if self.right_known else "PROBING",
                "escape_margin": "PRESERVED",
            },
            "visibility": "UNRELIABLE_AFTER_BLIND_DEATH" if self.unseen_deaths else "SPARSE_DECODER",
            "progress": "RECENT_SCORE" if self.frame - self.last_progress_frame < 180 else "NO_RECENT_SCORE",
            "current_policy": self.policy,
            "route": {
                "span": "WIDE" if number(route.get("hi"), 0.0) - number(route.get("lo"), 0.0) >= 70.0 else "NARROW",
                "direction": "RIGHT" if integer(route.get("direction"), self.direction) > 0 else "LEFT",
            },
            "living_columns": columns,
        }


class Harness:
    def __init__(self, ctx):
        self.ctx = ctx

    def _log(self, *args, **fields):
        try:
            self.ctx.log(*args, **fields)
        except Exception:
            pass

    def _legal(self, game):
        spec = getattr(game, "action_spec", None)
        try:
            names = set(getattr(spec, "names", ACTIONS))
        except Exception:
            names = set(ACTIONS)
        if not names:
            names = set(ACTIONS)
        noop = getattr(spec, "noop", "noop")
        if noop not in names:
            noop = "noop" if "noop" in names else next(iter(names))
        return names, noop

    def _post(self, action, legal, noop, controller):
        if action not in legal:
            base = "right" if "right" in action else "left" if "left" in action else "fire"
            if base not in legal:
                base = noop
            action = base
        try:
            self.ctx.act(action)
            controller.prev_action = action
            controller.actions += 1
            return True
        except Exception:
            if action != noop and noop in legal:
                try:
                    self.ctx.act(noop)
                    controller.prev_action = noop
                    controller.actions += 1
                    return True
                except Exception:
                    pass
            return False

    def run_episode(self, game):
        ctx = self.ctx
        legal, noop = self._legal(game)
        controller = Controller(ctx)
        mailbox = JevMailbox(ctx)
        last_observed_frame = None
        last_query_frame = -100000
        try:
            try:
                initial = ctx.observe()
            except Exception:
                initial = {}
            obs = controller.observe(initial)
            if not isinstance(obs, dict):
                obs = {}

            if mailbox.submit(
                controller.compact_state(),
                controller.frame,
                controller.lives,
                controller.wave,
                controller.policy_version,
            ):
                last_query_frame = controller.frame

            action, source = controller.action()
            if not self._post(action, legal, noop, controller):
                return
            last_observed_frame = controller.frame

            while not getattr(ctx, "done", False) and not bool(getattr(game, "done", False)):
                try:
                    ctx.wait_for_tick()
                except Exception:
                    break
                if getattr(ctx, "done", False) or bool(getattr(game, "done", False)):
                    break
                try:
                    raw = ctx.observe()
                except Exception:
                    break
                obs = controller.observe(raw)
                if not isinstance(obs, dict):
                    obs = {}
                if obs.get("game_over"):
                    break
                if last_observed_frame is not None and controller.frame == last_observed_frame:
                    # A malformed stand-in observation must not create a hot
                    # action loop; wait for the real-time clock again.
                    continue
                last_observed_frame = controller.frame

                result = mailbox.take()
                if result is not None:
                    controller.accept_advice(result)
                if controller.route is None:
                    controller.apply_advice_at_boundary()

                if (
                    controller.frame - last_query_frame >= 48
                    or controller.life_event
                    or controller.wave_event
                ):
                    if mailbox.submit(
                        controller.compact_state(),
                        controller.frame,
                        controller.lives,
                        controller.wave,
                        controller.policy_version,
                    ):
                        last_query_frame = controller.frame

                action, source = controller.action()
                if not self._post(action, legal, noop, controller):
                    break
                if controller.actions % 200 == 0:
                    self._log(
                        "sweep_controller_summary",
                        frame=controller.frame,
                        score=round(controller.score, 1),
                        lives=controller.lives,
                        source=source,
                        routes=controller.route_count,
                        jev_accepted=controller.accepted,
                        jev_effective=controller.effective,
                        unseen_deaths=controller.unseen_deaths,
                    )
        finally:
            mailbox.stop()
            self._log(
                "sweep_controller_finished",
                frame=controller.frame,
                score=round(controller.score, 1),
                lives=controller.lives,
                actions=controller.actions,
                routes=controller.route_count,
                jev_accepted=controller.accepted,
                jev_effective=controller.effective,
                left_limit=round(controller.left_limit, 1),
                right_limit=round(controller.right_limit, 1),
                left_known=controller.left_known,
                right_known=controller.right_known,
                safety_overrides=controller.safety_overrides,
            )


def build(ctx):
    return Harness(ctx)
