"""Candidate 0008: feedback-calibrated firing with an adaptive movement envelope.

The fast model is deliberately kept out of the actuator.  It can change the
ordering used when the next laser lane is selected, but it cannot turn a
controller into a long-lived ``relocate`` command.  The controller learns the
actual left/right travel limits from observed motion, keeps an interior escape
margin, and fires at a locked lane before bouncing through the interior.
"""
from __future__ import annotations

import json
import math

from metaharness.core.contract import BudgetExceeded, EpisodeOver
from metaharness.models.jev import Choice


ACTIONS = ("noop", "fire", "right", "left", "right_fire", "left_fire")
POLICIES = ("LOWEST_THREAT", "LEFT_OUTER", "RIGHT_OUTER", "NEAREST_LANE", "KEEP")


def number(value, default=0.0):
    try:
        value = float(value)
        return value if math.isfinite(value) else default
    except (TypeError, ValueError, OverflowError):
        return default


def integer(value, default=0):
    try:
        return int(number(value, default))
    except (TypeError, ValueError, OverflowError):
        return default


def clamp(value, low, high):
    return max(low, min(high, value))


def sign(value):
    return 1 if value > 0 else -1 if value < 0 else 0


def dict_obs(value):
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            decoded = json.loads(value)
            return decoded if isinstance(decoded, dict) else {}
        except (TypeError, ValueError, json.JSONDecodeError):
            return {}
    return {}


def objects(value):
    return [item for item in value if isinstance(item, dict)] if isinstance(value, list) else []


def alive(value):
    if isinstance(value, str):
        return value.lower() in {"true", "1", "yes"}
    return bool(value)


def answer_value(answer):
    if isinstance(answer, dict):
        return answer.get("value", answer.get("choice"))
    return getattr(answer, "value", None)


def answer_probability(answer, value):
    probabilities = (
        answer.get("probabilities", {})
        if isinstance(answer, dict)
        else getattr(answer, "probabilities", {})
    )
    if not isinstance(probabilities, dict):
        return 0.0
    for key, probability in probabilities.items():
        if str(key).upper() == str(value).upper():
            return number(probability, 0.0)
    return 0.0


def answer_confidence(answer):
    value = answer.get("confidence") if isinstance(answer, dict) else getattr(answer, "confidence", None)
    if value is None:
        return None
    parsed = number(value, -1.0)
    return parsed if parsed >= 0.0 else None


def _policy_questions(frame):
    options = {
        "LOWEST_THREAT": "Use the living column whose lowest invader is the most immediate threat; code chooses it exactly.",
        "LEFT_OUTER": "Use the leftmost living outer column for the next deliberate shot; do not invent a dead column.",
        "RIGHT_OUTER": "Use the rightmost living outer column for the next deliberate shot; do not invent a dead column.",
        "NEAREST_LANE": "Use the living target lane nearest the cannon's current interior lane; code computes the distance.",
        "KEEP": "Keep the current target column if it is still alive; otherwise let code use LOWEST_THREAT.",
    }
    ordered = list(options.items())
    if (integer(frame) // 32) % 2:
        ordered.reverse()
    first = dict(ordered)
    second = dict(reversed(ordered))
    instruction = (
        "Choose a target-ordering policy for a future laser only. The code owns exact "
        "coordinates, the one-shot lock, boundary safety, and the joystick action. "
        "Do not choose a movement string, count invaders, compare numbers, or assume "
        "an empty bomb list proves safety. Apply the policy only to the categorical "
        "columns and outcome in state."
    )
    return {
        "policy_a": Choice(instruction, first),
        "policy_b": Choice(instruction, second),
    }


class JevMailbox:
    """One short-lived request at a time; late judgments never block ticks."""

    def __init__(self, ctx):
        self.ctx = ctx
        self.future = None
        self.tag = None
        self.enabled = getattr(ctx, "system1", True) is not None

    def submit(self, state, frame, lives, wave):
        if not self.enabled or self.future is not None or getattr(self.ctx, "done", False):
            return
        try:
            if self.ctx.budget.remaining().get("system1_calls", 0) <= 0:
                self.enabled = False
                return
        except Exception:
            pass
        questions = _policy_questions(frame)
        tag = {"frame": integer(frame), "lives": integer(lives), "wave": wave}

        def request():
            try:
                return {"answers": self.ctx.decide(state, questions, timeout_s=0.70), "tag": tag}
            except Exception as exc:
                # A timeout or malformed response is a normal fallback event.
                # Backend-unavailable also makes ctx.done true; the main loop then
                # exits without allowing this marker to become an action.
                return {"error": type(exc).__name__, "tag": tag}

        try:
            self.future = self.ctx.spawn(request)
            self.tag = tag
        except Exception as exc:
            self.ctx_log("jev_spawn_failed", error=type(exc).__name__)
            self.future = None
            self.tag = None

    def take(self):
        future = self.future
        if future is None or not future.done():
            return None
        self.future = None
        tag = self.tag or {}
        self.tag = None
        try:
            result = future.result()
        except Exception as exc:
            return {"error": type(exc).__name__, "tag": tag}
        return result if isinstance(result, dict) else {"error": "non_dict", "tag": tag}

    def stop(self):
        # There is no plain Python worker to join.  An in-flight ctx.decide is
        # owned by the evaluator and is cancelled when the episode closes.
        self.future = None
        self.tag = None

    def ctx_log(self, *args, **fields):
        try:
            self.ctx.log(*args, **fields)
        except Exception:
            pass


class Controller:
    """Closed-loop lane controller with learned travel limits."""

    def __init__(self, ctx):
        self.ctx = ctx
        self.frame = 0
        self.lives = 3
        self.wave = 1
        self.score = 0.0
        self.px = 90.0
        self.py = 190.0
        self.prev_px = None
        self.prev_frame = None
        self.prev_action = "noop"
        self.prev_shot_active = False
        self.previous_alive = set()
        self.target_col = None
        self.target_policy = "LOWEST_THREAT"
        self.target_policy_age = 0
        self.last_outcome = "NONE"
        self.last_fire_frame = -100000
        self.last_shot_x = None
        self.last_shot_target_x = None
        self.last_shot_target_col = None
        self.shot_seen = False
        self.direction = 1
        self.recovery = 0
        self.dodge_until = -1
        self.stall_left = 0
        self.stall_right = 0
        # These are intentionally broad initial beliefs.  Repeated observed
        # motion at a limit tightens them; a single sticky-action frame cannot.
        self.left_limit = 32.0
        self.right_limit = 147.0
        self.min_seen = 159.0
        self.max_seen = 0.0
        self.aim_offset = 0.0
        self.observations = 0

    def log(self, *args, **fields):
        try:
            self.ctx.log(*args, **fields)
        except Exception:
            pass

    def player(self, obs):
        value = obs.get("player")
        if not isinstance(value, dict):
            return self.px, self.py
        return clamp(number(value.get("x"), self.px), 0.0, 159.0), clamp(number(value.get("y"), self.py), 0.0, 209.0)

    def shot(self, obs):
        value = obs.get("player_shot")
        return value if isinstance(value, dict) else {}

    def alive_by_column(self, obs):
        result = {}
        for item in objects(obs.get("invaders")):
            if not alive(item.get("alive")):
                continue
            col = integer(item.get("col"), -1)
            if 0 <= col <= 5 and isinstance(item.get("x"), (int, float)):
                result.setdefault(col, []).append(item)
        return result

    def alive_set(self, obs):
        return {
            (integer(item.get("row"), -1), integer(item.get("col"), -1))
            for item in objects(obs.get("invaders"))
            if alive(item.get("alive"))
        }

    def _update_boundary(self, px, dx):
        previous_direction = -1 if "left" in self.prev_action else 1 if "right" in self.prev_action else 0
        near_left = px <= self.left_limit + 10.0
        near_right = px >= self.right_limit - 10.0
        if previous_direction < 0:
            if near_left and abs(dx) < 0.75:
                self.stall_left += 1
            else:
                self.stall_left = 0
        else:
            self.stall_left = 0
        if previous_direction > 0:
            if near_right and abs(dx) < 0.75:
                self.stall_right += 1
            else:
                self.stall_right = 0
        else:
            self.stall_right = 0
        if self.stall_left >= 3:
            old = self.left_limit
            self.left_limit = max(self.left_limit, px + 3.0)
            if self.left_limit > old + 0.5:
                self.log("left_boundary_inferred", observed_x=round(px, 1), limit=round(self.left_limit, 1))
            self.stall_left = 0
        if self.stall_right >= 3:
            old = self.right_limit
            self.right_limit = min(self.right_limit, px - 3.0)
            if self.right_limit < old - 0.5:
                self.log("right_boundary_inferred", observed_x=round(px, 1), limit=round(self.right_limit, 1))
            self.stall_right = 0

    def safe_left(self):
        return clamp(self.left_limit + 4.0, 36.0, 72.0)

    def safe_right(self):
        return clamp(self.right_limit - 4.0, 104.0, 145.0)

    def _target_for_col(self, columns, col):
        values = columns.get(col) or []
        if not values:
            return None
        return max(values, key=lambda item: (number(item.get("y"), -999.0), integer(item.get("row"), -1)))

    def choose_target(self, columns):
        if not columns:
            return None
        if self.target_policy == "KEEP" and self.target_col in columns:
            return self.target_col
        if self.target_policy == "LEFT_OUTER":
            return min(columns)
        if self.target_policy == "RIGHT_OUTER":
            return max(columns)
        if self.target_policy == "NEAREST_LANE":
            return min(columns, key=lambda col: abs(number(columns[col][0].get("x"), self.px) + self.aim_offset - self.px))
        # LOWEST_THREAT: use the lowest surviving object in each living column,
        # then prefer the shorter interior travel on a tie.
        def rank(col):
            low = max(columns[col], key=lambda item: number(item.get("y"), -999.0))
            return (number(low.get("y"), -999.0), -abs(number(low.get("x"), self.px) + self.aim_offset - self.px), -col)
        return max(columns, key=rank)

    def target(self, obs, columns=None):
        columns = columns if columns is not None else self.alive_by_column(obs)
        if self.target_col not in columns:
            self.target_col = self.choose_target(columns)
            self.target_policy_age = 0
        if self.target_col is None:
            return None
        value = self._target_for_col(columns, self.target_col)
        if value is None:
            self.target_col = self.choose_target(columns)
            value = self._target_for_col(columns, self.target_col)
        return value

    def _bomb_escape(self, obs):
        """Use visible bombs as overrides, never as a safety certificate."""
        px = self.px
        candidates = []
        for bomb in objects(obs.get("bombs")):
            bx = number(bomb.get("x"), px)
            by = number(bomb.get("y"), -999.0)
            vertical = self.py - by
            if vertical < -4 or vertical > 105:
                continue
            if by < 112 and abs(bx - px) > 25:
                continue
            gap = abs(bx - px)
            if gap > 28:
                continue
            urgency = 0 if by >= 145 or (vertical <= 48 and gap <= 16) else 1
            candidates.append((urgency, vertical + gap, bx))
        if not candidates:
            return None
        _, _, bx = min(candidates)
        if bx <= px:
            wanted = 1
        else:
            wanted = -1
        if px <= self.safe_left() + 2:
            wanted = 1
        elif px >= self.safe_right() - 2:
            wanted = -1
        self.dodge_until = max(self.dodge_until, self.frame + 20)
        return wanted

    def _bounce_direction(self):
        left = self.safe_left()
        right = self.safe_right()
        if self.px <= left + 1.0:
            self.direction = 1
        elif self.px >= right - 1.0:
            self.direction = -1
        return self.direction

    def _move_action(self, wanted):
        wanted = -1 if wanted < 0 else 1
        if wanted < 0 and self.px <= self.safe_left():
            self.direction = 1
            return "right"
        if wanted > 0 and self.px >= self.safe_right():
            self.direction = -1
            return "left"
        self.direction = wanted
        return "left" if wanted < 0 else "right"

    def _record_hit_calibration(self, old_obs, old_alive, new_alive):
        killed = old_alive - new_alive
        if not killed or self.last_shot_x is None:
            return
        # The projectile's lane and the target's decoded x are the only
        # quantities needed.  Use a small update because a formation can move
        # between launch and the confirming observation.
        old_columns = self.alive_by_column(old_obs)
        target = self._target_for_col(old_columns, self.last_shot_target_col)
        if target is None:
            return
        residual = self.last_shot_x - number(target.get("x"), self.last_shot_x)
        if -12.0 <= residual <= 12.0:
            old_offset = self.aim_offset
            self.aim_offset = clamp(0.75 * self.aim_offset + 0.25 * residual, -8.0, 8.0)
            self.log("shot_lane_calibrated", old=round(old_offset, 2), new=round(self.aim_offset, 2), residual=round(residual, 2))

    def observe(self, obs):
        obs = dict_obs(obs)
        frame = integer(obs.get("frame"), self.frame + 4)
        old_obs = getattr(self, "last_obs", None)
        old_alive = self.previous_alive
        new_alive = self.alive_set(obs)
        old_lives = self.lives
        lives = integer(obs.get("lives"), self.lives)
        wave = integer(obs.get("wave"), self.wave)
        px, py = self.player(obs)
        dx = px - self.px if self.observations else 0.0
        self._update_boundary(px, dx)
        self.min_seen = min(self.min_seen, px)
        self.max_seen = max(self.max_seen, px)
        self.frame = frame
        self.px, self.py = px, py
        self.observations += 1
        self.last_outcome = "NONE"
        if old_obs is not None:
            killed = old_alive - new_alive
            if killed:
                self.last_outcome = "HIT"
                self._record_hit_calibration(old_obs, old_alive, new_alive)
                self.target_col = None
            elif self.prev_shot_active and not bool(self.shot(obs).get("active")):
                self.last_outcome = "MISS"
                self.target_col = None
        if lives < old_lives:
            self.target_col = None
            self.recovery = 12
            self.direction *= -1
            self.dodge_until = self.frame + 24
            self.log("life_recovery", frame=self.frame, lives=lives, observed_min=round(self.min_seen, 1), observed_max=round(self.max_seen, 1))
        if wave != self.wave:
            self.target_col = None
            self.target_policy_age = 0
            self.recovery = 8
        self.lives = lives
        self.wave = wave
        self.score = number(obs.get("score"), self.score)
        current_shot = self.shot(obs)
        shot_active = bool(current_shot.get("active"))
        if shot_active and not self.prev_shot_active:
            self.shot_seen = True
            self.last_shot_x = number(current_shot.get("x"), px)
            target = self.target(obs)
            self.last_shot_target_col = self.target_col
            self.last_shot_target_x = number(target.get("x"), px) if target else None
            # Leave the firing lane immediately; this is feedback-controlled
            # patrol, not a model-selected relocation destination.
            if self.last_shot_x <= (self.safe_left() + self.safe_right()) / 2.0:
                self.direction = 1
            else:
                self.direction = -1
        self.prev_shot_active = shot_active
        self.previous_alive = new_alive
        self.prev_px = px
        self.prev_frame = frame
        self.last_obs = obs
        if self.recovery > 0:
            self.recovery -= 1
        return obs

    def compact_state(self, obs):
        columns = self.alive_by_column(obs)
        target = self.target(obs, columns)
        alive_columns = []
        for col in sorted(columns):
            low = self._target_for_col(columns, col)
            side = "LEFT" if col <= 1 else "RIGHT" if col >= 4 else "CENTER"
            band = "BOTTOM" if number(low.get("y"), 0) >= 115 else "MIDDLE" if number(low.get("y"), 0) >= 75 else "TOP"
            alive_columns.append({"name": f"C{col}", "side": side, "lowest_band": band})
        if self.px <= self.safe_left() + 2:
            lane = "LEFT_MARGIN"
        elif self.px >= self.safe_right() - 2:
            lane = "RIGHT_MARGIN"
        else:
            lane = "INTERIOR"
        bomb_state = "UNSEEN_OR_EMPTY"
        for bomb in objects(obs.get("bombs")):
            by = number(bomb.get("y"), -999)
            bx = number(bomb.get("x"), self.px)
            if by >= 140 and abs(bx - self.px) <= 24:
                bomb_state = "NEAR_CANNON"
                break
            if by >= 112 and abs(bx - self.px) <= 30:
                bomb_state = "APPROACHING"
        return {
            "schema": "space_invaders_boundary_feedback_v1",
            "rules": "One player laser; movement is independent. Empty bombs is not proof of safety. Code handles geometry and legal actions.",
            "frame": self.frame,
            "lives": self.lives,
            "wave": self.wave,
            "phase": "LATE" if len(self.previous_alive) <= 12 else "MID" if len(self.previous_alive) <= 24 else "OPENING",
            "player_lane": lane,
            "motion": "LEFT_LIMIT_LEARNING" if self.stall_left else "RIGHT_LIMIT_LEARNING" if self.stall_right else "BOUNCE_INTERIOR",
            "shot": "ACTIVE" if self.prev_shot_active else "READY",
            "outcome": self.last_outcome,
            "bomb_evidence": bomb_state,
            "boundary_evidence": {
                "left": "observed_limit" if self.left_limit > 34 else "learning",
                "right": "observed_limit" if self.right_limit < 145 else "learning",
                "escape_margin": "preserved",
            },
            "current_target": None if target is None else {
                "column": f"C{self.target_col}",
                "side": "LEFT" if self.target_col <= 1 else "RIGHT" if self.target_col >= 4 else "CENTER",
                "lowest_band": "BOTTOM" if number(target.get("y"), 0) >= 115 else "MIDDLE" if number(target.get("y"), 0) >= 75 else "TOP",
            },
            "living_columns": alive_columns,
            "current_policy": self.target_policy,
        }

    def accept(self, result):
        if not isinstance(result, dict) or result.get("error"):
            return
        tag = result.get("tag")
        if not isinstance(tag, dict):
            return
        if tag.get("lives") != self.lives or tag.get("wave") != self.wave:
            self.log("jev_discarded", reason="life_or_wave_changed")
            return
        if self.frame - integer(tag.get("frame"), -100000) > 48:
            self.log("jev_discarded", reason="too_old")
            return
        answers = result.get("answers")
        if not isinstance(answers, dict):
            return
        first = answers.get("policy_a")
        second = answers.get("policy_b")
        left = answer_value(first)
        right = answer_value(second)
        if not isinstance(left, str) or not isinstance(right, str):
            return
        left = left.upper()
        right = right.upper()
        if left != right or left not in POLICIES:
            return
        for answer in (first, second):
            if answer_probability(answer, left) < 0.58:
                return
            confidence = answer_confidence(answer)
            if confidence is not None and confidence < 0.25:
                return
        self.target_policy = left
        self.target_policy_age = 0
        self.log("jev_target_policy", policy=left, age_frames=self.frame - integer(tag.get("frame"), self.frame))

    def action(self, obs):
        obs = dict_obs(obs)
        columns = self.alive_by_column(obs)
        target = self.target(obs, columns)
        escape = self._bomb_escape(obs)
        if escape is not None:
            return self._move_action(escape), "visible_bomb"
        if self.recovery > 0 and not self.prev_shot_active:
            return self._move_action(self._bounce_direction()), "recovery"
        if self.prev_shot_active:
            return self._move_action(self._bounce_direction()), "shot_patrol"
        # Decoder lag can briefly say READY after a fire command.  Move rather
        # than post another fire into the one-shot lock during that window.
        if self.frame - self.last_fire_frame < 8:
            return self._move_action(self._bounce_direction()), "fire_cooldown"
        if target is not None:
            raw_x = number(target.get("x"), self.px) + self.aim_offset
            desired = clamp(raw_x, self.safe_left(), self.safe_right())
            error_to_raw = raw_x - self.px
            if abs(error_to_raw) > 7.0 and abs(desired - self.px) > 5.0:
                return self._move_action(sign(desired - self.px)), "align"
            # Outer columns may be just outside the learned margin; a small
            # alignment tolerance lets the cannon fire from the safe interior.
            if abs(error_to_raw) <= 9.0:
                self.last_fire_frame = self.frame
                self.last_shot_target_col = self.target_col
                return "fire", "fire_aligned"
            return self._move_action(self._bounce_direction()), "target_margin"
        return self._move_action(self._bounce_direction()), "patrol_no_target"


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
        names = getattr(spec, "names", ACTIONS)
        try:
            legal = set(names)
        except Exception:
            legal = set(ACTIONS)
        noop = getattr(spec, "noop", "noop")
        if noop not in legal:
            noop = "noop" if "noop" in legal else (next(iter(legal)) if legal else "noop")
        return legal, noop

    def run_episode(self, game):
        ctx = self.ctx
        legal, noop = self._legal(game)
        controller = Controller(ctx)
        mailbox = JevMailbox(ctx)
        last_frame = None
        last_query = -100000
        try:
            obs = controller.observe(dict_obs(game.observe()))
        except Exception:
            return

        def post(action):
            if action not in legal:
                action = noop
            try:
                ctx.act(action)
                controller.prev_action = action
                return True
            except (BudgetExceeded, EpisodeOver):
                return False
            except Exception:
                if noop in legal and action != noop:
                    try:
                        ctx.act(noop)
                        controller.prev_action = noop
                        return True
                    except Exception:
                        return False
                return False

        def maybe_submit(obs):
            nonlocal last_query
            if controller.frame - last_query < 32:
                return
            if getattr(ctx, "done", False):
                return
            state = controller.compact_state(obs)
            mailbox.submit(state, controller.frame, controller.lives, controller.wave)
            last_query = controller.frame

        maybe_submit(obs)
        action, source = controller.action(obs)
        if not post(action):
            return

        try:
            while not getattr(ctx, "done", False) and not bool(getattr(game, "done", False)):
                try:
                    ctx.wait_for_tick()
                except (BudgetExceeded, EpisodeOver):
                    break
                except Exception:
                    break
                if getattr(ctx, "done", False) or bool(getattr(game, "done", False)):
                    break
                try:
                    raw = game.observe()
                except Exception:
                    break
                obs = controller.observe(dict_obs(raw))
                if obs.get("game_over"):
                    break
                result = mailbox.take()
                if result is not None:
                    controller.accept(result)
                maybe_submit(obs)
                action, source = controller.action(obs)
                if not post(action):
                    break
                if controller.frame == last_frame:
                    # This should not occur after wait_for_tick, but avoids a
                    # malformed stand-in observation creating a hot loop.
                    continue
                last_frame = controller.frame
        finally:
            mailbox.stop()
            self._log(
                "boundary_controller_finished",
                frame=controller.frame,
                lives=controller.lives,
                score=controller.score,
                left_limit=round(controller.left_limit, 1),
                right_limit=round(controller.right_limit, 1),
                aim_offset=round(controller.aim_offset, 2),
                policy=controller.target_policy,
            )


def build(ctx):
    return Harness(ctx)
