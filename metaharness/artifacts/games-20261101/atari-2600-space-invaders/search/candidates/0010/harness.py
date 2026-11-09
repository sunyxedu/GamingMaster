"""Candidate 0010: adaptive boundary sweep with hardware-rate firing.

The parent controller had useful closed-loop lane calibration, but its exact
alignment gate often turned the game into a movement-only controller.  This
version keeps the parent's feedback and interior margin while committing to a
broad, alternating sweep and holding FIRE on every legal tick.  The ROM's
one-shot lock makes repeated FIRE harmless while a laser is in flight; the
sweep supplies the alignment opportunities that the conservative gate missed.

Only the public symbolic observation and the HarnessContext contract are used.
"""
from __future__ import annotations

import json
import math
from concurrent.futures import CancelledError

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


def objects(value):
    return [item for item in value if isinstance(item, dict)] if isinstance(value, list) else []


def is_alive(value):
    if isinstance(value, str):
        return value.strip().lower() in ("true", "1", "yes")
    return bool(value)


def dict_observation(value):
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, dict) else {}
        except (TypeError, ValueError, json.JSONDecodeError):
            return {}
    return {}


def answer_value(answer):
    if isinstance(answer, dict):
        return answer.get("value", answer.get("choice"))
    return getattr(answer, "value", None)


def answer_probability(answer, option):
    if isinstance(answer, dict):
        probabilities = answer.get("probabilities", {})
    else:
        probabilities = getattr(answer, "probabilities", {})
    if not isinstance(probabilities, dict):
        return 0.0
    for key, value in probabilities.items():
        if str(key).upper() == str(option).upper():
            return number(value, 0.0)
    return 0.0


def answer_confidence(answer):
    value = answer.get("confidence") if isinstance(answer, dict) else getattr(answer, "confidence", None)
    if value is None:
        return None
    parsed = number(value, -1.0)
    return parsed if parsed >= 0.0 else None


# The model is asked only to order already-known target categories.  The
# options are deliberately described in terms of consequences, not raw
# arithmetic; geometry and joystick actions remain code-owned.
def policy_questions(frame):
    choices = {
        "LOWEST_THREAT": "Prefer the living column whose lowest surviving invader is the immediate threat; code selects it.",
        "LEFT_OUTER": "Prefer the leftmost living outer column for the next sweep; never invent a dead column.",
        "RIGHT_OUTER": "Prefer the rightmost living outer column for the next sweep; never invent a dead column.",
        "NEAREST_LANE": "Prefer the living target lane nearest the cannon's current interior lane; code computes distance.",
        "KEEP": "Keep the current target column if it remains alive; otherwise use the code fallback.",
    }
    ordered = list(choices.items())
    if (integer(frame) // 32) % 2:
        ordered.reverse()
    first = dict(ordered)
    second = dict(reversed(ordered))
    instruction = (
        "Choose a target-ordering policy for a future laser opportunity. The code "
        "owns exact coordinates, boundary safety, one-shot timing, and every "
        "joystick action. Do not select an action, count invaders, calculate a "
        "distance, or treat an empty bomb list as proof of safety. Use only the "
        "categorical facts in state."
    )
    return {
        "policy_a": Choice(instruction, first),
        "policy_b": Choice(instruction, second),
    }


class JevMailbox:
    """One latest-only asynchronous Jev request.

    A request is a single ctx.decide call rather than a persistent Python
    worker.  Thus a late answer can be dropped and there is no non-context
    thread to outlive the episode.
    """

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
        tag = {"frame": integer(frame), "lives": integer(lives), "wave": integer(wave)}
        questions = policy_questions(frame)

        def request():
            try:
                answers = self.ctx.decide(state, questions, timeout_s=0.70)
                return {"answers": answers, "tag": tag}
            except Exception as exc:
                # Timeouts, malformed stand-in answers, and ordinary backend
                # failures all leave the deterministic controller in charge.
                return {"error": type(exc).__name__, "tag": tag}

        try:
            self.future = self.ctx.spawn(request)
            self.tag = tag
        except Exception as exc:
            self.future = None
            self.tag = None
            try:
                self.ctx.log("jev_spawn_failed", error=type(exc).__name__)
            except Exception:
                pass

    def take(self):
        future = self.future
        if future is None or not future.done():
            return None
        tag = self.tag or {}
        self.future = None
        self.tag = None
        try:
            result = future.result()
        except CancelledError:
            return {"error": "cancelled", "tag": tag}
        except Exception as exc:
            return {"error": type(exc).__name__, "tag": tag}
        return result if isinstance(result, dict) else {"error": "non_dict", "tag": tag}

    def stop(self):
        future = self.future
        self.future = None
        self.tag = None
        if future is not None and not future.done():
            try:
                future.cancel()
            except Exception:
                pass


class Controller:
    """A deterministic sweep controller with feedback-calibrated lanes."""

    def __init__(self, ctx):
        self.ctx = ctx
        self.frame = 0
        self.lives = 3
        self.wave = 1
        self.score = 0.0
        self.px = 90.0
        self.py = 190.0
        self.prev_action = "noop"
        self.previous = None
        self.previous_alive = set()
        self.prev_shot_active = False
        self.target_col = None
        self.target_policy = "LOWEST_THREAT"
        self.last_outcome = "NONE"
        self.last_shot_x = None
        self.last_shot_col = None
        self.aim_offset = 0.0
        self.direction = 1
        self.route = None
        self.recovery = 0
        self.move_samples = []
        self.move_speed = 0.8
        # These are deliberately conservative initial beliefs.  A sustained
        # no-motion observation near a commanded side tightens them; one
        # sticky-action tick cannot do so.
        self.left_limit = 30.0
        self.right_limit = 150.0
        self.stall_left = 0
        self.stall_right = 0
        self.bomb_tracks = []
        self.last_bomb_frame = -100000
        self.last_progress = 0
        self.last_fire_frame = -100000
        self.last_obs = {}
        self.actions = 0
        self.hits = 0
        self.misses = 0
        self.life_losses = 0

    def log(self, *args, **fields):
        try:
            self.ctx.log(*args, **fields)
        except Exception:
            pass

    def _columns(self, obs):
        result = {}
        for item in objects(obs.get("invaders")):
            if not is_alive(item.get("alive")):
                continue
            col = integer(item.get("col"), -1)
            if 0 <= col <= 5 and isinstance(item.get("x"), (int, float)):
                result.setdefault(col, []).append(item)
        return result

    def _alive_set(self, obs):
        return {
            (integer(item.get("row"), -1), integer(item.get("col"), -1))
            for item in objects(obs.get("invaders"))
            if is_alive(item.get("alive"))
        }

    def _lowest(self, values):
        if not values:
            return None
        return max(values, key=lambda item: (number(item.get("y"), -999.0), integer(item.get("row"), -1)))

    def _target_for_policy(self, columns, policy=None):
        if not columns:
            return None
        policy = policy or self.target_policy
        if policy == "KEEP" and self.target_col in columns:
            return self.target_col
        if policy == "LEFT_OUTER":
            return min(columns)
        if policy == "RIGHT_OUTER":
            return max(columns)
        if policy == "NEAREST_LANE":
            return min(columns, key=lambda col: abs(number(self._lowest(columns[col]).get("x"), self.px) + self.aim_offset - self.px))

        # LOWEST_THREAT: lower (larger y) beats distance.  The tiny distance
        # tie-break is code-side and never exposed as a Jev arithmetic task.
        def rank(col):
            low = self._lowest(columns[col])
            x = number(low.get("x"), self.px) + self.aim_offset
            return (number(low.get("y"), -999.0), -abs(x - self.px), -col)

        return max(columns, key=rank)

    def _target(self, columns):
        if self.target_col not in columns:
            self.target_col = self._target_for_policy(columns)
        if self.target_col not in columns:
            self.target_col = self._target_for_policy(columns, "LOWEST_THREAT")
        return self._lowest(columns.get(self.target_col, [])) if self.target_col in columns else None

    def _safe_left(self):
        return clamp(self.left_limit + 5.0, 36.0, 72.0)

    def _safe_right(self):
        return clamp(self.right_limit - 5.0, 104.0, 145.0)

    def _update_boundary(self, px, dx):
        prior = -1 if "left" in self.prev_action else 1 if "right" in self.prev_action else 0
        if prior < 0 and px <= self._safe_left() + 5 and abs(dx) < 0.75:
            self.stall_left += 1
        else:
            self.stall_left = 0
        if prior > 0 and px >= self._safe_right() - 5 and abs(dx) < 0.75:
            self.stall_right += 1
        else:
            self.stall_right = 0
        if self.stall_left >= 3:
            old = self.left_limit
            self.left_limit = max(self.left_limit, px + 2.0)
            self.stall_left = 0
            if self.left_limit > old + 0.5:
                self.log("left_boundary_inferred", observed_x=round(px, 1), limit=round(self.left_limit, 1))
        if self.stall_right >= 3:
            old = self.right_limit
            self.right_limit = min(self.right_limit, px - 2.0)
            self.stall_right = 0
            if self.right_limit < old - 0.5:
                self.log("right_boundary_inferred", observed_x=round(px, 1), limit=round(self.right_limit, 1))

    def _update_bombs(self, obs):
        frame = self.frame
        current = []
        for bomb in objects(obs.get("bombs"))[:12]:
            if not isinstance(bomb.get("x"), (int, float)) or not isinstance(bomb.get("y"), (int, float)):
                continue
            bx, by = number(bomb.get("x"), self.px), number(bomb.get("y"), -999.0)
            matches = [old for old in self.bomb_tracks
                       if abs(old["x"] - bx) <= 3 and 0 < frame - old["frame"] <= 16]
            velocity = 1.2
            if matches:
                old = min(matches, key=lambda item: abs(by - item["y"] - 1.2 * (frame - item["frame"])))
                velocity = clamp((by - old["y"]) / max(1, frame - old["frame"]), 0.4, 3.0)
            current.append({"x": bx, "y": by, "v": velocity, "frame": frame})
        if current:
            self.last_bomb_frame = frame
        # Preserve only a short decoder gap.  This is a memory of observed
        # threats, never a claim that an empty list is safe.
        for old in self.bomb_tracks:
            if frame - old["frame"] <= 12 and not any(abs(old["x"] - item["x"]) <= 3 for item in current):
                current.append(old)
        self.bomb_tracks = current[:16]

    def _bomb_escape(self):
        candidates = []
        for bomb in self.bomb_tracks:
            age = max(0, self.frame - integer(bomb.get("frame"), self.frame))
            bx = number(bomb.get("x"), self.px)
            by = number(bomb.get("y"), -999.0) + number(bomb.get("v"), 1.2) * age
            if by < 95 or by > self.py + 10:
                continue
            gap = abs(bx - self.px)
            if by >= 142 and gap <= 25:
                urgency = 0
            elif by >= 112 and gap <= 32:
                urgency = 1
            elif gap <= 20:
                urgency = 2
            else:
                continue
            candidates.append((urgency, by, gap, bx))
        if not candidates:
            return None
        _, _, _, bx = min(candidates)
        wanted = 1 if bx <= self.px else -1
        if self.px <= self._safe_left() + 2:
            wanted = 1
        elif self.px >= self._safe_right() - 2:
            wanted = -1
        return wanted

    def _commit_route(self, columns):
        left = self._safe_left()
        right = self._safe_right()
        lane_xs = []
        for values in columns.values():
            low = self._lowest(values)
            if low is not None:
                lane_xs.append(number(low.get("x"), self.px) + self.aim_offset)
        if lane_xs:
            lo = clamp(min(lane_xs) - 12.0, left, right)
            hi = clamp(max(lane_xs) + 12.0, left, right)
        else:
            lo, hi = left, right
        # When only a narrow cluster remains, keep an escape traverse instead
        # of shuttling in a few pixels around one target lane.
        if hi - lo < 48.0:
            center = (lo + hi) / 2.0
            lo = clamp(center - 28.0, left, right)
            hi = clamp(center + 28.0, left, right)
        if hi - lo < 18.0:
            lo, hi = left, right

        old_direction = integer(self.route.get("direction"), self.direction) if self.route else 0
        if not self.route:
            center = (lo + hi) / 2.0
            direction = 1 if self.px <= center else -1
        else:
            direction = -1 if old_direction >= 0 else 1
        endpoint = hi if direction > 0 else lo
        self.direction = direction
        self.route = {
            "lo": lo,
            "hi": hi,
            "endpoint": endpoint,
            "direction": direction,
            "start": self.frame,
        }
        self.log("sweep_committed", lo=round(lo, 1), hi=round(hi, 1),
                 endpoint=round(endpoint, 1), direction=direction,
                 policy=self.target_policy)

    def _route_direction(self, columns):
        if self.route is None:
            self._commit_route(columns)
        route = self.route
        direction = integer(route.get("direction"), self.direction)
        endpoint = number(route.get("endpoint"), self.px)
        reached = ((direction > 0 and self.px >= endpoint - 1.0) or
                   (direction < 0 and self.px <= endpoint + 1.0))
        expired = self.frame - integer(route.get("start"), self.frame) > 260
        if reached or expired:
            self.log("sweep_finished", reason="reached" if reached else "expired",
                     frame=self.frame, x=round(self.px, 1))
            self._commit_route(columns)
            route = self.route
            direction = integer(route.get("direction"), self.direction)

        # Never continue pushing at the learned interior edge.  A visible
        # bomb can override the committed sweep, but the next tick resumes it.
        if self.px <= self._safe_left() and direction < 0:
            direction = 1
        elif self.px >= self._safe_right() and direction > 0:
            direction = -1
        self.direction = direction
        return direction

    def _calibrate_hit(self, old_obs):
        if self.last_shot_x is None or self.last_shot_col is None:
            return
        old_columns = self._columns(old_obs)
        target = self._lowest(old_columns.get(self.last_shot_col, []))
        if target is None:
            return
        residual = self.last_shot_x - number(target.get("x"), self.last_shot_x)
        if -12.0 <= residual <= 12.0:
            old_offset = self.aim_offset
            self.aim_offset = clamp(0.8 * self.aim_offset + 0.2 * residual, -8.0, 8.0)
            self.log("shot_lane_calibrated", old=round(old_offset, 2),
                     new=round(self.aim_offset, 2), residual=round(residual, 2))

    def observe(self, raw):
        obs = dict_observation(raw)
        old = self.previous
        old_frame = self.frame
        frame = integer(obs.get("frame"), old_frame + 4)
        old_px = self.px
        player = obs.get("player") if isinstance(obs.get("player"), dict) else {}
        px = clamp(number(player.get("x"), self.px), 0.0, 159.0)
        py = clamp(number(player.get("y"), self.py), 0.0, 209.0)
        lives = integer(obs.get("lives"), self.lives)
        wave = integer(obs.get("wave"), self.wave)
        current_alive = self._alive_set(obs)
        current_shot = obs.get("player_shot") if isinstance(obs.get("player_shot"), dict) else {}
        shot_active = bool(current_shot.get("active"))
        self.last_outcome = "NONE"

        if old is not None:
            self._update_boundary(px, px - old_px)
            killed = self.previous_alive - current_alive
            if killed:
                self.hits += len(killed)
                self.last_outcome = "HIT"
                self._calibrate_hit(old)
                self.target_col = None
            elif self.prev_shot_active and not shot_active:
                self.misses += 1
                self.last_outcome = "MISS"
                self.target_col = None

        if lives < self.lives:
            self.life_losses += self.lives - lives
            self.recovery = 16
            self.route = None
            self.target_col = None
            self.bomb_tracks = []
            self.direction *= -1
            self.log("life_loss", frame=frame, previous_x=round(self.px, 1),
                     last_bomb_frame=self.last_bomb_frame, score=self.score)
        if wave != self.wave:
            self.route = None
            self.target_col = None
            self.recovery = 10
            self.bomb_tracks = []
            self.log("wave_change", wave=wave, frame=frame)

        # A transition to active identifies the actual launch lane despite the
        # controller pressing FIRE on every tick.
        if shot_active and not self.prev_shot_active:
            columns = self._columns(obs)
            self.last_shot_x = number(current_shot.get("x"), px)
            self.last_shot_col = self.target_col
            if self.last_shot_col not in columns:
                self.last_shot_col = self._target_for_policy(columns)

        score = number(obs.get("score"), self.score)
        if score > self.score:
            self.last_progress = frame
        self.frame = frame
        self.px, self.py = px, py
        self.lives, self.wave, self.score = lives, wave, score
        self.previous_alive = current_alive
        self.prev_shot_active = shot_active
        self._update_bombs(obs)
        self.previous = obs
        return obs

    def accept(self, result):
        if not isinstance(result, dict) or result.get("error"):
            return
        tag = result.get("tag")
        if not isinstance(tag, dict):
            return
        if integer(tag.get("lives"), -1) != self.lives or integer(tag.get("wave"), -1) != self.wave:
            self.log("jev_discarded", reason="life_or_wave_changed")
            return
        age = self.frame - integer(tag.get("frame"), -100000)
        if age < 0 or age > 56:
            self.log("jev_discarded", reason="stale", age=age)
            return
        answers = result.get("answers")
        if not isinstance(answers, dict):
            return
        first, second = answers.get("policy_a"), answers.get("policy_b")
        left, right = answer_value(first), answer_value(second)
        if not isinstance(left, str) or not isinstance(right, str):
            return
        left, right = left.upper(), right.upper()
        if left != right or left not in POLICIES:
            return
        # Agreement plus a modest probability gate protects high-consequence
        # lane changes while still allowing the controller to run without Jev.
        for answer in (first, second):
            if answer_probability(answer, left) < 0.55:
                return
            confidence = answer_confidence(answer)
            if confidence is not None and confidence < 0.20:
                return
        previous = self.target_policy
        self.target_policy = left
        self.log("jev_target_policy", previous=previous, policy=left, age=age)

    def compact_state(self, obs):
        columns = self._columns(obs)
        target = self._target(columns)
        alive_count = len(self.previous_alive)
        if alive_count <= 12:
            phase = "LATE"
        elif alive_count <= 24:
            phase = "MID"
        else:
            phase = "OPENING"
        if self.px <= self._safe_left() + 2:
            lane = "LEFT_MARGIN"
        elif self.px >= self._safe_right() - 2:
            lane = "RIGHT_MARGIN"
        else:
            lane = "INTERIOR"
        bomb_state = "UNSEEN_OR_EMPTY"
        for bomb in self.bomb_tracks:
            age = max(0, self.frame - integer(bomb.get("frame"), self.frame))
            bx = number(bomb.get("x"), self.px)
            by = number(bomb.get("y"), -999.0) + number(bomb.get("v"), 1.2) * age
            if by >= 142 and abs(bx - self.px) <= 25:
                bomb_state = "NEAR_CANNON"
                break
            if by >= 112 and abs(bx - self.px) <= 32:
                bomb_state = "APPROACHING"
        living = []
        for col in sorted(columns):
            low = self._lowest(columns[col])
            y = number(low.get("y"), 0.0)
            side = "LEFT" if col <= 1 else "RIGHT" if col >= 4 else "CENTER"
            band = "BOTTOM" if y >= 115 else "MIDDLE" if y >= 75 else "TOP"
            living.append({"column": "C" + str(col), "side": side, "lowest_band": band})
        return {
            "schema": "space_invaders_fire_sweep_v1",
            "rules": "One laser is in flight at a time; repeated FIRE is harmless while it is active. Empty bombs is not proof of safety. Code owns geometry and legal actions.",
            "frame": self.frame,
            "lives": self.lives,
            "wave": self.wave,
            "phase": phase,
            "player_lane": lane,
            "sweep": "RIGHT" if self.direction > 0 else "LEFT",
            "shot": "ACTIVE" if self.prev_shot_active else "READY",
            "shot_outcome": self.last_outcome,
            "bomb_evidence": bomb_state,
            "progress": "RECENT" if self.frame - self.last_progress < 160 else "QUIET",
            "current_target": None if target is None else {
                "column": "C" + str(self.target_col),
                "side": "LEFT" if self.target_col <= 1 else "RIGHT" if self.target_col >= 4 else "CENTER",
            },
            "living_columns": living,
            "current_policy": self.target_policy,
            "boundary": {
                "left": "learned" if self.left_limit > 32 else "initial",
                "right": "learned" if self.right_limit < 148 else "initial",
                "escape_margin": "preserved",
            },
        }

    def action(self, obs):
        columns = self._columns(obs)
        self._target(columns)
        self._route_direction(columns)
        escape = self._bomb_escape()
        direction = escape if escape is not None else self.direction
        if self.px <= self._safe_left() and direction < 0:
            direction = 1
        if self.px >= self._safe_right() and direction > 0:
            direction = -1
        self.direction = direction
        base = "right" if direction > 0 else "left"
        # This is the tested change from the parent's alignment gate: keep
        # FIRE held during every sweep, including while a previous shot is
        # active.  The emulator's one-shot lock makes this a no-op until the
        # beam exits, while movement continues on the same tick.
        source = "visible_bomb_dodge" if escape is not None else "stream_sweep"
        if self.recovery > 0:
            source = "post_life_sweep" if escape is None else source
            self.recovery -= 1
        self.last_fire_frame = self.frame
        self.actions += 1
        return base + "_fire", source


class Harness:
    def __init__(self, ctx):
        self.ctx = ctx

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

    def _post(self, action, legal, noop):
        if action not in legal:
            action = noop
        try:
            self.ctx.act(action)
            return True
        except (BudgetExceeded, EpisodeOver):
            return False
        except Exception:
            if action != noop and noop in legal:
                try:
                    self.ctx.act(noop)
                    return True
                except Exception:
                    return False
            return False

    def _read(self):
        try:
            return self.ctx.observe()
        except (BudgetExceeded, EpisodeOver):
            return None
        except Exception:
            return None

    def run_episode(self, game) -> None:
        ctx = self.ctx
        legal, noop = self._legal(game)
        controller = Controller(ctx)
        mailbox = JevMailbox(ctx)
        last_query = -100000

        try:
            raw = self._read()
            obs = controller.observe(raw)
            if not isinstance(obs, dict):
                obs = {}
            if not obs.get("game_over") and not getattr(game, "done", False) and not getattr(ctx, "done", False):
                mailbox.submit(controller.compact_state(obs), controller.frame, controller.lives, controller.wave)
                last_query = controller.frame
                action, source = controller.action(obs)
                if not self._post(action, legal, noop):
                    return

            while not getattr(ctx, "done", False) and not getattr(game, "done", False):
                try:
                    ctx.wait_for_tick()
                except (BudgetExceeded, EpisodeOver):
                    break
                except Exception:
                    break
                if getattr(ctx, "done", False) or getattr(game, "done", False):
                    break
                raw = self._read()
                if raw is None:
                    break
                obs = controller.observe(raw)
                if not isinstance(obs, dict) or obs.get("game_over"):
                    break

                result = mailbox.take()
                if result is not None:
                    controller.accept(result)
                if controller.frame - last_query >= 32:
                    mailbox.submit(controller.compact_state(obs), controller.frame,
                                   controller.lives, controller.wave)
                    last_query = controller.frame
                action, source = controller.action(obs)
                if not self._post(action, legal, noop):
                    break
                if controller.actions % 150 == 0:
                    controller.log("controller_summary", frame=controller.frame,
                                   score=controller.score, lives=controller.lives,
                                   source=source, policy=controller.target_policy,
                                   hits=controller.hits, misses=controller.misses)
        finally:
            mailbox.stop()
            controller.log("fire_sweep_finished", frame=controller.frame,
                           score=controller.score, lives=controller.lives,
                           actions=controller.actions, hits=controller.hits,
                           misses=controller.misses, life_losses=controller.life_losses,
                           aim_offset=round(controller.aim_offset, 2),
                           left_limit=round(controller.left_limit, 1),
                           right_limit=round(controller.right_limit, 1))


def build(ctx):
    return Harness(ctx)
