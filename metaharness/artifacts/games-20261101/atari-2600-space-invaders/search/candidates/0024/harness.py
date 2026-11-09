"""Candidate 0024: late outer-column coverage excursions.

The controller deliberately keeps the proven fresh combined-FIRE sweep as its
normal policy.  It adds one bounded, observation-owned exception: when the
formation has become sparse or progress has gone quiet and a living outer
edge lies outside the working sweep, extend that edge once before returning
inside.  Jev can recognize whether the excursion is justified; code owns all
geometry, timing, legality, and fallback behavior.
"""
from __future__ import annotations

import json
import math

from metaharness.core.contract import BudgetExceeded, EpisodeOver
from metaharness.models.jev import Choice, Noul


ACTIONS = ("noop", "fire", "left", "right", "left_fire", "right_fire")


def number(value, default=0.0):
    try:
        result = float(value)
        return result if math.isfinite(result) else default
    except (TypeError, ValueError, OverflowError):
        return default


def integer(value, default=0):
    return int(number(value, default))


def bounded(value, low, high):
    return max(low, min(high, value))


def obj(value):
    return value if isinstance(value, dict) else {}


def rows(value):
    return [item for item in value if isinstance(item, dict)] if isinstance(value, list) else []


def yes(value):
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes")
    return bool(value)


def answer_value(answer, field, default=None):
    if isinstance(answer, dict):
        return answer.get(field, default)
    return getattr(answer, field, default)


class Mailbox:
    """One bounded model call; no polling loop lives on the worker."""

    def __init__(self, ctx, enabled=True):
        self.ctx = ctx
        self.enabled = bool(enabled)
        self.future = None
        self.tag = None

    def launch(self, function, tag):
        if not self.enabled or self.future is not None or self.ctx.done:
            return False

        def work():
            try:
                return {"ok": True, "value": function()}
            except Exception as exc:
                # Normal model failures are a fallback event.  Persistent
                # provider failures still mark ctx.done in the context.
                return {"ok": False, "error": type(exc).__name__}

        try:
            self.future = self.ctx.spawn(work)
            self.tag = tag
            return True
        except Exception:
            self.future = None
            self.tag = None
            return False

    def poll(self):
        if self.future is None or not self.future.done():
            return None
        future, tag = self.future, self.tag
        self.future = None
        self.tag = None
        try:
            result = future.result()
        except Exception as exc:
            result = {"ok": False, "error": type(exc).__name__}
        return result, tag

    def close(self):
        if self.future is not None:
            try:
                self.future.cancel()
            except Exception:
                pass
        self.future = None
        self.tag = None


class CoverageController:
    """A code-owned sweep with a late, bounded outer-coverage mode."""

    def __init__(self, ctx):
        self.ctx = ctx
        self.frame = 0
        self.previous = None
        self.px = 90.0
        self.py = 190.0
        self.score = 0.0
        self.lives = 3
        self.wave = 1
        self.last_score_frame = 0
        self.last_action = "noop"
        self.epoch = 0

        # This is the control that produced the best archived score.  It is
        # intentionally unchanged unless a late outer-coverage excursion is
        # armed by the evidence below.
        self.base_left = 38.0
        self.base_right = 116.0
        self.direction = 1
        self.active_side = 0              # -1, 0, +1
        self.extension_left = 30.0
        self.extension_right = 124.0
        self.side_cooldown = {-1: -1, 1: -1}
        self.last_extension = -10000

        self.alive = []
        self.alive_count = 36
        self.form_min = 26.0
        self.form_max = 106.0
        self.bottom_y = 125.0
        self.left_gap = False
        self.right_gap = False
        self.gap_started = None
        self.previous_gap = False
        self.score_quiet = 0
        self.bombs = []
        self.shield_pixels = [112, 112, 112]

        self.life_event = False
        self.wave_event = False
        self.reacquire_until = -1
        self.plan = "COMPLETE"     # Astra may tighten this to SURVIVE.
        self.plan_version = 0
        self.advice = None
        self.last_query = -10000

        self.stats = {
            "actions": 0,
            "extensions_armed": 0,
            "left_extensions": 0,
            "right_extensions": 0,
            "extensions_finished": 0,
            "visible_bomb_escapes": 0,
            "life_losses": 0,
            "jev_queries": 0,
            "jev_accepts": 0,
            "jev_changed_extensions": 0,
            "astra_updates": 0,
        }

    def log(self, event, **fields):
        try:
            self.ctx.log(event, frame=self.frame, **fields)
        except Exception:
            pass

    def corridor(self):
        low, high = self.base_left, self.base_right
        if self.active_side < 0:
            low = min(low, self.extension_left)
        elif self.active_side > 0:
            high = max(high, self.extension_right)
        return low, high

    def _shield_pixels(self, obs):
        values = []
        for shield in rows(obj(obs).get("shields"))[:3]:
            mask = shield.get("health_mask")
            if not isinstance(mask, list):
                values.append(0)
                continue
            total = 0
            for value in mask[:9]:
                total += integer(value, 0).bit_count()
            values.append(total)
        while len(values) < 3:
            values.append(0)
        return values

    def update(self, raw):
        obs = obj(raw)
        frame = integer(obs.get("frame"), self.frame + 4)
        if frame <= self.frame:
            frame = self.frame + 4
        player = obj(obs.get("player"))
        px = bounded(number(player.get("x"), self.px), 0.0, 159.0)
        py = bounded(number(player.get("y"), self.py), 150.0, 209.0)
        lives = integer(obs.get("lives"), self.lives)
        wave = integer(obs.get("wave"), self.wave)
        score = number(obs.get("score"), self.score)
        self.life_event = lives < self.lives
        self.wave_event = wave != self.wave
        jump = self.previous is not None and abs(px - self.px) > max(14.0, 1.5 * (frame - self.frame))
        discontinuity = self.life_event or self.wave_event or jump

        if score > self.score:
            self.last_score_frame = frame
        self.score_quiet = max(0, frame - self.last_score_frame)

        if discontinuity:
            self.epoch += 1
            self.active_side = 0
            self.advice = None
            self.reacquire_until = frame + 12
            if self.life_event:
                self.stats["life_losses"] += max(1, self.lives - lives)
                self.log("life_evidence", previous_x=round(self.px, 1),
                         lives=lives, score=round(score, 1),
                         warning="respawn position is not a death lane")
            if self.wave_event:
                self.log("wave_evidence", wave=wave)

        alive = []
        for item in rows(obs.get("invaders")):
            if not yes(item.get("alive")):
                continue
            if item.get("x") is None or item.get("y") is None:
                continue
            alive.append(item)
        self.alive = alive
        if alive:
            self.alive_count = len(alive)
            self.form_min = min(number(item.get("x"), self.form_min) for item in alive)
            self.form_max = max(number(item.get("x"), self.form_max) for item in alive)
            self.bottom_y = max(number(item.get("y"), self.bottom_y) for item in alive)
        else:
            # A malformed/abbreviated invader list must not create a false
            # edge excursion.  Keep the last reliable formation envelope.
            self.alive_count = max(0, 36 - integer(obs.get("invaders_killed"), 0))

        self.left_gap = bool(alive) and self.form_min < self.base_left - 5.0
        self.right_gap = bool(alive) and self.form_max > self.base_right + 5.0
        gap_now = self.left_gap or self.right_gap
        if gap_now and not self.previous_gap:
            self.gap_started = frame
            self.log("outer_coverage_gap", left=self.left_gap, right=self.right_gap,
                     form_min=round(self.form_min, 1), form_max=round(self.form_max, 1),
                     alive=self.alive_count)
        elif not gap_now:
            self.gap_started = None
        self.previous_gap = gap_now

        self.extension_left = bounded(self.form_min - 2.0, 27.0, self.base_left - 4.0)
        self.extension_right = bounded(self.form_max + 6.0, self.base_right + 4.0, 136.0)
        self.bombs = [b for b in rows(obs.get("bombs"))
                      if b.get("x") is not None and b.get("y") is not None][:12]
        self.shield_pixels = self._shield_pixels(obs)

        self.frame = frame
        self.px = px
        self.py = py
        self.score = score
        self.lives = lives
        self.wave = wave
        self.previous = obs
        return obs

    def pressure(self):
        if self.alive_count <= 12 or self.bottom_y >= 150:
            return "CRITICAL"
        if self.alive_count <= 24 or self.bottom_y >= 136:
            return "PRESSURE"
        return "OPENING"

    def can_extend(self, side):
        if side < 0:
            return self.left_gap and self.extension_left < self.base_left - 1
        return self.right_gap and self.extension_right > self.base_right + 1

    def strong_gap(self, side):
        if not self.can_extend(side):
            return False
        if self.plan == "SURVIVE":
            return self.alive_count <= 18 or self.score_quiet >= 200
        return self.alive_count <= 26 or self.score_quiet >= 120

    def _arm(self, side, reason):
        if side not in (-1, 1) or not self.can_extend(side):
            return False
        if self.frame < self.side_cooldown.get(side, -1):
            return False
        self.active_side = side
        self.last_extension = self.frame
        self.side_cooldown[side] = self.frame + 180
        self.stats["extensions_armed"] += 1
        if side < 0:
            self.stats["left_extensions"] += 1
        else:
            self.stats["right_extensions"] += 1
        self.log("outer_extension_armed", side="LEFT" if side < 0 else "RIGHT",
                 endpoint=round(self.extension_left if side < 0 else self.extension_right, 1),
                 reason=reason, alive=self.alive_count, score_quiet=self.score_quiet)
        return True

    def _advised_side(self):
        advice = self.advice
        if not isinstance(advice, dict):
            return 0
        tag = obj(advice.get("tag"))
        if tag.get("epoch") != self.epoch or tag.get("plan") != self.plan_version:
            return 0
        if self.frame - integer(tag.get("frame"), -10000) > 56:
            return 0
        value = advice.get("value")
        if value == "EXTEND_LEFT" and self.can_extend(-1):
            return -1
        if value == "EXTEND_RIGHT" and self.can_extend(1):
            return 1
        return 0

    def _boundary_side(self, low, high):
        if self.px <= low + 2.0:
            return -1
        if self.px >= high - 2.0:
            return 1
        return 0

    def maybe_arm(self):
        if self.active_side:
            return
        low, high = self.corridor()
        boundary = self._boundary_side(self.base_left, self.base_right)
        if boundary == 0:
            return
        advised = self._advised_side()
        if advised == boundary and self._arm(boundary, "jev"):
            self.stats["jev_changed_extensions"] += 1
            return
        # The deterministic fallback is deliberately stricter than the
        # normal coverage gap.  Thus a missing/late model cannot stall the
        # score-producing baseline, while a persistent late gap is still
        # exercised even with arbitrary stand-in model answers.
        if self.strong_gap(boundary):
            self._arm(boundary, "code_strong_gap")

    def _finish_extension_if_needed(self):
        if self.active_side < 0 and self.px <= self.extension_left + 2.0:
            self.active_side = 0
            self.direction = 1
            self.stats["extensions_finished"] += 1
            self.log("outer_extension_finished", side="LEFT")
        elif self.active_side > 0 and self.px >= self.extension_right - 2.0:
            self.active_side = 0
            self.direction = -1
            self.stats["extensions_finished"] += 1
            self.log("outer_extension_finished", side="RIGHT")

    def _bomb_risk(self, direction):
        risk = 0.0
        low, high = self.corridor()
        previous = -1 if "left" in self.last_action else 1 if "right" in self.last_action else 0
        for bomb in self.bombs:
            bx = number(bomb.get("x"), -999)
            by = number(bomb.get("y"), -999)
            if not (0 <= bx <= 159 and 70 <= by <= 209):
                continue
            # Coarse code-owned prediction; model answers never supply these
            # numbers.  Include one sticky-action delay.
            for delay in (0, 4):
                for future in range(0, 65, 4):
                    y = by + 1.25 * future
                    if not self.py - 18 <= y <= self.py + 8:
                        continue
                    travel = previous * min(delay, future) + direction * max(0, future - delay)
                    x = bounded(self.px + 0.55 * travel, low, high)
                    distance = abs(x - bx)
                    if distance <= 10:
                        risk = max(risk, 100.0 + 10.0 - distance)
                    elif distance < 18:
                        risk = max(risk, (18.0 - distance) / 4.0)
        return risk

    def action(self):
        self.maybe_arm()
        self._finish_extension_if_needed()
        low, high = self.corridor()

        if self.frame < self.reacquire_until:
            desired = 1 if self.px < 77 else -1
        else:
            desired = self.direction
            if self.px <= low + 2.0:
                desired = 1
            elif self.px >= high - 2.0:
                desired = -1

        risks = {d: self._bomb_risk(d) for d in (-1, 0, 1)}
        if risks[desired] >= 100.0:
            safe = min(risks, key=lambda d: (risks[d], d == 0, d != desired))
            if risks[safe] + 0.5 < risks[desired]:
                self.stats["visible_bomb_escapes"] += 1
                self.log("visible_bomb_escape", old=desired, new=safe,
                         risks={str(k): round(v, 1) for k, v in risks.items()})
                desired = safe

        if self.px <= low and desired < 0:
            desired = 1
        if self.px >= high and desired > 0:
            desired = -1
        if desired:
            self.direction = desired
        self.stats["actions"] += 1
        if desired < 0:
            return "left_fire"
        if desired > 0:
            return "right_fire"
        return "fire"

    def _life_bucket(self):
        return "THREE" if self.lives >= 3 else "TWO" if self.lives == 2 else "ONE" if self.lives == 1 else "NONE"

    def _population_bucket(self):
        return "MANY" if self.alive_count > 24 else "SOME" if self.alive_count > 12 else "FEW"

    def _shield_bucket(self):
        total = sum(self.shield_pixels)
        return "GOOD" if total >= 240 else "PARTIAL" if total >= 90 else "OPEN"

    def packet(self):
        if not (self.left_gap or self.right_gap):
            return None
        if self.alive_count > 30 and self.score_quiet < 96:
            return None
        if self.frame - self.last_query < 48:
            return None
        state = {
            "schema": "space_invaders_outer_coverage_v1",
            "episode": {"life": self._life_bucket(), "wave": max(1, self.wave),
                        "pressure": self.pressure()},
            "formation": {
                "population": self._population_bucket(),
                "left_coverage": "OUTSIDE_WORKING_SWEEP" if self.left_gap else "INSIDE_WORKING_SWEEP",
                "right_coverage": "OUTSIDE_WORKING_SWEEP" if self.right_gap else "INSIDE_WORKING_SWEEP",
                "progress": "QUIET" if self.score_quiet >= 96 else "RECENT",
            },
            "sweep": {
                "position": "LEFT_EDGE" if self.px <= self.base_left + 3 else
                            "RIGHT_EDGE" if self.px >= self.base_right - 3 else "INTERIOR",
                "heading": "LEFT" if self.direction < 0 else "RIGHT",
                "extension": "LEFT" if self.active_side < 0 else
                             "RIGHT" if self.active_side > 0 else "NONE",
            },
            "plan": {"posture": self.plan,
                     "meaning": "COMPLETE permits one late outer excursion; SURVIVE requires a stronger late gap."},
            "threat": "VISIBLE_NEAR" if any(number(b.get("y"), -1) >= 160 and
                                                abs(number(b.get("x"), 999) - self.px) <= 18
                                                for b in self.bombs) else
                      "VISIBLE_FAR" if self.bombs else "UNSEEN_NOT_SAFE",
            "shield": self._shield_bucket(),
            "rules": {
                "model_role": "recognize whether a listed outer coverage gap merits a bounded excursion",
                "no_math": "do not count, calculate coordinates, or invent bombs",
                "actuator": "code keeps moving and pressing FIRE; model never returns an action string",
            },
        }
        options = {
            "KEEP": "Keep the ordinary broad interior sweep at the next boundary; no listed late gap is worth the exposure now.",
            "EXTEND_LEFT": "At a left boundary, make one bounded excursion toward the listed living outer-left coverage gap, then return to the interior sweep.",
            "EXTEND_RIGHT": "At a right boundary, make one bounded excursion toward the listed living outer-right coverage gap, then return to the interior sweep.",
            "UNKNOWN": "The supplied categorical facts do not justify choosing an extension.",
        }
        instruction = (
            "Choose the next coverage posture from the supplied categorical state. "
            "Only an explicitly listed OUTSIDE_WORKING_SWEEP side may be extended. "
            "A quiet score or an empty bomb list is not proof that movement is safe. "
            "Prefer KEEP when the side is not at its matching boundary or when visible danger is near. "
            "Do not calculate coordinates, count invaders, invent threats, or output prose."
        )
        questions = {
            "coverage_a": Choice(instruction, options),
            "coverage_b": Choice(instruction, dict(reversed(list(options.items())))),
            "worth": Noul(
                "Does the supplied state support one bounded outer-coverage excursion now or at the next matching boundary?",
                {"true": "A listed outer gap is persistent and the posture/pressure make completion coverage worth a bounded excursion.",
                 "false": "The gap is absent, too early, or near visible danger makes KEEP safer."},
            ),
        }
        tag = {"frame": self.frame, "epoch": self.epoch, "plan": self.plan_version,
               "left": self.left_gap, "right": self.right_gap}
        return state, questions, tag

    def mark_query(self, tag):
        self.last_query = self.frame
        self.stats["jev_queries"] += 1

    def accept_jev(self, packet):
        if packet is None:
            return
        result, tag = packet
        if not isinstance(result, dict) or not result.get("ok"):
            return
        tag = obj(tag)
        if tag.get("epoch") != self.epoch or tag.get("plan") != self.plan_version:
            return
        if self.frame - integer(tag.get("frame"), -10000) > 40:
            return
        answers = obj(result.get("value"))
        first = answers.get("coverage_a")
        second = answers.get("coverage_b")
        value = answer_value(first, "value")
        if value not in ("KEEP", "EXTEND_LEFT", "EXTEND_RIGHT", "UNKNOWN"):
            return
        if answer_value(second, "value") != value:
            return
        for answer in (first, second):
            probabilities = obj(answer_value(answer, "probabilities", {}))
            if number(probabilities.get(value), 0.0) < 0.66:
                return
            confidence = answer_value(answer, "confidence")
            if confidence is not None and number(confidence, -1.0) < 0.22:
                return
        worth = number(answer_value(answers.get("worth"), "value"), 0.0)
        if value == "EXTEND_LEFT" and not self.left_gap:
            return
        if value == "EXTEND_RIGHT" and not self.right_gap:
            return
        if value in ("EXTEND_LEFT", "EXTEND_RIGHT") and worth < 0.55:
            return
        self.advice = {"value": value, "tag": tag}
        self.stats["jev_accepts"] += 1
        self.log("jev_coverage_accepted", value=value, worth=round(worth, 2))

    def astra_prompt(self):
        return (
            "Choose the risk posture for a real-time Atari 2600 Space Invaders controller. "
            "Reply with exactly one token: COMPLETE or SURVIVE. COMPLETE permits a late, "
            "bounded excursion when living outer invader coordinates lie outside a proven "
            "interior firing sweep; SURVIVE waits for a sparser formation or a longer score "
            "stall. The code always posts fresh left_fire/right_fire actions, retains an "
            "escape route, and owns all coordinates, bomb checks, and legality. Jev receives "
            "only qualitative coverage facts and may not choose an action. This is not a "
            "target lock, route jitter, shield inference, or empty-bomb safety assumption. "
            "Use COMPLETE if finishing the first wave is worth a small late exposure; use "
            "SURVIVE if preserving the established sweep is safer."
        )

    def apply_astra(self, packet):
        if packet is None:
            return
        result, unused_tag = packet
        if not isinstance(result, dict) or not result.get("ok"):
            return
        model_result = result.get("value")
        text = getattr(model_result, "text", None)
        if not isinstance(text, str):
            return
        token = text.strip().strip("`\"' .\n").upper()
        if token not in ("COMPLETE", "SURVIVE"):
            return
        if token != self.plan:
            old = self.plan
            self.plan = token
            self.plan_version += 1
            self.advice = None
            self.log("astra_posture", old=old, new=token)
        self.stats["astra_updates"] += 1


def build(ctx):
    return Harness(ctx)


class Harness:
    def __init__(self, ctx):
        self.ctx = ctx

    def run_episode(self, game) -> None:
        ctx = self.ctx
        controller = CoverageController(ctx)
        system1_enabled = getattr(ctx, "system1", True) is not None
        jev = Mailbox(ctx, system1_enabled)
        astra = Mailbox(ctx, True)
        spec = getattr(game, "action_spec", None)
        legal = set(getattr(spec, "names", ACTIONS))
        noop = getattr(spec, "noop", "noop")
        last_frame = None
        astra_sent = False

        try:
            while not ctx.done and not getattr(game, "done", False):
                raw = ctx.observe()
                obs = obj(raw)
                if yes(obs.get("game_over")):
                    break
                frame = integer(obs.get("frame"), controller.frame + 4)
                if last_frame is not None and frame <= last_frame:
                    ctx.wait_for_tick()
                    continue
                last_frame = frame

                controller.update(obs)
                controller.apply_astra(astra.poll())
                controller.accept_jev(jev.poll())

                action = controller.action()
                if action not in legal:
                    action = "fire" if "fire" in legal else noop
                ctx.act(action)
                controller.last_action = action

                remaining = obj(ctx.budget.remaining())
                if (not astra_sent and number(remaining.get("system2_calls"), 0) > 0
                        and number(remaining.get("wall_s"), 0) > 60):
                    astra_sent = astra.launch(
                        lambda prompt=controller.astra_prompt(): ctx.think(prompt, timeout_s=55.0),
                        {"frame": controller.frame},
                    )

                packet = controller.packet()
                if packet is not None and jev.future is None:
                    state, questions, tag = packet
                    if (number(remaining.get("system1_calls"), 0) > 1
                            and number(remaining.get("wall_s"), 0) > 2
                            and jev.launch(
                                lambda s=state, q=questions: ctx.decide(s, q, timeout_s=1.0),
                                tag,
                            )):
                        controller.mark_query(tag)

                if controller.stats["actions"] % 150 == 0:
                    controller.log("coverage_summary", score=round(controller.score, 1),
                                   lives=controller.lives, alive=controller.alive_count,
                                   left_gap=controller.left_gap, right_gap=controller.right_gap,
                                   active=controller.active_side, plan=controller.plan,
                                   stats=dict(controller.stats))
                ctx.wait_for_tick()
        except (BudgetExceeded, EpisodeOver):
            pass
        finally:
            jev.close()
            astra.close()
            controller.log("coverage_finished", score=round(controller.score, 1),
                           lives=controller.lives, alive=controller.alive_count,
                           active=controller.active_side, plan=controller.plan,
                           stats=dict(controller.stats))
