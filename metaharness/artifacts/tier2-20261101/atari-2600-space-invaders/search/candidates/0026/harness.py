"""0026: life-aware recovery corridors with a code-owned firing sweep.

The controller keeps the only mechanism that has repeatedly produced score:
post a fresh combined-FIRE action every tick and traverse a wide horizontal
lane.  Unlike the temporal-evidence candidates, model calls are not asked to
interpret unreliable projectile identities.  A delayed life decrement is
used as a reliable episode event: the next life gets a short, healthy-side
recovery corridor before the wide sweep resumes.  Jev may select that corridor
from qualitative, code-derived alternatives; code remains responsible for
coordinates, collision checks, and legal actions.
"""
from __future__ import annotations

import json
import math

from metaharness.core.contract import BudgetExceeded, EpisodeOver
from metaharness.models.jev import Choice


ACTIONS = ("noop", "fire", "right", "left", "right_fire", "left_fire")


def as_number(value, default=0.0):
    try:
        result = float(value)
        return result if math.isfinite(result) else default
    except (TypeError, ValueError, OverflowError):
        return default


def as_int(value, default=0):
    try:
        return int(as_number(value, default))
    except (TypeError, ValueError, OverflowError):
        return default


def clamp(value, low, high):
    return max(low, min(high, value))


def obj(value):
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            decoded = json.loads(value)
            return decoded if isinstance(decoded, dict) else {}
        except (TypeError, ValueError, json.JSONDecodeError):
            return {}
    return {}


def rows(value):
    return [item for item in value if isinstance(item, dict)] if isinstance(value, list) else []


def yes(value):
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "alive"}
    return bool(value)


def answer_field(answer, name, default=None):
    if isinstance(answer, dict):
        return answer.get(name, default)
    return getattr(answer, name, default)


def selected_probability(answer, value):
    probabilities = answer_field(answer, "probabilities", {})
    if not isinstance(probabilities, dict):
        return 0.0
    for key, probability in probabilities.items():
        if str(key).upper() == str(value).upper():
            return as_number(probability, 0.0)
    return 0.0


def agreed_choice(answers, prefix, allowed):
    first = answers.get(prefix + "_a")
    second = answers.get(prefix + "_b")
    value = answer_field(first, "value")
    if not isinstance(value, str):
        return None
    value = value.upper()
    if value not in allowed or str(answer_field(second, "value", "")).upper() != value:
        return None
    for answer in (first, second):
        if selected_probability(answer, value) < 0.62:
            return None
        confidence = answer_field(answer, "confidence")
        if confidence is not None and as_number(confidence, -1.0) < 0.20:
            return None
    return value


class Mailbox:
    """One bounded background call; the worker never owns a polling loop."""

    def __init__(self, ctx):
        self.ctx = ctx
        self.future = None
        self.tag = None

    def launch(self, function, tag):
        if self.future is not None or bool(getattr(self.ctx, "done", False)):
            return False

        def work():
            try:
                return {"result": function()}
            except Exception as exc:
                # The context records persistent unavailability itself.  A
                # timeout or malformed response is an ordinary fallback.
                return {"error": type(exc).__name__}

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
            value = future.result()
        except Exception as exc:
            value = {"error": type(exc).__name__}
        return value if isinstance(value, dict) else {"error": "non_dict", "tag": tag}

    def close(self):
        if self.future is not None:
            try:
                self.future.cancel()
            except Exception:
                pass
        self.future = None
        self.tag = None


class Controller:
    """A fresh-tick sweep plus a bounded post-life recovery state machine."""

    def __init__(self, ctx):
        self.ctx = ctx
        self.frame = 0
        self.previous = None
        self.last_action = "noop"
        self.px, self.py = 90.0, 190.0
        self.last_mobile_x = 90.0
        self.last_mobile_frame = 0
        self.lives = 3
        self.wave = 1
        self.score = 0.0
        self.alive_count = 36
        self.lowest_band = "TOP"
        self.lowest_side = "CENTER"

        # command_sign is the observed screen-x displacement of the command
        # named RIGHT.  It is code-derived, not model supplied.  Calibrating
        # here makes recovery corridors safe even if an adaptor's joystick
        # polarity is reversed.
        self.command_sign = 1
        self.polarity_votes = []
        self.epoch = 0

        self.screen_direction = 1
        self.recovery_side = None
        self.recovery_until = -1
        self.recovery_event_frame = -1
        self.death_x = None
        self.cover_left = 112
        self.cover_middle = 112
        self.cover_right = 112
        self.astra_rule = "AWAY"

        self.bombs = []
        self.dodge_until = -1
        self.dodge_direction = 0
        self.shield_damage_events = 0
        self.life_event = False
        self.wave_event = False
        self.actions = 0
        self.recoveries = 0
        self.model_recoveries = 0
        self.dodges = 0
        self.life_losses = 0
        self.scoreless_frames = 0
        self.last_score_frame = 0

    def log(self, event, **fields):
        try:
            self.ctx.log(event, frame=self.frame, **fields)
        except Exception:
            pass

    def player(self, observation):
        value = obj(observation.get("player"))
        return (
            clamp(as_number(value.get("x"), self.px), 0.0, 159.0),
            clamp(as_number(value.get("y"), self.py), 0.0, 209.0),
        )

    def living(self, observation):
        result = []
        for item in rows(observation.get("invaders")):
            if yes(item.get("alive")):
                result.append(item)
        return result

    def side_of(self, x):
        if as_number(x, 90.0) < 70.0:
            return "LEFT"
        if as_number(x, 90.0) > 100.0:
            return "RIGHT"
        return "CENTER"

    def cover_state(self, observation):
        shields = rows(observation.get("shields"))[:3]
        values = []
        for shield in shields:
            pixels = shield.get("pixels")
            if pixels is None:
                mask = shield.get("health_mask")
                pixels = sum(int(as_number(bits, 0)).bit_count() for bits in mask[:9]) if isinstance(mask, list) else 0
            values.append(max(0, as_int(pixels, 0)))
        while len(values) < 3:
            values.append(0)
        old = (self.cover_left, self.cover_middle, self.cover_right)
        self.cover_left, self.cover_middle, self.cover_right = values[:3]
        if sum(abs(a - b) for a, b in zip(old, values[:3])) >= 8:
            self.shield_damage_events += 1

    def cover_side(self):
        left = self.cover_left + 0.55 * self.cover_middle
        right = self.cover_right + 0.55 * self.cover_middle
        if left > right + 8:
            return "LEFT"
        if right > left + 8:
            return "RIGHT"
        return "CENTER"

    def update_formation_summary(self, observation):
        live = self.living(observation)
        self.alive_count = len(live)
        if not live:
            self.lowest_band, self.lowest_side = "NONE", "CENTER"
            return
        lowest = max(live, key=lambda item: as_number(item.get("y"), -1))
        y = as_number(lowest.get("y"), 0)
        self.lowest_band = "LOW" if y >= 105 else "MID" if y >= 70 else "TOP"
        self.lowest_side = self.side_of(as_number(lowest.get("x"), 90))

    def update_bombs(self, observation):
        current = []
        for bomb in rows(observation.get("bombs"))[:16]:
            if bomb.get("x") is None or bomb.get("y") is None:
                continue
            current.append({"x": as_number(bomb.get("x"), self.px),
                            "y": as_number(bomb.get("y"), -999)})
        self.bombs = current

    def deterministic_recovery_side(self):
        death_side = self.side_of(self.death_x if self.death_x is not None else self.px)
        # Move away from the last impact lane first.  Astra's COVER rule only
        # changes the genuinely ambiguous central case; it never overrides a
        # known left/right impact side.
        if death_side == "LEFT":
            return "RIGHT"
        if death_side == "RIGHT":
            return "LEFT"
        if self.astra_rule == "COVER":
            cover = self.cover_side()
            if cover in ("LEFT", "RIGHT"):
                return cover
        return "LEFT" if self.px >= 80 else "RIGHT"

    def begin_recovery(self):
        self.recovery_side = self.deterministic_recovery_side()
        # Aim for the selected half immediately; otherwise the preceding
        # sweep heading could carry a right-side recovery into the wrong half.
        self.screen_direction = -1 if self.recovery_side == "LEFT" else 1
        self.recovery_event_frame = self.frame
        self.recovery_until = self.frame + 180
        self.recoveries += 1
        self.log("recovery_started", side=self.recovery_side,
                 death_x=None if self.death_x is None else round(self.death_x, 1),
                 cover_side=self.cover_side(), rule=self.astra_rule)

    def update(self, raw):
        observation = obj(raw)
        frame = as_int(observation.get("frame"), self.frame + 4)
        if self.previous is not None and frame <= self.frame:
            frame = self.frame + 1
        old_x, old_lives, old_wave = self.px, self.lives, self.wave
        old_frame = self.frame
        px, py = self.player(observation)
        df = max(1, frame - old_frame)
        command = 1 if "right" in self.last_action else -1 if "left" in self.last_action else 0
        teleport = self.previous is not None and abs(px - old_x) > max(14.0, 2.5 * df)
        lives = as_int(observation.get("lives"), old_lives)
        wave = as_int(observation.get("wave"), old_wave)
        self.life_event = lives < old_lives
        self.wave_event = wave != old_wave

        # Learn command polarity only from ordinary movement, never from a
        # respawn/teleport.  Two consistent votes are enough to replace the
        # standard sign; the sweep itself remains active during calibration.
        if self.previous is not None and command and not teleport and not self.life_event:
            dx = px - old_x
            if 0.5 <= abs(dx) <= 1.5 * df:
                vote = 1 if dx * command > 0 else -1
                self.polarity_votes.append(vote)
                self.polarity_votes = self.polarity_votes[-6:]
                if len(self.polarity_votes) >= 2:
                    positive = sum(1 for item in self.polarity_votes if item > 0)
                    negative = len(self.polarity_votes) - positive
                    if positive >= 2 or negative >= 2:
                        self.command_sign = 1 if positive >= negative else -1
        if self.previous is None or teleport:
            self.last_mobile_x = px
            self.last_mobile_frame = frame
        elif abs(px - old_x) >= 0.5:
            self.last_mobile_x = px
            self.last_mobile_frame = frame

        self.frame, self.px, self.py = frame, px, py
        self.lives, self.wave = lives, wave
        score = as_number(observation.get("score"), self.score)
        if score > self.score:
            self.last_score_frame = frame
            self.scoreless_frames = 0
        else:
            self.scoreless_frames = max(0, frame - self.last_score_frame)
        self.score = score
        self.cover_state(observation)
        self.update_formation_summary(observation)
        self.update_bombs(observation)

        if self.life_event:
            self.life_losses += old_lives - lives
            self.epoch += 1
            self.death_x = self.last_mobile_x if self.last_mobile_x is not None else old_x
            self.dodge_until = -1
            self.dodge_direction = 0
            self.begin_recovery()
            self.log("life_restart", old_lives=old_lives, lives=lives,
                     death_x=round(self.death_x, 1),
                     last_mobile_frame=self.last_mobile_frame)
        if self.wave_event:
            self.epoch += 1
            self.recovery_side = None
            self.recovery_until = -1
            self.death_x = None
            self.log("wave_restart", wave=wave)
        if self.recovery_side is not None and self.frame >= self.recovery_until:
            self.log("recovery_finished", side=self.recovery_side,
                     frames=self.frame - self.recovery_event_frame)
            self.recovery_side = None
            self.recovery_until = -1
            self.death_x = None

        self.previous = observation
        return observation

    def visible_dodge(self):
        urgent = []
        for bomb in self.bombs:
            bx, by = bomb["x"], bomb["y"]
            # A decoded bomb is high-value evidence.  Keep the ordinary
            # sweep and FIRE, but select a screen direction away from the
            # collision lane for a short commitment.
            if by >= 140 and abs(bx - self.px) <= 25:
                wanted = -1 if bx > self.px else 1
                urgent.append((max(0.0, self.py - by), wanted))
        if not urgent:
            return None
        _, wanted = min(urgent)
        if self.px <= 42:
            wanted = 1
        elif self.px >= 116:
            wanted = -1
        self.dodge_until = max(self.dodge_until, self.frame + 28)
        self.dodge_direction = wanted
        return wanted

    def sweep_direction(self):
        if self.px <= 40:
            self.screen_direction = 1
        elif self.px >= 116:
            self.screen_direction = -1
        return self.screen_direction

    def recovery_direction(self):
        # Stay in the outer half for a short post-life interval.  This is not
        # a permanent death wall: after 180 frames the normal wide sweep is
        # restored, so firing coverage and wave progress remain the default.
        if self.recovery_side == "LEFT":
            lo, hi = 42.0, 72.0
        else:
            lo, hi = 88.0, 116.0
        if self.px <= lo:
            self.screen_direction = 1
        elif self.px >= hi:
            self.screen_direction = -1
        return self.screen_direction

    def command_for_screen(self, direction):
        if direction == 0:
            return "fire"
        # command_sign is dx_screen for the command named RIGHT.
        command = "right" if direction * self.command_sign > 0 else "left"
        return command + "_fire"

    def action(self):
        dodge = self.visible_dodge()
        if dodge is not None:
            self.screen_direction = dodge
            source = "visible_bomb"
            self.dodges += 1
        elif self.frame < self.dodge_until and self.dodge_direction:
            self.screen_direction = self.dodge_direction
            source = "dodge_commit"
        elif self.recovery_side is not None:
            self.screen_direction = self.recovery_direction()
            source = "life_recovery"
        else:
            self.screen_direction = self.sweep_direction()
            source = "wide_sweep"
        self.actions += 1
        return self.command_for_screen(self.screen_direction), source

    def recovery_state(self):
        fallback = self.deterministic_recovery_side()
        return {
            "schema": "space_invaders_life_recovery_v1",
            "rules": {
                "fire": "Every action should retain FIRE; code owns the one-shot lock.",
                "recovery": "A short outer-half corridor follows a life loss; then the wide sweep resumes.",
                "model": "Choose among supplied corridors only; do not calculate coordinates or invent threats.",
            },
            "event": "LIFE_RESTART",
            "life_epoch": self.epoch,
            "lives": self.lives,
            "death_zone": self.side_of(self.death_x),
            "fallback": fallback,
            "cover": {
                "left": "GOOD" if self.cover_left >= 90 else "DAMAGED" if self.cover_left >= 45 else "OPEN",
                "middle": "GOOD" if self.cover_middle >= 90 else "DAMAGED" if self.cover_middle >= 45 else "OPEN",
                "right": "GOOD" if self.cover_right >= 90 else "DAMAGED" if self.cover_right >= 45 else "OPEN",
                "healthier_outer": self.cover_side(),
            },
            "formation": {
                "phase": "LATE" if self.alive_count <= 12 else "MID" if self.alive_count <= 24 else "OPENING",
                "lowest_band": self.lowest_band,
                "lowest_side": self.lowest_side,
            },
            "recent_control": "responsive" if self.frame - self.last_mobile_frame < 32 else "post_impact_or_motion_gap",
        }


def recovery_questions(frame):
    options = {
        "LEFT": "Use the left outer recovery corridor for the next short interval, keeping FIRE and avoiding a central re-entry.",
        "RIGHT": "Use the right outer recovery corridor for the next short interval, keeping FIRE and avoiding a central re-entry.",
        "FULL_SWEEP": "Do not use a recovery corridor; resume the code-owned wide sweep immediately.",
    }
    ordered = list(options.items())
    if (as_int(frame) // 4) % 2:
        ordered.reverse()
    instruction = (
        "Choose the safest supplied post-life posture for this real-time Atari Space Invaders state. "
        "The code has already summarized the death side, shield condition, formation phase, and fallback. "
        "LEFT and RIGHT are temporary outer-half corridors, not a permanent wall; FULL_SWEEP resumes broad coverage. "
        "Do not count, compare raw coordinates, plan a wave, or invent a bomb. Return only the best named option."
    )
    first = dict(ordered)
    second = dict(reversed(ordered))
    return {
        "recovery_a": Choice(instruction, first),
        "recovery_b": Choice(instruction, second),
    }


class Harness:
    def __init__(self, ctx):
        self.ctx = ctx

    def log(self, event, **fields):
        try:
            self.ctx.log(event, **fields)
        except Exception:
            pass

    def legal_actions(self, game):
        spec = getattr(game, "action_spec", None)
        names = getattr(spec, "names", ACTIONS)
        try:
            legal = set(names)
        except Exception:
            legal = set(ACTIONS)
        if not legal:
            legal = set(ACTIONS)
        noop = getattr(spec, "noop", "noop")
        if noop not in legal:
            noop = "noop" if "noop" in legal else next(iter(legal))
        return legal, noop

    def post(self, action, legal, noop, controller):
        if action not in legal:
            if "fire" in action:
                action = "fire" if "fire" in legal else noop
            else:
                action = noop
        try:
            self.ctx.act(action)
            controller.last_action = action
            return True
        except Exception:
            # A malformed stand-in action spec should not turn model garbage
            # into a harness error.  Real canonical actions are all legal.
            if noop in legal:
                try:
                    self.ctx.act(noop)
                    controller.last_action = noop
                    return True
                except Exception:
                    return False
            return False

    def start_astra(self, controller, mailbox):
        if bool(getattr(self.ctx, "done", False)):
            return
        try:
            remaining = self.ctx.budget.remaining()
        except Exception:
            remaining = {}
        if not isinstance(remaining, dict):
            remaining = {}
        if as_number(remaining.get("system2_calls"), 0) <= 0 or as_number(remaining.get("wall_s"), 0) <= 55:
            return
        prompt = (
            "You are the slow strategic reviewer for a real-time Atari 2600 Space Invaders controller. "
            "The code must keep a fresh combined-FIRE wide sweep. When a life decrement is observed, "
            "it can use either a short outer-half corridor away from the last impact lane (AWAY), "
            "or the outer corridor with the healthier shield as the tie-breaker (COVER). "
            "Choose the durable tie-break rule, not an action or prose. Reply with exactly one token: AWAY or COVER. "
            "AWAY is the conservative default when the death lane is known; COVER is only a tie-break for a central death."
        )
        mailbox.launch(lambda: self.ctx.think(prompt, timeout_s=52.0), {"kind": "astra_rule"})

    def apply_astra(self, packet, controller):
        if not isinstance(packet, dict) or packet.get("error"):
            return
        result = packet.get("result")
        text = getattr(result, "text", None)
        if not isinstance(text, str):
            return
        token = text.strip().strip("`'\" .\n").upper()
        if token in ("AWAY", "COVER"):
            controller.astra_rule = token
            self.log("astra_rule", rule=token)

    def submit_life_review(self, controller, mailbox):
        if controller.epoch <= 0 or controller.recovery_event_frame != controller.frame:
            return
        try:
            remaining = self.ctx.budget.remaining()
        except Exception:
            remaining = {}
        if not isinstance(remaining, dict):
            remaining = {}
        if as_number(remaining.get("system1_calls"), 0) <= 1:
            return
        state = controller.recovery_state()
        tag = {
            "frame": controller.frame,
            "epoch": controller.epoch,
            "lives": controller.lives,
            "wave": controller.wave,
        }
        questions = recovery_questions(controller.frame)

        def request():
            try:
                return {"answers": self.ctx.decide(state, questions, timeout_s=0.9), "tag": tag}
            except Exception as exc:
                return {"error": type(exc).__name__, "tag": tag}

        mailbox.launch(request, tag)

    def apply_life_review(self, packet, controller):
        if not isinstance(packet, dict) or packet.get("error"):
            return
        tag = packet.get("tag")
        if not isinstance(tag, dict):
            return
        if (as_int(tag.get("epoch"), -1) != controller.epoch or
                as_int(tag.get("lives"), -1) != controller.lives or
                as_int(tag.get("wave"), -1) != controller.wave or
                controller.frame - as_int(tag.get("frame"), -100000) > 48):
            self.log("life_review_discard", reason="stale")
            return
        answers = packet.get("answers")
        if not isinstance(answers, dict):
            return
        choice = agreed_choice(answers, "recovery", {"LEFT", "RIGHT", "FULL_SWEEP"})
        if choice is None:
            return
        old = controller.recovery_side
        if choice == "FULL_SWEEP":
            controller.recovery_side = None
            controller.recovery_until = -1
        else:
            controller.recovery_side = choice
            controller.recovery_until = max(controller.recovery_until, controller.frame + 96)
        if choice != old:
            controller.model_recoveries += 1
        self.log("life_review_applied", choice=choice, previous=old)

    def run_episode(self, game):
        ctx = self.ctx
        legal, noop = self.legal_actions(game)
        controller = Controller(ctx)
        jev = Mailbox(ctx)
        astra = Mailbox(ctx)
        last_frame = None
        try:
            try:
                initial = ctx.observe()
            except Exception:
                initial = {}
            controller.update(initial)
            action, source = controller.action()
            if not self.post(action, legal, noop, controller):
                return
            self.log("controller_start", source=source, action=action)
            self.start_astra(controller, astra)

            while not bool(getattr(ctx, "done", False)) and not bool(getattr(game, "done", False)):
                ctx.wait_for_tick()
                if bool(getattr(ctx, "done", False)) or bool(getattr(game, "done", False)):
                    break
                raw = ctx.observe()
                observation = obj(raw)
                if yes(observation.get("game_over")):
                    break
                frame_before = controller.frame
                controller.update(observation)
                if last_frame is not None and controller.frame == last_frame:
                    continue
                last_frame = controller.frame

                self.apply_astra(astra.poll(), controller)
                self.apply_life_review(jev.poll(), controller)
                if controller.life_event:
                    self.submit_life_review(controller, jev)

                action, source = controller.action()
                if not self.post(action, legal, noop, controller):
                    break
                if controller.actions % 200 == 0:
                    self.log("controller_summary", score=round(controller.score, 1),
                             lives=controller.lives, source=source,
                             recovery=controller.recovery_side,
                             life_losses=controller.life_losses,
                             model_recoveries=controller.model_recoveries,
                             dodges=controller.dodges)
        except (BudgetExceeded, EpisodeOver):
            pass
        finally:
            jev.close()
            astra.close()
            self.log("controller_finished", frame=controller.frame,
                     score=round(controller.score, 1), lives=controller.lives,
                     actions=controller.actions, life_losses=controller.life_losses,
                     recoveries=controller.recoveries,
                     model_recoveries=controller.model_recoveries,
                     polarity=controller.command_sign,
                     dodges=controller.dodges,
                     shield_damage_events=controller.shield_damage_events)


def build(ctx):
    return Harness(ctx)
