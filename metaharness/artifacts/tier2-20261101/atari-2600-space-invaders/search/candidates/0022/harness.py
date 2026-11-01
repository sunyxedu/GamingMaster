"""Candidate 0022: shot-window acquisition over a continuous sweep.

The controller keeps the empirically useful fresh lateral FIRE sweep for most
of a wave.  When the laser is empty and the formation is late (or has stopped
making score), it briefly acquires the current lowest living invader before
launching the next shot.  The shot then gets a full sweep while it is in
flight.  This makes the change about shot timing, rather than a model-selected
route or an always-on target lock.
"""
from __future__ import annotations

import json
import math
from concurrent.futures import CancelledError

from metaharness.core.contract import BudgetExceeded, EpisodeOver
from metaharness.models.jev import Choice


ACTIONS = ("noop", "fire", "left", "right", "left_fire", "right_fire")
CHOICES = ("AIM_TARGET", "SWEEP_FIRE", "KEEP", "EMERGENCY_LEFT", "EMERGENCY_RIGHT", "UNKNOWN")


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


def mapping(value):
    return value if isinstance(value, dict) else {}


def records(value):
    return [item for item in value if isinstance(item, dict)] if isinstance(value, list) else []


def truth(value):
    if isinstance(value, str):
        return value.strip().lower() in ("true", "yes", "1")
    return bool(value)


def observation(value):
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, dict) else {}
        except (TypeError, ValueError):
            return {}
    return {}


def answer_field(answer, name, default=None):
    if isinstance(answer, dict):
        return answer.get(name, default)
    return getattr(answer, name, default)


def answer_value(answer):
    value = answer_field(answer, "value")
    return value if isinstance(value, str) else None


def answer_probability(answer, option):
    probabilities = answer_field(answer, "probabilities", {})
    if not isinstance(probabilities, dict):
        return 0.0
    if option in probabilities:
        return number(probabilities.get(option), 0.0)
    for key, value in probabilities.items():
        if str(key).upper() == option.upper():
            return number(value, 0.0)
    return 0.0


def answer_confidence(answer):
    value = answer_field(answer, "confidence")
    if value is None:
        return None
    value = number(value, -1.0)
    return value if value >= 0.0 else None


class AsyncCall:
    """A single latest-only model request; the game loop never waits for it."""

    def __init__(self, ctx, kind):
        self.ctx = ctx
        self.kind = kind
        self.future = None
        self.tag = None
        self.enabled = kind == "jev" and getattr(ctx, "system1", True) is not None or kind == "astra"

    def launch(self, fn, tag=None):
        if not self.enabled or self.future is not None or self.ctx.done:
            return False
        try:
            remaining = self.ctx.budget.remaining()
            key = "system1_calls" if self.kind == "jev" else "system2_calls"
            if number(remaining.get(key), 0.0) <= 0:
                self.enabled = False
                return False
        except Exception:
            pass
        try:
            self.future = self.ctx.spawn(fn)
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
        except (CancelledError, TimeoutError, RuntimeError, TypeError, ValueError):
            return {"error": "model_call_failed", "tag": tag}
        except Exception:
            return {"error": "model_call_failed", "tag": tag}
        return {"result": result, "tag": tag}

    def close(self):
        future = self.future
        self.future = None
        self.tag = None
        if future is not None and not future.done():
            try:
                future.cancel()
            except Exception:
                pass


class ShotWindowController:
    """Code owns geometry; models only advise when a shot window is used."""

    def __init__(self, ctx):
        self.ctx = ctx
        self.frame = 0
        self.px = 90.0
        self.py = 190.0
        self.score = 0.0
        self.lives = 3
        self.wave = 1
        self.previous = None
        self.previous_shot_active = False
        self.shot_active = False
        self.last_action = "noop"
        self.direction = 1
        self.left_limit = 38.0
        self.right_limit = 116.0
        self.epoch = 0

        self.alive = []
        self.alive_count = 36
        self.bottom_y = 125.0
        self.last_score_frame = 0
        self.last_target_key = None
        self.aiming = False
        self.aim_ticks = 0
        self.policy = "LATE_AIM"
        self.je_mode = None
        self.je_mode_until = -1
        self.last_query_frame = -10000
        self.escape_direction = 0
        self.escape_until = -1
        self.bombs = []
        self.stats = {
            "actions": 0,
            "shot_windows": 0,
            "aim_attempts": 0,
            "aim_fires": 0,
            "aim_aborts": 0,
            "sweep_fires": 0,
            "visible_escapes": 0,
            "life_losses": 0,
            "jev_accepted": 0,
            "jev_changed_modes": 0,
            "astra_updates": 0,
        }

    def log(self, event, **fields):
        try:
            self.ctx.log(event, frame=self.frame, **fields)
        except Exception:
            pass

    def _formation(self, obs):
        self.alive = [item for item in records(obs.get("invaders"))
                      if truth(item.get("alive")) and item.get("x") is not None]
        self.alive_count = len(self.alive)
        ys = [number(item.get("y"), self.bottom_y) for item in self.alive]
        if ys:
            self.bottom_y = max(ys)

    def _update_bombs(self, obs):
        current = []
        for bomb in records(obs.get("bombs"))[:16]:
            if bomb.get("x") is None or bomb.get("y") is None:
                continue
            x = number(bomb.get("x"), -999.0)
            y = number(bomb.get("y"), -999.0)
            if not (0.0 <= x <= 159.0 and 0.0 <= y <= 209.0):
                continue
            matches = [old for old in self.bombs
                       if abs(old["x"] - x) <= 5.0
                       and 0 < self.frame - old["frame"] <= 20
                       and y >= old["y"] - 4.0]
            velocity = 1.1
            if matches:
                old = min(matches, key=lambda item: abs(y - item["y"]))
                velocity = clamp((y - old["y"]) / max(1, self.frame - old["frame"]), 0.35, 3.5)
            current.append({"x": x, "y": y, "v": velocity, "frame": self.frame})
        # Keep a short memory through the one-frame gray-pixel flicker, but
        # discard old projectiles quickly enough not to create permanent walls.
        for old in self.bombs:
            if self.frame - old["frame"] <= 20 and not any(abs(old["x"] - item["x"]) <= 5 for item in current):
                current.append(old)
        self.bombs = current[:24]

    def update(self, raw):
        obs = observation(raw)
        old_frame = self.frame
        frame = integer(obs.get("frame"), old_frame + 4)
        df = max(1, frame - old_frame) if self.previous is not None else 4
        player = mapping(obs.get("player"))
        px = clamp(number(player.get("x"), self.px), 0.0, 159.0)
        py = clamp(number(player.get("y"), self.py), 150.0, 209.0)
        lives = integer(obs.get("lives"), self.lives)
        wave = integer(obs.get("wave"), self.wave)
        life_loss = self.previous is not None and lives < self.lives
        wave_change = self.previous is not None and wave != self.wave
        teleport = self.previous is not None and abs(px - self.px) > max(12.0, 1.6 * df)

        self.frame = frame
        if life_loss:
            self.stats["life_losses"] += max(1, self.lives - lives)
        if life_loss or wave_change or teleport:
            self.epoch += 1
            self.aiming = False
            self.aim_ticks = 0
            self.last_target_key = None
            self.escape_until = -1
            self.escape_direction = 0
            self.bombs = []
            # Re-phasing is deliberately only a reset detail.  The normal
            # controller still reverses at measured movement limits.
            self.direction = 1 if (self.epoch % 2 == 0) else -1
            if life_loss:
                self.log("shot_window_life_reset", lives=lives, previous_x=round(self.px, 1))
            elif wave_change:
                self.log("shot_window_wave_reset", wave=wave)
        else:
            dx = px - self.px
            if abs(dx) >= 0.35:
                commanded = -1 if "left" in self.last_action else 1 if "right" in self.last_action else 0
                if commanded and commanded * dx > 0:
                    self.direction = commanded

        shot = mapping(obs.get("player_shot"))
        self.previous_shot_active = self.shot_active
        self.shot_active = truth(shot.get("active"))
        if self.previous_shot_active and not self.shot_active:
            self.aiming = False
            self.aim_ticks = 0
        score = number(obs.get("score"), self.score)
        if score > self.score:
            self.last_score_frame = frame
        self.score = score
        self.lives = lives
        self.wave = wave
        self.px = px
        self.py = py
        self._formation(obs)
        self._update_bombs(obs)
        self.previous = obs
        return obs

    def phase(self):
        if self.alive_count >= 27:
            return "OPENING"
        if self.alive_count >= 15:
            return "MIDWAVE"
        return "LATEWAVE"

    def _target(self):
        if not self.alive:
            return None
        bottom = max(number(item.get("y"), self.bottom_y) for item in self.alive)
        # Prefer the currently front row, then the outer side in the direction
        # of travel.  This is code-side bookkeeping, not a Jev counting task.
        pool = [item for item in self.alive if number(item.get("y"), bottom) >= bottom - 3.0]
        side = 1 if self.direction >= 0 else -1
        def key(item):
            x = number(item.get("x"), self.px)
            col = integer(item.get("col"), 0)
            side_rank = -x if side > 0 else x
            return (abs(number(item.get("y"), bottom) - bottom), side_rank, abs(x - self.px), col)
        item = min(pool or self.alive, key=key)
        x = clamp(number(item.get("x"), self.px), self.left_limit, self.right_limit)
        return {
            "key": (integer(item.get("row"), -1), integer(item.get("col"), -1)),
            "x": x,
            "side": "LEFT" if x < self.px - 6 else "RIGHT" if x > self.px + 6 else "CENTER",
            "height": "LOW" if number(item.get("y"), bottom) >= bottom - 3 else "HIGHER",
        }

    def _hazard_direction(self):
        """Return an immediate lateral escape only for a present near bomb."""
        candidates = []
        for bomb in self.bombs:
            age = max(0, self.frame - integer(bomb.get("frame"), self.frame))
            y = number(bomb.get("y"), -999.0) + number(bomb.get("v"), 1.1) * age
            x = number(bomb.get("x"), -999.0)
            if y < 118.0 or y > 211.0:
                continue
            lateral = abs(x - self.px)
            if (y >= 154.0 and lateral <= 18.0) or (y >= 128.0 and lateral <= 11.0):
                urgency = (y - 117.0) - lateral * 0.8
                candidates.append((urgency, x))
        if not candidates:
            return 0
        _, x = max(candidates)
        away = -1 if x >= self.px else 1
        if self.px <= self.left_limit + 2.0 and away < 0:
            away = 1
        if self.px >= self.right_limit - 2.0 and away > 0:
            away = -1
        return away

    def _sweep_direction(self):
        if self.px <= self.left_limit:
            self.direction = 1
        elif self.px >= self.right_limit:
            self.direction = -1
        if self.direction == 0:
            self.direction = 1
        return self.direction

    def _aim_allowed(self):
        if self.policy != "LATE_AIM":
            return False
        if self.je_mode == "SWEEP_FIRE" and self.frame < self.je_mode_until:
            return False
        late = self.phase() == "LATEWAVE"
        stagnant = self.frame - self.last_score_frame >= 192 and self.lives <= 2
        return late or stagnant

    def _relation(self, target):
        if target is None:
            return "NO_TARGET"
        distance = target["x"] - self.px
        if abs(distance) <= 7:
            return "ALIGNED"
        return "LEFT_NEAR" if distance < 0 and abs(distance) <= 30 else \
               "RIGHT_NEAR" if distance > 0 and abs(distance) <= 30 else \
               "LEFT_FAR" if distance < 0 else "RIGHT_FAR"

    def state_packet(self):
        target = self._target()
        hazard = self._hazard_direction()
        if hazard < 0:
            threat = "VISIBLE_NEAR_LEFT"
        elif hazard > 0:
            threat = "VISIBLE_NEAR_RIGHT"
        elif self.bombs:
            threat = "VISIBLE_NOT_IMMEDIATE"
        else:
            threat = "UNSEEN_NOT_SAFE"
        state = {
            "schema": "space_invaders_shot_window_v1",
            "episode": {
                "wave": self.wave,
                "life_stage": "THREE_LIVES" if self.lives >= 3 else "RECOVERY" if self.lives == 2 else "LAST_LIFE",
                "phase": self.phase(),
                "score_progress": "RECENT" if self.frame - self.last_score_frame < 128 else "QUIET",
            },
            "shot": {
                "state": "IN_FLIGHT" if self.shot_active else "READY",
                "window": "only_empty_laser_may_launch",
            },
            "formation": {
                "population": "MANY" if self.alive_count >= 27 else "SOME" if self.alive_count >= 15 else "FEW",
                "pressure": "HIGH" if self.bottom_y >= 145 else "RISING" if self.bottom_y >= 110 else "LOW",
                "target": "AVAILABLE" if target else "NONE",
                "target_height": target["height"] if target else "UNKNOWN",
                "target_relation": self._relation(target),
                "target_side": target["side"] if target else "UNKNOWN",
            },
            "movement": {
                "sweep": "LEFT" if self.direction < 0 else "RIGHT",
                "aiming": "YES" if self.aiming else "NO",
                "aim_age": "SHORT" if self.aim_ticks < 7 else "LONG",
            },
            "threat": threat,
            "policy": self.policy,
            "rules": [
                "Code computes target geometry and legal actions.",
                "AIM_TARGET means acquire the supplied lowest target only while shot.state is READY.",
                "SWEEP_FIRE means preserve fresh lateral movement with FIRE; it is the fallback.",
                "Do not invent a bomb from an empty bomb list or choose coordinates.",
            ],
        }
        options = {
            "AIM_TARGET": "Use one short, code-gated acquisition window for the supplied lowest living invader; never aim while a shot is IN_FLIGHT.",
            "SWEEP_FIRE": "Keep the proven full-rate lateral sweep and FIRE; do not pause to acquire a target.",
            "KEEP": "Leave the current shot-window mode unchanged because the supplied facts do not require a change.",
            "EMERGENCY_LEFT": "Only if the code-visible threat is immediately on the right: request a short left escape; code still verifies it.",
            "EMERGENCY_RIGHT": "Only if the code-visible threat is immediately on the left: request a short right escape; code still verifies it.",
            "UNKNOWN": "The supplied categorical facts are insufficient; use the deterministic sweep fallback.",
        }
        items = list(options.items())
        if (self.frame // 64) % 2:
            items.reverse()
        instruction = (
            "Choose a near-term control mode from the listed programs, not a raw joystick coordinate. "
            "Read the exact fields `shot.state`, `formation.target_relation`, `formation.phase`, and `threat`. "
            "A target acquisition is useful only when the laser is READY and the target is AVAILABLE; "
            "a shot already IN_FLIGHT must be allowed to travel while code sweeps. "
            "An empty bomb list means UNSEEN_NOT_SAFE, not no danger. Do not count invaders, calculate distances, "
            "invent projectiles, or override a visible collision."
        )
        questions = {
            "mode_a": Choice(instruction, dict(items)),
            "mode_b": Choice(instruction, dict(reversed(items))),
        }
        tag = {"frame": self.frame, "epoch": self.epoch, "lives": self.lives, "wave": self.wave}
        return state, questions, tag

    def should_query(self):
        if self.frame < 32 or self.frame - self.last_query_frame < 96:
            return False
        return self._target() is not None and (self.phase() != "OPENING" or self.frame - self.last_score_frame >= 128)

    def accept_jev(self, packet):
        if not isinstance(packet, dict) or packet.get("error"):
            return
        tag = mapping(packet.get("tag"))
        if (tag.get("epoch") != self.epoch or tag.get("lives") != self.lives
                or tag.get("wave") != self.wave):
            return
        age = self.frame - integer(tag.get("frame"), -100000)
        if age < 0 or age > 48:
            return
        answers = mapping(packet.get("result"))
        first = answers.get("mode_a")
        second = answers.get("mode_b")
        choice = answer_value(first)
        if choice not in CHOICES or answer_value(second) != choice:
            return
        for answer in (first, second):
            if answer_probability(answer, choice) < 0.64:
                return
            confidence = answer_confidence(answer)
            if confidence is not None and confidence < 0.25:
                return
        old = self.je_mode
        if choice in ("AIM_TARGET", "SWEEP_FIRE"):
            self.je_mode = choice
            self.je_mode_until = self.frame + 128
        elif choice == "KEEP":
            # Preserve an existing short lease; KEEP is not a reset command.
            pass
        elif choice == "EMERGENCY_LEFT" and self._hazard_direction() < 0:
            self.escape_direction = -1
            self.escape_until = self.frame + 32
        elif choice == "EMERGENCY_RIGHT" and self._hazard_direction() > 0:
            self.escape_direction = 1
            self.escape_until = self.frame + 32
        else:
            return
        self.stats["jev_accepted"] += 1
        if old != self.je_mode:
            self.stats["jev_changed_modes"] += 1
        self.log("jev_shot_mode", choice=choice, age=age, prior=old)

    def apply_astra(self, packet):
        if not isinstance(packet, dict) or packet.get("error"):
            return
        result = packet.get("result")
        text = getattr(result, "text", "")
        if not isinstance(text, str):
            return
        upper = text.upper()
        # Accept only the two explicit tokens, allowing a JSON wrapper but not
        # allowing free-form prose to silently change the safety policy.
        if "LATE_AIM" in upper:
            choice = "LATE_AIM"
        elif "SWEEP_ONLY" in upper:
            choice = "SWEEP_ONLY"
        else:
            return
        new_policy = "LATE_AIM" if choice == "LATE_AIM" else "SWEEP_ONLY"
        if new_policy != self.policy:
            old = self.policy
            self.policy = new_policy
            self.stats["astra_updates"] += 1
            self.log("astra_shot_policy", old=old, new=new_policy)

    def astra_prompt(self):
        target = self._target()
        evidence = {
            "frame": self.frame,
            "score": self.score,
            "lives": self.lives,
            "wave": self.wave,
            "phase": self.phase(),
            "alive_bucket": "MANY" if self.alive_count >= 27 else "SOME" if self.alive_count >= 15 else "FEW",
            "shot": "IN_FLIGHT" if self.shot_active else "READY",
            "target": "AVAILABLE" if target else "NONE",
            "visible_bombs": len(self.bombs),
        }
        return (
            "Choose the durable firing policy for canonical Atari 2600 Space Invaders. "
            "Reply with exactly one token: LATE_AIM or SWEEP_ONLY. The controller always posts "
            "fresh lateral actions with FIRE while a player laser is in flight. LATE_AIM enables "
            "a short code-owned alignment window only when the laser is READY and the formation is "
            "late or score has stalled; after launch, it returns to the broad moving sweep. "
            "SWEEP_ONLY preserves the baseline everywhere. Code handles coordinates, visible-bomb "
            "escapes, action legality, and one-laser timing; an empty bomb list is not safety. "
            "The experiment tests whether random launch positions leave the last low invaders alive, "
            "not whether a model can steer every tick. Current categorical evidence: "
            + json.dumps(evidence, separators=(",", ":"), allow_nan=False)
        )

    def action(self):
        hazard = self._hazard_direction()
        source = "sweep"
        if hazard:
            self.escape_direction = hazard
            self.escape_until = self.frame + 32
            self.stats["visible_escapes"] += 1
            self.log("shot_window_visible_escape", direction=hazard)
        if self.frame < self.escape_until and self.escape_direction:
            direction = self.escape_direction
            source = "visible_escape"
        else:
            direction = self._sweep_direction()
            target = self._target()
            if not self.shot_active and self._aim_allowed() and target is not None:
                relation = self._relation(target)
                if not self.aiming:
                    self.aiming = True
                    self.aim_ticks = 0
                    self.last_target_key = target["key"]
                    self.stats["aim_attempts"] += 1
                    self.log("shot_window_open", target=target["key"], side=target["side"])
                if relation == "ALIGNED":
                    self.aiming = False
                    self.aim_ticks = 0
                    self.stats["aim_fires"] += 1
                    self.stats["shot_windows"] += 1
                    self.log("shot_window_fire", target=target["key"])
                    self.stats["actions"] += 1
                    return "fire", "aim_fire"
                if self.aim_ticks < 14:
                    self.aim_ticks += 1
                    direction = -1 if target["x"] < self.px else 1
                    self.direction = direction
                    source = "aim_move"
                    self.stats["actions"] += 1
                    return ("left" if direction < 0 else "right"), source
                self.aiming = False
                self.aim_ticks = 0
                self.stats["aim_aborts"] += 1
                self.log("shot_window_abort", reason="alignment_timeout")
            self.stats["sweep_fires"] += 1
        base = "left" if direction < 0 else "right" if direction > 0 else "noop"
        action = base + "_fire" if base in ("left", "right") else "fire"
        self.stats["actions"] += 1
        return action, source


class Harness:
    def __init__(self, ctx):
        self.ctx = ctx

    @staticmethod
    def legal_actions(game):
        spec = getattr(game, "action_spec", None)
        try:
            legal = set(getattr(spec, "names", ACTIONS))
        except Exception:
            legal = set(ACTIONS)
        legal = legal or set(ACTIONS)
        noop = getattr(spec, "noop", "noop")
        if noop not in legal:
            noop = "noop" if "noop" in legal else next(iter(legal))
        return legal, noop

    def run_episode(self, game) -> None:
        ctx = self.ctx
        controller = ShotWindowController(ctx)
        jev = AsyncCall(ctx, "jev")
        astra = AsyncCall(ctx, "astra")
        legal, noop = self.legal_actions(game)
        last_frame = None
        astra_sent = False
        try:
            while not ctx.done and not getattr(game, "done", False):
                try:
                    raw = ctx.observe()
                except (BudgetExceeded, EpisodeOver):
                    break
                except Exception:
                    if ctx.done:
                        break
                    raw = {}
                obs = observation(raw)
                if truth(obs.get("game_over")):
                    break
                frame = integer(obs.get("frame"), (last_frame + 4) if last_frame is not None else 0)
                if last_frame is not None and frame == last_frame:
                    try:
                        ctx.wait_for_tick()
                    except Exception:
                        break
                    continue
                last_frame = frame
                controller.update(obs)
                controller.accept_jev(jev.poll())
                controller.apply_astra(astra.poll())
                action, source = controller.action()
                if action not in legal:
                    action = "fire" if "fire" in legal else noop
                try:
                    ctx.act(action)
                except (BudgetExceeded, EpisodeOver):
                    break
                except Exception:
                    break
                controller.last_action = action

                if jev.future is None and controller.should_query():
                    state, questions, tag = controller.state_packet()
                    try:
                        remaining = ctx.budget.remaining()
                    except Exception:
                        remaining = {}
                    if number(remaining.get("system1_calls"), 0.0) > 1 and number(remaining.get("wall_s"), 0.0) > 2:
                        if jev.launch(
                            lambda s=state, q=questions: ctx.decide(s, q, timeout_s=0.90),
                            tag,
                        ):
                            controller.last_query_frame = controller.frame

                if (not astra_sent and controller.frame >= 4 and not ctx.done):
                    try:
                        remaining = ctx.budget.remaining()
                    except Exception:
                        remaining = {}
                    if number(remaining.get("system2_calls"), 0.0) > 0 and number(remaining.get("wall_s"), 0.0) > 60:
                        prompt = controller.astra_prompt()
                        if astra.launch(lambda p=prompt: ctx.think(p, timeout_s=55.0)):
                            astra_sent = True
                            controller.log("astra_shot_policy_requested")

                if controller.stats["actions"] % 180 == 0:
                    controller.log("shot_window_summary", score=controller.score,
                                   lives=controller.lives, alive=controller.alive_count,
                                   policy=controller.policy, stats=dict(controller.stats))
                try:
                    ctx.wait_for_tick()
                except Exception:
                    break
        except (BudgetExceeded, EpisodeOver):
            pass
        finally:
            jev.close()
            astra.close()
            controller.log("shot_window_finished", score=controller.score,
                           lives=controller.lives, policy=controller.policy,
                           **controller.stats)


def build(ctx):
    return Harness(ctx)
