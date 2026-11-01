"""Candidate 0019: reversible evasion pulses over a productive firing sweep.

The controller treats a single shield chip as ambiguous.  It keeps the
high-throughput combined-FIRE sweep, but a corroborated lane cue (or a real
incoming trajectory) starts a short pulse away from that lane and then returns
to the old sweep phase.  This is deliberately different from turning a weak
cue into a permanent wall: movement coverage and firing cadence survive the
sensor's uncertainty.
"""
from __future__ import annotations

import json
import math
from collections import deque
from concurrent.futures import CancelledError

from metaharness.core.contract import BudgetExceeded, EpisodeOver
from metaharness.models.jev import Choice


ACTIONS = ("noop", "fire", "right", "left", "right_fire", "left_fire")


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


def mapping(value):
    return value if isinstance(value, dict) else {}


def records(value):
    return [item for item in value if isinstance(item, dict)] if isinstance(value, list) else []


def truth(value):
    if isinstance(value, str):
        return value.strip().lower() in ("true", "yes", "1")
    return bool(value)


def clamp(value, low, high):
    return max(low, min(high, value))


def answer_field(answer, name, default=None):
    if isinstance(answer, dict):
        return answer.get(name, default)
    return getattr(answer, name, default)


def answer_probability(answer, option):
    probs = answer_field(answer, "probabilities", {})
    return number(probs.get(option), 0.0) if isinstance(probs, dict) else 0.0


def agreed_choice(answers, first, second, options, minimum=0.60):
    a = answers.get(first)
    b = answers.get(second)
    value = answer_field(a, "value")
    if value not in options or answer_field(b, "value") != value:
        return None
    for answer in (a, b):
        if answer_probability(answer, value) < minimum:
            return None
        confidence = answer_field(answer, "confidence")
        if confidence is not None and number(confidence, -1.0) < 0.25:
            return None
    return value


def choice_questions(frame):
    """Two reordered views counter Jev's documented first-option bias."""
    choices = {
        "PULSE_LEFT": {
            "what": "Use one short leftward evasion pulse, only if the code-computed left path is safe and the cue is on or near the right side.",
            "not_for": "Do not choose for an isolated shield chip that is explained by the player's own shot.",
        },
        "PULSE_RIGHT": {
            "what": "Use one short rightward evasion pulse, only if the code-computed right path is safe and the cue is on or near the left side.",
            "not_for": "Do not choose for an isolated shield chip that is explained by the player's own shot.",
        },
        "KEEP_SWEEP": {
            "what": "Keep the current broad sweep because the supplied cue is weak, contradictory, or not actionable.",
            "not_for": "This does not mean that an explicit code-predicted collision may be ignored.",
        },
        "UNSURE": {
            "what": "The supplied evidence does not justify a reversible pulse choice.",
            "not_for": "Never invent a projectile or use an empty bomb list as proof of safety.",
        },
    }
    ordered = list(choices.items())
    if (integer(frame) // 24) % 2:
        ordered.reverse()
    instruction = (
        "Classify the near-term movement response from the supplied `cue` and `paths`. "
        "The code already predicts projectile contact, clamps movement, and always adds FIRE. "
        "A one-off shield occupancy loss is not a surviving bomb; repeated same-lane erosion "
        "is stronger but still only supports a short reversible pulse. Use only the named "
        "options, do not calculate coordinates, and do not select a target or a different action."
    )
    return {
        "judgment_a": Choice(instruction, dict(ordered)),
        "judgment_b": Choice(instruction, dict(reversed(ordered))),
    }


class Mailbox:
    """One bounded asynchronous Jev request; no worker-side polling loop."""

    def __init__(self, ctx):
        self.ctx = ctx
        self.future = None
        self.tag = None
        self.enabled = getattr(ctx, "system1", True) is not None

    def submit(self, state, questions, tag):
        if not self.enabled or self.future is not None or self.ctx.done:
            return False
        try:
            if self.ctx.budget.remaining().get("system1_calls", 0) <= 0:
                self.enabled = False
                return False
        except Exception:
            pass

        def work():
            try:
                return {"answers": self.ctx.decide(state, questions, timeout_s=0.85)}
            except Exception as exc:
                # A failed/garbage Jev response only removes optional advice.
                return {"error": type(exc).__name__}

        try:
            self.future = self.ctx.spawn(work)
            self.tag = dict(tag)
            return True
        except Exception:
            self.future = None
            self.tag = None
            return False

    def poll(self):
        if self.future is None or not self.future.done():
            return None
        future, tag = self.future, self.tag or {}
        self.future = None
        self.tag = None
        try:
            result = future.result()
        except (CancelledError, TimeoutError, RuntimeError, TypeError, ValueError):
            return {"error": "failed", "tag": tag}
        except Exception:
            return {"error": "failed", "tag": tag}
        if not isinstance(result, dict):
            return {"error": "malformed", "tag": tag}
        result["tag"] = tag
        return result

    def close(self):
        future = self.future
        self.future = None
        self.tag = None
        if future is not None and not future.done():
            try:
                future.cancel()
            except Exception:
                pass


class Controller:
    """Fast code policy plus a narrow evidence interface for Jev."""

    def __init__(self, ctx):
        self.ctx = ctx
        self.frame = 0
        self.score = 0.0
        self.lives = 3
        self.wave = 1
        self.player_x = 90.0
        self.player_y = 190.0
        self.speed = 0.55
        self.direction = 1
        self.last_action = "noop"
        self.previous = None
        self.previous_mask = None
        self.previous_shot_active = False
        self.launches = deque(maxlen=24)
        self.tracks = {}
        self.cues = deque(maxlen=32)
        self.erosion_history = deque(maxlen=32)
        self.serial = 0
        self.epoch = 0
        self.alive_count = 36
        self.lowest_invader_y = 0.0
        self.life_event = False
        self.wave_event = False
        self.reset_event = False
        self.mask_event = None
        self.last_bomb_frame = -100000
        self.recovery_until = -1
        self.pulse_until = -1
        self.pulse_direction = 0
        self.resume_direction = 1
        self.pulse_frames = 36
        self.advice = None
        self.last_query_frame = -100000
        self.actions = 0
        self.counters = {
            "life_losses": 0,
            "shield_chips": 0,
            "repeated_lanes": 0,
            "explicit_or_tracked_threats": 0,
            "pulses": 0,
            "model_pulses": 0,
            "model_changes": 0,
            "tracks": 0,
        }

    def log(self, event, **fields):
        try:
            self.ctx.log(event, frame=self.frame, **fields)
        except Exception:
            pass

    def new_id(self, prefix):
        self.serial += 1
        return prefix + str(self.serial)

    def masks(self, obs):
        cells = {}
        for shield_index, shield in enumerate(records(obs.get("shields"))[:3]):
            raw_masks = shield.get("health_mask")
            if not isinstance(raw_masks, list):
                continue
            x0 = number(shield.get("x"), 0.0) - 4.0
            for row, raw in enumerate(raw_masks[:9]):
                bits = integer(raw, 0) & 255
                for col in range(8):
                    if bits & (1 << (7 - col)):
                        cells[(shield_index, row, col)] = x0 + col
        return cells

    def own_shot_near(self, x):
        return any(abs(x - launch_x) <= 5.0 and 0 <= self.frame - launch_frame <= 52
                   for launch_x, launch_frame in self.launches)

    def add_cue(self, kind, x, y=None, velocity=0.0, strength=1, track_id=None):
        x = number(x, self.player_x)
        # Coalesce the same physical cue instead of allowing a stale lane to
        # trigger a pulse on every observation.
        for cue in reversed(self.cues):
            if (cue.get("track_id") == track_id and track_id is not None) or (
                    cue.get("kind") == kind and abs(number(cue.get("x")) - x) <= 5
                    and self.frame - integer(cue.get("frame"), self.frame) <= 12):
                cue["x"] = (number(cue.get("x")) + x) / 2.0
                cue["frame"] = self.frame
                cue["strength"] = max(integer(cue.get("strength"), 1), strength)
                if y is not None:
                    cue["y"] = number(y)
                if velocity:
                    cue["velocity"] = velocity
                return cue
        cue = {
            "id": self.new_id("Q"),
            "kind": kind,
            "x": x,
            "y": None if y is None else number(y),
            "velocity": number(velocity),
            "strength": integer(strength, 1),
            "frame": self.frame,
            "used": False,
            "reviewed": False,
            "track_id": track_id,
        }
        self.cues.append(cue)
        return cue

    def record_shield_change(self, obs):
        current = self.masks(obs)
        old = self.previous_mask
        self.previous_mask = current
        self.mask_event = None
        if self.reset_event or old is None or len(records(obs.get("shields"))) != 3:
            return
        removed = sorted(old.keys() - current.keys())
        if not removed:
            return
        if len(removed) > 24 or (len(old) > 12 and not current):
            self.log("shield_reset_or_bulk_clear", removed=len(removed))
            return

        unexplained = []
        for shield_index, row, col in removed:
            top_before = min(
                (old_row for old_shield, old_row, old_col in old
                 if old_shield == shield_index and old_col == col),
                default=9,
            )
            x = number(old[(shield_index, row, col)])
            # A bottom-originating beam and a recently launched own shot are
            # not treated as enemy evidence.
            if row > top_before + 2 or self.own_shot_near(x):
                continue
            unexplained.append(x)
        if not unexplained:
            return

        x = sum(unexplained) / len(unexplained)
        self.counters["shield_chips"] += 1
        self.erosion_history.append((x, self.frame))
        self.erosion_history = deque(
            (item for item in self.erosion_history if self.frame - item[1] <= 80),
            maxlen=32,
        )
        repeat = any(abs(old_x - x) <= 8 and 0 < self.frame - old_frame <= 80
                     for old_x, old_frame in self.erosion_history
                     if old_frame != self.frame)
        if repeat:
            self.counters["repeated_lanes"] += 1
        self.mask_event = "repeated_lane" if repeat else "single_chip"
        cue = self.add_cue("shield_repeat" if repeat else "shield_chip", x, strength=2 if repeat else 1)
        self.log("shield_lane_observed", x=round(x, 1), kind=cue["kind"],
                 lost_pixels=len(unexplained), corroborated=repeat)

    def track_projectiles(self, obs):
        points = []
        for bomb in records(obs.get("bombs"))[:12]:
            if bomb.get("x") is not None and bomb.get("y") is not None:
                points.append((number(bomb.get("x")), number(bomb.get("y")), "bomb"))
        shot = mapping(obs.get("player_shot"))
        shot_active = truth(shot.get("active"))
        if shot_active and shot.get("x") is not None and shot.get("y") is not None:
            points.append((number(shot.get("x")), number(shot.get("y")), "shot"))
            if (not self.previous_shot_active or number(shot.get("y"), 999) >= self.player_y - 45):
                self.launches.append((number(shot.get("x"), self.player_x), self.frame))
        used = set()
        for x, y, source in points:
            candidates = []
            for key, track in self.tracks.items():
                age = self.frame - integer(track.get("frame"), self.frame)
                if key in used or not 0 < age <= 12 or abs(x - number(track.get("x"))) > 4.5:
                    continue
                predicted = number(track.get("y")) + number(track.get("velocity")) * age
                if abs(y - predicted) <= max(9.0, 3.5 * age):
                    candidates.append((abs(y - predicted) + abs(x - number(track.get("x"))), key))
            if candidates:
                _, key = min(candidates)
                track = self.tracks[key]
                age = max(1, self.frame - integer(track.get("frame"), self.frame))
                dy = y - number(track.get("y"))
                velocity = clamp(dy / age, -3.5, 3.5)
                if velocity > 0.20:
                    track["down"] = integer(track.get("down")) + 1
                    track["up"] = 0
                elif velocity < -0.20:
                    track["up"] = integer(track.get("up")) + 1
                    track["down"] = 0
                track["velocity"] = velocity
                track["x"], track["y"], track["frame"] = x, y, self.frame
                if source == "bomb":
                    track["explicit"] = True
            else:
                key = self.new_id("T")
                own = source == "shot" and y >= self.player_y - 35 and abs(x - self.player_x) <= 9
                track = {
                    "id": key, "x": x, "y": y, "frame": self.frame,
                    "velocity": 0.0, "down": 0, "up": 0,
                    "own": own, "explicit": source == "bomb",
                }
                self.tracks[key] = track
                self.counters["tracks"] += 1
            used.add(key)
            if track.get("explicit") and y >= 110:
                cue = self.add_cue("explicit_bomb", x, y=y,
                                   velocity=max(0.6, number(track.get("velocity"), 1.2)),
                                   strength=3, track_id=key)
                if cue.get("frame") == self.frame:
                    self.counters["explicit_or_tracked_threats"] += 1
                    self.log("explicit_bomb_track", x=round(x, 1), y=round(y, 1))
            elif (integer(track.get("down")) >= 2 and not track.get("own")
                  and not integer(track.get("up")) and y >= 110):
                cue = self.add_cue("descending_track", x, y=y,
                                   velocity=max(0.6, number(track.get("velocity"), 1.2)),
                                   strength=2, track_id=key)
                if cue.get("frame") == self.frame:
                    self.counters["explicit_or_tracked_threats"] += 1
                    self.log("descending_track", x=round(x, 1), y=round(y, 1))
        self.tracks = {
            key: track for key, track in self.tracks.items()
            if self.frame - integer(track.get("frame"), self.frame) <= 24
        }
        self.cues = deque(
            (cue for cue in self.cues if self.frame - integer(cue.get("frame"), self.frame)
             <= (30 if cue.get("kind") in ("explicit_bomb", "descending_track") else 96)),
            maxlen=32,
        )
        self.previous_shot_active = shot_active

    def update(self, raw):
        obs = mapping(raw)
        old = self.previous
        old_frame = self.frame
        frame = integer(obs.get("frame"), old_frame + 4)
        df = max(1, frame - old_frame)
        p = mapping(obs.get("player"))
        px = clamp(number(p.get("x"), self.player_x), 0.0, 159.0)
        py = clamp(number(p.get("y"), self.player_y), 150.0, 209.0)
        lives = integer(obs.get("lives"), self.lives)
        wave = integer(obs.get("wave"), self.wave)
        old_px = self.player_x
        self.life_event = old is not None and lives < self.lives
        self.wave_event = old is not None and wave != self.wave
        jump = old is not None and abs(px - old_px) > max(12.0, 1.5 * df)
        self.reset_event = self.life_event or self.wave_event or jump
        self.frame = frame
        if self.life_event:
            self.counters["life_losses"] += max(1, self.lives - lives)
            self.epoch += 1
            self.tracks.clear()
            self.cues.clear()
            self.launches.clear()
            self.recovery_until = frame + 96
            self.pulse_until = -1
            self.pulse_direction = 0
            # Reverse the phase, but do not remember the respawn x as a death lane.
            self.direction = -self.direction if self.direction else 1
            self.log("life_recovery", previous_x=round(old_px, 1), lives=lives,
                     note="phase reversed; no death-lane memory")
        elif self.wave_event:
            self.epoch += 1
            self.tracks.clear()
            self.cues.clear()
            self.launches.clear()
            self.recovery_until = frame + 80
            self.pulse_until = -1
            self.direction = 1 if px < 78 else -1
            self.log("wave_reset", wave=wave)
        elif old is not None and abs(px - old_px) > 0.4:
            action_direction = -1 if "left" in self.last_action else 1 if "right" in self.last_action else 0
            measured = abs(px - old_px) / df
            if action_direction * (px - old_px) > 0:
                self.speed = clamp(0.85 * self.speed + 0.15 * measured, 0.25, 1.15)

        self.player_x, self.player_y = px, py
        self.lives, self.wave = lives, wave
        self.record_shield_change(obs)
        self.track_projectiles(obs)
        invaders = [item for item in records(obs.get("invaders")) if truth(item.get("alive"))]
        self.alive_count = len(invaders)
        self.lowest_invader_y = max((number(item.get("y"), 0.0) for item in invaders), default=0.0)
        score = number(obs.get("score"), self.score)
        self.score = score
        self.previous = obs
        return obs

    def bounds(self):
        # A short central recovery corridor keeps an immediately respawned
        # cannon from repeating a blind edge-to-edge path.  Late waves get a
        # modest margin without abandoning the outer columns altogether.
        if self.frame < self.recovery_until:
            return 48.0, 108.0
        if self.lives <= 1 or self.alive_count <= 12:
            return 42.0, 114.0
        return 36.0, 118.0

    def base_direction(self):
        left, right = self.bounds()
        if self.player_x <= left:
            self.direction = 1
        elif self.player_x >= right:
            self.direction = -1
        return self.direction or 1

    def projected_x(self, direction, delta):
        left, right = self.bounds()
        prior = -1 if "left" in self.last_action else 1 if "right" in self.last_action else 0
        # Include one possible sticky tick; the result is a safety category,
        # not a numerical command for Jev.
        x = self.player_x + self.speed * (prior * min(4, max(0, delta))
                                          + direction * max(0, delta - 4))
        return clamp(x, left, right)

    def cue_risk(self, cue, direction):
        age = max(0, self.frame - integer(cue.get("frame"), self.frame))
        kind = cue.get("kind")
        x = number(cue.get("x"), self.player_x)
        if kind in ("explicit_bomb", "descending_track"):
            y = number(cue.get("y"), 120.0)
            velocity = max(0.6, number(cue.get("velocity"), 1.2))
            for delta in range(0, 81, 4):
                yy = y + velocity * (age + delta)
                if self.player_y - 15 <= yy <= self.player_y + 8:
                    if abs(self.projected_x(direction, delta) - x) <= 11:
                        return 100.0 + integer(cue.get("strength"), 1) * 4
            return 0.0
        # An erosion cue has no reliable y coordinate.  It is only actionable
        # while fresh and only if the committed path crosses its lane soon.
        if age > 64:
            return 0.0
        for delta in range(0, 57, 4):
            if abs(self.projected_x(direction, delta) - x) <= 12:
                return 30.0 + integer(cue.get("strength"), 1) * 8
        return 0.0

    def live_cue(self):
        valid = []
        for cue in self.cues:
            age = self.frame - integer(cue.get("frame"), self.frame)
            if age >= 0 and not cue.get("used") and age <= (26 if cue.get("kind") in ("explicit_bomb", "descending_track") else 72):
                valid.append(cue)
        if not valid:
            return None
        return max(valid, key=lambda cue: (integer(cue.get("strength"), 1), -self.frame + integer(cue.get("frame"), 0)))

    def escape_direction(self, cue, risks):
        x = number(cue.get("x"), self.player_x)
        if x > self.player_x + 3:
            preferred = -1
        elif x < self.player_x - 3:
            preferred = 1
        else:
            preferred = -1 if self.player_x > 78 else 1
        left, right = self.bounds()
        candidates = [preferred, -preferred]
        candidates = [d for d in candidates if not (d < 0 and self.player_x <= left + 1)
                      and not (d > 0 and self.player_x >= right - 1)]
        if not candidates:
            return 0
        # A high direct risk wins; for a weak erosion cue use the away side
        # unless it is itself a code-computed collision path.
        safe = [d for d in candidates if risks.get(d, 0.0) < 100.0]
        return (safe or candidates)[0]

    def derived(self):
        base = self.base_direction()
        cue = self.live_cue()
        risks = {-1: 0.0, 0: 0.0, 1: 0.0}
        if cue is not None:
            for direction in risks:
                risks[direction] = self.cue_risk(cue, direction)
        phase = "RECOVERY" if self.frame < self.recovery_until else "SWEEP"
        if self.frame < self.pulse_until:
            phase = "PULSE"
        safe_left = self.player_x > self.bounds()[0] + 1 and risks[-1] < 100.0
        safe_right = self.player_x < self.bounds()[1] - 1 and risks[1] < 100.0
        side = "NONE"
        if cue is not None:
            x = number(cue.get("x"), self.player_x)
            side = "LEFT" if x < self.player_x - 5 else "RIGHT" if x > self.player_x + 5 else "UNDER"
        state = {
            "schema": "space_invaders_reversible_pulse_v1",
            "frame": self.frame,
            "lives": self.lives,
            "wave": self.wave,
            "phase": phase,
            "player_lane": "LEFT" if self.player_x < 65 else "RIGHT" if self.player_x > 110 else "CENTER",
            "formation": "LATE" if self.alive_count <= 12 else "MID_OR_OPENING",
            "cue": {
                "kind": cue.get("kind", "none") if cue else "none",
                "side": side,
                "age": "fresh" if cue and self.frame - integer(cue.get("frame"), self.frame) <= 20 else "aging" if cue else "none",
                "strength": "direct" if cue and integer(cue.get("strength")) >= 3 else "repeated" if cue and integer(cue.get("strength")) == 2 else "single_or_none",
                "own_shot_explanation": bool(cue and cue.get("kind") == "shield_chip"),
            },
            "paths": {
                "sweep": "RIGHT" if base > 0 else "LEFT",
                "safe_left": safe_left,
                "safe_right": safe_right,
                "left_risk": "HIGH" if risks[-1] >= 100 else "LOW_OR_UNOBSERVED",
                "right_risk": "HIGH" if risks[1] >= 100 else "LOW_OR_UNOBSERVED",
                "code_priority": "direct_collision_first",
            },
            "rules": "One laser is in flight at a time; every posted action includes FIRE. A pulse is temporary and returns to the prior sweep. Empty bombs do not certify safety; Jev cannot invent a projectile.",
        }
        tag = {
            "frame": self.frame,
            "epoch": self.epoch,
            "cue_id": cue.get("id") if cue else None,
            "cue_frame": cue.get("frame") if cue else None,
        }
        return {"state": state, "tag": tag, "cue": cue, "risks": risks, "base": base}

    def accept_advice(self, packet):
        if packet is None:
            return
        result = mapping(packet)
        tag = mapping(result.get("tag"))
        if result.get("error") or tag.get("epoch") != self.epoch:
            return
        if self.frame - integer(tag.get("frame"), -100000) < 0 or self.frame - integer(tag.get("frame"), -100000) > 32:
            return
        cue = next((item for item in self.cues if item.get("id") == tag.get("cue_id")), None)
        if cue is None or cue.get("used") or cue.get("frame") != tag.get("cue_frame"):
            return
        answers = mapping(result.get("answers"))
        choice = agreed_choice(answers, "judgment_a", "judgment_b",
                               ("PULSE_LEFT", "PULSE_RIGHT", "KEEP_SWEEP", "UNSURE"))
        if choice is None:
            self.log("jev_advice_rejected", reason="disagreement_or_low_confidence")
            return
        self.advice = {"cue_id": cue.get("id"), "choice": choice, "frame": self.frame}
        self.log("jev_advice_accepted", cue=cue.get("id"), choice=choice)

    def review_packet(self, derived):
        cue = derived.get("cue")
        if cue is None or cue.get("reviewed"):
            return None
        if self.frame - self.last_query_frame < 24:
            return None
        age = self.frame - integer(cue.get("frame"), self.frame)
        if age > 32:
            return None
        cue["reviewed"] = True
        self.last_query_frame = self.frame
        tag = dict(derived["tag"])
        return derived["state"], choice_questions(self.frame), tag

    def start_pulse(self, direction, source, cue):
        if direction == 0:
            return False
        left, right = self.bounds()
        if (direction < 0 and self.player_x <= left + 1) or (direction > 0 and self.player_x >= right - 1):
            return False
        self.resume_direction = self.base_direction()
        self.pulse_direction = direction
        self.pulse_until = self.frame + self.pulse_frames
        self.direction = direction
        if cue is not None:
            cue["used"] = True
        self.counters["pulses"] += 1
        if source == "jev":
            self.counters["model_pulses"] += 1
        self.log("evasion_pulse", direction="LEFT" if direction < 0 else "RIGHT",
                 source=source, cue=cue.get("id") if cue else None,
                 duration=self.pulse_frames)
        return True

    def choose_action(self, derived):
        cue = derived.get("cue")
        risks = derived.get("risks", {-1: 0.0, 0: 0.0, 1: 0.0})
        base = derived.get("base", self.direction)
        selected = base
        source = "sweep"

        # Strong code-owned evidence never waits for Jev.
        if cue is not None and cue.get("kind") in ("explicit_bomb", "descending_track"):
            candidates = [d for d in (-1, 1) if risks.get(d, 0.0) < 100.0]
            if candidates:
                best = min(candidates, key=lambda d: risks.get(d, 0.0))
                if risks.get(base, 0.0) >= 100.0 and best != base:
                    if self.start_pulse(best, "code_direct", cue):
                        selected = best
                        source = "code_direct"

        # A repeated shield lane is enough for one deterministic short pulse,
        # unlike 0015's permanent direction reversal for every chip.
        if source == "sweep" and cue is not None and cue.get("kind") == "shield_repeat":
            away = self.escape_direction(cue, risks)
            if away and risks.get(away, 0.0) < 100.0 and away != base:
                if self.start_pulse(away, "repeated_shield_lane", cue):
                    selected = away
                    source = "repeated_shield_lane"

        # Jev can promote one weak cue to the same bounded pulse, but cannot
        # override the live safety check or choose a target.
        advice = self.advice
        if source == "sweep" and advice is not None:
            age = self.frame - integer(advice.get("frame"), -100000)
            if 0 <= age <= 20 and cue is not None and cue.get("id") == advice.get("cue_id"):
                proposed = -1 if advice.get("choice") == "PULSE_LEFT" else 1 if advice.get("choice") == "PULSE_RIGHT" else 0
                if proposed and risks.get(proposed, 0.0) < 100.0:
                    expected = self.escape_direction(cue, risks)
                    if proposed == expected and self.start_pulse(proposed, "jev", cue):
                        selected = proposed
                        source = "jev_pulse"
                        self.counters["model_changes"] += 1
        self.advice = None

        if self.frame < self.pulse_until and self.pulse_direction:
            selected = self.pulse_direction
            source = source if source != "sweep" else "pulse_commit"
        elif self.frame >= self.pulse_until and self.pulse_direction:
            self.pulse_direction = 0
            self.direction = self.resume_direction or base
            selected = self.base_direction()
            source = "sweep_resumed"
        if selected == 0:
            selected = base
        self.direction = selected
        self.actions += 1
        self.last_action = "left_fire" if selected < 0 else "right_fire"
        return self.last_action, source


class Harness:
    def __init__(self, ctx):
        self.ctx = ctx

    def legal_actions(self, game):
        spec = getattr(game, "action_spec", None)
        try:
            legal = set(getattr(spec, "names", ACTIONS))
        except Exception:
            legal = set(ACTIONS)
        noop = getattr(spec, "noop", "noop")
        if noop not in legal:
            noop = "noop" if "noop" in legal else ("fire" if "fire" in legal else next(iter(legal), "noop"))
        return legal, noop

    def start_astra(self, controller):
        """Late escalation only: short episodes already showed early S2 was inert."""
        ctx = self.ctx
        try:
            remaining = ctx.budget.remaining()
            if remaining.get("system2_calls", 0) <= 0 or remaining.get("wall_s", 0) <= 120:
                return None
        except Exception:
            return None
        evidence = {
            "frame": controller.frame,
            "score": controller.score,
            "lives": controller.lives,
            "alive_invaders": controller.alive_count,
            "life_losses": controller.counters["life_losses"],
            "repeated_lanes": controller.counters["repeated_lanes"],
            "pulses": controller.counters["pulses"],
        }
        prompt = (
            "You are revising a late-wave real-time Atari 2600 Space Invaders "
            "controller. Code already keeps combined FIRE, predicts explicit "
            "incoming trajectories, and uses short reversible pulses for repeated "
            "shield-lane evidence. Choose only the pulse duration policy. Return "
            "exactly one token: SHORT, MEDIUM, or LONG. SHORT=24 emulator frames "
            "when evidence is noisy or movement coverage matters; MEDIUM=36 for "
            "ordinary corroborated cues; LONG=56 only after repeated blind losses "
            "or a dangerous late wave. Do not redesign geometry, targets, or the "
            "action mapping. Evidence: "
            + json.dumps(evidence, separators=(",", ":"), allow_nan=False)
        )

        def work():
            try:
                return ctx.think(prompt, timeout_s=55.0)
            except Exception as exc:
                return {"error": type(exc).__name__}

        try:
            return ctx.spawn(work)
        except Exception:
            return None

    def poll_astra(self, future, controller):
        if future is None or not future.done():
            return future
        try:
            result = future.result()
            text = getattr(result, "text", "")
            if isinstance(text, str):
                token = text.strip().strip("`'\".,:;()[]{}\n ").upper().split()
                token = token[0] if token else ""
                duration = {"SHORT": 24, "MEDIUM": 36, "LONG": 56}.get(token)
                if duration is not None:
                    old = controller.pulse_frames
                    controller.pulse_frames = duration
                    controller.log("astra_duration_update", old=old, new=duration)
        except (CancelledError, TimeoutError, BudgetExceeded, EpisodeOver, RuntimeError, TypeError, ValueError):
            controller.log("astra_unavailable")
        except Exception:
            controller.log("astra_unavailable")
        return None

    def run_episode(self, game) -> None:
        ctx = self.ctx
        controller = Controller(ctx)
        mailbox = Mailbox(ctx)
        legal, noop = self.legal_actions(game)
        last_frame = None
        astra_future = None
        astra_started = False
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
                obs = mapping(raw)
                if truth(obs.get("game_over")):
                    break
                frame = integer(obs.get("frame"), controller.frame + 4)
                if last_frame is not None and frame <= last_frame:
                    try:
                        ctx.wait_for_tick()
                    except (BudgetExceeded, EpisodeOver):
                        break
                    continue
                last_frame = frame
                controller.update(obs)
                controller.accept_advice(mailbox.poll())
                astra_future = self.poll_astra(astra_future, controller)
                derived = controller.derived()
                action, source = controller.choose_action(derived)
                if action not in legal:
                    combined = action
                    if combined not in legal:
                        action = "fire" if "fire" in legal else noop
                try:
                    ctx.act(action)
                except (BudgetExceeded, EpisodeOver):
                    break

                packet = controller.review_packet(derived)
                if packet is not None and mailbox.future is None:
                    state, questions, tag = packet
                    mailbox.submit(state, questions, tag)

                # A late Astra revision is useful only after the sweep has
                # survived long enough to expose late-wave evidence.  This
                # avoids spending an early 30-50s call that cannot reach the
                # actuator before the usual game-over seen in the archive.
                if (not astra_started and astra_future is None and controller.actions >= 720
                        and (controller.alive_count <= 18 or controller.lives <= 1)):
                    astra_future = self.start_astra(controller)
                    astra_started = astra_future is not None
                    if astra_started:
                        controller.log("astra_requested", reason="late_wave_revision")

                if controller.actions % 160 == 0:
                    controller.log("pulse_controller_summary", score=controller.score,
                                   lives=controller.lives, alive=controller.alive_count,
                                   phase=derived.get("state", {}).get("phase"),
                                   source=source, cues=len(controller.cues),
                                   counters=dict(controller.counters))
                controller.last_action = action
                try:
                    ctx.wait_for_tick()
                except (BudgetExceeded, EpisodeOver):
                    break
        finally:
            mailbox.close()
            if astra_future is not None and not astra_future.done():
                try:
                    astra_future.cancel()
                except Exception:
                    pass
            controller.log("pulse_controller_finished", score=controller.score,
                           lives=controller.lives, actions=controller.actions,
                           astra_started=astra_started, pulse_frames=controller.pulse_frames,
                           counters=dict(controller.counters))


def build(ctx):
    return Harness(ctx)
