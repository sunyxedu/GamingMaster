"""Candidate 0018: life-adaptive, aperiodic firing patrols.

The proven actuator is retained: move on every tick and hold FIRE.  Unlike the
fixed triangular sweeps in the parent, the controller commits to long
cross-field legs with changing endpoints, and changes its patrol phase after a
life loss.  Jev chooses a *future patrol posture* from code-computed choices;
it never chooses a joystick action.  Astra chooses the durable life-level
posture off the actuator path.
"""
from __future__ import annotations

import json
import math
from collections import deque
from concurrent.futures import CancelledError

from metaharness.core.contract import BudgetExceeded, EpisodeOver
from metaharness.models.jev import Choice


ACTIONS = ("noop", "fire", "right", "left", "right_fire", "left_fire")
POSTURES = (
    "WIDE_DITHER",
    "COVER_LEFT",
    "COVER_CENTER",
    "COVER_RIGHT",
    "REVERSE",
    "KEEP",
    "UNKNOWN",
)


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


def observation(value):
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, dict) else {}
        except (TypeError, ValueError, json.JSONDecodeError):
            return {}
    return {}


def records(value):
    return [item for item in value if isinstance(item, dict)] if isinstance(value, list) else []


def truth(value):
    if isinstance(value, str):
        return value.strip().lower() in ("true", "yes", "1")
    return bool(value)


def answer_value(answer):
    if isinstance(answer, dict):
        return answer.get("value", answer.get("choice"))
    return getattr(answer, "value", None)


def answer_prob(answer, option):
    probabilities = (answer.get("probabilities", {}) if isinstance(answer, dict)
                     else getattr(answer, "probabilities", {}))
    if not isinstance(probabilities, dict):
        return 0.0
    direct = probabilities.get(option)
    if direct is not None:
        return number(direct, 0.0)
    for key, value in probabilities.items():
        if str(key).upper() == str(option).upper():
            return number(value, 0.0)
    return 0.0


def answer_confidence(answer):
    value = (answer.get("confidence") if isinstance(answer, dict)
             else getattr(answer, "confidence", None))
    if value is None:
        return None
    result = number(value, -1.0)
    return result if result >= 0.0 else None


def signed(value):
    return 1 if value > 0 else -1 if value < 0 else 0


class AdviceMailbox:
    """One latest-only asynchronous typed request, with no polling worker."""

    def __init__(self, ctx):
        self.ctx = ctx
        self.future = None
        self.tag = None
        self.enabled = getattr(ctx, "system1", True) is not None

    def launch(self, state, tag):
        if not self.enabled or self.future is not None or self.ctx.done:
            return False
        try:
            if self.ctx.budget.remaining().get("system1_calls", 0) <= 0:
                self.enabled = False
                return False
        except Exception:
            pass

        questions = posture_questions(tag.get("frame", 0))

        def request():
            try:
                return {"answers": self.ctx.decide(state, questions, timeout_s=0.90),
                        "tag": tag}
            except Exception as exc:
                # A typed answer is optional.  The route controller is the
                # executable fallback for timeout, garbage, and bad stand-ins.
                return {"error": type(exc).__name__, "tag": tag}

        try:
            self.future = self.ctx.spawn(request)
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
        return result if isinstance(result, dict) else {"error": "malformed", "tag": tag}

    def close(self):
        future = self.future
        self.future = None
        self.tag = None
        if future is not None and not future.done():
            try:
                future.cancel()
            except Exception:
                pass


def posture_questions(frame):
    options = {
        "WIDE_DITHER": {
            "what": "At the next route boundary, make a long cross-field traverse with a changed endpoint; keep firing.",
            "not_for": "Do not use this to ignore a currently predicted visible collision.",
        },
        "COVER_LEFT": {
            "what": "At the next route boundary, patrol the code-selected left shield neighborhood while still traversing and firing.",
            "not_for": "Not a command to park the cannon or to assume a shield is permanent.",
        },
        "COVER_CENTER": {
            "what": "At the next route boundary, patrol the code-selected center shield neighborhood while still traversing and firing.",
            "not_for": "Not a command to stop moving or to infer safety from an empty bomb list.",
        },
        "COVER_RIGHT": {
            "what": "At the next route boundary, patrol the code-selected right shield neighborhood while still traversing and firing.",
            "not_for": "Not a command to stop moving or to infer safety from an empty bomb list.",
        },
        "REVERSE": {
            "what": "Use the opposite long leg at the next boundary because the supplied risk or motion facts support changing phase.",
            "not_for": "Do not reverse immediately for an unobserved or merely possible threat.",
        },
        "KEEP": {
            "what": "Keep the deterministic route posture already committed; code continues to hold FIRE and move.",
            "not_for": "This is not permission to remain at a movement limit.",
        },
        "UNKNOWN": {
            "what": "The supplied categorical facts do not justify changing the code fallback.",
            "not_for": "Never invent a bomb, target, or coordinate.",
        },
    }
    ordered = list(options.items())
    if (integer(frame) // 64) % 2:
        ordered.reverse()
    first = dict(ordered)
    second = dict(reversed(ordered))
    instruction = (
        "Choose the patrol posture for the NEXT route boundary, not an immediate action. "
        "Use only the supplied categorical facts and the listed program preconditions. "
        "The controller computes coordinates, visible-bomb collision paths, legal actions, "
        "formation geometry, and firing. A long traverse is preferred when there is no "
        "explicit collision. Empty bombs are UNSEEN, never SAFE. Do not count, calculate, "
        "select a column, or invent an object."
    )
    return {
        "posture_a": Choice(instruction, first),
        "posture_b": Choice(instruction, second),
    }


class PatrolController:
    """Code-owned geometry and actuator with a slow-changing posture interface."""

    def __init__(self, ctx):
        self.ctx = ctx
        self.frame = 0
        self.score = 0.0
        self.lives = 3
        self.wave = 1
        self.px = 90.0
        self.py = 190.0
        self.previous = None
        self.previous_alive = set()
        self.previous_masks = None
        self.previous_shot_active = False
        self.last_action = "noop"

        self.direction = 1
        self.speed = 0.50
        self.last_motion_frame = 0
        self.stall_frames = 0
        self.route = None
        self.route_index = 0
        self.epoch = 0
        self.rng = 0x51F15EED
        self.astra_policy = "DITHER"
        self.advice = None
        self.last_query_frame = -100000
        self.last_life_event = False
        self.last_wave_event = False
        self.last_progress_frame = 0

        self.bomb_tracks = []
        self.last_bomb_frame = -100000
        self.shot_launches = deque(maxlen=24)
        self.lane_heat = {}
        self.damage_events = 0
        self.cover = []
        self.alive = []
        self.alive_count = 36
        self.form_center = 66.0
        self.bottom_y = 125.0
        self.escape_until = -1
        self.escape_direction = 0

        self.stats = {
            "actions": 0,
            "routes": 0,
            "route_reversals": 0,
            "visible_dodges": 0,
            "life_losses": 0,
            "shield_damage_events": 0,
            "jev_accepted": 0,
            "jev_applied": 0,
            "jev_changed_routes": 0,
            "astra_applied": 0,
        }

    def log(self, event, **fields):
        try:
            self.ctx.log(event, frame=self.frame, **fields)
        except Exception:
            pass

    def _mask_cells(self, obs):
        cells = {}
        for shield_index, shield in enumerate(records(obs.get("shields"))[:3]):
            raw = shield.get("health_mask")
            if not isinstance(raw, list):
                continue
            x0 = number(shield.get("x"), 0.0) - 4.0
            for row, bits in enumerate(raw[:9]):
                value = integer(bits, 0) & 255
                for column in range(8):
                    if value & (1 << (7 - column)):
                        cells[(shield_index, row, column)] = x0 + column
        return cells

    def _cover_facts(self, obs=None):
        if obs is not None:
            shields = records(obs.get("shields"))
        else:
            shields = []
        result = []
        for index in range(3):
            cells = [x for (shield, _row, _column), x in (self.previous_masks or {}).items()
                     if shield == index]
            if not cells and index < len(shields):
                shield = shields[index]
                cells = []
                for bits in (shield.get("health_mask") if isinstance(shield.get("health_mask"), list) else [])[:9]:
                    cells.extend([1] * (integer(bits, 0) & 255).bit_count())
            occupied = len(cells)
            quality = "GOOD" if occupied >= 42 else "DAMAGED" if occupied >= 18 else "OPEN"
            x = number(shields[index].get("x"), 46 + 32 * index) if index < len(shields) else 46 + 32 * index
            result.append({"name": ("LEFT", "CENTER", "RIGHT")[index],
                           "quality": quality, "occupied_band": "MANY" if occupied >= 42 else "SOME" if occupied >= 18 else "FEW",
                           "x": x})
        return result

    def _own_shot_near(self, x):
        return any(abs(x - launch_x) <= 6.0 and 0 <= self.frame - launch_frame <= 64
                   for launch_x, launch_frame in self.shot_launches)

    def _note_shield_damage(self, obs, discontinuity):
        current = self._mask_cells(obs)
        old = self.previous_masks
        self.previous_masks = current
        if discontinuity or old is None or len(records(obs.get("shields"))) != 3:
            return
        removed = list(old.keys() - current.keys())
        if not removed or len(removed) > 24 or (len(old) > 12 and not current):
            return
        event_xs = []
        for key in removed:
            _shield, row, _column = key
            same_column_top = min((r for s, r, c in old if s == key[0] and c == key[2]), default=9)
            x = number(old.get(key), 0.0)
            # Bottom-origin chips or chips near a recently launched player
            # beam are deliberately not treated as enemy evidence.
            if row > same_column_top + 2 or self._own_shot_near(x):
                continue
            event_xs.append(x)
        if not event_xs:
            return
        self.damage_events += 1
        self.stats["shield_damage_events"] += 1
        for x in event_xs:
            bucket = int(round(x / 8.0))
            self.lane_heat[bucket] = min(8.0, self.lane_heat.get(bucket, 0.0) + 1.0)
        self.log("unattributed_shield_lane", x=round(sum(event_xs) / len(event_xs), 1),
                 lost_pixels=len(event_xs), warning="weak, not a surviving bomb")

    def _track_bombs(self, obs):
        current = []
        for bomb in records(obs.get("bombs"))[:12]:
            if bomb.get("x") is None or bomb.get("y") is None:
                continue
            x, y = number(bomb.get("x"), -999.0), number(bomb.get("y"), -999.0)
            if not (0 <= x <= 159 and 0 <= y <= 209):
                continue
            matches = [old for old in self.bomb_tracks
                       if abs(old["x"] - x) <= 3.0 and 0 < self.frame - old["frame"] <= 16
                       and y >= old["y"] - 2]
            velocity = 1.2
            if matches:
                old = min(matches, key=lambda item: abs(y - item["y"] - 1.2 * (self.frame - item["frame"])))
                velocity = clamp((y - old["y"]) / max(1, self.frame - old["frame"]), 0.4, 3.0)
            current.append({"x": x, "y": y, "v": velocity, "frame": self.frame})
        if current:
            self.last_bomb_frame = self.frame
        for old in self.bomb_tracks:
            if self.frame - old["frame"] <= 12 and not any(abs(old["x"] - item["x"]) <= 3 for item in current):
                current.append(old)
        self.bomb_tracks = current[:16]

    def _risk(self, direction):
        if direction == 0:
            direction = 0
        prior = -1 if "left" in self.last_action else 1 if "right" in self.last_action else 0
        worst = 0.0
        for bomb in self.bomb_tracks:
            age = max(0, self.frame - integer(bomb.get("frame"), self.frame))
            velocity = number(bomb.get("v"), 1.2)
            y0 = number(bomb.get("y"), -999.0) + velocity * age
            for dt in range(0, 65, 4):
                y = y0 + velocity * dt
                if not self.py - 13 <= y <= self.py + 8:
                    continue
                for delay in (0, 4):
                    x = clamp(self.px + self.speed * (
                        prior * min(delay, dt) + direction * max(0, dt - delay)), 16, 143)
                    gap = abs(x - number(bomb.get("x"), x))
                    if gap <= 10:
                        worst = max(worst, 100.0 + (10.0 - gap) + (64.0 - dt) / 64.0)
                    elif gap < 18:
                        worst = max(worst, (18.0 - gap) / 8.0)
        return worst

    def _decay_heat(self, df):
        for key in list(self.lane_heat):
            self.lane_heat[key] -= df / 160.0
            if self.lane_heat[key] <= 0:
                del self.lane_heat[key]

    def _formation(self, obs):
        self.alive = [item for item in records(obs.get("invaders"))
                      if truth(item.get("alive")) and item.get("x") is not None]
        self.alive_count = len(self.alive)
        xs = [number(item.get("x"), self.form_center) for item in self.alive]
        ys = [number(item.get("y"), self.bottom_y) for item in self.alive]
        if xs:
            self.form_center = clamp((min(xs) + max(xs)) / 2.0, 24.0, 136.0)
        if ys:
            self.bottom_y = max(ys)

    def update(self, raw):
        obs = observation(raw)
        old = self.previous
        old_frame = self.frame
        frame = integer(obs.get("frame"), old_frame + 4)
        df = max(1, frame - old_frame) if old is not None else 4
        player = mapping(obs.get("player"))
        px = clamp(number(player.get("x"), self.px), 0.0, 159.0)
        py = clamp(number(player.get("y"), self.py), 150.0, 209.0)
        lives = integer(obs.get("lives"), self.lives)
        wave = integer(obs.get("wave"), self.wave)
        life_event = old is not None and lives < self.lives
        wave_event = old is not None and wave != self.wave
        jump = old is not None and abs(px - self.px) > max(12.0, 1.5 * df)
        discontinuity = life_event or wave_event or jump

        self.frame = frame
        self.last_life_event = life_event
        self.last_wave_event = wave_event
        if life_event:
            self.stats["life_losses"] += max(1, self.lives - lives)
            self.epoch += 1
            self.route = None
            self.advice = None
            self.bomb_tracks = []
            self.escape_until = -1
            self.escape_direction = 0
            self.direction = -self.direction if self.direction else 1
            self.stall_frames = 0
            self.last_motion_frame = frame
            self.log("life_phase_reset", previous_x=round(self.px, 1), lives=lives,
                     route_change="new_aperiodic_leg", visible_bombs=len(records(obs.get("bombs"))))
        elif wave_event:
            self.epoch += 1
            self.route = None
            self.advice = None
            self.bomb_tracks = []
            self.direction = 1 if px < 80 else -1
            self.stall_frames = 0
            self.log("wave_phase_reset", wave=wave)
        elif old is not None:
            dx = px - self.px
            commanded = -1 if "left" in self.last_action else 1 if "right" in self.last_action else 0
            if abs(dx) >= 0.35:
                self.last_motion_frame = frame
                self.stall_frames = 0
                if commanded and commanded * dx > 0:
                    self.speed = clamp(0.82 * self.speed + 0.18 * abs(dx) / df, 0.25, 1.2)
            elif commanded and 20.0 < px < 140.0:
                self.stall_frames += df

        shot = mapping(obs.get("player_shot"))
        shot_active = truth(shot.get("active"))
        if shot_active and not self.previous_shot_active:
            self.shot_launches.append((number(shot.get("x"), px), frame))
        self.shot_launches = deque(
            [(x, f) for x, f in self.shot_launches if frame - f <= 80], maxlen=24
        )

        score = number(obs.get("score"), self.score)
        if score > self.score:
            self.last_progress_frame = frame
        self.score = score
        self.lives = lives
        self.wave = wave
        self.px = px
        self.py = py
        self._decay_heat(df)
        self._note_shield_damage(obs, discontinuity)
        self._track_bombs(obs)
        self._formation(obs)
        self.cover = self._cover_facts(obs)
        self.previous_alive = {
            (integer(item.get("row"), -1), integer(item.get("col"), -1))
            for item in records(obs.get("invaders")) if truth(item.get("alive"))
        }
        self.previous_shot_active = shot_active
        self.previous = obs
        return obs

    def _mix(self):
        # The endpoint sequence is deterministic but not periodic in frame
        # time.  Score, formation position, and observed masks desynchronise
        # the next leg without using a training seed or a hidden game API.
        mask_hash = sum(integer(x * 3.0) for x in self.lane_heat.values())
        self.rng = (1664525 * self.rng + 1013904223 + self.frame
                    + integer(self.score * 7.0) + integer(self.form_center * 11.0)
                    + mask_hash + self.epoch * 65537) & 0xFFFFFFFF
        return self.rng

    def _wide_edges(self):
        center = self.form_center
        left = clamp(center - 62.0, 20.0, 50.0)
        right = clamp(center + 62.0, 108.0, 140.0)
        if right - left < 72.0:
            left, right = 20.0, 132.0
        return left, right

    def _best_cover(self, requested=None):
        if requested in ("COVER_LEFT", "COVER_CENTER", "COVER_RIGHT"):
            wanted = requested.split("_", 1)[1]
            for item in self.cover:
                if item["name"] == wanted:
                    return item
        if not self.cover:
            return {"name": "CENTER", "quality": "DAMAGED", "x": 78.0}
        rank = {"GOOD": 2, "DAMAGED": 1, "OPEN": 0}
        return max(self.cover, key=lambda item: (rank.get(item.get("quality"), 0),
                                                  -abs(number(item.get("x"), 78.0) - self.px)))

    def _endpoint(self, side, posture):
        left, right = self._wide_edges()
        if posture in ("COVER_LEFT", "COVER_CENTER", "COVER_RIGHT"):
            shield = self._best_cover(posture)
            center = number(shield.get("x"), 78.0)
            band_left = clamp(center - 28.0, 18.0, 112.0)
            band_right = clamp(center + 28.0, 48.0, 141.0)
            return band_left if side < 0 else band_right
        if posture == "WIDE":
            return left if side < 0 else right

        # DITHER uses long legs, but changes the exact endpoint.  Every
        # candidate remains well inside the legal lane and spans a useful
        # portion of the screen; this is not a rapid jitter policy.
        self._mix()
        if side < 0:
            candidates = [left, clamp(left + 10.0, 22.0, 55.0),
                          clamp(self.form_center - 34.0, 24.0, 58.0)]
        else:
            candidates = [right, clamp(right - 10.0, 105.0, 138.0),
                          clamp(self.form_center + 34.0, 96.0, 136.0)]
        choice = int(self.rng % len(candidates))
        goal = candidates[choice]
        # Do not accidentally turn a long traverse into a tiny reversal.
        if abs(goal - self.px) < 42.0:
            goal = left if side < 0 else right
        return goal

    def _posture_for_new_route(self):
        if self.advice:
            tag = mapping(self.advice.get("tag"))
            age = self.frame - integer(tag.get("frame"), self.frame)
            if (tag.get("epoch") == self.epoch and tag.get("lives") == self.lives
                    and tag.get("wave") == self.wave and 0 <= age <= 100):
                choice = self.advice.get("choice")
                self.advice = None
                if choice in ("COVER_LEFT", "COVER_CENTER", "COVER_RIGHT"):
                    # Cover is a recovery option, not a normal opening
                    # policy.  The code also rejects an actually open mask.
                    requested = self._best_cover(choice)
                    if self.lives >= 3 or requested.get("quality") == "OPEN":
                        self.log("jev_cover_rejected", reason="precondition")
                        choice = "WIDE_DITHER"
                if choice in POSTURES:
                    self.stats["jev_applied"] += 1
                    self.log("jev_patrol_posture", posture=choice, age=age)
                    return choice
            else:
                self.advice = None

        # After a loss the default is still a broad moving firing policy, but
        # its endpoint generator is newly re-phased by epoch and observations.
        # Astra's cover rotation is deliberately enabled only on a recovery or
        # last life, and selects the healthiest observed shield.
        if self.astra_policy == "COVER_ROTATE" and self.lives <= 2:
            cover = self._best_cover()
            if cover.get("quality") != "OPEN":
                return "COVER_" + str(cover.get("name", "CENTER"))
        return "WIDE" if self.astra_policy == "WIDE" else "WIDE_DITHER"

    def _start_route(self):
        if self.route is not None:
            return
        posture = self._posture_for_new_route()
        if posture == "UNKNOWN" or posture == "KEEP":
            posture = "WIDE" if self.astra_policy == "WIDE" else "WIDE_DITHER"
        if posture == "REVERSE":
            # Finishing a route already flips direction.  Only the initial
            # pre-route advice needs an additional reversal.
            if self.route_index == 0:
                self.direction = -self.direction if self.direction else 1
            self.stats["route_reversals"] += 1
            posture = "WIDE_DITHER"
        side = 1 if self.direction >= 0 else -1
        goal = self._endpoint(side, posture)
        self.route = {
            "start": self.frame,
            "goal": goal,
            "side": side,
            "posture": posture,
            "score": self.score,
            "index": self.route_index,
        }
        self.stats["routes"] += 1
        self.log("patrol_route_started", goal=round(goal, 1), side="RIGHT" if side > 0 else "LEFT",
                 posture=posture, route_index=self.route_index,
                 life_stage=max(0, 3 - self.lives))

    def _prepare_route(self):
        if self.route is not None:
            goal = number(self.route.get("goal"), self.px)
            side = integer(self.route.get("side"), self.direction)
            reached = (self.px >= goal - 2.0 if side > 0 else self.px <= goal + 2.0)
            expired = self.frame - integer(self.route.get("start"), self.frame) > 520
            stalled = self.stall_frames >= 36 and self.frame - self.last_motion_frame >= 36
            if reached or expired or stalled:
                reason = "reached" if reached else "stalled" if stalled else "expired"
                self.log("patrol_route_finished", reason=reason,
                         duration=self.frame - integer(self.route.get("start"), self.frame),
                         score_gain=round(self.score - number(self.route.get("score"), self.score), 1))
                self.route = None
                self.route_index += 1
                self.direction = -side
                self.stats["route_reversals"] += 1
        self._start_route()

    def _lane_heat_side(self):
        left = sum(v for k, v in self.lane_heat.items() if k * 8 < 62)
        center = sum(v for k, v in self.lane_heat.items() if 62 <= k * 8 <= 94)
        right = sum(v for k, v in self.lane_heat.items() if k * 8 > 94)
        if max(left, center, right) < 1.5:
            return "NONE"
        if left >= center and left >= right:
            return "LEFT"
        if right >= left and right >= center:
            return "RIGHT"
        return "CENTER"

    def _bomb_category(self, risks=None):
        risks = risks if risks is not None else {d: self._risk(d) for d in (-1, 0, 1)}
        if max(risks.values()) >= 100:
            return "NEAR_COLLISION"
        if self.bomb_tracks:
            return "VISIBLE_APPROACH"
        return "UNSEEN"

    def _phase(self):
        if self.alive_count >= 28:
            return "OPENING"
        if self.alive_count >= 14:
            return "MIDWAVE"
        return "LATEWAVE"

    def _program_facts(self):
        cover_good = any(item.get("quality") in ("GOOD", "DAMAGED") for item in self.cover)
        visible = bool(self.bomb_tracks)
        return {
            "WIDE_DITHER": "available",
            "REVERSE": "available" if visible or self.stall_frames >= 24 else "not_needed_without_explicit_trigger",
            "COVER_LEFT": "available" if cover_good and self.lives <= 2 else "reserve_for_late_life_or_visible_danger",
            "COVER_CENTER": "available" if cover_good and self.lives <= 2 else "reserve_for_late_life_or_visible_danger",
            "COVER_RIGHT": "available" if cover_good and self.lives <= 2 else "reserve_for_late_life_or_visible_danger",
            "KEEP": "available",
        }

    def packet(self):
        risks = {d: self._risk(d) for d in (-1, 0, 1)}
        route = self.route or {}
        return {
            "schema": "space_invaders_patrol_posture_v1",
            "episode": {
                "frame": self.frame,
                "wave": self.wave,
                "life_stage": "OPENING_LIVES" if self.lives >= 3 else "RECOVERY_LIFE" if self.lives == 2 else "LAST_LIFE",
                "phase": self._phase(),
                "progress": "RECENT" if self.frame - self.last_progress_frame <= 160 else "QUIET",
            },
            "formation": {
                "pressure": "LOW" if self.bottom_y < 110 else "RISING" if self.bottom_y < 145 else "HIGH",
                "phase": self._phase(),
                "geometry": "wide" if self.alive_count >= 24 else "reduced",
            },
            "route": {
                "posture": route.get("posture", "NONE"),
                "side": "RIGHT" if integer(route.get("side"), self.direction) > 0 else "LEFT",
                "age": "new" if self.frame - integer(route.get("start"), self.frame) < 80 else "committed",
                "movement": "STALLED" if self.stall_frames >= 24 else "RESPONSIVE",
                "next_boundary": "near" if self.route is None or self.frame - integer(route.get("start"), self.frame) > 360 else "not_near",
            },
            "threat": {
                "visible_bomb": self._bomb_category(risks),
                "lane_damage": self._lane_heat_side(),
                "empty_bombs_mean": "UNSEEN_NOT_SAFE",
            },
            "cover": {item["name"]: item["quality"] for item in self.cover},
            "programs": self._program_facts(),
            "plan": {
                "astra_policy": self.astra_policy,
                "rule": "At the boundary choose a long route posture; every tick moves and holds FIRE; code rechecks visible collisions.",
            },
            "limits": "No option selects coordinates, counts, a target column, or a permanent shield assumption.",
        }

    def query_tag(self):
        return {"frame": self.frame, "epoch": self.epoch, "lives": self.lives,
                "wave": self.wave, "route_index": self.route_index}

    def should_query(self):
        if self.frame - self.last_query_frame < 96:
            return False
        if self.last_query_frame < 0 or self.last_life_event or self.last_wave_event or self.route is None:
            return True
        route_age = self.frame - integer(self.route.get("start"), self.frame)
        return route_age >= 320

    def mark_query(self):
        self.last_query_frame = self.frame

    def accept(self, packet):
        if not isinstance(packet, dict) or packet.get("error"):
            return
        tag = mapping(packet.get("tag"))
        if (tag.get("epoch") != self.epoch or tag.get("lives") != self.lives
                or tag.get("wave") != self.wave):
            return
        age = self.frame - integer(tag.get("frame"), -100000)
        if age < 0 or age > 100:
            self.log("jev_posture_discarded", reason="stale", age=age)
            return
        answers = mapping(packet.get("answers"))
        a, b = answers.get("posture_a"), answers.get("posture_b")
        choice = answer_value(a)
        if choice not in POSTURES or answer_value(b) != choice:
            return
        for answer in (a, b):
            if answer_prob(answer, choice) < 0.62:
                return
            confidence = answer_confidence(answer)
            if confidence is not None and confidence < 0.25:
                return
        self.advice = {"choice": choice, "tag": tag}
        self.stats["jev_accepted"] += 1
        self.log("jev_posture_accepted", posture=choice, age=age)

    def apply_astra(self, text):
        if not isinstance(text, str):
            return False
        try:
            start, end = text.find("{"), text.rfind("}")
            data = json.loads(text[start:end + 1]) if 0 <= start < end else {}
        except (TypeError, ValueError, json.JSONDecodeError):
            self.log("astra_posture_rejected", reason="malformed")
            return False
        if not isinstance(data, dict):
            return False
        policy = data.get("patrol")
        if not isinstance(policy, str):
            return False
        policy = policy.strip().upper()
        if policy not in ("DITHER", "WIDE", "COVER_ROTATE"):
            self.log("astra_posture_rejected", reason="unknown_token")
            return False
        old = self.astra_policy
        self.astra_policy = policy
        if old != policy:
            self.stats["astra_applied"] += 1
            self.log("astra_posture_applied", old=old, new=policy,
                     takes_effect="next_route_boundary")
        return True

    def astra_prompt(self):
        cover = {item["name"]: item["quality"] for item in self.cover}
        evidence = {
            "frame": self.frame,
            "score": self.score,
            "lives": self.lives,
            "wave": self.wave,
            "remaining_invader_phase": self._phase(),
            "visible_bomb_tracks": len(self.bomb_tracks),
            "weak_shield_lane_events": self.damage_events,
            "cover": cover,
            "route_count": self.stats["routes"],
            "life_losses": self.stats["life_losses"],
        }
        return (
            "Choose one durable patrol posture for a real-time Atari 2600 Space Invaders "
            "controller. Reply with exactly one short JSON object and no explanation: "
            '{"patrol":"DITHER"}. Allowed patrol values are DITHER, WIDE, COVER_ROTATE. '
            "DITHER means long cross-field legs with changing safe interior endpoints; WIDE "
            "means fixed outer-margin traverses; COVER_ROTATE means use the healthiest of the "
            "three observed shield neighborhoods only after a life loss, while still moving "
            "and firing. This is a policy for future route boundaries, not an immediate action. "
            "Code already holds FIRE, predicts any explicitly decoded bomb, uses legal actions, "
            "and never treats an empty bomb list or weak shield chip as proof of safety. The "
            "strong archived controller scored by continuous FIRE but repeated nearly periodic "
            "left/right legs and had zero accepted incoming-projectile dodges. Choose based on "
            "the evidence, not on a seed-specific replay. Current evidence: "
            + json.dumps(evidence, separators=(",", ":"), allow_nan=False)
        )

    def action(self):
        self._prepare_route()
        goal = number(self.route.get("goal"), self.px) if self.route else self.px
        planned = 1 if goal > self.px + 1.5 else -1 if goal < self.px - 1.5 else -self.direction
        if planned == 0:
            planned = self.direction or 1
        risks = {d: self._risk(d) for d in (-1, 0, 1)}
        selected = planned
        source = "aperiodic_patrol"

        # Explicit projectile geometry always outranks a route posture.  The
        # detour is short and the route remains committed afterwards.
        if self.frame < self.escape_until and self.escape_direction:
            if risks.get(self.escape_direction, 0.0) < 100:
                selected = self.escape_direction
                source = "visible_bomb_escape_commit"
        elif risks.get(planned, 0.0) >= 100:
            alternatives = sorted((-1, 0, 1), key=lambda direction: (
                risks[direction], direction == 0, direction != planned))
            chosen = alternatives[0]
            if chosen != planned and risks[chosen] + 1.0 < risks[planned]:
                selected = chosen
                self.escape_direction = chosen
                self.escape_until = self.frame + 32
                self.stats["visible_dodges"] += 1
                source = "visible_bomb_escape"
                self.log("visible_bomb_escape", direction=chosen, planned=planned)
        if self.px <= 17 and selected < 0:
            selected = 1
        elif self.px >= 143 and selected > 0:
            selected = -1
        self.direction = selected if selected else self.direction
        base = "right" if selected > 0 else "left" if selected < 0 else "noop"
        action = base + "_fire" if base in ("right", "left") else "fire"
        self.stats["actions"] += 1
        return action, source


class Harness:
    def __init__(self, ctx):
        self.ctx = ctx

    def _legal(self, game):
        spec = getattr(game, "action_spec", None)
        try:
            legal = set(getattr(spec, "names", ACTIONS))
        except Exception:
            legal = set(ACTIONS)
        if not legal:
            legal = set(ACTIONS)
        noop = getattr(spec, "noop", "noop")
        if noop not in legal:
            noop = "noop" if "noop" in legal else next(iter(legal))
        return legal, noop

    def run_episode(self, game) -> None:
        ctx = self.ctx
        controller = PatrolController(ctx)
        mailbox = AdviceMailbox(ctx)
        legal, noop = self._legal(game)
        astra_future = None
        astra_sent = False
        last_frame = None

        def poll_astra():
            nonlocal astra_future
            if astra_future is None or not astra_future.done():
                return
            future = astra_future
            astra_future = None
            try:
                result = future.result()
                controller.apply_astra(getattr(result, "text", ""))
            except (CancelledError, TimeoutError, BudgetExceeded, EpisodeOver,
                    RuntimeError, TypeError, ValueError):
                controller.log("astra_posture_unavailable")
            except Exception:
                controller.log("astra_posture_unavailable")

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
                frame = integer(obs.get("frame"), (last_frame + 4) if last_frame is not None else 0)
                if last_frame is not None and frame == last_frame:
                    try:
                        ctx.wait_for_tick()
                    except (BudgetExceeded, EpisodeOver):
                        break
                    except Exception:
                        break
                    continue
                last_frame = frame
                controller.update(obs)
                controller.accept(mailbox.poll())
                poll_astra()
                action, source = controller.action()
                if action not in legal:
                    action = "fire" if "fire" in legal else noop
                try:
                    ctx.act(action)
                except (BudgetExceeded, EpisodeOver):
                    break
                except Exception:
                    # Legal-action construction is deterministic; a game-side
                    # failure should not turn malformed model advice into an
                    # escaping harness exception.
                    break
                controller.last_action = action

                if mailbox.future is None and controller.should_query():
                    state = controller.packet()
                    tag = controller.query_tag()
                    if mailbox.launch(state, tag):
                        controller.mark_query()

                if (not astra_sent and not ctx.done
                        and controller.frame >= 4):
                    try:
                        remaining = ctx.budget.remaining()
                    except Exception:
                        remaining = {}
                    if (remaining.get("system2_calls", 0) > 0
                            and remaining.get("wall_s", 0) > 50):
                        prompt = controller.astra_prompt()
                        try:
                            astra_future = ctx.spawn(
                                lambda p=prompt: ctx.think(p, timeout_s=55.0)
                            )
                            astra_sent = True
                            controller.log("astra_posture_requested")
                        except Exception:
                            astra_future = None

                if controller.stats["actions"] % 180 == 0:
                    controller.log("patrol_summary", score=controller.score,
                                   lives=controller.lives, alive=controller.alive_count,
                                   routes=controller.stats["routes"],
                                   visible_dodges=controller.stats["visible_dodges"],
                                   posture=controller.astra_policy)
                try:
                    ctx.wait_for_tick()
                except (BudgetExceeded, EpisodeOver):
                    break
                except Exception:
                    break
        finally:
            mailbox.close()
            if astra_future is not None and not astra_future.done():
                try:
                    astra_future.cancel()
                except Exception:
                    pass
            controller.log("patrol_finished", score=controller.score, lives=controller.lives,
                           alive=controller.alive_count, **controller.stats,
                           astra_policy=controller.astra_policy)


def build(ctx):
    return Harness(ctx)
