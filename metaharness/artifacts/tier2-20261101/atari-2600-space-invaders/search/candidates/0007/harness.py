"""Candidate 0007: empirical lane sweep with bounded asynchronous advice.

The game loop never waits for a model.  Code owns the one-shot lock, target
geometry, movement limits, and visible-threat response.  Jev chooses only a
short-lived movement intent; Astra can revise a small wave policy off the
critical path.  A wide patrol is deliberately used after an unobserved death:
the public bomb list is a sample, not a proof that a lane is safe.
"""
from __future__ import annotations

import json
import math
import re
from collections import deque
from concurrent.futures import CancelledError
from typing import Any

from metaharness.core.contract import BudgetExceeded, EpisodeOver
from metaharness.models.jev import Choice


ACTIONS = ("noop", "fire", "right", "left", "right_fire", "left_fire")

# These are the observed canonical player limits, not screen limits.  They are
# widened when a future observation proves that this build exposes more lane.
INITIAL_LEFT = 37.0
INITIAL_RIGHT = 119.0
ALIGN = 5.0
SHOT_LOCK_FRAMES = 8
PATROL_SEGMENT_FRAMES = 72


def number(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
        return result if math.isfinite(result) else default
    except (TypeError, ValueError, OverflowError):
        return default


def integer(value: Any, default: int = 0) -> int:
    result = number(value, float(default))
    try:
        return int(result)
    except (TypeError, ValueError, OverflowError):
        return default


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def sign(value: float) -> int:
    return 1 if value > 0 else -1 if value < 0 else 0


def objects(value: Any) -> list[dict[str, Any]]:
    return [x for x in value if isinstance(x, dict)] if isinstance(value, list) else []


def as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def alive(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes"}
    return bool(value)


def answer_value(answer: Any) -> Any:
    if isinstance(answer, dict):
        return answer.get("value", answer.get("choice"))
    return getattr(answer, "value", None)


def answer_prob(answer: Any, value: Any) -> float:
    probabilities = (
        answer.get("probabilities", {})
        if isinstance(answer, dict)
        else getattr(answer, "probabilities", {})
    )
    if not isinstance(probabilities, dict):
        return 0.0
    wanted = str(value).upper()
    for key, probability in probabilities.items():
        if str(key).upper() == wanted:
            return number(probability, 0.0)
    return 0.0


def answer_choice(answer: Any) -> str:
    value = answer_value(answer)
    return value.upper() if isinstance(value, str) else ""


def safe_json_object(text: Any) -> dict[str, Any]:
    """Accept arbitrary model text without making it a harness error."""
    if not isinstance(text, str):
        return {}
    candidates = [text]
    match = re.search(r"\{.*?\}", text, flags=re.DOTALL)
    if match:
        candidates.insert(0, match.group(0))
    for candidate in candidates:
        try:
            value = json.loads(candidate.strip())
        except (TypeError, ValueError, json.JSONDecodeError):
            continue
        if isinstance(value, dict):
            return value
    return {}


def _json_number(value: Any, default: int = 0) -> int:
    """Keep state strictly JSON-safe even if a decoder is malformed."""
    return integer(value, default)


class Controller:
    """A deterministic per-tick controller with a short semantic mailbox."""

    def __init__(self, ctx: Any):
        self.ctx = ctx
        self.plan: dict[str, str] = {
            "target_bias": "NEAREST",
            "posture": "WIDE",
            "dodge": "EARLY",
        }
        self.plan_version = 0

        self.previous: dict[str, Any] | None = None
        self.frame = 0
        self.wave = 1
        self.lives = 3
        self.score = 0.0
        self.last_score_frame = 0
        self.px = 90.0
        self.py = 190.0
        self.left_bound = INITIAL_LEFT
        self.right_bound = INITIAL_RIGHT
        self.alive: list[dict[str, Any]] = []
        self.shot: dict[str, Any] = {}
        self.shot_expected_until = -10000
        self.fire_lock_until = -10000
        self.target_col: int | None = None
        self.last_fired_col: int | None = None

        self.patrol_dir = 1
        self.patrol_until = PATROL_SEGMENT_FRAMES
        self.last_action = "noop"
        self.recovery_until = -10000
        self.recovery_center = False
        self.death_lanes: deque[tuple[float, int]] = deque(maxlen=8)
        self.blind = False
        self.unseen_deaths = 0
        self.life_event = False
        self.wave_event = False

        # Short bomb tracks bridge a decoder gap, but never turn an empty list
        # into a certificate of safety.
        self.bomb_tracks: list[dict[str, float]] = []
        self.last_bomb_frame = -10000

        # A Jev result is an intent, never a cached joystick command.
        self.advice: dict[str, Any] | None = None
        self.advice_version = 0

        self.events: deque[dict[str, Any]] = deque(maxlen=8)

    # ---------- observation and exact derived facts ----------

    def _alive_columns(self) -> dict[int, list[dict[str, Any]]]:
        columns: dict[int, list[dict[str, Any]]] = {}
        for invader in self.alive:
            col = integer(invader.get("col"), -1)
            if 0 <= col <= 5:
                columns.setdefault(col, []).append(invader)
        return columns

    def _target_for_column(self, values: list[dict[str, Any]]) -> dict[str, Any] | None:
        if not values:
            return None
        # The lowest surviving invader is the immediate threat and is a useful
        # first obstruction to remove from a column.
        return max(values, key=lambda item: (number(item.get("y"), 0), integer(item.get("row"), 0)))

    def _target_lane(self, target: dict[str, Any] | None) -> float:
        if target is None:
            return self.px
        # In the supplied decoder the player/laser lane and invader x use the
        # same left-edge convention.  Clamp to the learned legal lane.
        return clamp(number(target.get("x"), self.px), self.left_bound + 1, self.right_bound - 1)

    def _death_penalty(self, lane: float) -> float:
        penalty = 0.0
        for old_lane, old_frame in self.death_lanes:
            if self.frame - old_frame < 900 and abs(lane - old_lane) <= 9:
                penalty += 80.0 if self.blind else 30.0
        return penalty

    def _choose_column(self, columns: dict[int, list[dict[str, Any]]]) -> int | None:
        if not columns:
            return None
        candidates: list[tuple[float, int]] = []
        outer_left = min(columns)
        outer_right = max(columns)
        for col, values in columns.items():
            target = self._target_for_column(values)
            if target is None:
                continue
            lane = self._target_lane(target)
            cost = abs(lane - self.px)
            # Lower aliens matter more than a high-value upper-row shot.
            cost -= min(14.0, number(target.get("y"), 0) * 0.08)
            if self.plan.get("target_bias") == "OUTER" and col in {outer_left, outer_right}:
                cost -= 10.0
            if self.last_fired_col is not None and col == self.last_fired_col and len(columns) > 1:
                cost += 16.0
            if self.patrol_dir > 0 and lane < self.px - 10:
                cost += 8.0
            if self.patrol_dir < 0 and lane > self.px + 10:
                cost += 8.0
            cost += self._death_penalty(lane)
            # Deterministic tie breaking follows current sweep direction.
            tie = col if self.patrol_dir > 0 else -col
            candidates.append((cost + tie * 0.0001, col))
        return min(candidates)[1] if candidates else next(iter(columns))

    def target_info(self) -> dict[str, Any]:
        columns = self._alive_columns()
        if self.target_col not in columns:
            self.target_col = self._choose_column(columns)
        if self.target_col is None or self.target_col not in columns:
            return {
                "col": None,
                "lane": self.px,
                "side": "NONE",
                "row_band": "NONE",
                "alignment": "NONE",
            }
        target = self._target_for_column(columns[self.target_col])
        lane = self._target_lane(target)
        error = lane - self.px
        row = integer(target.get("row"), 0) if target else 0
        side = "LEFT" if self.target_col <= 1 else "RIGHT" if self.target_col >= 4 else "CENTER"
        return {
            "col": self.target_col,
            "lane": lane,
            "side": side,
            "row_band": "BOTTOM" if row >= 5 else "MIDDLE" if row >= 3 else "TOP",
            "alignment": "ALIGNED" if abs(error) <= ALIGN else "APPROACHING" if abs(error) <= 18 else "FAR",
            "target": target,
        }

    def _update_bombs(self, raw_bombs: list[dict[str, Any]], frame: int) -> None:
        current: list[dict[str, float]] = []
        for bomb in raw_bombs[:12]:
            if not isinstance(bomb.get("x"), (int, float)) or not isinstance(bomb.get("y"), (int, float)):
                continue
            bx, by = number(bomb.get("x")), number(bomb.get("y"))
            speed = 1.2
            matches = [
                old
                for old in self.bomb_tracks
                if abs(old["x"] - bx) <= 4 and 0 < frame - integer(old["frame"], frame) <= 20
            ]
            if matches:
                old = min(matches, key=lambda item: abs(by - item["y"]))
                dt = max(1, frame - integer(old["frame"], frame))
                speed = clamp((by - old["y"]) / dt, 0.5, 3.5)
            current.append({"x": bx, "y": by, "v": speed, "frame": float(frame)})
        if current:
            self.last_bomb_frame = frame
        # Keep only a short extrapolation.  This handles sparse observations,
        # but it is not a long-lived imagined projectile.
        for old in self.bomb_tracks:
            age = frame - integer(old["frame"], frame)
            if 0 < age <= 12 and not any(abs(old["x"] - item["x"]) <= 4 for item in current):
                current.append(dict(old))
        self.bomb_tracks = current[:16]

    def observe(self, obs: dict[str, Any]) -> None:
        self.life_event = False
        self.wave_event = False
        old = self.previous
        frame = integer(obs.get("frame"), self.frame + 4)
        if old is not None:
            frame = max(frame, self.frame + 1)
        self.frame = frame

        old_lives = self.lives
        old_wave = self.wave
        new_lives = integer(obs.get("lives"), self.lives)
        new_wave = integer(obs.get("wave"), self.wave)
        player = as_dict(obs.get("player"))
        self.px = clamp(number(player.get("x"), self.px), 0.0, 159.0)
        self.py = clamp(number(player.get("y"), self.py), 150.0, 209.0)
        # Learn only observations in the plausible cannon lane.  The initial
        # 37..119 envelope protects against a malformed model-facing state.
        if 20.0 <= self.px <= 145.0:
            if self.px < self.left_bound:
                self.left_bound = self.px
            if self.px > self.right_bound:
                self.right_bound = self.px
        self.left_bound = clamp(self.left_bound, 20.0, 70.0)
        self.right_bound = clamp(self.right_bound, 90.0, 145.0)

        self.alive = [item for item in objects(obs.get("invaders")) if alive(item.get("alive"))]
        shot = as_dict(obs.get("player_shot"))
        self.shot = shot
        was_active = bool(as_dict(old.get("player_shot")).get("active")) if old else False
        active = bool(shot.get("active"))
        if active and not was_active:
            self.shot_expected_until = frame + 8
        if not active and frame >= self.shot_expected_until:
            # The lock is released by the decoder, not by a timer alone.
            pass

        raw_bombs = objects(obs.get("bombs"))
        self._update_bombs(raw_bombs, frame)

        self.score = number(obs.get("score"), self.score)
        if old is not None:
            if self.score > number(old.get("score"), self.score):
                self.last_score_frame = frame
        elif self.score > 0:
            self.last_score_frame = frame

        if new_lives < old_lives:
            self.life_event = True
            death_lane = self.px
            self.death_lanes.append((death_lane, frame))
            # A life loss with no recent decoded bomb is evidence against a
            # lane-local quiet policy, not evidence that the game was safe.
            unseen = frame - self.last_bomb_frame > 48
            if unseen:
                self.unseen_deaths += 1
                self.blind = True
            self.recovery_until = frame + (32 if unseen else 20)
            self.recovery_center = True
            self.target_col = None
            self.advice = None
            self.fire_lock_until = frame + 12
            self.patrol_dir = 1 if self.px < (self.left_bound + self.right_bound) / 2 else -1
            self.patrol_until = frame + 90
            event = {
                "frame": frame,
                "previous_lives": old_lives,
                "lives": new_lives,
                "death_lane": round(self.px, 1),
                "unseen_bomb": bool(unseen),
            }
            self.events.append(event)
            try:
                self.ctx.log("life_loss_evidence", **event)
            except Exception:
                pass

        if old is not None and new_wave > old_wave:
            self.wave_event = True
            self.target_col = None
            self.last_fired_col = None
            self.recovery_until = frame + 28
            self.patrol_dir = 1
            self.patrol_until = frame + 90

        self.lives = new_lives
        self.wave = new_wave
        self.previous = obs

    def _danger(self) -> dict[str, Any]:
        best: tuple[int, float] | None = None
        escape = 0
        for bomb in self.bomb_tracks:
            age = max(0, self.frame - integer(bomb.get("frame"), self.frame))
            bx = number(bomb.get("x"), self.px)
            by = number(bomb.get("y"), -100) + number(bomb.get("v"), 1.2) * age
            gap = self.py - by
            lateral = abs(bx - self.px)
            if gap <= 0 or (gap <= 104 and lateral <= 25):
                urgent = gap <= 48 or lateral <= 11
                rank = 0 if urgent else 1
                score = gap + lateral
                if best is None or (rank, score) < best:
                    best = (rank, score)
                    if bx < self.px - 5:
                        escape = 1
                    elif bx > self.px + 5:
                        escape = -1
                    else:
                        escape = 1 if self.px <= (self.left_bound + self.right_bound) / 2 else -1
        lowest_y = max((number(item.get("y"), 0) for item in self.alive), default=0)
        formation = "URGENT" if lowest_y >= 145 else "WATCH" if lowest_y >= 135 else "NONE"
        if formation == "URGENT":
            return {"urgency": "URGENT", "escape": escape or self.patrol_dir, "bomb": "NONE", "formation": formation}
        if best is None:
            return {"urgency": formation, "escape": 0, "bomb": "NONE", "formation": formation}
        rank = best[0]
        return {
            "urgency": "URGENT" if rank == 0 else "WATCH",
            "escape": escape or self.patrol_dir,
            "bomb": "LEFT" if escape > 0 else "RIGHT" if escape < 0 else "CENTER",
            "formation": formation,
        }

    def compact_state(self) -> dict[str, Any]:
        target = self.target_info()
        danger = self._danger()
        lane = "LEFT" if self.px < self.left_bound + 18 else "RIGHT" if self.px > self.right_bound - 18 else "CENTER"
        boundary = "LEFT" if self.px <= self.left_bound + 3 else "RIGHT" if self.px >= self.right_bound - 3 else "OPEN"
        alive_count = len(self.alive)
        phase = "OPENING" if alive_count >= 30 else "MIDWAVE" if alive_count >= 15 else "LATEWAVE"
        return {
            "schema": "space_invaders_lane_sweep_v1",
            "frame": _json_number(self.frame),
            "wave": _json_number(self.wave),
            "lives": _json_number(self.lives),
            "phase": phase,
            "player": {
                "lane": lane,
                "boundary": boundary,
                "patrol": "RIGHT" if self.patrol_dir > 0 else "LEFT",
                "motion_policy": "WIDE_SWEEP" if self.blind or self.plan.get("posture") == "WIDE" else "TARGET_AIM",
            },
            "shot": "ACTIVE" if bool(self.shot.get("active")) else "LOCKED" if self.frame < self.fire_lock_until else "READY",
            "target": {
                "side": target["side"],
                "row_band": target["row_band"],
                "alignment": target["alignment"],
            },
            "threat": {
                "bomb": danger["bomb"],
                "urgency": danger["urgency"],
                "escape": "LEFT" if danger["escape"] < 0 else "RIGHT" if danger["escape"] > 0 else "NONE",
                "formation": danger["formation"],
            },
            "exposure": "BLIND_AFTER_DEATH" if self.blind else "BOMB_SAMPLES_SPARSE",
            "formation": {
                "alive_band": "MANY" if alive_count >= 24 else "SOME" if alive_count >= 10 else "FEW",
                "outer_left": any(integer(x.get("col"), -1) == 0 for x in self.alive),
                "outer_right": any(integer(x.get("col"), -1) == 5 for x in self.alive),
            },
            "plan": dict(self.plan),
            "recent_action": self.last_action,
        }

    # ---------- actions ----------

    def _boundary_safe_direction(self, direction: int) -> int:
        if self.px <= self.left_bound + 2:
            return 1
        if self.px >= self.right_bound - 2:
            return -1
        return 1 if direction >= 0 else -1

    def _patrol_direction(self) -> int:
        if self.px <= self.left_bound + 3:
            self.patrol_dir = 1
            self.patrol_until = self.frame + PATROL_SEGMENT_FRAMES
        elif self.px >= self.right_bound - 3:
            self.patrol_dir = -1
            self.patrol_until = self.frame + PATROL_SEGMENT_FRAMES
        elif self.frame >= self.patrol_until:
            self.patrol_dir *= -1
            self.patrol_until = self.frame + PATROL_SEGMENT_FRAMES
        return self._boundary_safe_direction(self.patrol_dir)

    def _advised_direction(self, fallback: int, shot_active: bool) -> int:
        advice = self.advice
        if not advice:
            return fallback
        if self.frame - integer(advice.get("frame"), self.frame) > 28:
            return fallback
        choice = str(advice.get("choice", ""))
        if choice == "SWEEP_LEFT":
            if self._danger()["urgency"] != "URGENT":
                return self._boundary_safe_direction(-1)
        if choice == "SWEEP_RIGHT":
            if self._danger()["urgency"] != "URGENT":
                return self._boundary_safe_direction(1)
        # A typed dodge is accepted only when it agrees with a current exact
        # danger calculation.  This prevents stale Jev choices from steering
        # into a visible bomb.
        danger = self._danger()
        if choice == "DODGE_LEFT" and danger["escape"] < 0:
            return self._boundary_safe_direction(-1)
        if choice == "DODGE_RIGHT" and danger["escape"] > 0:
            return self._boundary_safe_direction(1)
        return fallback

    def action(self) -> str:
        target = self.target_info()
        danger = self._danger()
        active = bool(self.shot.get("active")) or self.frame < self.shot_expected_until

        # Exact safety checks precede any typed answer.
        if danger["urgency"] == "URGENT":
            direction = self._boundary_safe_direction(integer(danger["escape"], self.patrol_dir))
            fire = False
        elif self.recovery_center and self.frame < self.recovery_until:
            center = (self.left_bound + self.right_bound) / 2
            direction = self._boundary_safe_direction(sign(center - self.px) or self.patrol_dir)
            fire = False
        elif active:
            direction = self._patrol_direction()
            direction = self._advised_direction(direction, True)
            fire = False
        else:
            lane = number(target.get("lane"), self.px)
            error = lane - self.px
            if abs(error) > ALIGN:
                direction = sign(error)
            else:
                direction = self._patrol_direction()
                direction = self._advised_direction(direction, False)
            # A known death lane is avoided before aiming or firing.  The
            # controller may cross it while sweeping, but never deliberately
            # dwell there after a blind death.
            if self.blind and self._death_penalty(self.px) > 0 and abs(self.px - lane) <= ALIGN:
                direction = 1 if self.px <= (self.left_bound + self.right_bound) / 2 else -1
            fire = (
                self.frame >= self.fire_lock_until
                and self.frame >= self.recovery_until - 12
                and bool(target.get("col") is not None)
                and abs(error) <= ALIGN
                and danger["urgency"] not in {"WATCH", "URGENT"}
                and danger["formation"] != "URGENT"
                and not (self.blind and self._death_penalty(self.px) > 0)
            )
            if fire:
                self.fire_lock_until = self.frame + SHOT_LOCK_FRAMES
                self.shot_expected_until = self.frame + 8
                self.last_fired_col = target.get("col")
                # Once a shot is launched, leave its lane rather than parking
                # below the same column for the full flight.
                self.patrol_dir = 1 if self.px <= (self.left_bound + self.right_bound) / 2 else -1
                self.patrol_until = self.frame + PATROL_SEGMENT_FRAMES

        direction = self._boundary_safe_direction(direction)
        base = "right" if direction > 0 else "left" if direction < 0 else "noop"
        if fire:
            return base + "_fire" if base in {"left", "right"} else "fire"
        return base

    # ---------- typed advice ----------

    def questions(self) -> dict[str, Any]:
        if (self.frame // 16) % 2:
            names = ["SWEEP_RIGHT", "AIM", "DODGE_LEFT", "DODGE_RIGHT", "SWEEP_LEFT", "HOLD", "UNKNOWN"]
        else:
            names = ["SWEEP_LEFT", "AIM", "DODGE_RIGHT", "DODGE_LEFT", "SWEEP_RIGHT", "HOLD", "UNKNOWN"]
        descriptions = {
            "AIM": "Use the exact code-selected living target and align for progress; code still owns firing legality.",
            "SWEEP_LEFT": "Continue a broad leftward patrol while no urgent threat contradicts it; movement reduces exposure to an unobserved bomb lane.",
            "SWEEP_RIGHT": "Continue a broad rightward patrol while no urgent threat contradicts it; movement reduces exposure to an unobserved bomb lane.",
            "DODGE_LEFT": "The supplied threat.escape says left is the immediate safer direction; never choose this for an absent or distant threat.",
            "DODGE_RIGHT": "The supplied threat.escape says right is the immediate safer direction; never choose this for an absent or distant threat.",
            "HOLD": "Briefly preserve the lane only when the shot/target state supports it and no exposure emergency is present.",
            "UNKNOWN": "The evidence does not support a described maneuver; code fallback must control.",
        }
        criteria = {name: descriptions[name] for name in names}
        reverse = dict(reversed(list(criteria.items())))
        instruction = (
            "Choose one short movement intent from the current JSON. Use only the named fields in "
            "player, target, threat, shot, exposure, and plan. Do not count, calculate, or invent "
            "bombs. An empty bomb sample is not safety proof. Code will recheck the latest state, "
            "reject stale or unsafe intents, and decide the exact action string."
        )
        return {
            "maneuver": Choice(instruction, criteria),
            "crosscheck": Choice(instruction, reverse),
        }

    def accept_advice(self, result: Any, tag: dict[str, Any]) -> None:
        if not isinstance(result, dict):
            return
        first = result.get("maneuver")
        second = result.get("crosscheck")
        choice = answer_choice(first)
        if choice not in {"AIM", "SWEEP_LEFT", "SWEEP_RIGHT", "DODGE_LEFT", "DODGE_RIGHT", "HOLD"}:
            return
        if choice != answer_choice(second):
            return
        if answer_prob(first, choice) < 0.58 or answer_prob(second, choice) < 0.58:
            return
        if tag.get("version") != self.plan_version or tag.get("wave") != self.wave or tag.get("lives") != self.lives:
            return
        age = self.frame - integer(tag.get("frame"), self.frame)
        if age < 0 or age > 28:
            return
        self.advice = {"choice": choice, "frame": integer(tag.get("frame"), self.frame), "version": self.plan_version}
        try:
            self.ctx.log("jev_intent_accepted", maneuver=choice, age_frames=age)
        except Exception:
            pass

    def advice_packet(self) -> tuple[dict[str, Any], dict[str, Any]]:
        state = self.compact_state()
        tag = {"frame": self.frame, "wave": self.wave, "lives": self.lives, "version": self.plan_version}
        return state, tag

    # ---------- Astra policy ----------

    def astra_prompt(self, game: Any, reason: str) -> str:
        try:
            description = game.describe()
        except Exception:
            description = "Atari 2600 Space Invaders, canonical one-player mode."
        evidence = {
            "reason": reason,
            "plan": self.plan,
            "frame": self.frame,
            "wave": self.wave,
            "lives": self.lives,
            "score": self.score,
            "alive_count": len(self.alive),
            "unseen_deaths": self.unseen_deaths,
            "recent_events": list(self.events),
            "current_state": self.compact_state(),
        }
        return (
            "You are the slow wave strategist for a real-time Atari 2600 Space Invaders "
            "controller. The emulator advances while you think. Return only a short JSON "
            "object with target_bias (NEAREST or OUTER), posture (WIDE or AIM), and "
            "dodge (EARLY or NORMAL). Code owns exact coordinates, learned legal bounds, "
            "one-laser locking, visible bomb prediction, and every tick; do not output "
            "actions or timed sequences. Empty bomb lists are not evidence of safety. "
            "Prefer WIDE after an unobserved death and do not make UFO chasing a goal.\n\n"
            f"Game description:\n{description}\n\nEvidence:\n{json.dumps(evidence, sort_keys=True)}"
        )

    def apply_astra(self, text: Any) -> bool:
        value = safe_json_object(text)
        if not value:
            return False
        new_plan = dict(self.plan)
        if str(value.get("target_bias", "")).upper() in {"NEAREST", "OUTER"}:
            new_plan["target_bias"] = str(value["target_bias"]).upper()
        if str(value.get("posture", "")).upper() in {"WIDE", "AIM"}:
            new_plan["posture"] = str(value["posture"]).upper()
        if str(value.get("dodge", "")).upper() in {"EARLY", "NORMAL"}:
            new_plan["dodge"] = str(value["dodge"]).upper()
        if self.blind:
            new_plan["posture"] = "WIDE"
        changed = new_plan != self.plan
        self.plan = new_plan
        if changed:
            self.plan_version += 1
            self.target_col = None
            self.advice = None
            try:
                self.ctx.log("astra_policy_updated", plan=dict(self.plan), version=self.plan_version)
            except Exception:
                pass
        return True


class Harness:
    def __init__(self, ctx: Any):
        self.ctx = ctx

    def _launch_jev(self, controller: Controller, slot: dict[str, Any]) -> None:
        if slot.get("future") is not None or getattr(self.ctx, "system1", True) is None:
            return
        try:
            if integer(self.ctx.budget.remaining().get("system1_calls"), 0) <= 0:
                return
        except Exception:
            pass
        state, tag = controller.advice_packet()
        questions = controller.questions()

        def call() -> Any:
            try:
                return self.ctx.decide(state, questions, timeout_s=0.50)
            except (TimeoutError, RuntimeError, TypeError, BudgetExceeded, EpisodeOver) as exc:
                return {"_error": type(exc).__name__}
            except Exception as exc:
                # Garbage stand-ins and transient failures are ordinary control
                # events; the deterministic loop remains executable.
                return {"_error": type(exc).__name__}

        try:
            slot["future"] = self.ctx.spawn(call)
            slot["tag"] = tag
        except Exception:
            slot["future"] = None
            slot["tag"] = None

    def _poll_jev(self, controller: Controller, slot: dict[str, Any]) -> None:
        future = slot.get("future")
        if future is None:
            return
        try:
            if not future.done():
                return
            result = future.result()
        except CancelledError:
            result = {"_error": "cancelled"}
        except Exception:
            result = {"_error": "future_error"}
        tag = slot.get("tag") or {}
        slot["future"] = None
        slot["tag"] = None
        if isinstance(result, dict) and not result.get("_error"):
            controller.accept_advice(result, tag)

    def _launch_astra(self, game: Any, controller: Controller, slot: dict[str, Any], reason: str) -> None:
        if slot.get("future") is not None:
            return
        if getattr(self.ctx, "system2", True) is None:
            return
        try:
            if integer(self.ctx.budget.remaining().get("system2_calls"), 0) <= 0:
                return
        except Exception:
            pass
        prompt = controller.astra_prompt(game, reason)
        session = slot.get("session")

        def call() -> Any:
            try:
                return self.ctx.think(prompt, session_id=session, timeout_s=60.0)
            except Exception as exc:
                return {"_error": type(exc).__name__}

        try:
            slot["future"] = self.ctx.spawn(call)
            slot["count"] = integer(slot.get("count"), 0) + 1
        except Exception:
            slot["future"] = None

    def _poll_astra(self, controller: Controller, slot: dict[str, Any]) -> None:
        future = slot.get("future")
        if future is None:
            return
        try:
            if not future.done():
                return
            result = future.result()
        except Exception:
            result = {"_error": "future_error"}
        slot["future"] = None
        if isinstance(result, dict) and result.get("_error"):
            return
        text = getattr(result, "text", None)
        session = getattr(result, "session_id", None)
        if isinstance(result, dict):
            text = result.get("text", text)
            session = result.get("session_id", session)
        if isinstance(session, str) and session:
            slot["session"] = session
        controller.apply_astra(text)

    @staticmethod
    def _legal(action: str, legal: set[str], noop: str) -> str:
        if action in legal:
            return action
        if noop in legal:
            return noop
        return next(iter(legal), noop)

    def run_episode(self, game: Any) -> None:
        ctx = self.ctx
        legal = set(getattr(game.action_spec, "names", ACTIONS))
        noop = getattr(game.action_spec, "noop", "noop")
        controller = Controller(ctx)
        jev_slot: dict[str, Any] = {"future": None, "tag": None}
        astra_slot: dict[str, Any] = {"future": None, "count": 0, "session": None}
        next_query_frame = -10000
        initial = True

        try:
            raw = game.observe()
            obs = raw if isinstance(raw, dict) else {}
        except Exception:
            obs = {}
        controller.observe(obs)
        self._launch_astra(game, controller, astra_slot, "initial wave policy")
        self._launch_jev(controller, jev_slot)

        while not ctx.done and not bool(getattr(game, "done", False)):
            self._poll_astra(controller, astra_slot)
            self._poll_jev(controller, jev_slot)
            if not initial:
                # Events are acted on at the next tick, after their state has
                # been rechecked.  The first observation is handled below.
                pass
            action = self._legal(controller.action(), legal, noop)
            try:
                ctx.act(action)
            except (BudgetExceeded, EpisodeOver):
                break
            except Exception:
                # The action was filtered against the public action spec.  A
                # final noop attempt is only a recovery from a transient game
                # adapter error.
                try:
                    ctx.act(noop if noop in legal else "noop")
                except Exception:
                    break
            controller.last_action = action
            initial = False

            try:
                ctx.wait_for_tick()
            except (BudgetExceeded, EpisodeOver):
                break
            except Exception:
                break
            if ctx.done or bool(getattr(game, "done", False)):
                break

            try:
                raw = game.observe()
                obs = raw if isinstance(raw, dict) else {}
            except Exception:
                break
            if not isinstance(obs, dict):
                continue
            if obs.get("game_over"):
                break
            controller.observe(obs)
            self._poll_astra(controller, astra_slot)
            self._poll_jev(controller, jev_slot)

            if controller.life_event:
                # One revision is useful only if it can affect future ticks;
                # do not pile up slow calls during a dying episode.
                if integer(astra_slot.get("count"), 0) < 2:
                    self._launch_astra(game, controller, astra_slot, "a life was lost; revise lane exposure and firing posture")
            if controller.wave_event and integer(astra_slot.get("count"), 0) < 2:
                self._launch_astra(game, controller, astra_slot, "a new wave was inferred; choose the next sweep posture")

            if controller.frame >= next_query_frame and jev_slot.get("future") is None:
                self._launch_jev(controller, jev_slot)
                next_query_frame = controller.frame + 16

        try:
            ctx.log(
                "controller_finished",
                frame=controller.frame,
                score=controller.score,
                unseen_deaths=controller.unseen_deaths,
                plan=dict(controller.plan),
                astra_requests=integer(astra_slot.get("count"), 0),
            )
        except Exception:
            pass


def build(ctx: Any) -> Harness:
    return Harness(ctx)
