"""Candidate 0012: infer decoder-hidden danger from shield erosion.

The low-level policy keeps the productive, full-rate strafing fire of the
strong candidates.  Its new sensor is the *change* in the observed shield
occupancy mask: erosion not explained by a recent player-shot lane becomes a
short-lived, uncertain danger lane.  Code computes the geometry and safe
fallback; Jev classifies only a compact set of already-computed choices.
"""
from __future__ import annotations

import json
import math
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


def clamp(value, low, high):
    return max(low, min(high, value))


def sign(value):
    return 1 if value > 0 else -1 if value < 0 else 0


def dict_or_empty(value):
    return value if isinstance(value, dict) else {}


def objects(value):
    return [item for item in value if isinstance(item, dict)] if isinstance(value, list) else []


def alive(value):
    if isinstance(value, str):
        return value.strip().lower() in ("true", "1", "yes")
    return bool(value)


def answer_value(answer):
    return answer.get("value", answer.get("choice")) if isinstance(answer, dict) else getattr(answer, "value", None)


def answer_probability(answer, option):
    probabilities = answer.get("probabilities", {}) if isinstance(answer, dict) else getattr(answer, "probabilities", {})
    if not isinstance(probabilities, dict):
        return 0.0
    return number(probabilities.get(option), 0.0)


def answer_confidence(answer):
    value = answer.get("confidence") if isinstance(answer, dict) else getattr(answer, "confidence", None)
    if value is None:
        return None
    parsed = number(value, -1.0)
    return parsed if parsed >= 0 else None


def questions(frame):
    options = {
        "DODGE_LEFT": "Move left now, only when `safe.left` is true and the recent damage lane makes left an escape.",
        "DODGE_RIGHT": "Move right now, only when `safe.right` is true and the recent damage lane makes right an escape.",
        "CONTINUE": "Continue the committed broad sweep; use when no current lane evidence supports a safe dodge.",
        "UNKNOWN": "The compact evidence does not justify choosing among the available controls.",
    }
    ordered = list(options.items())
    if (integer(frame) // 24) % 2:
        ordered.reverse()
    instruction = (
        "Choose only among the code-computed controls using `signals`, `safe`, and `sweep`. "
        "An unexplained loss of occupied shield pixels is weak evidence of a bomb lane, "
        "not proof. `own_shot_explains_change` means recent player fire could have caused "
        "the damage; do not call that an enemy threat. Never infer danger merely because "
        "the bomb list is empty. Geometry, movement limits, visible-bomb collision checks, "
        "and continuous firing are code-owned."
    )
    return {
        "intent_a": Choice(instruction, dict(ordered)),
        "intent_b": Choice(instruction, dict(reversed(ordered))),
    }


class AdviceMailbox:
    """At most one asynchronous, latest-state Jev call; no queued backlog."""
    def __init__(self, ctx):
        self.ctx = ctx
        self.future = None
        self.tag = None
        self.enabled = getattr(ctx, "system1", True) is not None

    def submit(self, state, tag):
        if not self.enabled or self.future is not None or self.ctx.done:
            return
        try:
            if self.ctx.budget.remaining().get("system1_calls", 0) <= 0:
                self.enabled = False
                return
        except Exception:
            pass
        qs = questions(tag["frame"])

        def call():
            try:
                return {"answers": self.ctx.decide(state, qs, timeout_s=0.85), "tag": tag}
            except Exception as exc:
                # Bad stand-in output, timeout, and ordinary transient failures
                # only remove advice; the deterministic controller keeps acting.
                return {"error": type(exc).__name__, "tag": tag}

        try:
            self.future = self.ctx.spawn(call)
            self.tag = dict(tag)
        except Exception:
            self.future = None
            self.tag = None

    def take(self):
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


class Controller:
    def __init__(self, ctx):
        self.ctx = ctx
        self.frame = 0
        self.score = 0.0
        self.lives = 3
        self.wave = 1
        self.px = 90.0
        self.py = 190.0
        self.direction = 1
        self.left_edge = 36.0
        self.right_edge = 118.0
        self.move_speed = 0.55
        self.last_action = "noop"
        self.old_obs = None
        self.old_masks = None
        self.old_alive = set()
        self.old_shot_active = False
        self.shot_history = []
        self.bomb_tracks = []
        self.last_bomb_frame = -100000
        self.lane_events = []
        self.avoid_frames = 180
        self.life_event = False
        self.wave_event = False
        self.mask_event = "none"
        self.mask_event_x = None
        self.own_shot_explains = False
        self.last_query_frame = -100000
        self.intent = None
        self.actions = 0
        self.foreign_chips = 0
        self.life_losses = 0
        self.score_at_last_progress = 0.0
        self.last_progress_frame = 0
        self.astra_requested = False
        self.astra_effective = 0
        self.last_escape_frame = -100000
        self.jev_changed = 0

    def log(self, event, **fields):
        try:
            self.ctx.log(event, **fields)
        except Exception:
            pass

    def _mask_cells(self, obs):
        """Map occupied shield bits to stable shield-row-bit cells and x pixels."""
        result = {}
        shields = objects(obs.get("shields"))
        for si, shield in enumerate(shields[:3]):
            x0 = number(shield.get("x"), 0.0) - 4.0
            masks = shield.get("health_mask")
            if not isinstance(masks, list):
                continue
            for row, raw in enumerate(masks[:9]):
                mask = integer(raw, 0) & 255
                for bit in range(8):
                    if mask & (1 << (7 - bit)):
                        result[(si, row, bit)] = x0 + bit
        return result

    def _record_shield_damage(self, obs):
        self.mask_event = "none"
        self.mask_event_x = None
        self.own_shot_explains = False
        if self.life_event or self.wave_event:
            # A respawn or wave transition can refill/reinitialize shield
            # pixels; never interpret that discontinuity as projectile damage.
            self.old_masks = self._mask_cells(obs)
            return
        if len(objects(obs.get("shields"))) != 3:
            self.old_masks = None
            return
        current = self._mask_cells(obs)
        if self.old_masks is None:
            self.old_masks = current
            return
        previous = self.old_masks
        lost = [(key, previous[key]) for key in previous.keys() - current.keys()]
        self.old_masks = current
        if len(lost) > 24 or (len(previous) > 12 and not current):
            self.mask_event = "bulk_clear_not_projectile"
            self.lane_events = []
            self.log("shield_bulk_clear_ignored", frame=self.frame, lost_pixels=len(lost))
            return
        if not lost:
            return
        # Attribute likely outgoing-laser damage before treating shield erosion
        # as an enemy-fire clue. A few pixels of decoder jitter are tolerated.
        unexplained = []
        for key, x in lost:
            si, row, bit = key
            top_before = min((r for sh, r, b in previous if sh == si and b == bit), default=9)
            # Bottom-origin chips are ambiguous outgoing fire. Only top-entry
            # erosion can supply the new enemy-lane cue.
            if row > top_before + 2:
                self.own_shot_explains = True
                continue
            if any(abs(x - shot_x) <= 4.5 and 0 <= self.frame - shot_frame <= 48
                   for shot_x, shot_frame in self.shot_history):
                self.own_shot_explains = True
            else:
                unexplained.append(x)
        if not unexplained:
            self.mask_event = "shot_aligned_erosion"
            return
        center = sum(unexplained) / len(unexplained)
        self.mask_event = "unexplained_erosion"
        self.mask_event_x = center
        self.foreign_chips += 1
        # A changed mask can be a large destructive burst. Keep only the
        # newly lost cell x-coordinates so direction is not biased by the
        # shield's center or by surviving occupancy.
        for x in sorted(set(unexplained)):
            remaining = sum(1 for xx in current.values() if abs(xx-x) < 0.1)
            self.lane_events.append({"x": x, "frame": self.frame,
                                     "expires": self.frame + self.avoid_frames,
                                     "exposed": remaining <= 2, "used": False,
                                     "kind": "shield_erosion"})
        self.lane_events = self.lane_events[-64:]
        self.log("shield_lane_inferred", frame=self.frame, x=round(center, 1),
                 lost_x=sorted(round(x, 1) for x in unexplained),
                 lost_pixels=len(unexplained), confidence="weak")

    def _track_bombs(self, obs):
        current = []
        for bomb in objects(obs.get("bombs"))[:12]:
            if not isinstance(bomb.get("x"), (int, float)) or not isinstance(bomb.get("y"), (int, float)):
                continue
            x, y = number(bomb.get("x")), number(bomb.get("y"))
            matches = [b for b in self.bomb_tracks
                       if abs(b["x"] - x) <= 3 and 0 < self.frame - b["frame"] <= 16 and y >= b["y"]]
            velocity = 1.2
            if matches:
                old = min(matches, key=lambda b: abs(y - b["y"] - 1.2 * (self.frame - b["frame"])))
                velocity = clamp((y - old["y"]) / max(1, self.frame - old["frame"]), 0.4, 3.0)
            current.append({"x": x, "y": y, "v": velocity, "frame": self.frame})
        if current:
            self.last_bomb_frame = self.frame
        for old in self.bomb_tracks:
            if self.frame - old["frame"] <= 12 and not any(abs(old["x"] - b["x"]) <= 3 for b in current):
                current.append(old)
        self.bomb_tracks = current[:16]

    def _visible_bomb_risk(self, direction):
        worst = 0.0
        prior = -1 if "left" in self.last_action else 1 if "right" in self.last_action else 0
        for bomb in self.bomb_tracks:
            age = max(0, self.frame - integer(bomb.get("frame"), self.frame))
            y0 = number(bomb.get("y")) + age * number(bomb.get("v"), 1.2)
            for dt in range(0, 57, 4):
                y = y0 + number(bomb.get("v"), 1.2) * dt
                if not self.py - 12 <= y <= self.py + 7:
                    continue
                for sticky_delay in (0, 4):
                    x = clamp(self.px + self.move_speed * (
                        prior * min(sticky_delay, dt) + direction * max(0, dt - sticky_delay)), 16, 143)
                    gap = abs(x - number(bomb.get("x"), x))
                    if gap < 11:
                        worst = max(worst, 100.0 + 11.0 - gap + (56 - dt) / 56.0)
                    elif gap < 18:
                        worst = max(worst, (18.0 - gap) / 7.0)
        return worst

    def observe(self, raw):
        obs = raw if isinstance(raw, dict) else {}
        old = self.old_obs
        old_frame = self.frame
        frame = integer(obs.get("frame"), old_frame + 4)
        df = max(1, frame - old_frame) if old is not None else 4
        self.frame = frame
        p = dict_or_empty(obs.get("player"))
        px = clamp(number(p.get("x"), self.px), 0.0, 159.0)
        self.py = clamp(number(p.get("y"), self.py), 150.0, 209.0)
        lives = integer(obs.get("lives"), self.lives)
        wave = integer(obs.get("wave"), self.wave)
        self.life_event = old is not None and lives < self.lives
        self.wave_event = old is not None and wave != self.wave
        self.mask_event = "none"
        self.mask_event_x = None
        shot = dict_or_empty(obs.get("player_shot"))
        shot_active = bool(shot.get("active"))
        if old is not None:
            op = dict_or_empty(old.get("player"))
            dx = px - number(op.get("x"), px)
            if abs(dx) > max(12, 1.5 * df):
                # Respawn jumps are not movement-speed observations.
                dx = 0.0
            if abs(dx) > 0.4:
                action_dir = -1 if "left" in self.last_action else 1 if "right" in self.last_action else 0
                if action_dir and 0.08 <= dx * action_dir / df <= 1.5:
                    self.move_speed = clamp(0.75 * self.move_speed + 0.25 * abs(dx) / df, 0.25, 1.2)
            old_shot = dict_or_empty(old.get("player_shot"))
            if shot_active and (not self.old_shot_active or number(shot.get("y")) >= 145):
                self.shot_history.append((number(shot.get("x"), px), frame))
            if "fire" in self.last_action and (not old_shot.get("active") or number(old_shot.get("y")) >= 145):
                # Some outgoing beams hit the shield between observations.
                self.shot_history.append((number(op.get("x"), px) + 1, frame))
            if self.life_event:
                self.life_losses += max(1, self.lives - lives)
                death_x = number(op.get("x"), self.px)
                unseen = self.frame - self.last_bomb_frame > 90
                # Do not repeat the refuted reset-position death-lane memory.
                # Life loss is diagnostic only and invalidates old lane cues.
                self.lane_events = []
                self.intent = None
                self.log("life_loss_signal", frame=frame, previous_x=round(death_x, 1),
                         recent_bomb=not unseen)
            if self.wave_event:
                self.lane_events = []
                self.intent = None
                self.direction = 1 if px < 77 else -1
            self._record_shield_damage(obs)
            # Keep lane evidence bounded and forget it at the policy-defined age.
            self.lane_events = [e for e in self.lane_events if e.get("expires", 0) >= frame]
            self.shot_history = [(x, f) for x, f in self.shot_history if frame - f <= 60][-16:]
        else:
            self.old_masks = self._mask_cells(obs)
        score = number(obs.get("score"), self.score)
        if score > self.score:
            self.last_progress_frame = frame
        self.score = score
        self.lives, self.wave, self.px = lives, wave, px
        self._track_bombs(obs)
        self.old_shot_active = shot_active
        self.old_alive = {(integer(a.get("row"), -1), integer(a.get("col"), -1))
                          for a in objects(obs.get("invaders")) if alive(a.get("alive"))}
        self.old_obs = obs
        return obs

    def _choose_lane_event(self):
        active = [e for e in self.lane_events if self.frame <= e.get("expires", -1)]
        self.lane_events = active
        if not active:
            return None
        # Retain the nearest active lane as compact evidence, but use the
        # strongest lane for the deterministic safety choice when several
        # shield rows were damaged in one observation.
        relevant = [e for e in active if not e.get("used", False)
                    and self.frame - self.last_escape_frame >= 48
                    and abs(number(e.get("x")) - self.px) <= 28]
        if not relevant:
            return None
        return min(relevant, key=lambda e: (abs(number(e.get("x")) - self.px),
                                             -integer(e.get("frame"), 0)))

    def _sweep_direction(self):
        if self.px >= self.right_edge - 2:
            self.direction = -1
        elif self.px <= self.left_edge + 2:
            self.direction = 1
        return self.direction

    def _safe_signal_direction(self, event):
        if event is None:
            return 0
        hx = number(event.get("x"), self.px)
        gap = hx - self.px
        base = self._sweep_direction()
        # Stay outside the inferred danger band. If heading toward the lane,
        # turn away; if already within the band, escape until clear.
        if abs(gap) <= 23 or (abs(gap) <= 40 and base * gap > 0):
            if gap > 1.5:
                return -1
            if gap < -1.5:
                return 1
            left_room = self.px - self.left_edge
            right_room = self.right_edge - self.px
            return 1 if right_room >= left_room else -1
        return 0

    def _derived(self):
        event = self._choose_lane_event()
        escape = self._safe_signal_direction(event) if event and event.get("exposed") else 0
        risk = {d: self._visible_bomb_risk(d) for d in (-1, 0, 1)}
        base = self._sweep_direction()
        bomb_escape = None
        if risk[base] >= 100:
            candidate = min(risk, key=lambda d: (risk[d], d == 0, d != base))
            if risk[candidate] + 1.0 < risk[base]:
                bomb_escape = candidate
        if bomb_escape is not None:
            fallback = bomb_escape
            reason = "visible_bomb_collision"
        elif escape and risk[escape] < 100:
            fallback = escape
            reason = "exposed_shield_lane"
        else:
            fallback = base
            reason = "committed_sweep"
        if self.px <= self.left_edge and fallback < 0:
            fallback = 1
        elif self.px >= self.right_edge and fallback > 0:
            fallback = -1
        if risk[fallback] >= 100:
            fallback = min(risk, key=lambda d: (risk[d], d == 0, d != base))
            reason = "visible_bomb_collision"
        event_name = "none"
        event_age = None
        event_x = None
        if event:
            event_name = event.get("kind", "lane_event")
            event_age = max(0, self.frame - integer(event.get("frame"), self.frame))
            event_x = number(event.get("x"), self.px)
        cover_count = 0
        if self.old_masks:
            cover_count = sum(1 for x in self.old_masks.values() if abs(x - self.px) <= 3.5)
        return {
            "event": event,
            "fallback": fallback,
            "reason": reason,
            "risk": risk,
            "state": {
                "schema": "space_invaders_shield_erosion_v1",
                "frame": self.frame,
                "lives": self.lives,
                "wave": self.wave,
                "player_lane": "LEFT" if self.px < 65 else "RIGHT" if self.px > 110 else "CENTER",
                "sweep": "RIGHT" if base > 0 else "LEFT",
                "signals": {
                    "shield_change": self.mask_event,
                    "signal_source": event_name,
                    "remaining_cover": "THIN" if event and event.get("exposed") else "PRESENT_OR_UNKNOWN",
                    "own_shot_explains_change": bool(self.own_shot_explains),
                    "lane_signal": "recent_unexplained" if event else "none_or_expired",
                    "lane_side": "LEFT" if event_x is not None and event_x < self.px - 5 else "RIGHT" if event_x is not None and event_x > self.px + 5 else "UNDER_PLAYER" if event_x is not None else "NONE",
                    "lane_age": "fresh" if event_age is not None and event_age <= 32 else "aging" if event_age is not None else "NONE",
                    "direct_bomb_risk": "HIGH" if max(risk.values()) >= 100 else "LOW_OR_UNOBSERVED",
                },
                "safe": {
                    "left": self.px > self.left_edge + 1 and risk[-1] < 100,
                    "right": self.px < self.right_edge - 1 and risk[1] < 100,
                    "fallback_escape": "LEFT" if fallback < 0 else "RIGHT" if fallback > 0 else "HOLD",
                    "cover_at_player": "OCCUPIED" if cover_count >= 14 else "PARTIAL_OR_OPEN",
                },
                "rules": "One laser at a time; combined movement/fire is legal. Shield loss aligned with a recent player shot is not enemy evidence. Empty bomb observations do not certify safety.",
            },
            "tag": {"frame": self.frame, "lives": self.lives, "wave": self.wave,
                    "signal_frame": event.get("frame") if event else None,
                    "signal_x": event.get("x") if event else None},
        }

    def accept(self, result):
        if not isinstance(result, dict) or result.get("error"):
            return
        tag = dict_or_empty(result.get("tag"))
        if (integer(tag.get("lives"), -1) != self.lives or integer(tag.get("wave"), -1) != self.wave
                or not 0 <= self.frame - integer(tag.get("frame"), -100000) <= 36):
            return
        answers = dict_or_empty(result.get("answers"))
        a, b = answers.get("intent_a"), answers.get("intent_b")
        choice = answer_value(a)
        if choice not in ("DODGE_LEFT", "DODGE_RIGHT", "CONTINUE") or answer_value(b) != choice:
            return
        for answer in (a, b):
            if answer_probability(answer, choice) < 0.60:
                return
            confidence = answer_confidence(answer)
            if confidence is not None and confidence < 0.25:
                return
        self.intent = {"choice": choice, **tag}

    def action(self, derived):
        fallback = integer(derived.get("fallback"), self.direction)
        event = derived.get("event")
        intent = self.intent
        selected = fallback
        source = derived.get("reason", "committed_sweep")
        if intent and event and intent.get("lives") == self.lives and intent.get("wave") == self.wave:
            age = self.frame - integer(intent.get("frame"), -100000)
            if (0 <= age <= 24 and abs(number(event.get("x")) - self.px) <= 28
                    and intent.get("signal_frame") == event.get("frame")
                    and abs(number(intent.get("signal_x"), -999) - number(event.get("x"))) <= 2):
                choice = intent.get("choice")
                proposed = -1 if choice == "DODGE_LEFT" else 1 if choice == "DODGE_RIGHT" else 0
                safe = (proposed < 0 and derived["state"]["safe"]["left"]
                        or proposed > 0 and derived["state"]["safe"]["right"])
                # The signal may veto a dangerous course but cannot override a
                # clear code safety advantage or move out of the escape region.
                if (safe and source != "visible_bomb_collision"
                        and proposed == self._safe_signal_direction(event)):
                    selected = proposed
                    source = "jev_selected_escape"
        if event and source in ("exposed_shield_lane", "jev_selected_escape"):
            # Consume the entire chip event once. Preserve the new sweep
            # heading, rather than treating a weak lane as a permanent wall.
            for item in self.lane_events:
                if item.get("frame") == event.get("frame"):
                    item["used"] = True
            self.last_escape_frame = self.frame
            if selected != fallback:
                self.jev_changed += 1
                self.log("jev_changed_control", frame=self.frame, fallback=fallback, selected=selected)
            self.log("shield_escape_applied", frame=self.frame, direction=selected, source=source)
        self.direction = selected if selected else self.direction
        base = "right" if selected > 0 else "left" if selected < 0 else "noop"
        # Repeated FIRE is deliberately held: the ROM's single-shot lock keeps
        # one beam in flight, while fresh beams are launched whenever possible.
        action = base + "_fire" if base in ("right", "left") else "fire"
        self.actions += 1
        return action, source

    def packet(self, derived):
        return derived["state"], derived["tag"]

    def astra_prompt(self):
        event = self._choose_lane_event()
        evidence = {
            "frame": self.frame,
            "score": self.score,
            "lives": self.lives,
            "wave": self.wave,
            "unexplained_shield_erosion_events": self.foreign_chips,
            "recent_lane_evidence": ("unexplained shield damage near x=" + str(round(number(event.get("x")), 1))
                                     if event else "no live inferred lane"),
            "recent_bombs_visible": self.frame - self.last_bomb_frame <= 90,
            "current_avoid_window_frames": self.avoid_frames,
        }
        return (
            "Choose a cautious memory duration for weak shield-damage evidence in a real-time "
            "Atari 2600 Space Invaders controller. The screen decoder sometimes omits bombs. "
            "Code already attributes recent player-shot lanes, runs a full-rate broad firing "
            "sweep, and treats visible collision predictions as stronger than this cue. "
            "Do not claim shield damage proves a bomb; choose only how long a future "
            "unexplained lane cue should remain relevant. Return exactly a short JSON object: "
            '{"avoid_window":"brief"}. Allowed values: brief, normal, extended. '
            "brief is appropriate if the signal is noisy; extended only if recent unexplained "
            "erosion or a death without a decoded bomb suggests repeat lane risk. Evidence: "
            + json.dumps(evidence, separators=(",", ":"), allow_nan=False)
        )

    def apply_astra(self, text):
        if not isinstance(text, str):
            return False
        try:
            start, end = text.find("{"), text.rfind("}")
            data = json.loads(text[start:end + 1]) if 0 <= start < end else {}
        except (TypeError, ValueError):
            return False
        if not isinstance(data, dict):
            return False
        window = data.get("avoid_window")
        if not isinstance(window, str):
            return False
        frames = {"brief": 96, "normal": 180, "extended": 300}.get(window)
        if frames is None:
            return False
        old = self.avoid_frames
        self.avoid_frames = frames
        for item in self.lane_events:
            item["expires"] = item.get("frame", self.frame) + frames
        self.log("astra_window_updated", old_frames=old, new_frames=frames)
        self.astra_effective += 1
        return True


class Harness:
    def __init__(self, ctx):
        self.ctx = ctx

    def _legal(self, game):
        spec = getattr(game, "action_spec", None)
        try:
            legal = set(getattr(spec, "names", ACTIONS))
        except Exception:
            legal = set(ACTIONS)
        noop = getattr(spec, "noop", "noop")
        if noop not in legal:
            noop = "noop" if "noop" in legal else (next(iter(legal)) if legal else "noop")
        return legal, noop

    def run_episode(self, game) -> None:
        ctx = self.ctx
        controller = Controller(ctx)
        mailbox = AdviceMailbox(ctx)
        legal, noop = self._legal(game)
        astra_future = None
        astra_started = False
        last_frame = None
        last_query_frame = -100000
        astra_calls = 0

        def poll_astra():
            nonlocal astra_future
            if astra_future is None or not astra_future.done():
                return
            future = astra_future
            astra_future = None
            try:
                result = future.result()
                controller.apply_astra(getattr(result, "text", ""))
            except (CancelledError, TimeoutError, BudgetExceeded, EpisodeOver, RuntimeError,
                    TypeError, ValueError):
                controller.log("astra_unavailable")
            except Exception:
                controller.log("astra_unavailable")

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
                obs = raw if isinstance(raw, dict) else {}
                if obs.get("game_over"):
                    break
                frame = integer(obs.get("frame"), (last_frame + 4) if last_frame is not None else 0)
                if last_frame is not None and frame == last_frame:
                    try:
                        ctx.wait_for_tick()
                    except (BudgetExceeded, EpisodeOver):
                        break
                    continue
                last_frame = frame
                controller.observe(obs)
                controller.accept(mailbox.take())
                poll_astra()
                derived = controller._derived()
                action, source = controller.action(derived)
                if action not in legal:
                    action = "fire" if "fire" in legal else noop
                try:
                    ctx.act(action)
                except (BudgetExceeded, EpisodeOver):
                    break
                controller.last_action = action

                state, tag = controller.packet(derived)
                state["plan"] = {"signal_retention_frames": controller.avoid_frames,
                                 "rule": "One detour per erosion event; keep firing; thin cover raises urgency."}
                urgent_query = (controller.mask_event == "unexplained_erosion" or controller.life_event)
                if (urgent_query or controller.frame - last_query_frame >= 20):
                    if derived.get("event") is not None:
                        mailbox.submit(state, tag)
                    last_query_frame = controller.frame

                # Astra revises the duration of a weak inferred danger signal
                # only after real evidence (first blind loss or unexplained chip).
                trigger = controller.life_event or controller.mask_event == "unexplained_erosion"
                remaining = ctx.budget.remaining()
                if (trigger and not astra_started and astra_calls < 1
                        and remaining.get("system2_calls", 0) > 0
                        and remaining.get("wall_s", 0) > 45):
                    prompt = controller.astra_prompt()
                    try:
                        astra_future = ctx.spawn(lambda p=prompt: ctx.think(p, timeout_s=55.0))
                        astra_started = True
                        astra_calls += 1
                        controller.log("astra_requested", frame=controller.frame,
                                       reason="unexplained_shield_or_life_signal")
                    except Exception:
                        astra_future = None
                if controller.actions % 150 == 0:
                    controller.log("shield_controller_summary", frame=controller.frame,
                                   score=controller.score, lives=controller.lives,
                                   lane_events=len(controller.lane_events),
                                   unexplained_chips=controller.foreign_chips, source=source)
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
            controller.log("shield_controller_finished", actions=controller.actions,
                           score=controller.score, lives=controller.lives,
                           lane_events=len(controller.lane_events),
                           unexplained_chips=controller.foreign_chips,
                           life_losses=controller.life_losses, astra_calls=astra_calls,
                           jev_changed_actions=controller.jev_changed,
                           astra_effective=controller.astra_effective)


def build(ctx):
    return Harness(ctx)
