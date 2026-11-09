"""Closed-loop, phase-based controller for Atari 2600 Donkey Kong.

The important difference from a reactive nearest-object controller is that a
ladder is not considered a goal merely because it is above Mario.  The
controller follows the verified bottom-to-top ladder levels, and it only
advances a level after an UP command has produced measured vertical progress.
Walking, climbing, jumping, and screen restarts are all short feedback loops;
no model answer is allowed to turn a stale observation into a long hold.
"""
from __future__ import annotations

import json
import math

from metaharness.models.jev import Choice, Noul


# Coordinates are the public decoder coordinates (top-left of each fixed
# ladder box).  On the ramp screen the first usable ladder is the bottom-right
# one.  The second choice at a two-ladder level is deliberately retained as a
# recovery, not as a free-form route search.
# Public ladder coordinates are top-left coordinates.  The game reports the
# player box with the same convention, so horizontal alignment is done on raw
# left edges rather than mixing a player center with a ladder edge.
RAMP_LEVELS = (
    (176, (108,)),
    (148, (48, 80)),
    (120, (108, 88)),
    (92, (48, 68)),
    (64, (108,)),
    (40, (32, 76)),
)

RIVET_ROWS_Y = (176, 148, 120, 92)
RIVET_X = {"left": 48, "right": 108}

# A four-frame ALE quantum is about 66 ms.  A cartridge jump lasts about
# thirty-one frames, so holding the latched direction for nine quanta covers
# the jump without repeatedly pressing FIRE after landing.
JUMP_TICKS = 9
X_TOLERANCE = 2
Y_TOLERANCE = 10


def _finite_number(value, default=0.0):
    try:
        result = float(value)
        if math.isfinite(result):
            return result
    except (TypeError, ValueError):
        pass
    return default


def _integer(value, default=0):
    try:
        if isinstance(value, bool):
            return int(value)
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _visible(obj):
    return isinstance(obj, dict) and obj.get("visible", True) is not False


def _box_center(obj):
    if not isinstance(obj, dict):
        return 0.0, 0.0
    x = _finite_number(obj.get("x", obj.get("left", 0)))
    y = _finite_number(obj.get("y", obj.get("top", 0)))
    w = _finite_number(obj.get("w", obj.get("width", 0)))
    h = _finite_number(obj.get("h", obj.get("height", 0)))
    return x + w / 2.0, y + h / 2.0


def _safe_rows(obs):
    raw = obs.get("rivet_rows", []) if isinstance(obs, dict) else []
    if not isinstance(raw, (list, tuple)):
        return [0, 0, 0, 0]
    result = []
    for value in list(raw)[:4]:
        n = _integer(value, 0)
        result.append(n if n in (0, 6, 12, 18) else 0)
    while len(result) < 4:
        result.append(0)
    return result


def _stage(obs):
    if not isinstance(obs, dict):
        return "unknown"
    value = obs.get("stage", "unknown")
    return value if isinstance(value, str) else "unknown"


def _same_row(player, hazard):
    px, py = _box_center(player)
    hx, hy = _box_center(hazard)
    # Object boxes are more useful than their nominal walkway labels: on the
    # ramp trace the player top is 176 and a barrel top is 189.
    return abs(hy - py) <= 25.0, px, py, hx, hy


def _hazard_facts(obs):
    player = obs.get("player", {}) if isinstance(obs, dict) else {}
    px, py = _box_center(player)
    best = None
    hazards = obs.get("hazards", []) if isinstance(obs, dict) else []
    if not isinstance(hazards, list):
        hazards = []

    for raw in hazards:
        if not _visible(raw):
            continue
        same, _, _, hx, hy = _same_row(player, raw)
        if not same:
            continue
        dx = _finite_number(raw.get("dx", 0))
        horizontal = hx - px
        distance = abs(horizontal)
        closing = (
            (horizontal > 0 and dx < -0.2)
            or (horizontal < 0 and dx > 0.2)
            or (distance <= 17 and abs(dx) <= 0.2)
        )
        if not closing and distance > 16:
            continue
        urgency = "clear"
        if closing and distance <= 38:
            urgency = "immediate"
        elif closing and distance <= 61:
            urgency = "soon"
        elif distance <= 20:
            urgency = "immediate"
        elif closing:
            urgency = "watch"
        item = {
            "side": "right" if horizontal > 0 else "left",
            "motion": "left" if dx < -0.2 else "right" if dx > 0.2 else "still",
            "distance": round(distance, 1),
            "closing": bool(closing),
            "urgency": urgency,
            "x": round(hx, 1),
            "y": round(hy, 1),
        }
        rank = (0 if urgency == "immediate" else 1 if urgency == "soon" else 2, distance)
        if best is None or rank < best[0]:
            best = rank, item

    if best is None:
        return {
            "present": False,
            "side": "none",
            "motion": "none",
            "distance_bucket": "none",
            "closing": False,
            "urgency": "clear",
        }

    item = best[1]
    distance = item["distance"]
    item["present"] = True
    item["distance_bucket"] = (
        "overlapping" if distance <= 20 else
        "near" if distance <= 38 else
        "approaching" if distance <= 61 else "far"
    )
    # Preserve only named buckets in the Jev state.  Code retains the numeric
    # distance for its own threshold, but Jev is not asked to compare numbers.
    return item


def _mask_has(mask, side):
    # The decoder/study convention is 6 for the left cleared rivet and 12 for
    # the right cleared rivet.  Thus 6 means left remains, 12 means right
    # remains, and 0 means both remain.
    if side == "left":
        return mask in (0, 6)
    return mask in (0, 12)


def _remaining_side(mask, preferred):
    if mask == 18:
        return None
    if _mask_has(mask, preferred):
        return preferred
    other = "right" if preferred == "left" else "left"
    if _mask_has(mask, other):
        return other
    # An invalid/intermediate byte should not make the controller walk blindly.
    return preferred


def _motion_name(player):
    dx = _finite_number(player.get("dx", 0)) if isinstance(player, dict) else 0
    dy = _finite_number(player.get("dy", 0)) if isinstance(player, dict) else 0
    if abs(dx) <= 1 and abs(dy) <= 1:
        return "still"
    if abs(dx) >= abs(dy):
        return "right" if dx > 0 else "left"
    return "down" if dy > 0 else "up"


class Controller:
    def __init__(self, ctx):
        self.ctx = ctx
        self.last_action = "noop"
        self.loop_tick = 0
        self.last_stage = None
        self.last_lives = None
        self.last_screens = 0

        # Ramp feedback state.
        self.ramp_level = 0
        self.ramp_choice = 0
        self.climb_started = False
        self.climb_start_y = None
        self.climb_last_y = None
        self.climb_ticks = 0
        self.no_progress_ticks = 0

        # Rivet feedback state.
        self.rivet_row = 0
        self.last_rows = None
        self.sweep_started = False
        self.gap_pending = False
        self.gap_direction = "right"
        self.rivet_target_side = "left"
        self.rivet_first_preference = "left"
        self.rivet_climb_side = "right"

        # A jump is a committed short action sequence.  No Jev answer can
        # replace its direction halfway through the sequence.
        self.jump_ticks_left = 0
        self.jump_direction = "right"
        self.last_hazard_urgency = "clear"

        # Transition/restart state.
        self.transition_age = 0
        self.transition_attempts = 0
        self.idle_attempts = 0

        # Asynchronous model state.  Results are advisory; executable route
        # and all lethal guards remain in this object.
        self.jev_future = None
        self.jev_requested_at = -10000
        self.jev_result = None
        self.jev_result_tick = -10000
        self.jev_result_stage = None
        self.astra_future = None
        self.astra_requested_at = -10000
        self.astra_calls = 0
        self.plan_version = 0

    def _log(self, event, **fields):
        try:
            self.ctx.log(event, **fields)
        except Exception:
            pass

    def _is_done(self, game=None):
        try:
            if bool(self.ctx.done):
                return True
        except Exception:
            return True
        try:
            if game is not None and bool(game.done):
                return True
        except Exception:
            pass
        return False

    def _valid_action(self, game, action):
        if not isinstance(action, str):
            action = "noop"
        try:
            names = set(game.action_spec.names)
            if action in names:
                return action
            noop = game.action_spec.noop
            if noop in names:
                return noop
        except Exception:
            pass
        return action if action in {
            "noop", "fire", "up", "right", "left", "down",
            "upright", "upleft", "downright", "downleft", "upfire",
            "rightfire", "leftfire", "downfire", "uprightfire",
            "upleftfire", "downrightfire", "downleftfire",
        } else "noop"

    def _post(self, game, action):
        action = self._valid_action(game, action)
        try:
            self.ctx.act(action)
            self.last_action = action
            return True
        except Exception as exc:
            self._log("act_failed", error=type(exc).__name__, action=action)
            return False

    def _reset_active_route(self, stage, obs):
        self.jump_ticks_left = 0
        self.climb_started = False
        self.climb_start_y = None
        self.climb_last_y = None
        self.climb_ticks = 0
        self.no_progress_ticks = 0
        if stage == "ramp":
            self.ramp_level = 0
            self.ramp_choice = 0
        elif stage == "rivet":
            self.rivet_row = 0
            self.last_rows = None
            self.sweep_started = False
            self.gap_pending = False
            self.rivet_target_side = "left"
            self.rivet_climb_side = "right"

    def _sync_observation(self, obs):
        stage = _stage(obs)
        lives = _integer(obs.get("lives_remaining", 0), 0) if isinstance(obs, dict) else 0
        screens = _integer(obs.get("screens_completed", 0), 0) if isinstance(obs, dict) else 0

        stage_changed = stage != self.last_stage
        life_changed = self.last_lives is not None and lives != self.last_lives
        screen_changed = screens != self.last_screens

        if stage_changed:
            self._log("stage_change", old=self.last_stage, new=stage,
                      lives=lives, screens=screens)
            self.transition_age = 0
            self.transition_attempts = 0
            self.idle_attempts = 0
            if stage in ("ramp", "rivet"):
                self._reset_active_route(stage, obs)
        if life_changed:
            self._log("life_change", old=self.last_lives, new=lives)
            self._reset_active_route(stage, obs)
        if screen_changed:
            self._log("screen_complete", screens=screens)

        if stage == "transition":
            self.transition_age += 1
        else:
            self.transition_age = 0

        self.last_stage = stage
        self.last_lives = lives
        self.last_screens = screens
        return stage, lives, screens, stage_changed, life_changed

    def _start_astra(self, obs, reason):
        if self.astra_future is not None or self._is_done():
            return
        if self.astra_calls >= 8:
            return
        try:
            remaining = self.ctx.budget.remaining()
            if remaining.get("system2_calls", 0) <= 0:
                return
        except Exception:
            return

        state = self._model_state(obs, "macro")
        prompt = (
            "You are the slow macro planner for the NTSC Atari 2600 Donkey Kong "
            "cartridge, not the arcade game. The controller is already executing "
            "a closed-loop route. Choose only validated variants; do not provide "
            "a long action sequence. Return one compact JSON object, with no prose: "
            "{\"ramp_variant\":\"nominal|alternate\","
            "\"rivet_first\":\"left|right\","
            "\"risk\":\"conservative|balanced\"}. "
            "Nominal is the verified bottom-right then alternating ladder prior. "
            "Use alternate only when the supplied recent observations show that "
            "the nominal ladder failed to produce vertical progress. "
            "Astra cannot change the action mapping or override a jump guard.\n"
            "Reason: " + str(reason)[:120] + "\n"
            "State: " + json.dumps(state, separators=(",", ":"), allow_nan=False)
        )
        self.astra_calls += 1
        self.astra_requested_at = self.loop_tick

        def work():
            return self.ctx.think(prompt, timeout_s=95.0)

        try:
            self.astra_future = self.ctx.spawn(work)
        except Exception as exc:
            self.astra_future = None
            self._log("astra_spawn_failed", error=type(exc).__name__)

    def _collect_astra(self):
        future = self.astra_future
        if future is None or not future.done():
            return
        self.astra_future = None
        try:
            result = future.result()
            raw = getattr(result, "text", "")
            if not isinstance(raw, str):
                return
            parsed = None
            try:
                parsed = json.loads(raw)
            except (TypeError, ValueError):
                start = raw.find("{")
                end = raw.rfind("}")
                if start >= 0 and end > start:
                    try:
                        parsed = json.loads(raw[start:end + 1])
                    except (TypeError, ValueError):
                        parsed = None
            if not isinstance(parsed, dict):
                return

            # Only exact, low-risk fields are accepted.  This makes arbitrary
            # stand-in output harmless during pre-flight.
            first = parsed.get("rivet_first")
            if first in ("left", "right"):
                self.rivet_first_preference = first
            risk = parsed.get("risk")
            if risk not in ("conservative", "balanced"):
                risk = "balanced"
            self.plan_version += 1
            self._log("astra_plan", version=self.plan_version,
                      rivet_first=self.rivet_first_preference, risk=risk)
        except Exception as exc:
            self._log("astra_failed", error=type(exc).__name__)

    def _model_state(self, obs, mode):
        player = obs.get("player", {}) if isinstance(obs, dict) else {}
        px, py = _box_center(player)
        hazard = _hazard_facts(obs if isinstance(obs, dict) else {})
        rows = _safe_rows(obs if isinstance(obs, dict) else {})
        stage = _stage(obs if isinstance(obs, dict) else {})
        if stage == "ramp":
            level = min(self.ramp_level, len(RAMP_LEVELS) - 1)
            row_y, choices = RAMP_LEVELS[level]
            target_x = choices[min(self.ramp_choice, len(choices) - 1)]
            target_kind = "ladder"
            target_side = "left" if target_x < px else "right"
        elif stage == "rivet":
            row = min(self.rivet_row, 3)
            target_x = RIVET_X.get(self.rivet_target_side, 48)
            row_y = RIVET_ROWS_Y[row]
            target_kind = "rivet_or_gap"
            target_side = self.rivet_target_side
        else:
            target_x, row_y, target_kind, target_side = 0, 0, "none", "none"
        return {
            "game": "Atari 2600 Donkey Kong NTSC",
            "mode": mode,
            "stage": stage,
            "player": {
                "horizontal_band": "left" if px < 52 else "right" if px > 100 else "middle",
                "vertical_band": "bottom" if py > 160 else "lower" if py > 132 else "middle" if py > 78 else "top",
                "motion": _motion_name(player),
                "jump_committed": self.jump_ticks_left > 0,
            },
            "route": {
                "target_kind": target_kind,
                "target_side": target_side,
                "target_band": "bottom" if row_y >= 160 else "lower" if row_y >= 132 else "middle" if row_y >= 78 else "top",
                "ramp_level": self.ramp_level,
                "rivet_row": self.rivet_row,
                "rivet_rows": rows,
                "climb_started": self.climb_started,
                "no_progress": self.no_progress_ticks > 8,
                "gap_pending": self.gap_pending,
            },
            "hazard": hazard,
            "progress": {
                "lives_remaining": _integer(obs.get("lives_remaining", 0)),
                "screens_completed": _integer(obs.get("screens_completed", 0)),
                "rivets_removed": _integer(obs.get("rivets_removed", 0)),
                "bonus_bucket": "none" if _integer(obs.get("current_bonus", 0)) <= 0 else "active",
            },
            "last_action": self.last_action,
            "target_horizontal_band": "left" if target_x < 52 else "right" if target_x > 100 else "middle",
        }

    def _start_jev(self, obs, fallback):
        if self.jev_future is not None or self._is_done():
            return
        if self.loop_tick - self.jev_requested_at < 7:
            return
        try:
            if self.ctx.budget.remaining().get("system1_calls", 0) <= 0:
                return
        except Exception:
            return

        state = self._model_state(obs, "tactical")
        criteria = {
            "hold_route": {
                "what": "Continue the code-computed walk or climb toward the displayed target.",
                "not_for": "Do not choose this when hazard.urgency is immediate or a gap jump is pending.",
            },
            "jump_left": {
                "what": "Commit to a LEFTFIRE jump now to clear a closing hazard or cleared gap.",
                "not_for": "Do not invent a jump when hazard is clear and gap_pending is false.",
            },
            "jump_right": {
                "what": "Commit to a RIGHTFIRE jump now to clear a closing hazard or cleared gap.",
                "not_for": "Do not invent a jump when hazard is clear and gap_pending is false.",
            },
            "wait": {
                "what": "Use NOOP for one short quantum while a transition or uncertain state settles.",
                "not_for": "Do not wait instead of a safe route action during active play.",
            },
        }
        questions = {
            "tactic": Choice(
                "Choose one tactic for this exact normalized state. Code has already "
                "computed geometry and the legal route. Treat an immediate hazard "
                "or gap_pending as a strong instruction to select the matching jump; "
                "otherwise select hold_route. Do not compare or calculate numbers.",
                criteria,
            ),
            "closing": Noul(
                "Is hazard.present a closing same-row obstacle that needs a jump "
                "before the next few quanta? Answer from hazard.urgency and hazard.closing, "
                "not from raw coordinates.",
                {
                    "true": "The normalized hazard facts show an immediate or soon closing obstacle.",
                    "false": "No immediate jump-worthy closing obstacle is shown.",
                },
            ),
        }
        self.jev_requested_at = self.loop_tick
        stage_at_request = _stage(obs)
        self.jev_result = None

        def work():
            return self.ctx.decide(state, questions, timeout_s=0.9)

        try:
            self.jev_future = self.ctx.spawn(work)
            self.jev_result_stage = stage_at_request
        except Exception as exc:
            self.jev_future = None
            self._log("jev_spawn_failed", error=type(exc).__name__)

    @staticmethod
    def _answer_probability(answer, option):
        probabilities = getattr(answer, "probabilities", None)
        if not isinstance(probabilities, dict):
            return 0.0
        try:
            value = float(probabilities.get(option, 0.0))
            return value if math.isfinite(value) else 0.0
        except (TypeError, ValueError):
            return 0.0

    def _collect_jev(self, obs):
        future = self.jev_future
        if future is None or not future.done():
            return
        self.jev_future = None
        try:
            answers = future.result()
            if not isinstance(answers, dict):
                return
            answer = answers.get("tactic")
            value = getattr(answer, "value", None) if answer is not None else None
            if isinstance(value, str):
                self.jev_result = value
                self.jev_result_tick = self.loop_tick
                self.jev_result_stage = _stage(obs)
                self._log("jev_tactic", tactic=value,
                          confidence=getattr(answer, "confidence", None))
        except Exception as exc:
            self._log("jev_failed", error=type(exc).__name__)

    def _choose_jump_direction(self, obs, hazard):
        if self.gap_pending:
            return self.gap_direction
        side = hazard.get("side", "none")
        px, _ = _box_center(obs.get("player", {}))
        # Usually jump away from a closing obstacle.  At an edge, use the
        # interior side; a late turn into the wall is worse than a committed
        # jump across the approaching sprite.
        away = "left" if side == "right" else "right"
        if away == "left" and px < 31:
            return "right"
        if away == "right" and px > 126:
            return "left"
        return away if side in ("left", "right") else ("left" if px > 80 else "right")

    def _jump_action(self, direction):
        return "leftfire" if direction == "left" else "rightfire"

    def _hazard_action(self, obs):
        hazard = _hazard_facts(obs)
        self.last_hazard_urgency = hazard.get("urgency", "clear")
        if self.jump_ticks_left > 0:
            self.jump_ticks_left -= 1
            return self._jump_action(self.jump_direction)
        if self.gap_pending:
            self.gap_pending = False
            self.jump_direction = self.gap_direction
            self.jump_ticks_left = JUMP_TICKS - 1
            return self._jump_action(self.jump_direction)
        if hazard.get("urgency") in ("immediate", "soon"):
            self.jump_direction = self._choose_jump_direction(obs, hazard)
            self.jump_ticks_left = JUMP_TICKS - 1
            self._log("jump_started", reason="hazard", direction=self.jump_direction,
                      urgency=hazard.get("urgency"))
            return self._jump_action(self.jump_direction)
        return None

    def _ramp_action(self, obs):
        player = obs.get("player", {}) if isinstance(obs, dict) else {}
        # The public object decoder reports all boxes as left/top coordinates.
        # Keep the horizontal controller in that same coordinate frame.
        px = _finite_number(player.get("x", 0))
        py = _finite_number(player.get("y", 0))
        level = min(max(self.ramp_level, 0), len(RAMP_LEVELS) - 1)
        row_y, choices = RAMP_LEVELS[level]
        self.ramp_choice = min(self.ramp_choice, len(choices) - 1)
        target_x = choices[self.ramp_choice]

        # A level is complete only after the controller was trying to climb at
        # its ladder and the player moved materially above that level.  This
        # prevents the FIRE start animation from being mistaken for a climb.
        if self.climb_started:
            # Wait until Mario is near the next walkway, not merely a dozen
            # pixels above the current one.  The first climb reports y=163
            # briefly while the bottom ladder animation is still active.
            next_y = RAMP_LEVELS[level + 1][0] if level < len(RAMP_LEVELS) - 1 else None
            reached_next = next_y is not None and py <= next_y + 8
            if reached_next:
                self.ramp_level += 1
                self.ramp_choice = 0
                self.climb_started = False
                self.climb_start_y = None
                self.climb_last_y = None
                self.climb_ticks = 0
                self.no_progress_ticks = 0
                self._log("ramp_level_advanced", level=self.ramp_level)
                return self._ramp_action(obs)
            if self.climb_last_y is not None and abs(py - self.climb_last_y) >= 2:
                self.no_progress_ticks = 0
            else:
                self.no_progress_ticks += 1
            self.climb_last_y = py
            self.climb_ticks += 1
            if len(choices) > 1 and self.no_progress_ticks >= 14:
                self.ramp_choice = (self.ramp_choice + 1) % len(choices)
                self.climb_started = False
                self.climb_start_y = None
                self.climb_last_y = None
                self.climb_ticks = 0
                self.no_progress_ticks = 0
                return self._ramp_action(obs)
            # Add a small diagonal nudge after a few exact UP quanta.  This
            # recovers from a one-pixel alignment miss without abandoning the
            # phase or pressing FIRE.
            if self.climb_ticks in (4, 5):
                return "upright" if target_x >= px else "upleft"
            return "up"

        if abs(px - target_x) > X_TOLERANCE:
            self.no_progress_ticks = 0
            return "right" if px < target_x else "left"

        # If a walking slope left Mario a little above the nominal row, do not
        # advance the phase; align horizontally first and then test UP.
        if py > row_y + Y_TOLERANCE:
            return "up"
        self.climb_started = True
        self.climb_start_y = py
        self.climb_last_y = py
        self.climb_ticks = 0
        self.no_progress_ticks = 0
        return "up"

    def _nearest_ladder_for_side(self, obs, py, side):
        ladders = obs.get("ladders", []) if isinstance(obs, dict) else []
        candidates = []
        if not isinstance(ladders, list):
            ladders = []
        desired = 108 if side == "right" else 48
        for item in ladders:
            if not _visible(item):
                continue
            x = _finite_number(item.get("x", 0))
            y = _finite_number(item.get("y", 0))
            if abs(y - py) <= 18 or abs(y - (py - 8)) <= 18:
                candidates.append((abs(x - desired) + abs(y - py) * 0.1, x, y))
        if not candidates:
            return desired
        candidates.sort(key=lambda item: item[0])
        return candidates[0][1]

    def _row_target_side(self, row, mask):
        preferred = self.rivet_first_preference if row % 2 == 0 else (
            "right" if self.rivet_first_preference == "left" else "left"
        )
        return _remaining_side(mask, preferred) or preferred

    def _rivet_action(self, obs):
        player = obs.get("player", {}) if isinstance(obs, dict) else {}
        px = _finite_number(player.get("x", 0))
        py = _finite_number(player.get("y", 0))
        rows = _safe_rows(obs)
        row = next((i for i, value in enumerate(rows) if value < 18), 4)
        if row >= 4:
            return "up"

        if row != self.rivet_row:
            self.rivet_row = row
            self.sweep_started = False
            self.climb_started = False
            self.climb_start_y = None
            self.climb_last_y = None
            self.no_progress_ticks = 0
            self.rivet_target_side = self._row_target_side(row, rows[row])
            # The preceding serpentine row ends on the side opposite the
            # first side of this row.  That is also the ladder side to use.
            self.rivet_climb_side = "right" if self.rivet_target_side == "left" else "left"

        mask = rows[row]
        target_side = _remaining_side(mask, self.rivet_target_side)
        if target_side is None:
            return "up"
        self.rivet_target_side = target_side
        target_x = RIVET_X[target_side]
        target_y = RIVET_ROWS_Y[row]

        # If Mario is below the incomplete row, approach the ladder at the
        # side where the preceding serpentine row should have ended.
        if py > target_y + Y_TOLERANCE:
            ladder_x = self._nearest_ladder_for_side(obs, py, self.rivet_climb_side)
            if abs(px - ladder_x) > X_TOLERANCE:
                self.climb_started = False
                return "right" if px < ladder_x else "left"
            self.climb_started = True
            self.no_progress_ticks += 1
            if py > target_y + 4:
                return "up"

        # If Mario is accidentally above the row, descend only toward the
        # row's usable level; this is safer than wandering horizontally.
        if py < target_y - Y_TOLERANCE:
            return "down"

        # Approach the selected rivet from the outside edge.  Once close,
        # sweep across it; the row byte, not a guessed x crossing, confirms
        # that it was actually removed.
        if target_side == "left":
            if px > target_x + 10:
                return "left"
            if px < target_x - 8:
                return "right"
            self.sweep_started = True
            return "right"
        else:
            if px < target_x - 10:
                return "right"
            if px > target_x + 8:
                return "left"
            self.sweep_started = True
            return "left"

    def _update_rivet_change(self, obs):
        rows = _safe_rows(obs)
        if self.last_rows is None:
            self.last_rows = list(rows)
            return
        for i in range(4):
            old, new = self.last_rows[i], rows[i]
            if new != old and new in (0, 6, 12, 18):
                if i == self.rivet_row:
                    # If left was removed, cross right; if right was removed,
                    # cross left.  For a complete row the same direction gets
                    # Mario to the row's finishing/ladder side.
                    if new == 6:
                        self.gap_direction = "right"
                    elif new == 12:
                        self.gap_direction = "left"
                    else:
                        self.gap_direction = "right" if i % 2 == 0 else "left"
                    self.gap_pending = True
                    self._log("rivet_changed", row=i, old=old, new=new,
                              gap_direction=self.gap_direction)
                break
        self.last_rows = list(rows)

    def _fallback_action(self, obs, stage, lives):
        if bool(obs.get("terminated", False)) or bool(obs.get("truncated", False)):
            return "noop"
        if stage == "game_over":
            return "noop"
        if stage == "idle":
            if self.idle_attempts < 3:
                self.idle_attempts += 1
                return "fire"
            return "noop"
        if stage == "transition":
            if lives <= 0:
                return "noop"
            # Release during the death/bonus animation, then use one FIRE
            # attempt.  If the transition did not accept it, retry sparingly.
            if self.transition_age >= 18 and self.transition_attempts < 3:
                if (self.transition_age - 18) % 28 == 0:
                    self.transition_attempts += 1
                    return "fire"
            return "noop"
        if stage == "ramp":
            return self._ramp_action(obs)
        if stage == "rivet":
            return self._rivet_action(obs)
        return "noop"

    def _apply_jev(self, obs, fallback):
        answer = self.jev_result
        if answer is None:
            return fallback
        # Consume a result once.  It is a recognition hint, not a persistent
        # action, and it must belong to the current stage.
        self.jev_result = None
        if self.jev_result_stage != _stage(obs):
            return fallback
        hazard = _hazard_facts(obs)
        if hazard.get("urgency") in ("immediate", "soon") and answer in ("jump_left", "jump_right"):
            # The code-level hazard guard below will choose the direction; use
            # Jev only when it agrees with that guard and has enough support.
            direction = answer.rsplit("_", 1)[-1]
            if direction == self._choose_jump_direction(obs, hazard):
                return self._jump_action(direction)
        # Never let a low-confidence/late answer replace a route action.  The
        # model still contributes a traceable tactical judgment above.
        return fallback

    def _act_one_tick(self, game, obs):
        stage, lives, screens, stage_changed, life_changed = self._sync_observation(obs)
        self._collect_astra()
        self._collect_jev(obs)
        self._update_rivet_change(obs)

        if stage_changed or life_changed or screens != self.last_screens:
            self._start_astra(obs, "stage/life/screen change")

        # Keep the actor alive while Astra is thinking.  A strategic review is
        # periodic, never synchronous with the game clock.
        if self.loop_tick == 1:
            self._start_astra(obs, "initial route review")
        elif self.loop_tick - self.astra_requested_at >= 900:
            self._start_astra(obs, "periodic evidence review")

        fallback = self._fallback_action(obs, stage, lives)

        hazard = _hazard_facts(obs)
        jump = self._hazard_action(obs) if stage in ("ramp", "rivet") else None
        if jump is not None:
            action = jump
        else:
            action = self._apply_jev(obs, fallback)

        # Ask Jev at a leisurely cadence.  The code guard is deliberately
        # usable at every tick, so a late or failed call cannot create stale
        # movement.  A hazard edge is an additional trigger, but only one
        # request may be outstanding.
        if (
            self.loop_tick - self.jev_requested_at >= 8
            or hazard.get("urgency") in ("immediate", "soon")
            or stage_changed
        ):
            self._start_jev(obs, fallback)

        if self.loop_tick % 10 == 0:
            p = obs.get("player", {}) if isinstance(obs, dict) else {}
            self._log(
                "control_snapshot",
                tick=self.loop_tick,
                stage=stage,
                px=_integer(p.get("x")),
                py=_integer(p.get("y")),
                action=action,
                ramp_level=self.ramp_level,
                ramp_choice=self.ramp_choice,
                climb_started=self.climb_started,
                climb_ticks=self.climb_ticks,
                jump_ticks=self.jump_ticks_left,
                hazard=hazard.get("urgency"),
            )

        if stage == "transition" and lives <= 0:
            return "noop", True
        return action, False

    def run(self, game):
        self._log("controller_started", tick_s=getattr(game, "tick_s", None))
        try:
            obs = self.ctx.observe()
        except Exception:
            return
        if not isinstance(obs, dict):
            obs = {}

        # Start is an explicit edge, not a standing FIRE policy.  This avoids
        # turning every future tactical decision into an unnecessary jump.
        stage = _stage(obs)
        if stage == "idle":
            self._sync_observation(obs)
            self._post(game, "fire")
        else:
            self._sync_observation(obs)

        while not self._is_done(game):
            try:
                self.ctx.wait_for_tick()
            except Exception:
                break
            if self._is_done(game):
                break
            try:
                obs = self.ctx.observe()
            except Exception:
                break
            if not isinstance(obs, dict):
                obs = {}
            self.loop_tick += 1
            action, stop = self._act_one_tick(game, obs)
            if stop:
                break
            if self._is_done(game):
                break
            if not self._post(game, action):
                break

        self._log("controller_finished", ticks=self.loop_tick,
                  score=getattr(game, "score", None), action=self.last_action)


class Program:
    def __init__(self, ctx):
        self.ctx = ctx

    def run_episode(self, game) -> None:
        Controller(self.ctx).run(game)


def build(ctx):
    return Program(ctx)
