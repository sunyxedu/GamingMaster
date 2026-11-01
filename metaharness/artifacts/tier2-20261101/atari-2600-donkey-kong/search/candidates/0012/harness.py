"""Closed-loop Donkey Kong controller with calibrated Mario/ladder alignment.

The useful coordinates in the public observation are ladder-box coordinates, while
Mario's x is the left edge of his wider bounding box.  This controller therefore
plans against a player-edge route (ladder_x - 2), and only changes route phase after
an observed vertical climb.  It also synchronizes jump ownership to RAM rather than
guessing that every jump lasts the same number of four-frame quanta.
"""
from __future__ import annotations

import json
import math

from metaharness.models.jev import Choice, Noul


# The observation gives ladder left edges.  Traces show Mario's left edge at
# 106/46/106/... when he is centered over 108/48/108/... ladders.
LADDER_X = (108, 48, 108, 48, 108, 32)
PLAYER_ROUTE_X = tuple(x - 2 for x in LADDER_X)
ROUTE_Y = (176, 148, 120, 92, 64, 40)
RIVET_Y = (176, 148, 120, 92)
RIVET_X = {"left": 46, "right": 106}

_ACTIONS = {
    "noop", "fire", "up", "right", "left", "down", "upright", "upleft",
    "downright", "downleft", "upfire", "rightfire", "leftfire", "downfire",
    "uprightfire", "upleftfire", "downrightfire", "downleftfire",
}


def number(value, default=0.0):
    try:
        result = float(value)
        return result if math.isfinite(result) else default
    except (TypeError, ValueError):
        return default


def integer(value, default=0):
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def stage_of(obs):
    return obs.get("stage", "unknown") if isinstance(obs, dict) else "unknown"


def player_of(obs):
    value = obs.get("player", {}) if isinstance(obs, dict) else {}
    return value if isinstance(value, dict) else {}


def visible(item):
    return isinstance(item, dict) and item.get("visible", True) is not False


def rows_of(obs):
    values = obs.get("rivet_rows", []) if isinstance(obs, dict) else []
    if not isinstance(values, (list, tuple)):
        values = []
    result = []
    for value in list(values)[:4]:
        n = integer(value, 0)
        result.append(n if n in (0, 6, 12, 18) else 0)
    while len(result) < 4:
        result.append(0)
    return result


def action_for(direction, fire=False):
    if direction not in ("left", "right"):
        return "fire" if fire else "noop"
    return direction + ("fire" if fire else "")


def raw_jump_state(obs):
    """The documented $85 byte is index 5 in the 128-byte wrapped snapshot."""
    raw = obs.get("raw_ram", []) if isinstance(obs, dict) else []
    if isinstance(raw, list) and len(raw) > 5:
        return integer(raw[5], 0)
    return 0


def same_row_hazard(obs):
    """Return a compact closing obstacle on Mario's current girder.

    Hazard y and player y are top edges in this adaptor.  A 18-pixel gate keeps
    an object on the neighboring ramp from stealing a jump, while the horizontal
    motion comes directly from the observation and is never estimated by Jev.
    """
    p = player_of(obs)
    px = number(p.get("x"), 45.0)
    py = number(p.get("y"), 176.0)
    best = None
    hazards = obs.get("hazards", []) if isinstance(obs, dict) else []
    if not isinstance(hazards, list):
        hazards = []
    for item in hazards:
        if not visible(item):
            continue
        hx = number(item.get("x"), 0.0)
        hy = number(item.get("y"), 0.0)
        if abs(hy - py) > 18.0:
            continue
        dx = number(item.get("dx"), 0.0)
        delta = hx - px
        distance = abs(delta)
        closing = ((delta < 0 and dx > 0.2) or
                   (delta > 0 and dx < -0.2) or
                   (distance <= 16 and abs(dx) <= 0.2))
        if not closing and distance > 18:
            continue
        if distance <= 22:
            urgency = "immediate"
        elif closing and distance <= 36:
            urgency = "soon"
        elif closing and distance <= 72:
            urgency = "watch"
        else:
            urgency = "near"
        fact = {
            "present": True,
            "side": "left" if delta < 0 else "right",
            "motion": "left" if dx < -0.2 else "right" if dx > 0.2 else "still",
            "distance_bucket": (
                "overlapping" if distance <= 16 else
                "near" if distance <= 32 else
                "approaching" if distance <= 56 else "far"
            ),
            "closing": bool(closing),
            "urgency": urgency,
            "distance": round(distance, 1),
            "slot": integer(item.get("slot"), -1),
        }
        rank = (0 if urgency == "immediate" else
                1 if urgency == "soon" else
                2 if urgency == "watch" else 3, distance)
        if best is None or rank < best[0]:
            best = (rank, fact)
    if best is None:
        return {
            "present": False, "side": "none", "motion": "none",
            "distance_bucket": "none", "closing": False, "urgency": "clear",
            "distance": 999.0, "slot": -1,
        }
    return best[1]


def remaining_side(mask, preferred):
    """Interpret the normal 0/6/12/18 row status without arithmetic by Jev."""
    if mask == 18:
        return None
    if mask == 0:
        return preferred
    if mask == 6:
        return "right" if preferred == "left" else "left"
    if mask == 12:
        return "left" if preferred == "right" else "right"
    return preferred


class Controller:
    def __init__(self, ctx):
        self.ctx = ctx
        self.ticks = 0
        self.last_action = "noop"
        self.last_stage = None
        self.last_lives = None
        self.last_screens = 0

        # Ramp transaction: phase is the walkway currently occupied, not an
        # optimistic action count.  climb_open owns UP until measured progress.
        self.phase = 0
        self.climb_open = False
        self.climb_ticks = 0
        self.climb_start_y = None
        self.climb_best_y = None
        self.no_progress = 0
        self.alignment_bias = 0

        # Jump transaction: FIRE is posted until the RAM jump state clears, then
        # a short steering tail lets Mario land before route control resumes.
        self.jump_direction = None
        self.jump_tail = 0
        self.jump_lock = 0
        self.jump_guard = 0
        self.jump_seen_active = False
        self.post_jump_up = 0
        self.jump_starts = 0

        # Death and idle are edge-triggered, never a continuously held FIRE.
        self.transition_age = 0
        self.restart_edges = 0
        self.idle_edges = 0

        # Rivet state is observational and is reset after every life/screen.
        self.rivet_row = 0
        self.rivet_preference = "left"
        self.previous_rows = None
        self.gap_direction = None

        # Jev/Astra are advisory and cannot block the actor.  They are normally
        # dormant on the ramp, where the finite route is more reliable than a
        # 150 ms answer.  They become useful only if the rivet screen is reached.
        self.jev_future = None
        self.jev_requested = -10000
        self.jev_hint = None
        self.astra_future = None
        self.astra_started = False
        self.astra_hint = None

    def log(self, event, **fields):
        try:
            self.ctx.log(event, **fields)
        except Exception:
            pass

    def is_done(self, game):
        try:
            if bool(self.ctx.done):
                return True
        except Exception:
            return True
        try:
            return bool(game.done)
        except Exception:
            return False

    def post(self, game, action):
        if not isinstance(action, str) or action not in _ACTIONS:
            action = "noop"
        try:
            self.ctx.act(action)
            self.last_action = action
            return True
        except Exception as exc:
            self.log("act_failed", action=action, error=type(exc).__name__)
            return False

    def reset_play_state(self, stage):
        self.phase = 0
        self.climb_open = False
        self.climb_ticks = 0
        self.climb_start_y = None
        self.climb_best_y = None
        self.no_progress = 0
        self.alignment_bias = 0
        self.clear_jump()
        self.gap_direction = None
        if stage == "rivet":
            self.rivet_row = 0
            self.rivet_preference = self.astra_hint if self.astra_hint in ("left", "right") else "left"
            self.previous_rows = None

    def clear_jump(self):
        self.jump_direction = None
        self.jump_tail = 0
        self.jump_lock = 0
        self.jump_guard = 0
        self.jump_seen_active = False
        self.post_jump_up = 0

    def sync(self, obs):
        stage = stage_of(obs)
        lives = integer(obs.get("lives_remaining"), 0) if isinstance(obs, dict) else 0
        screens = integer(obs.get("screens_completed"), 0) if isinstance(obs, dict) else 0
        changed = stage != self.last_stage
        life_changed = self.last_lives is not None and lives != self.last_lives
        screen_changed = screens != self.last_screens
        if changed or life_changed:
            self.log("state_boundary", old_stage=self.last_stage, stage=stage,
                     old_lives=self.last_lives, lives=lives, phase=self.phase)
            if stage in ("ramp", "rivet"):
                self.reset_play_state(stage)
        if changed:
            self.transition_age = 0
            self.restart_edges = 0
            self.idle_edges = 0
        if stage == "transition":
            self.transition_age += 1
        else:
            self.transition_age = 0
        self.last_stage = stage
        self.last_lives = lives
        self.last_screens = screens
        return stage, lives, screens, changed, life_changed, screen_changed

    # ---------- Jump transaction ----------

    def begin_jump(self, obs, direction, reason):
        if direction not in ("left", "right"):
            direction = "right"
        self.jump_direction = direction
        self.jump_tail = 0
        self.jump_lock = 20
        # Allow the emulator a few quanta to expose the RAM edge, then use
        # that edge to start the landing tail instead of assuming eight ticks.
        self.jump_guard = 10
        self.jump_seen_active = False
        self.jump_starts += 1
        self.log("jump_begin", tick=self.ticks, direction=direction, reason=reason,
                 hazard=same_row_hazard(obs), phase=self.phase)
        return action_for(direction, True)

    def jump_action(self, obs):
        if self.jump_direction is not None:
            ram_state = raw_jump_state(obs)
            if ram_state > 0:
                self.jump_seen_active = True
                self.jump_guard = 10
                return action_for(self.jump_direction, True)
            if self.jump_seen_active:
                # The first zero after an active jump is the landing edge.
                if self.jump_tail <= 0:
                    self.jump_tail = 5
                self.jump_tail -= 1
                if self.jump_tail <= 0:
                    self.jump_direction = None
                    # 0009's successful crossing gives the cartridge several
                    # UP quanta before the next horizontal target.  This is a
                    # bounded recovery, not a new route phase.
                    self.post_jump_up = 8
                    return "up"
                # After meeting the obstacle in the air, steer away from its
                # side during the landing tail.
                landing = "left" if self.jump_direction == "right" else "right"
                return action_for(landing, False)
            if self.jump_guard > 0:
                # A post can be observed one quantum before the cartridge's
                # jump byte changes.  Keep FIRE briefly rather than cancelling
                # the jump on that first zero.
                self.jump_guard -= 1
                return action_for(self.jump_direction, True)
            self.jump_direction = None
            return None
        if self.post_jump_up > 0:
            self.post_jump_up -= 1
            if self.jump_lock > 0:
                self.jump_lock -= 1
            return "up"
        if self.jump_lock > 0:
            self.jump_lock -= 1
        return None

    def choose_jump_direction(self, obs, hazard):
        motion = hazard.get("motion")
        if motion == "right":
            return "left"
        if motion == "left":
            return "right"
        # A still sprite is handled toward the open side of the current route.
        px = number(player_of(obs).get("x"), 45)
        target = PLAYER_ROUTE_X[min(max(self.phase, 0), len(PLAYER_ROUTE_X) - 1)]
        if px < target:
            return "right"
        if px > target:
            return "left"
        return "right" if hazard.get("side") == "left" else "left"

    def emergency_jump(self, obs, stage):
        if stage not in ("ramp", "rivet"):
            return None
        if self.jump_direction is not None or self.jump_lock > 0:
            return None
        hazard = same_row_hazard(obs)
        if hazard.get("urgency") != "immediate":
            return None
        # While Mario is visibly climbing, UP is the safer escape from a hazard
        # on the old girder.  Only interrupt before vertical progress or when an
        # overlapping sprite leaves no climb escape.
        if self.climb_open and self.climb_best_y is not None:
            current_y = number(player_of(obs).get("y"), 176)
            made_height = self.climb_start_y is not None and current_y < self.climb_start_y - 4
            if made_height and hazard.get("urgency") != "immediate":
                return None
        direction = self.choose_jump_direction(obs, hazard)
        return self.begin_jump(obs, direction, "hazard")

    # ---------- Ramp route ----------

    def route_target(self):
        return PLAYER_ROUTE_X[min(max(self.phase, 0), len(PLAYER_ROUTE_X) - 1)]

    def open_climb(self, y):
        self.climb_open = True
        self.climb_ticks = 0
        self.climb_start_y = y
        self.climb_best_y = y
        self.no_progress = 0

    def ramp_action(self, obs):
        p = player_of(obs)
        x = number(p.get("x"), 45)
        y = number(p.get("y"), 176)
        dy = number(p.get("dy"), 0)
        phase = min(max(self.phase, 0), len(PLAYER_ROUTE_X) - 1)
        target = PLAYER_ROUTE_X[phase] + self.alignment_bias

        if self.climb_open:
            # A jump can move Mario away from the rung.  Reacquire the calibrated
            # edge rather than holding UP in empty space.
            if abs(x - target) > 2:
                self.climb_open = False
                self.climb_ticks = 0
                self.climb_start_y = None
                self.climb_best_y = None
                self.no_progress = 0
                return self.ramp_action(obs)
            self.climb_ticks += 1
            if self.climb_best_y is None or y < self.climb_best_y - 1:
                self.climb_best_y = y
                self.no_progress = 0
            elif abs(dy) <= 1:
                self.no_progress += 1
            if phase + 1 < len(ROUTE_Y):
                next_y = ROUTE_Y[phase + 1]
                reached = y <= next_y + 6
                enough_motion = self.climb_start_y is not None and self.climb_start_y - y >= 8
                if reached and enough_motion and self.climb_ticks >= 5:
                    self.phase += 1
                    self.climb_open = False
                    self.climb_ticks = 0
                    self.climb_start_y = None
                    self.climb_best_y = None
                    self.no_progress = 0
                    self.alignment_bias = 0
                    self.log("ramp_phase", phase=self.phase, x=round(x), y=round(y))
                    return self.ramp_action(obs)
            # Never switch to the other ladder merely because a few frames have
            # passed.  The calibrated x-1 retry is only a recovery after a full
            # no-progress window and preserves the transaction's purpose.
            if self.no_progress >= 28:
                self.alignment_bias = -1 if self.alignment_bias == 0 else 0
                self.climb_open = False
                self.climb_ticks = 0
                self.climb_start_y = None
                self.climb_best_y = None
                self.no_progress = 0
                self.log("alignment_retry", phase=phase, bias=self.alignment_bias,
                         x=round(x), y=round(y))
                return self.ramp_action(obs)
            return "up"

        # Calibrated player-edge alignment is deliberately strict: x=49 at a
        # ladder box beginning at 48 was the repeated stall in prior traces.
        if x < target - 1:
            return "right"
        if x > target + 1:
            return "left"
        self.open_climb(y)
        return "up"

    # ---------- Rivet route ----------

    def nearest_route_ladder(self, obs, preferred):
        """Use visible ladder boxes only to choose a side, then apply edge offset."""
        desired_box_x = 108 if preferred == "right" else 48
        ladders = obs.get("ladders", []) if isinstance(obs, dict) else []
        candidates = []
        if isinstance(ladders, list):
            py = number(player_of(obs).get("y"), 176)
            for item in ladders:
                if not visible(item):
                    continue
                lx = number(item.get("x"), 0)
                ly = number(item.get("y"), 0)
                if abs(ly - py) <= 22 or abs(ly - (py - 8)) <= 22:
                    candidates.append((abs(lx - desired_box_x), lx - 2))
        if candidates:
            candidates.sort()
            return candidates[0][1]
        return desired_box_x - 2

    def rivet_action(self, obs):
        rows = rows_of(obs)
        if self.previous_rows is not None:
            for index, (old, new) in enumerate(zip(self.previous_rows, rows)):
                if index == self.rivet_row and old != new and new in (0, 6, 12, 18):
                    # Continue in the direction of the sweep; a removed rivet is
                    # a gap, so a normal walk across it is not allowed.
                    self.gap_direction = "right" if self.rivet_preference == "left" else "left"
                    self.log("rivet_change", row=index, old=old, new=new,
                             gap=self.gap_direction)
        self.previous_rows = list(rows)
        row = next((i for i, value in enumerate(rows) if value < 18), 4)
        if row >= 4:
            return "up"
        if row != self.rivet_row:
            self.rivet_row = row
            self.rivet_preference = (
                self.astra_hint if self.astra_hint in ("left", "right") and row == 0
                else ("left" if row % 2 == 0 else "right")
            )
            self.gap_direction = None
        if self.gap_direction in ("left", "right"):
            direction = self.gap_direction
            self.gap_direction = None
            if self.jump_direction is None and self.jump_lock == 0:
                return self.begin_jump(obs, direction, "rivet_gap")

        p = player_of(obs)
        x = number(p.get("x"), 45)
        y = number(p.get("y"), 176)
        mask = rows[row]
        side = remaining_side(mask, self.rivet_preference)
        if side is None:
            ladder_side = "right" if row % 2 == 0 else "left"
            target = self.nearest_route_ladder(obs, ladder_side)
            if x < target - 1:
                return "right"
            if x > target + 1:
                return "left"
            return "up"
        target_y = RIVET_Y[row]
        if y > target_y + 9:
            ladder_side = "right" if row % 2 == 0 else "left"
            target = self.nearest_route_ladder(obs, ladder_side)
            if x < target - 1:
                return "right"
            if x > target + 1:
                return "left"
            return "up"
        target_x = RIVET_X[side]
        if x < target_x - 8:
            return "right"
        if x > target_x + 8:
            return "left"
        return "right" if side == "left" else "left"

    # ---------- Advisory model bridge ----------

    def state_for_models(self, obs):
        p = player_of(obs)
        hazard = same_row_hazard(obs)
        return {
            "game": "Atari 2600 Donkey Kong NTSC retail port",
            "stage": stage_of(obs),
            "player": {
                "horizontal_band": "left" if number(p.get("x"), 45) < 52 else
                                   "right" if number(p.get("x"), 45) > 100 else "middle",
                "vertical_band": "bottom" if number(p.get("y"), 176) > 160 else
                                 "lower" if number(p.get("y"), 176) > 132 else
                                 "middle" if number(p.get("y"), 176) > 78 else "top",
                "motion": "up" if number(p.get("dy"), 0) < -1 else
                          "down" if number(p.get("dy"), 0) > 1 else "steady",
                "jump_committed": self.jump_direction is not None,
            },
            "route": {"ramp_phase": self.phase, "climb_open": self.climb_open,
                      "rivet_row": self.rivet_row, "rivet_rows": rows_of(obs)},
            "hazard": {key: hazard.get(key) for key in
                       ("present", "side", "motion", "distance_bucket", "closing", "urgency")},
            "progress": {"lives_remaining": integer(obs.get("lives_remaining"), 0),
                         "screens_completed": integer(obs.get("screens_completed"), 0),
                         "rivets_removed": integer(obs.get("rivets_removed"), 0)},
            "last_action": self.last_action,
        }

    def start_jev(self, obs):
        if self.jev_future is not None or stage_of(obs) != "rivet":
            return
        interval = 5 if same_row_hazard(obs).get("urgency") in ("watch", "soon", "immediate") else 14
        if self.ticks - self.jev_requested < interval:
            return
        try:
            if self.ctx.budget.remaining().get("system1_calls", 0) <= 0:
                return
        except Exception:
            return
        state = self.state_for_models(obs)
        questions = {
            "sweep": Choice(
                "Choose the legal rivet-route label supported by `route.rivet_rows` and the current row. Do not count or calculate; choose wait if a jump or climb is active.",
                {
                    "left": "Sweep toward the left remaining rivet on the current row.",
                    "right": "Sweep toward the right remaining rivet on the current row.",
                    "wait": "Keep the deterministic controller's current climb/jump transaction.",
                },
            ),
            "danger": Noul(
                "Is `hazard` an immediate or soon closing obstacle on Mario's current row?",
                {"true": "The named hazard is soon/immediate and deserves a jump guard.",
                 "false": "No soon/immediate closing current-row hazard is shown."},
            ),
        }
        self.jev_requested = self.ticks
        try:
            self.jev_future = self.ctx.spawn(
                lambda: self.ctx.decide(state, questions, timeout_s=0.65)
            )
        except Exception as exc:
            self.log("jev_start_failed", error=type(exc).__name__)
            self.jev_future = None

    def collect_jev(self):
        future = self.jev_future
        if future is None or not future.done():
            return
        self.jev_future = None
        try:
            answers = future.result()
            answer = answers.get("sweep") if isinstance(answers, dict) else None
            value = getattr(answer, "value", None)
            probabilities = getattr(answer, "probabilities", None)
            probability = None
            if isinstance(probabilities, dict):
                probability = number(probabilities.get(value), 0.0)
            accepted = value in ("left", "right") and (
                probability is None or probability >= 0.55
            )
            if accepted:
                self.jev_hint = value
                self.log("jev_hint", side=value, probability=probability)
        except Exception as exc:
            self.log("jev_failed", error=type(exc).__name__)

    def start_astra(self, obs, reason):
        if self.astra_started or self.astra_future is not None or stage_of(obs) != "rivet":
            return
        try:
            if self.ctx.budget.remaining().get("system2_calls", 0) <= 0:
                return
        except Exception:
            return
        self.astra_started = True
        state = self.state_for_models(obs)
        prompt = (
            "You are a slow route reviewer for the Atari 2600 Donkey Kong NTSC "
            "retail port. The actor owns timing, calibrated ladder alignment, "
            "hazard jumps, raw rivet bookkeeping, and lifecycle. Review only the "
            "future rivet sweep. Return exactly JSON {\"first_side\":\"left\"} "
            "or {\"first_side\":\"right\"}; never return an action sequence. "
            "Use the supplied row bytes and route facts, not arcade rules. State="
            + json.dumps(state, separators=(",", ":"), allow_nan=False)
        )
        try:
            self.astra_future = self.ctx.spawn(
                lambda: self.ctx.think(prompt, timeout_s=80.0)
            )
        except Exception as exc:
            self.astra_future = None
            self.log("astra_start_failed", error=type(exc).__name__)

    def collect_astra(self):
        future = self.astra_future
        if future is None or not future.done():
            return
        self.astra_future = None
        try:
            text = getattr(future.result(), "text", "")
            if not isinstance(text, str):
                return
            parsed = None
            try:
                parsed = json.loads(text)
            except (TypeError, ValueError):
                left, right = text.find("{"), text.rfind("}")
                if left >= 0 and right > left:
                    try:
                        parsed = json.loads(text[left:right + 1])
                    except (TypeError, ValueError):
                        parsed = None
            side = parsed.get("first_side") if isinstance(parsed, dict) else None
            if side in ("left", "right"):
                self.astra_hint = side
                self.log("astra_hint", side=side)
        except Exception as exc:
            self.log("astra_failed", error=type(exc).__name__)

    # ---------- Top-level policy ----------

    def choose(self, obs):
        stage, lives, screens, stage_changed, life_changed, screen_changed = self.sync(obs)
        self.collect_jev()
        self.collect_astra()
        if stage == "idle":
            if self.idle_edges < 2:
                self.idle_edges += 1
                return "fire"
            return "noop"
        if stage == "game_over" or bool(obs.get("terminated", False)):
            return "noop"
        if stage == "transition":
            if lives <= 0:
                return "noop"
            # FIRE is a sparse edge after the death animation, not a hold.
            if (self.transition_age >= 18 and self.restart_edges < 4 and
                    (self.transition_age - 18) % 28 == 0):
                self.restart_edges += 1
                return "fire"
            return "noop"
        if stage not in ("ramp", "rivet"):
            return "noop"

        if stage_changed and stage == "rivet":
            self.start_astra(obs, "ramp completed")
        elif stage == "rivet" and self.astra_future is None and not self.astra_started:
            self.start_astra(obs, "rivet route active")

        # A committed jump owns the action until the RAM edge and tail finish.
        ongoing = self.jump_action(obs)
        if ongoing is not None:
            return ongoing
        emergency = self.emergency_jump(obs, stage)
        if emergency is not None:
            return emergency

        action = self.ramp_action(obs) if stage == "ramp" else self.rivet_action(obs)
        self.start_jev(obs)
        if stage == "rivet" and self.jev_hint in ("left", "right"):
            # Jev may only select the already legal current-row side.  It cannot
            # suppress a climb, gap jump, or the deterministic hazard guard.
            row = min(max(self.rivet_row, 0), 3)
            side = remaining_side(rows_of(obs)[row], self.jev_hint)
            if side == self.jev_hint and not self.climb_open:
                self.rivet_preference = self.jev_hint
            self.jev_hint = None
        if stage_changed or life_changed or screen_changed:
            self.log("decision_boundary", stage=stage, phase=self.phase, action=action,
                     lives=lives, screens=screens)
        if self.ticks % 20 == 0:
            p = player_of(obs)
            self.log("snapshot", tick=self.ticks, stage=stage, phase=self.phase,
                     x=round(number(p.get("x"), 0)), y=round(number(p.get("y"), 0)),
                     action=action, climb=self.climb_open, climb_ticks=self.climb_ticks,
                     jump=self.jump_direction, raw_jump=raw_jump_state(obs),
                     hazard=same_row_hazard(obs).get("urgency"), lives=lives,
                     score=number(obs.get("score_total"), 0))
        return action

    def run(self, game):
        try:
            obs = self.ctx.observe()
        except Exception:
            return
        if not isinstance(obs, dict):
            obs = {}
        self.last_stage = stage_of(obs)
        self.last_lives = integer(obs.get("lives_remaining"), 0)
        self.last_screens = integer(obs.get("screens_completed"), 0)
        if self.last_stage == "idle":
            if not self.post(game, "fire"):
                return
        while not self.is_done(game):
            try:
                self.ctx.wait_for_tick()
            except Exception:
                break
            if self.is_done(game):
                break
            try:
                obs = self.ctx.observe()
            except Exception:
                break
            if not isinstance(obs, dict):
                obs = {}
            self.ticks += 1
            action = self.choose(obs)
            if not self.post(game, action):
                break
        self.log("finished", ticks=self.ticks, score=getattr(game, "score", None),
                 phase=self.phase, jumps=self.jump_starts)


class Program:
    def __init__(self, ctx):
        self.ctx = ctx

    def run_episode(self, game) -> None:
        Controller(self.ctx).run(game)


def build(ctx):
    return Program(ctx)
