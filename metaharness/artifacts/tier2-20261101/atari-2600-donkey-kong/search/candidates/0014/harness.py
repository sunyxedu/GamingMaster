"""Adaptive ladder-connector controller for Atari 2600 Donkey Kong.

The previous best controller knew the right vertical sequence, but treated the
left edge of each ladder as a single magic x coordinate.  This controller makes
that assumption observable: every ladder is a small transaction with a
sprite-to-ladder alignment probe, progress confirmation, and an alternate
connector.  Hazards still use a committed, direction-latched jump.  Jev is
asked only when the cartridge has failed to accept a climb; it can arbitrate
between already-known recovery probes without ever blocking the actor.
"""
from __future__ import annotations

import math

from metaharness.models.jev import Choice


# The public ladder slots, grouped by the next upward walkway.  The first
# element of a group is the route prior; the second is a legal alternate at a
# split.  Slot order is from the adaptor's fixed ten-ladder observation.
RAMP_ROWS = (176, 148, 120, 92, 64, 40)
RAMP_LADDER_GROUPS = (
    (0,),       # bottom: x=108
    (1, 2),     # row 148: x=48, x=80
    (4, 3),     # row 120: x=108, x=88
    (5, 6),     # row 92: x=48, x=68
    (7,),       # row 64: x=108
    (9, 8),     # top: x=32, x=76
)
# Mario's left coordinate is about two pixels left of a ladder's reported x.
# Three nearby probes handle the integer collision window without guessing
# that visual overlap means an accepted climb.
ALIGNMENT_VARIANTS = (0, -1, 1)

RAMP_FALLBACK_X = (
    (106,),
    (46, 78),
    (106, 86),
    (46, 66),
    (106,),
    (30, 74),
)

RIVET_Y = (176, 148, 120, 92)
RIVET_X = {"left": 48, "right": 108}

JUMP_QUANTA = 8
LANDING_QUANTA = 10
HAZARD_LOCK_QUANTA = 36
NO_PROGRESS_QUANTA = 12
SETTLE_QUANTA = 4

_ACTIONS = {
    "noop", "fire", "up", "right", "left", "down", "upright", "upleft",
    "downright", "downleft", "upfire", "rightfire", "leftfire",
    "downfire", "uprightfire", "upleftfire", "downrightfire", "downleftfire",
}


def _num(value, default=0.0):
    try:
        result = float(value)
        if math.isfinite(result):
            return result
    except (TypeError, ValueError):
        pass
    return default


def _int(value, default=0):
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _stage(obs):
    if isinstance(obs, dict) and isinstance(obs.get("stage"), str):
        return obs["stage"]
    return "unknown"


def _player(obs):
    value = obs.get("player", {}) if isinstance(obs, dict) else {}
    return value if isinstance(value, dict) else {}


def _visible(value):
    return isinstance(value, dict) and value.get("visible", True) is not False


def _rows(obs):
    values = obs.get("rivet_rows", []) if isinstance(obs, dict) else []
    if not isinstance(values, (list, tuple)):
        values = []
    answer = []
    for value in list(values)[:4]:
        number = _int(value, 0)
        answer.append(number if number in (0, 6, 12, 18) else 0)
    while len(answer) < 4:
        answer.append(0)
    return answer


def _motion(value):
    dx = _num(value.get("dx", 0)) if isinstance(value, dict) else 0.0
    dy = _num(value.get("dy", 0)) if isinstance(value, dict) else 0.0
    if abs(dx) <= 1 and abs(dy) <= 1:
        return "still"
    if abs(dx) >= abs(dy):
        return "right" if dx > 0 else "left"
    return "down" if dy > 0 else "up"


def _hazard_facts(obs):
    """Normalize the nearest relevant closing obstacle without asking Jev to do math."""
    player = _player(obs)
    px = _num(player.get("x", 45))
    py = _num(player.get("y", 176))
    nearest = None
    hazards = obs.get("hazards", []) if isinstance(obs, dict) else []
    if not isinstance(hazards, list):
        hazards = []
    for raw in hazards:
        if not _visible(raw):
            continue
        hx = _num(raw.get("x", 0))
        hy = _num(raw.get("y", 0))
        # The public decoder reports an obstacle on a nearby slope before its
        # sprite is exactly level.  Keep the same broad early-warning gate as
        # the best controller; the distance/velocity test below is stricter.
        if abs(hy - py) > 40:
            continue
        dx = _num(raw.get("dx", 0))
        delta = hx - px
        distance = abs(delta)
        closing = (
            (delta < 0 and dx > 0.2)
            or (delta > 0 and dx < -0.2)
            or (distance <= 15 and abs(dx) <= 0.2)
        )
        if not closing:
            continue
        urgency = "immediate" if distance <= 38 else "soon" if distance <= 66 else "watch"
        item = {
            "present": True,
            "side": "left" if delta < 0 else "right",
            "motion": "left" if dx < -0.2 else "right" if dx > 0.2 else "still",
            "distance_bucket": (
                "overlapping" if distance <= 18 else
                "near" if distance <= 38 else
                "approaching" if distance <= 66 else "far"
            ),
            "closing": True,
            "urgency": urgency,
        }
        rank = (0 if urgency == "immediate" else 1 if urgency == "soon" else 2, distance)
        if nearest is None or rank < nearest[0]:
            nearest = (rank, item)
    if nearest is None:
        return {
            "present": False,
            "side": "none",
            "motion": "none",
            "distance_bucket": "none",
            "closing": False,
            "urgency": "clear",
        }
    return nearest[1]


def _action(direction, fire=False):
    if direction not in ("left", "right"):
        return "fire" if fire else "noop"
    return direction + ("fire" if fire else "")


def _raw_jump_active(obs):
    raw = obs.get("raw_ram", []) if isinstance(obs, dict) else []
    return isinstance(raw, list) and len(raw) > 5 and _int(raw[5], 0) > 0


def _row_side(row, first):
    return first if row % 2 == 0 else ("right" if first == "left" else "left")


class Controller:
    def __init__(self, ctx):
        self.ctx = ctx
        self.tick = 0
        self.last_action = "noop"
        self.last_stage = None
        self.last_lives = None
        self.last_screens = 0
        self.current_obs = {}

        # Ramp connector state.  level names the current walkway, while
        # candidate/variant names the exact ladder alignment currently being
        # tested.  A level advances only after observed vertical progress.
        self.level = 0
        self.candidate = 0
        self.variant = 0
        self.climb_active = False
        self.climb_age = 0
        self.climb_start_y = None
        self.progress_seen = False
        self.settle_age = 0
        self.last_y = None
        self.route_failures = 0
        self.failed_key = None

        # Committed hazard/gap crossing.  Once FIRE begins, route planning is
        # suspended until the jump and landing tail have elapsed.
        self.jump_direction = "right"
        self.jump_left = 0
        self.landing_left = 0
        self.hazard_lock = 0
        self.jump_cooldown = 0
        self.last_walk_direction = "right"

        # Rivet bookkeeping is entirely raw-byte driven.
        self.rivet_first = "left"
        self.rivet_row = 0
        self.rivet_target = "left"
        self.rivet_previous = None
        self.gap_direction = None
        self.rivet_climb_side = "right"
        self.rivet_climbing = False

        # Lifecycle is code-owned.  A transition gets a sparse FIRE edge; a
        # final-life transition returns rather than spending 900 seconds on
        # noop ticks.
        self.transition_age = 0
        self.restart_attempts = 0
        self.idle_attempts = 0

        # Jev is an event-level arbiter, not a per-tick pilot.  It is invoked
        # only after a climb probe has demonstrably made no vertical progress.
        self.jev_future = None
        self.jev_hint = None
        self.jev_key = None

    def log(self, event, **fields):
        try:
            self.ctx.log(event, **fields)
        except Exception:
            pass

    def done(self, game):
        try:
            if bool(self.ctx.done) or bool(game.done):
                return True
        except Exception:
            return True
        return False

    def post(self, game, action):
        if not isinstance(action, str) or action not in _ACTIONS:
            action = "noop"
        try:
            names = set(game.action_spec.names)
            if action not in names:
                action = game.action_spec.noop if game.action_spec.noop in names else "noop"
        except Exception:
            pass
        try:
            self.ctx.act(action)
            self.last_action = action
            return True
        except Exception as exc:
            self.log("action_failed", action=action, error=type(exc).__name__)
            return False

    def reset_route(self, stage):
        self.level = 0
        self.candidate = 0
        self.variant = 0
        self.climb_active = False
        self.climb_age = 0
        self.climb_start_y = None
        self.progress_seen = False
        self.settle_age = 0
        self.last_y = None
        self.route_failures = 0
        self.failed_key = None
        self.jump_left = 0
        self.landing_left = 0
        self.hazard_lock = 0
        self.jump_cooldown = 0
        self.gap_direction = None
        self.rivet_climbing = False
        if stage == "rivet":
            self.rivet_row = 0
            self.rivet_target = self.rivet_first
            self.rivet_previous = None
            self.rivet_climb_side = "right" if self.rivet_first == "left" else "left"
        self.jev_hint = None
        self.jev_key = None

    def sync(self, obs):
        stage = _stage(obs)
        lives = _int(obs.get("lives_remaining", 0), 0) if isinstance(obs, dict) else 0
        screens = _int(obs.get("screens_completed", 0), 0) if isinstance(obs, dict) else 0
        stage_changed = stage != self.last_stage
        life_changed = self.last_lives is not None and lives != self.last_lives
        screen_changed = screens != self.last_screens
        if stage_changed:
            self.log("stage_change", old=self.last_stage, new=stage,
                     lives=lives, screens=screens)
            self.transition_age = 0
            self.restart_attempts = 0
            self.idle_attempts = 0
            if stage in ("ramp", "rivet"):
                self.reset_route(stage)
        if life_changed:
            self.log("life_change", old=self.last_lives, new=lives)
            if stage in ("ramp", "rivet"):
                self.reset_route(stage)
        if stage == "transition":
            self.transition_age += 1
        else:
            self.transition_age = 0
        self.last_stage = stage
        self.last_lives = lives
        self.last_screens = screens
        return stage, lives, screens, stage_changed, life_changed, screen_changed

    def ladder_targets(self, obs, level):
        """Return player-left-coordinate targets for the current ladder group."""
        group = RAMP_LADDER_GROUPS[min(max(level, 0), len(RAMP_LADDER_GROUPS) - 1)]
        ladders = obs.get("ladders", []) if isinstance(obs, dict) else []
        by_slot = {}
        if isinstance(ladders, list):
            for ladder in ladders:
                if _visible(ladder):
                    by_slot[_int(ladder.get("slot", -1), -1)] = _num(ladder.get("x", 0))
        fallback = RAMP_FALLBACK_X[min(max(level, 0), len(RAMP_FALLBACK_X) - 1)]
        result = []
        for index, slot in enumerate(group):
            if slot in by_slot:
                # Observation x is the ladder's left edge; Mario's accepted
                # alignment is its left edge minus roughly two pixels.
                base = int(round(by_slot[slot])) - 2
            else:
                base = fallback[min(index, len(fallback) - 1)]
            result.append((base, slot))
        return result

    def active_target(self, obs):
        targets = self.ladder_targets(obs, self.level)
        if not targets:
            return 46, -1
        index = min(self.candidate, len(targets) - 1)
        base, slot = targets[index]
        delta = ALIGNMENT_VARIANTS[min(self.variant, len(ALIGNMENT_VARIANTS) - 1)]
        return base + delta, slot

    def model_state(self, obs):
        player = _player(obs)
        hazard = _hazard_facts(obs)
        targets = self.ladder_targets(obs, self.level)
        return {
            "game": "Atari 2600 Donkey Kong NTSC retail port",
            "stage": _stage(obs),
            "player": {
                "horizontal_band": "left" if _num(player.get("x", 0)) < 52 else "right" if _num(player.get("x", 0)) > 100 else "middle",
                "vertical_band": "bottom" if _num(player.get("y", 176)) > 160 else "upper" if _num(player.get("y", 176)) < 135 else "middle",
                "vertical_motion": _motion(player),
            },
            "climb_probe": {
                "level": int(self.level),
                "candidate": int(self.candidate),
                "candidate_count": int(len(targets)),
                "alignment_variant": int(self.variant),
                "attempt_age": int(self.climb_age),
                "progress_seen": bool(self.progress_seen),
                "next_row": int(min(self.level + 1, len(RAMP_ROWS) - 1)),
            },
            "hazard": {key: hazard.get(key) for key in ("present", "side", "motion", "distance_bucket", "urgency")},
            "rivet_rows": _rows(obs),
            "last_action": self.last_action,
            "lives_remaining": int(_int(obs.get("lives_remaining", 0), 0)),
            "screens_completed": int(_int(obs.get("screens_completed", 0), 0)),
        }

    def request_jev(self, obs):
        if self.jev_future is not None:
            return
        if _stage(obs) != "ramp" or not self.climb_active:
            return
        key = (self.level, self.candidate, self.variant, self.last_lives)
        if self.jev_key == key:
            return
        try:
            if bool(self.ctx.done) or self.ctx.budget.remaining().get("system1_calls", 0) <= 0:
                return
        except Exception:
            return
        state = self.model_state(obs)
        questions = {
            "recovery": Choice(
                "A climb input has been held but the derived facts show no vertical progress. "
                "Choose one already-described recovery operation; never invent an action or "
                "calculate from coordinates.",
                {
                    "retry_alignment": "Keep the same ladder but retry a nearby sprite-to-ladder alignment.",
                    "alternate_ladder": "Abandon this ladder candidate and probe the other fixed ladder on this walkway.",
                    "continue_climb": "The attempt is plausibly still settling; keep UP for a short bounded continuation.",
                },
            )
        }
        self.jev_key = key

        def work():
            return self.ctx.decide(state, questions, timeout_s=0.7)

        try:
            self.jev_future = self.ctx.spawn(work)
        except Exception as exc:
            self.jev_future = None
            self.log("jev_spawn_failed", error=type(exc).__name__)

    def collect_jev(self):
        future = self.jev_future
        if future is None or not future.done():
            return
        self.jev_future = None
        try:
            answers = future.result()
            answer = answers.get("recovery") if isinstance(answers, dict) else None
            value = getattr(answer, "value", None) if answer is not None else None
            probabilities = getattr(answer, "probabilities", None)
            probability = probabilities.get(value, 0.0) if isinstance(probabilities, dict) else 1.0
            if value in ("retry_alignment", "alternate_ladder", "continue_climb") and _num(probability, 0.0) >= 0.55:
                self.jev_hint = value
                self.log("jev_recovery", choice=value, probability=round(float(probability), 3))
        except Exception as exc:
            # Timeouts, arbitrary stand-in text, and cancelled futures are
            # ordinary for this advisory branch; the deterministic recovery is
            # always executable.
            self.log("jev_failed", error=type(exc).__name__)

    def route_direction(self, obs):
        player = _player(obs)
        px = _num(player.get("x", 45))
        if _stage(obs) == "ramp":
            target, _ = self.active_target(obs)
        else:
            target = RIVET_X.get(self.rivet_target, px)
        if target > px:
            return "right"
        if target < px:
            return "left"
        return self.last_walk_direction

    def choose_jump_direction(self, obs, hazard):
        # A moving obstacle is crossed head-on: Mario's direction opposes its
        # measured horizontal velocity.  A still obstacle uses the current
        # route direction, matching the successful parent behavior.
        if hazard.get("motion") == "left":
            return "right"
        if hazard.get("motion") == "right":
            return "left"
        preferred = self.route_direction(obs)
        if preferred in ("left", "right"):
            return preferred
        return "right"

    def start_jump(self, obs, direction, reason):
        self.jump_direction = direction if direction in ("left", "right") else "right"
        self.jump_left = JUMP_QUANTA
        self.landing_left = 0
        self.hazard_lock = HAZARD_LOCK_QUANTA
        self.jump_cooldown = 0
        self.log("crossing_started", tick=self.tick, direction=self.jump_direction,
                 reason=reason, hazard=_hazard_facts(obs))
        return _action(self.jump_direction, fire=True)

    def crossing_action(self, obs):
        hazard = _hazard_facts(obs)
        if self.jump_left > 0:
            self.jump_left -= 1
            if self.jump_left == 0:
                self.landing_left = LANDING_QUANTA
            return _action(self.jump_direction, fire=True), hazard
        if self.landing_left > 0:
            self.landing_left -= 1
            if self.landing_left == 0:
                self.jump_cooldown = 6
            return _action(self.jump_direction), hazard
        if self.jump_cooldown > 0:
            self.jump_cooldown -= 1
        if self.hazard_lock > 0:
            self.hazard_lock -= 1
            if not hazard.get("present") or hazard.get("urgency") == "clear":
                self.hazard_lock = 0

        if self.gap_direction in ("left", "right") and self.jump_cooldown == 0 and self.hazard_lock == 0:
            direction = self.gap_direction
            self.gap_direction = None
            self.jump_direction = direction
            self.jump_left = JUMP_QUANTA
            self.landing_left = 0
            self.hazard_lock = 10
            self.log("crossing_started", tick=self.tick, direction=direction, reason="cleared_gap")
            return _action(direction, fire=True), hazard
        if self.hazard_lock == 0 and self.jump_cooldown == 0 and hazard.get("urgency") in ("soon", "immediate"):
            return self.start_jump(obs, self.choose_jump_direction(obs, hazard), "hazard"), hazard
        return None, hazard

    def begin_climb(self, y):
        self.climb_active = True
        self.climb_age = 0
        self.climb_start_y = y
        self.progress_seen = False
        self.settle_age = 0
        self.last_y = y

    def advance_level_if_settled(self, obs, y, dy):
        if self.level + 1 >= len(RAMP_ROWS):
            return False
        next_y = RAMP_ROWS[self.level + 1]
        if y > next_y + 6:
            return False
        if abs(dy) <= 1:
            self.settle_age += 1
        else:
            self.settle_age = 0
        if self.progress_seen and self.settle_age >= SETTLE_QUANTA:
            self.level += 1
            self.candidate = 0
            self.variant = 0
            self.climb_active = False
            self.climb_age = 0
            self.climb_start_y = None
            self.progress_seen = False
            self.settle_age = 0
            self.last_y = None
            self.failed_key = None
            self.log("connector_accepted", level=self.level, x=_int(_player(obs).get("x", 0)), y=_int(y))
            return True
        return False

    def recover_failed_climb(self, obs):
        targets = self.ladder_targets(obs, self.level)
        hint = self.jev_hint
        self.jev_hint = None
        if hint == "alternate_ladder" and self.candidate + 1 < len(targets):
            self.candidate += 1
            self.variant = 0
        elif hint == "continue_climb":
            # Give one bounded continuation, then use the deterministic probe.
            self.variant = self.variant
        elif hint == "retry_alignment" and self.variant + 1 < len(ALIGNMENT_VARIANTS):
            self.variant += 1
        elif self.variant + 1 < len(ALIGNMENT_VARIANTS):
            self.variant += 1
        elif self.candidate + 1 < len(targets):
            self.candidate += 1
            self.variant = 0
        else:
            # A final bounded retry of the first candidate is safer than
            # wandering horizontally forever; the next hazard crossing can
            # still re-open the transaction at a different x.
            self.variant = 0
            self.candidate = 0
        self.route_failures += 1
        self.climb_active = False
        self.climb_age = 0
        self.climb_start_y = None
        self.progress_seen = False
        self.settle_age = 0
        self.last_y = None
        self.log("connector_retry", level=self.level, candidate=self.candidate,
                 variant=self.variant, hint=hint, failures=self.route_failures)

    def ramp_action(self, obs):
        player = _player(obs)
        x = _num(player.get("x", 45))
        y = _num(player.get("y", 176))
        dy = _num(player.get("dy", 0))
        self.level = min(max(self.level, 0), len(RAMP_ROWS) - 1)
        target, slot = self.active_target(obs)

        if self.climb_active:
            self.climb_age += 1
            if abs(x - target) > 3:
                # A crossing jump or sticky action displaced Mario.  Do not
                # keep sending UP at the old rung; reacquire its connector.
                self.climb_active = False
                self.climb_age = 0
                self.climb_start_y = None
                self.progress_seen = False
                self.settle_age = 0
                self.last_y = None
                return self.ramp_action(obs)
            if self.last_y is not None and y < self.last_y - 1:
                self.progress_seen = True
            if self.climb_start_y is not None and y < self.climb_start_y - 2:
                self.progress_seen = True
            self.last_y = y
            if self.advance_level_if_settled(obs, y, dy):
                return self.ramp_action(obs)
            if self.climb_age >= NO_PROGRESS_QUANTA and not self.progress_seen:
                self.request_jev(obs)
                self.recover_failed_climb(obs)
                return self.ramp_action(obs)
            return "up"

        # Only start a climb at the calibrated integer target.  This is the
        # central difference from 0009: x=49 is visually close to the x=48
        # ladder, but the successful trace accepted the rung near x=46.
        if int(round(x)) != int(round(target)):
            direction = "right" if x < target else "left"
            self.last_walk_direction = direction
            return direction
        self.begin_climb(y)
        self.log("climb_probe", level=self.level, candidate=self.candidate,
                 variant=self.variant, slot=slot, x=int(round(x)), y=int(round(y)))
        return "up"

    def nearest_ladder(self, obs, y, side):
        preferred = 108 if side == "right" else 48
        values = obs.get("ladders", []) if isinstance(obs, dict) else []
        choices = []
        if isinstance(values, list):
            for item in values:
                if not _visible(item):
                    continue
                lx = _num(item.get("x", 0))
                ly = _num(item.get("y", 0))
                if abs(ly - y) <= 20 or abs(ly - (y - 8)) <= 20:
                    choices.append((abs(lx - preferred), lx - 2))
        choices.sort()
        return choices[0][1] if choices else preferred - 2

    def remaining_rivet_side(self, mask, preferred):
        if mask == 18:
            return None
        if preferred == "left":
            if mask in (0, 6):
                return "left"
            if mask == 12:
                return "right"
        else:
            if mask in (0, 12):
                return "right"
            if mask == 6:
                return "left"
        return preferred

    def update_rivets(self, obs):
        current = _rows(obs)
        if self.rivet_previous is not None:
            for index, (old, new) in enumerate(zip(self.rivet_previous, current)):
                if index == self.rivet_row and old != new and new in (0, 6, 12, 18):
                    # The sweep has just made a hole.  Cross in the direction
                    # of the next side rather than walking over it.
                    self.gap_direction = "right" if self.rivet_target == "left" else "left"
                    self.log("rivet_byte_change", row=index, old=old, new=new,
                             direction=self.gap_direction)
        self.rivet_previous = list(current)

    def rivet_action(self, obs):
        self.update_rivets(obs)
        rows = _rows(obs)
        row = next((i for i, value in enumerate(rows) if value < 18), 4)
        if row >= 4:
            return "up"
        if row != self.rivet_row:
            self.rivet_row = row
            self.rivet_target = _row_side(row, self.rivet_first)
            self.rivet_climbing = False
            self.gap_direction = None
        target_side = self.remaining_rivet_side(rows[row], self.rivet_target)
        if target_side is None:
            ladder = self.nearest_ladder(obs, _num(_player(obs).get("y", 176)),
                                         "right" if row % 2 == 0 else "left")
            x = _num(_player(obs).get("x", 45))
            if abs(x - ladder) > 2:
                direction = "right" if x < ladder else "left"
                self.last_walk_direction = direction
                return direction
            return "up"
        self.rivet_target = target_side
        player = _player(obs)
        x = _num(player.get("x", 45))
        y = _num(player.get("y", 176))
        target_y = RIVET_Y[row]
        if y > target_y + 8:
            ladder = self.nearest_ladder(obs, y, self.rivet_climb_side)
            if abs(x - ladder) > 2:
                direction = "right" if x < ladder else "left"
                self.last_walk_direction = direction
                return direction
            self.rivet_climbing = True
            return "up"
        if y < target_y - 12:
            return "down"
        target_x = RIVET_X[target_side]
        if target_side == "left":
            if x > target_x + 8:
                return "left"
            return "right"
        if x < target_x - 8:
            return "right"
        return "left"

    def fallback(self, obs, stage, lives):
        if bool(obs.get("terminated", False)) or bool(obs.get("truncated", False)):
            return "noop"
        if stage == "game_over":
            return "noop"
        if stage == "idle":
            if self.idle_attempts < 2:
                self.idle_attempts += 1
                return "fire"
            return "noop"
        if stage == "transition":
            if lives <= 0:
                return "noop"
            if self.transition_age >= 18 and self.restart_attempts < 4 and (self.transition_age - 18) % 30 == 0:
                self.restart_attempts += 1
                return "fire"
            return "noop"
        if stage == "ramp":
            return self.ramp_action(obs)
        if stage == "rivet":
            return self.rivet_action(obs)
        return "noop"

    def choose(self, obs):
        stage, lives, screens, stage_changed, life_changed, screen_changed = self.sync(obs)
        self.current_obs = obs
        self.collect_jev()
        if stage == "idle":
            return "fire"
        if stage == "game_over" or bool(obs.get("terminated", False)):
            return "noop"
        if stage == "transition":
            return self.fallback(obs, stage, lives)
        if stage not in ("ramp", "rivet"):
            return "noop"

        emergency, hazard = self.crossing_action(obs)
        if emergency is not None:
            return emergency
        action = self.fallback(obs, stage, lives)
        if stage == "ramp" and self.climb_active and self.climb_age >= NO_PROGRESS_QUANTA - 2 and not self.progress_seen:
            # Start the advisory call before the deterministic probe changes
            # the state; the actor never waits for it.
            self.request_jev(obs)
        if stage == "rivet" and self.rivet_previous is None:
            self.rivet_previous = _rows(obs)
        if (self.tick % 20 == 0 or stage_changed or life_changed or screen_changed):
            player = _player(obs)
            self.log("snapshot", tick=self.tick, stage=stage, level=self.level,
                     candidate=self.candidate, variant=self.variant,
                     x=_int(player.get("x", 0)), y=_int(player.get("y", 0)),
                     action=action, climb=self.climb_active,
                     climb_age=self.climb_age, jump=self.jump_left,
                     hazard=hazard.get("urgency"), lives=lives,
                     score=obs.get("score_total"))
        return action

    def run(self, game):
        try:
            obs = self.ctx.observe()
        except Exception:
            return
        if not isinstance(obs, dict):
            obs = {}
        self.last_stage = _stage(obs)
        self.last_lives = _int(obs.get("lives_remaining", 0), 0)
        self.last_screens = _int(obs.get("screens_completed", 0), 0)
        if self.last_stage == "idle":
            if not self.post(game, "fire"):
                return
        while not self.done(game):
            try:
                self.ctx.wait_for_tick()
            except Exception:
                break
            if self.done(game):
                break
            try:
                obs = self.ctx.observe()
            except Exception:
                break
            if not isinstance(obs, dict):
                obs = {}
            self.tick += 1
            if _stage(obs) == "transition" and _int(obs.get("lives_remaining", 0), 0) <= 0:
                self.log("last_life_transition", tick=self.tick)
                break
            action = self.choose(obs)
            if not self.post(game, action):
                break
        self.log("finished", ticks=self.tick, score=getattr(game, "score", None))


class Program:
    def __init__(self, ctx):
        self.ctx = ctx

    def run_episode(self, game) -> None:
        Controller(self.ctx).run(game)


def build(ctx):
    return Program(ctx)
