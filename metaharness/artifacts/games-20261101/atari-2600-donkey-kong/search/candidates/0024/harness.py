"""Phase-aware Donkey Kong controller with a post-jump hazard lease.

The controller is code-first.  It uses the public fixed ladder graph and the
observed player height to route, while a small asynchronous Jev request can
choose between already-known connector branches after a measured stall.  The
important distinction from earlier route controllers is that a jump cooldown
is not treated as collision immunity: a new critical hazard, especially one
revealed by entering a new ramp phase, can preempt the cooldown.
"""
from __future__ import annotations

import json
import math

from metaharness.models.jev import Choice, Noul


ACTIONS = frozenset({
    "noop", "fire", "up", "right", "left", "down", "upright", "upleft",
    "downright", "downleft", "upfire", "rightfire", "leftfire", "downfire",
    "uprightfire", "upleftfire", "downrightfire", "downleftfire",
})

# These are Mario's useful alignment x values, two pixels left of the ladder
# boxes in the JSON observation.  Height observations, rather than action
# count, decide whether a connector was actually climbed.
RAMP_Y = (176.0, 148.0, 120.0, 92.0, 64.0, 40.0)
CONNECTORS = (
    (106.0,),
    (46.0, 78.0),
    (106.0, 86.0),
    (46.0, 66.0),
    (106.0,),
    (30.0, 74.0),
)
RIVET_Y = (176.0, 148.0, 120.0, 92.0)
RIVET_X = {"left": 46.0, "right": 106.0}


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


def xy_of(obs):
    player = player_of(obs)
    return number(player.get("x"), 45.0), number(player.get("y"), 176.0)


def visible(value):
    return isinstance(value, dict) and value.get("visible", True) is not False


def raw_byte(obs, address):
    values = obs.get("raw_ram", []) if isinstance(obs, dict) else []
    if not isinstance(values, (list, tuple)):
        return 0
    index = int(address) & 0x7F
    return integer(values[index], 0) if index < len(values) else 0


def jump_byte(obs):
    # The adaptor exposes the 128-byte RAM snapshot with $80 at index zero.
    return raw_byte(obs, 0x85)


def rows_of(obs):
    values = obs.get("rivet_rows", []) if isinstance(obs, dict) else []
    if not isinstance(values, (list, tuple)):
        values = []
    result = []
    for value in list(values)[:4]:
        item = integer(value, 0)
        result.append(item if item in (0, 6, 12, 18) else 0)
    while len(result) < 4:
        result.append(0)
    return result


def move(direction, fire=False):
    if direction not in ("left", "right"):
        return "fire" if fire else "noop"
    return direction + ("fire" if fire else "")


def probability(answer, option):
    values = getattr(answer, "probabilities", None)
    if not isinstance(values, dict):
        return None
    value = number(values.get(option), -1.0)
    return value if value >= 0.0 else None


def parse_json_object(text):
    if not isinstance(text, str):
        return None
    candidates = [text.strip()]
    left, right = text.find("{"), text.rfind("}")
    if left >= 0 and right > left:
        candidates.insert(0, text[left:right + 1])
    for candidate in candidates:
        try:
            result = json.loads(candidate)
        except (TypeError, ValueError):
            continue
        if isinstance(result, dict):
            return result
    return None


def remaining_side(mask, preference):
    # 6 means left is cleared, 12 means right is cleared.
    if mask == 18:
        return None
    if mask == 6:
        return "right"
    if mask == 12:
        return "left"
    return preference if preference in ("left", "right") else "left"


class RampPlane:
    """Short-lived grounded samples for deciding which ramp a hazard occupies."""

    def __init__(self):
        self.phase = None
        self.samples = []

    def reset(self, phase):
        self.phase = int(phase)
        self.samples = []

    def add(self, phase, x, y):
        if self.phase != int(phase):
            self.reset(phase)
        if math.isfinite(x) and math.isfinite(y):
            self.samples.append((float(x), float(y)))
            if len(self.samples) > 48:
                self.samples = self.samples[-48:]

    def fit(self):
        if len(self.samples) < 6:
            return None
        xs = [item[0] for item in self.samples]
        ys = [item[1] for item in self.samples]
        lo, hi = min(xs), max(xs)
        if hi - lo < 18.0:
            return None
        mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
        variance = sum((x - mx) ** 2 for x in xs)
        if variance < 20.0:
            return None
        slope = sum((x - mx) * (y - my) for x, y in self.samples) / variance
        slope = max(-0.75, min(0.75, slope))
        intercept = my - slope * mx
        residual = sum(abs(y - (slope * x + intercept))
                       for x, y in self.samples) / len(self.samples)
        if residual > 8.0:
            return None
        return slope, intercept, residual, len(self.samples), hi - lo


class Controller:
    def __init__(self, ctx, game):
        self.ctx = ctx
        self.game = game
        self.tick = 0
        self.last_action = "noop"
        self.last_stage = None
        self.last_lives = None
        self.last_screens = 0
        self.last_obs = {}

        # Route state is an observed walkway, not an UP-command counter.
        self.phase = 0
        self.branch = 0
        self.climb_open = False
        self.climb_origin_y = None
        self.climb_best_y = None
        self.climb_age = 0
        self.climb_stall = 0
        self.walk_stall = 0
        self.previous_x = None
        self.route_failures = 0
        self.route_event = None
        self.recheck_hazard = False
        self.plane = RampPlane()

        # After an actual life loss, the finite route attempt is over and a
        # proven phase-1 reward lane is used as a floor.  This is deliberately
        # activated on a boundary, never merely on entering phase one.
        self.farm_mode = False
        self.pending_life_reset = False
        self.route_attempts = 0

        # Jump transaction and its separate post-jump hazard lease.  The lock
        # suppresses duplicate FIRE for the same object; it is not an absolute
        # ban on a different critical object or a newly entered phase.
        self.jump_direction = None
        self.jump_seen = False
        self.jump_guard = 8
        self.jump_tail = 0
        self.jump_lock = 0
        self.post_jump_up = 0
        self.jump_reason = None
        self.jump_count = 0
        self.jump_phase = -1
        self.jump_signature = None
        self.landing_window = 0

        # Lifecycle state.
        self.transition_age = 0
        self.restart_edges = 0
        self.idle_edges = 0
        self.stop_requested = False

        # Rivet route state.
        self.rivet_row = 0
        self.rivet_first = "left"
        self.previous_rows = None
        self.gap_direction = None
        self.rivet_climb = False

        # Optional fast arbiter.  It can only select an already-known branch at
        # a measured stall; hazard timing and all lethal actions remain code.
        self.jev_future = None
        self.jev_stamp = None
        self.jev_started = -10000
        self.jev_hint = None
        self.jev_hint_stamp = None

    # ---------- safe execution ----------
    def log(self, event, **fields):
        try:
            self.ctx.log(event, **fields)
        except Exception:
            pass

    def done(self):
        try:
            if bool(self.ctx.done):
                return True
        except Exception:
            return True
        try:
            return bool(self.game.done)
        except Exception:
            return False

    def post(self, action):
        if not isinstance(action, str) or action not in ACTIONS:
            action = "noop"
        try:
            self.ctx.act(action)
        except Exception as exc:
            self.log("act_failed", action=action, error=type(exc).__name__)
            return False
        self.last_action = action
        return True

    def budget_allows_s1(self):
        try:
            if bool(self.ctx.done):
                return False
            return integer(self.ctx.budget.remaining().get("system1_calls"), 0) > 0
        except Exception:
            return False

    # ---------- lifecycle and route reset ----------
    def clear_jump(self):
        self.jump_direction = None
        self.jump_seen = False
        self.jump_guard = 8
        self.jump_tail = 0
        self.post_jump_up = 0
        self.jump_reason = None

    def reset_active(self, stage):
        self.phase = 0
        self.branch = 0
        self.climb_open = False
        self.climb_origin_y = None
        self.climb_best_y = None
        self.climb_age = 0
        self.climb_stall = 0
        self.walk_stall = 0
        self.previous_x = None
        self.route_failures = 0
        self.route_event = None
        self.recheck_hazard = False
        self.plane.reset(0)
        self.clear_jump()
        self.jump_lock = 0
        self.jump_phase = -1
        self.jump_signature = None
        self.landing_window = 0
        self.post_jump_up = 0
        self.rivet_climb = False
        self.gap_direction = None
        self.previous_rows = None
        if stage == "ramp":
            # pending_life_reset is set when the transition decrements the
            # spare-life byte, and is consumed only on active restart.
            if self.pending_life_reset:
                # Preserve the established first-death score floor.  The new
                # phase-aware hazard lease is tested on every route attempt
                # that survives long enough; spending a second Mario on the
                # same upper crossing made the final farm too short.
                self.route_attempts += 1
                self.farm_mode = True
                self.log("farm_enabled", reason="life boundary",
                         route_attempts=self.route_attempts)
            # Initial start remains a route experiment.
            self.pending_life_reset = False
        else:
            self.farm_mode = False
            self.rivet_row = 0
            self.rivet_first = "left"
        self.log("active_reset", stage=stage, phase=self.phase,
                 farm=self.farm_mode)

    def synchronize(self, obs):
        stage = stage_of(obs)
        lives = integer(obs.get("lives_remaining"), 0) if isinstance(obs, dict) else 0
        screens = integer(obs.get("screens_completed"), 0) if isinstance(obs, dict) else 0
        changed = stage != self.last_stage
        life_changed = self.last_lives is not None and lives != self.last_lives
        screen_changed = screens != self.last_screens
        if self.last_lives is not None and lives < self.last_lives:
            self.pending_life_reset = True
        if changed or life_changed or screen_changed:
            self.log("boundary", old_stage=self.last_stage, stage=stage,
                     old_lives=self.last_lives, lives=lives,
                     old_screens=self.last_screens, screens=screens,
                     phase=self.phase)
        if stage in ("ramp", "rivet") and (changed or life_changed or screen_changed):
            self.reset_active(stage)
        if changed:
            self.restart_edges = 0
            self.idle_edges = 0
        if stage == "transition":
            self.transition_age = self.transition_age + 1 if self.last_stage == "transition" else 1
        else:
            self.transition_age = 0
        self.last_stage = stage
        self.last_lives = lives
        self.last_screens = screens
        self.last_obs = obs if isinstance(obs, dict) else {}
        return stage, lives, screens, changed, life_changed, screen_changed

    def lifecycle_action(self, obs, stage, lives):
        if stage == "idle":
            if self.idle_edges < 3:
                self.idle_edges += 1
                return "fire"
            return "noop"
        if stage == "game_over" or bool(obs.get("terminated")) or bool(obs.get("truncated")):
            return "noop"
        if stage == "transition":
            # This edge schedule was validated for both death and screen
            # transition.  Zero spare lives still describes the current Mario.
            restart_points = (18, 46, 74, 102, 130, 158)
            if self.transition_age in restart_points and self.restart_edges < len(restart_points):
                self.restart_edges += 1
                self.log("restart_edge", age=self.transition_age,
                         lives=lives, edge=self.restart_edges)
                return "fire"
            # A zero-bonus, zero-spare shell is not an active game.  Allow a
            # short grace for the terminal flag, then preserve the score.
            if (self.transition_age > 80 and lives <= 0 and
                    number(obs.get("current_bonus"), 1.0) <= 0.0):
                self.stop_requested = True
            return "noop"
        if stage not in ("ramp", "rivet"):
            return "noop"
        return None

    # ---------- route geometry ----------
    def route_options(self):
        index = max(0, min(self.phase, len(CONNECTORS) - 1))
        return CONNECTORS[index]

    def route_target(self):
        options = self.route_options()
        index = max(0, min(self.branch, len(options) - 1))
        return float(options[index])

    def learn_plane(self, obs):
        if self.farm_mode or stage_of(obs) != "ramp":
            return
        if self.jump_direction is not None or self.climb_open:
            return
        if jump_byte(obs) != 0 or self.last_action not in ("left", "right"):
            return
        x, y = xy_of(obs)
        self.plane.add(self.phase, x, y)

    def switch_branch(self, reason):
        options = self.route_options()
        old = self.branch
        if len(options) > 1:
            self.branch = 1 if old == 0 else 0
        else:
            self.branch = 0
        self.route_failures += 1
        self.climb_open = False
        self.climb_origin_y = None
        self.climb_best_y = None
        self.climb_age = 0
        self.climb_stall = 0
        self.walk_stall = 0
        self.previous_x = None
        self.plane.reset(self.phase)
        self.route_event = "stall"
        self.log("branch_switch", phase=self.phase, old=old,
                 new=self.branch, target=self.route_target(),
                 reason=reason, failures=self.route_failures)

    def adopt_landing(self, obs):
        if self.landing_window <= 0 or self.phase >= len(CONNECTORS):
            return
        options = self.route_options()
        if len(options) < 2:
            return
        x, y = xy_of(obs)
        alternate = options[1]
        if abs(x - alternate) <= 7.0 and y <= RAMP_Y[self.phase] + 10.0:
            if self.branch != 1:
                self.log("landing_branch_adopted", phase=self.phase,
                         old=self.branch, new=1, x=round(x, 1), y=round(y, 1))
            self.branch = 1
        self.landing_window -= 1

    def advance_phase(self, obs, y):
        old = self.phase
        self.phase = min(self.phase + 1, len(CONNECTORS) - 1)
        self.branch = 0
        self.climb_open = False
        self.climb_origin_y = None
        self.climb_best_y = None
        self.climb_age = 0
        self.climb_stall = 0
        self.walk_stall = 0
        self.previous_x = None
        self.route_event = "advance"
        self.plane.reset(self.phase)
        self.recheck_hazard = True
        self.log("ramp_advance", old_phase=old, phase=self.phase,
                 x=round(xy_of(obs)[0], 1), y=round(y, 1))

    def ramp_action(self, obs):
        self.learn_plane(obs)
        x, y = xy_of(obs)

        if self.farm_mode:
            # Hold the established reward lane.  A jump remains an action
            # transaction, but the normal connector graph is not guessed from
            # the repeated landing at y~154.
            target = 106.0
            if abs(x - target) > 1.5:
                return "right" if x < target else "left"
            if self.post_jump_up > 0:
                self.post_jump_up -= 1
            return "up"

        self.adopt_landing(obs)
        target = self.route_target()
        options = self.route_options()

        if self.climb_open:
            self.climb_age += 1
            if abs(x - target) > 4.0:
                self.climb_open = False
                self.climb_origin_y = None
                self.climb_best_y = None
                self.climb_age = 0
                self.climb_stall = 0
                return self.ramp_action(obs)
            if self.climb_best_y is None or y < self.climb_best_y - 1.0:
                self.climb_best_y = y
                self.climb_stall = 0
            else:
                self.climb_stall += 1
            if self.phase + 1 < len(RAMP_Y):
                next_y = RAMP_Y[self.phase + 1]
                gain = self.climb_origin_y - y if self.climb_origin_y is not None else 0.0
                if self.climb_age >= 5 and gain >= 8.0 and y <= next_y + 6.0:
                    self.advance_phase(obs, y)
                    return self.ramp_action(obs)
            if self.climb_stall >= 28:
                if len(options) > 1:
                    self.switch_branch("vertical_stall")
                    return self.ramp_action(obs)
                self.climb_open = False
                self.climb_origin_y = None
                self.climb_best_y = None
                self.climb_age = 0
                self.climb_stall = 0
            else:
                return "up"

        if self.post_jump_up > 0:
            self.post_jump_up -= 1
            return "up"

        if self.previous_x is not None and self.last_action in ("left", "right"):
            self.walk_stall = self.walk_stall + 1 if abs(x - self.previous_x) < 0.6 else 0
        else:
            self.walk_stall = 0
        self.previous_x = x

        if abs(x - target) > 1.5:
            if self.walk_stall >= 12 and len(options) > 1:
                self.switch_branch("horizontal_stall")
                return self.ramp_action(obs)
            return "right" if x < target else "left"

        if self.phase >= len(RAMP_Y) - 1:
            return "up"
        self.climb_open = True
        self.climb_origin_y = y
        self.climb_best_y = y
        self.climb_age = 0
        self.climb_stall = 0
        self.route_event = None
        self.log("climb_begin", phase=self.phase, branch=self.branch,
                 target=round(target, 1), x=round(x, 1), y=round(y, 1))
        return "up"

    # ---------- hazard perception and jump transaction ----------
    def hazard_facts(self, obs, stage=None):
        stage = stage or stage_of(obs)
        p = player_of(obs)
        px, py = xy_of(obs)
        ph = number(p.get("h"), 15.0)
        pdx = number(p.get("dx"), 0.0)
        fitted = self.plane.fit() if stage == "ramp" else None
        hazards = obs.get("hazards", []) if isinstance(obs, dict) else []
        if not isinstance(hazards, list):
            hazards = []
        best = None
        for item in hazards:
            if not visible(item):
                continue
            hx = number(item.get("x"), 0.0)
            hy = number(item.get("y"), 0.0)
            hw = number(item.get("w"), 8.0)
            hh = number(item.get("h"), 8.0)
            dx = number(item.get("dx"), 0.0)
            delta = hx - px
            distance = abs(delta)
            if fitted is not None:
                expected = fitted[0] * hx + fitted[1]
                row_error = abs((hy + hh * 0.5) - (expected + ph * 0.5))
            else:
                row_error = abs((hy + hh * 0.5) - (py + ph * 0.5))
            same_row = row_error <= 7.0
            if not same_row and not (distance <= 9.0 and row_error <= 6.0):
                continue
            relative = dx - pdx
            closing = (delta * relative < -0.45 or
                       (distance <= 9.0 and abs(relative) <= 0.5))
            if not closing and distance > 18.0:
                continue
            if distance <= 10.0:
                urgency = "overlap"
            elif closing and distance <= 24.0:
                urgency = "commit"
            elif closing and distance <= 38.0:
                urgency = "soon"
            elif closing:
                urgency = "watch"
            else:
                urgency = "near"
            fact = {
                "present": True,
                "slot": integer(item.get("slot"), -1),
                "side": "left" if delta < 0.0 else "right",
                "motion": "left" if dx < -0.2 else "right" if dx > 0.2 else "still",
                "distance": round(distance, 1),
                "distance_bucket": ("overlap" if distance <= 10.0 else
                                    "near" if distance <= 24.0 else
                                    "approaching" if distance <= 42.0 else "far"),
                "relative_motion": "closing" if closing else "opening",
                "closing": bool(closing),
                "row_error": round(row_error, 1),
                "urgency": urgency,
            }
            rank = ({"overlap": 0, "commit": 1, "soon": 2,
                     "watch": 3, "near": 4}.get(urgency, 5), distance)
            if best is None or rank < best[0]:
                best = (rank, fact)
        if best is not None:
            return best[1]
        return {
            "present": False, "slot": -1, "side": "none", "motion": "none",
            "distance": 999.0, "distance_bucket": "none",
            "relative_motion": "none", "closing": False,
            "row_error": 999.0, "urgency": "clear",
        }

    def same_jump_object(self, hazard):
        if self.jump_signature is None:
            return False
        return (hazard.get("slot", -1), hazard.get("side", "none"),
                hazard.get("motion", "none")) == self.jump_signature

    def jump_direction_for(self, obs, hazard):
        if self.farm_mode:
            # The score-lane traces are most stable when FIRE faces into the
            # moving object, rather than steering toward its current side.
            if hazard.get("motion") == "left":
                direction = "right"
            elif hazard.get("motion") == "right":
                direction = "left"
            else:
                direction = "right" if hazard.get("side") == "left" else "left"
            x, _ = xy_of(obs)
            if direction == "left" and x < 28.0:
                direction = "right"
            if direction == "right" and x > 116.0:
                direction = "left"
            return direction
        # Preserve the route crossing that produced the deepest public
        # progress: face into the barrel's motion, so Mario meets it while
        # rising.  The phase-aware lease is the experiment here; changing
        # steering direction at the same time would confound it.
        if hazard.get("motion") == "left":
            return "right"
        if hazard.get("motion") == "right":
            return "left"
        side = hazard.get("side")
        if side == "left":
            return "right"
        if side == "right":
            return "left"
        return "right" if self.route_target() > xy_of(obs)[0] else "left"

    def begin_jump(self, obs, direction, reason, hazard):
        if direction not in ("left", "right"):
            direction = "right"
        self.jump_direction = direction
        self.jump_seen = False
        self.jump_guard = 8
        self.jump_tail = 0
        self.jump_reason = reason
        self.jump_lock = 0
        self.post_jump_up = 0
        self.jump_phase = self.phase
        self.jump_signature = (hazard.get("slot", -1), hazard.get("side", "none"),
                               hazard.get("motion", "none"))
        self.climb_open = False
        self.climb_origin_y = None
        self.climb_best_y = None
        self.climb_age = 0
        self.plane.reset(self.phase)
        self.jump_count += 1
        self.log("jump_begin", tick=self.tick, phase=self.phase,
                 direction=direction, reason=reason,
                 hazard={key: hazard.get(key) for key in
                         ("slot", "side", "motion", "distance", "row_error", "urgency")})
        return move(direction, True)

    def landing_direction(self, obs, default):
        if self.farm_mode:
            return default
        x, _ = xy_of(obs)
        target = self.route_target()
        if x < target - 2.0:
            return "right"
        if x > target + 2.0:
            return "left"
        return default

    def jump_action(self, obs):
        if self.jump_direction is None:
            if self.jump_lock > 0:
                self.jump_lock -= 1
            return None
        direction = self.jump_direction
        state = jump_byte(obs)
        if state > 0:
            self.jump_seen = True
            self.jump_guard = 8
            return move(direction, True)
        if not self.jump_seen and self.jump_guard > 0:
            self.jump_guard -= 1
            return move(direction, True)
        if self.jump_seen:
            if self.jump_tail < 5:
                self.jump_tail += 1
                return move(self.landing_direction(obs, direction), False)
            self.log("jump_end", tick=self.tick, phase=self.phase,
                     direction=direction, reason=self.jump_reason,
                     x=round(xy_of(obs)[0], 1), y=round(xy_of(obs)[1], 1))
            self.clear_jump()
            # This is a duplicate-action lease, not hazard immunity.  A new
            # object or phase can interrupt it on the very next observation.
            self.jump_lock = 24
            self.post_jump_up = 5
            self.landing_window = 8
            self.plane.reset(self.phase)
            return None
        self.log("jump_unconfirmed", tick=self.tick, phase=self.phase)
        self.clear_jump()
        self.jump_lock = 7
        return None

    def hazard_action(self, obs, stage):
        if stage not in ("ramp", "rivet") or self.jump_direction is not None:
            return None
        hazard = self.hazard_facts(obs, stage)
        if not hazard.get("present"):
            return None
        if self.farm_mode:
            if hazard.get("distance", 999.0) > 16.0:
                return None
        elif hazard.get("urgency") not in ("overlap", "commit"):
            return None

        # Ordinary lockout filters the same object just after a successful
        # crossing.  It deliberately yields to an overlap, a different slot,
        # or a phase change: this is the critical fix under test.
        if self.jump_lock > 0 and self.jump_phase == self.phase:
            if self.same_jump_object(hazard) and hazard.get("urgency") != "overlap":
                return None

        # Do not abandon a climb already showing height gain for a merely
        # projected object.  A true overlap is always an emergency.
        if (self.climb_open and self.climb_origin_y is not None and
                hazard.get("urgency") != "overlap"):
            _, y = xy_of(obs)
            if self.climb_origin_y - y >= 5.0:
                return None
        direction = self.jump_direction_for(obs, hazard)
        return self.begin_jump(obs, direction, "hazard", hazard)

    # ---------- rivet screen ----------
    def nearest_ladder(self, obs, side):
        desired = 108.0 if side == "right" else 48.0
        _, py = xy_of(obs)
        ladders = obs.get("ladders", []) if isinstance(obs, dict) else []
        candidates = []
        if isinstance(ladders, list):
            for item in ladders:
                if not visible(item):
                    continue
                lx = number(item.get("x"), desired) - 2.0
                ly = number(item.get("y"), py)
                if abs(ly - py) <= 23.0 or abs(ly - (py - 8.0)) <= 23.0:
                    candidates.append((abs(lx - (desired - 2.0)) +
                                      0.05 * abs(ly - py), lx))
        return min(candidates)[1] if candidates else desired - 2.0

    def note_rivet_change(self, old, new, row):
        removed = None
        if old == 0 and new == 6:
            removed = "left"
        elif old == 0 and new == 12:
            removed = "right"
        elif old == 6 and new == 18:
            removed = "right"
        elif old == 12 and new == 18:
            removed = "left"
        if removed is not None:
            self.gap_direction = "right" if removed == "left" else "left"
            self.log("rivet_changed", row=row, old=old, new=new,
                     removed=removed, gap=self.gap_direction)

    def rivet_action(self, obs):
        current = rows_of(obs)
        if self.previous_rows is not None:
            for index, (old, new) in enumerate(zip(self.previous_rows, current)):
                if index == self.rivet_row and old != new:
                    self.note_rivet_change(old, new, index)
        self.previous_rows = list(current)
        row = next((i for i, value in enumerate(current) if value < 18), 4)
        if row >= 4:
            return "up"
        if row != self.rivet_row:
            self.rivet_row = row
            self.rivet_climb = False
            self.gap_direction = None
        preference = self.rivet_first if row % 2 == 0 else (
            "right" if self.rivet_first == "left" else "left")
        if self.gap_direction in ("left", "right") and self.jump_lock == 0:
            direction = self.gap_direction
            self.gap_direction = None
            return self.begin_jump(obs, direction, "rivet_gap",
                                   {"slot": -1, "side": direction,
                                    "motion": "none", "distance": 0,
                                    "row_error": 0, "urgency": "gap"})

        x, y = xy_of(obs)
        side = remaining_side(current[row], preference)
        if side is None:
            ladder_side = "right" if row % 2 == 0 else "left"
            target = self.nearest_ladder(obs, ladder_side)
            if abs(x - target) > 2.0:
                return "right" if x < target else "left"
            self.rivet_climb = True
            return "up"
        target_y = RIVET_Y[row]
        if y > target_y + 10.0 or self.rivet_climb:
            ladder_side = "right" if row % 2 == 0 else "left"
            target = self.nearest_ladder(obs, ladder_side)
            if abs(x - target) > 2.0:
                self.rivet_climb = False
                return "right" if x < target else "left"
            if y <= target_y + 8.0:
                self.rivet_climb = False
            return "up"
        target_x = RIVET_X[side]
        if x < target_x - 8.0:
            return "right"
        if x > target_x + 8.0:
            return "left"
        return "right" if side == "left" else "left"

    # ---------- compact Jev bridge ----------
    def model_state(self, obs, event, fallback):
        p = player_of(obs)
        x, y = xy_of(obs)
        hazard = self.hazard_facts(obs, stage_of(obs))
        fitted = self.plane.fit()
        route = {
            "phase": self.phase,
            "branch": self.branch,
            "connector_count": len(self.route_options()),
            "alternate_available": len(self.route_options()) > 1,
            "climb_open": self.climb_open,
            "plane": "learned" if fitted is not None else "uncertain",
            "rivet_row": self.rivet_row,
            "rivet_rows": rows_of(obs),
        }
        return {
            "game": "Atari 2600 Donkey Kong NTSC retail cartridge",
            "event": event,
            "stage": stage_of(obs),
            "player": {
                "horizontal_band": "left" if x < 52 else "right" if x > 100 else "middle",
                "vertical_band": "bottom" if y > 160 else "lower" if y > 132 else "middle" if y > 78 else "top",
                "motion": "up" if number(p.get("dy"), 0) < -1 else
                          "down" if number(p.get("dy"), 0) > 1 else "steady",
                "jump_committed": self.jump_direction is not None,
            },
            "route": route,
            "hazard": {key: hazard.get(key) for key in
                       ("present", "slot", "side", "motion", "distance_bucket",
                        "relative_motion", "closing", "urgency")},
            "fallback_action": fallback,
            "progress": {
                "lives_remaining": integer(obs.get("lives_remaining"), 0),
                "screens_completed": integer(obs.get("screens_completed"), 0),
                "rivets_removed": integer(obs.get("rivets_removed"), 0),
            },
            "last_action": self.last_action,
        }

    def request_jev(self, obs, event, fallback):
        if self.jev_future is not None or not self.budget_allows_s1():
            return
        if event != "stall":
            return
        hazard = self.hazard_facts(obs, stage_of(obs))
        stamp = (event, stage_of(obs), self.phase, self.branch,
                 self.route_failures, hazard.get("slot", -1),
                 hazard.get("urgency", "clear"))
        if stamp == self.jev_stamp and self.tick - self.jev_started < 12:
            return
        state = self.model_state(obs, event, fallback)
        questions = {
            "branch": Choice(
                "At this measured connector stall, use only the explicitly listed fixed branches. Choose hold during a live climb or when no alternate exists.",
                {
                    "hold": "Keep the deterministic fallback and do not change branch.",
                    "nominal": "Retry branch zero, the nominal connector already in the route graph.",
                    "alternate": "Try the other connector explicitly listed for this phase.",
                },
            ),
            "current_row": Noul(
                "Do the supplied normalized hazard facts establish a current-walkway closing threat?",
                {
                    "true": "The object is a current-row overlap or commit threat.",
                    "false": "The object is absent, opening, or not established on the current walkway.",
                },
            ),
        }
        self.jev_stamp = stamp
        self.jev_started = self.tick
        try:
            self.jev_future = self.ctx.spawn(
                lambda: self.ctx.decide(state, questions, timeout_s=0.60)
            )
        except Exception as exc:
            self.jev_future = None
            self.log("jev_start_failed", error=type(exc).__name__)

    def collect_jev(self):
        future = self.jev_future
        if future is None:
            return
        try:
            if not future.done():
                return
        except Exception:
            self.jev_future = None
            return
        self.jev_future = None
        try:
            answers = future.result()
            if not isinstance(answers, dict):
                return
            answer = answers.get("branch")
            value = getattr(answer, "value", None)
            p = probability(answer, value)
            accepted = (
                value == "hold" and (p is None or p >= 0.65) or
                value in ("nominal", "alternate") and p is not None and p >= 0.65
            )
            if accepted:
                self.jev_hint = value
                self.jev_hint_stamp = self.jev_stamp
                self.log("jev_branch", value=value, probability=p)
        except Exception as exc:
            self.log("jev_failed", error=type(exc).__name__)

    def apply_jev(self, obs, action):
        hint = self.jev_hint
        hint_stamp = self.jev_hint_stamp
        self.jev_hint = None
        self.jev_hint_stamp = None
        if hint not in ("nominal", "alternate"):
            return action
        hazard = self.hazard_facts(obs, stage_of(obs))
        current_stamp = ("stall", stage_of(obs), self.phase, self.branch,
                         self.route_failures, hazard.get("slot", -1),
                         hazard.get("urgency", "clear"))
        if current_stamp != hint_stamp:
            self.log("jev_stale", answer=hint)
            return action
        if self.route_event != "stall" or self.climb_open or self.farm_mode:
            return action
        if hint == "alternate" and len(self.route_options()) > 1:
            if self.branch == 0:
                self.switch_branch("jev_alternate")
                return self.ramp_action(obs)
        elif hint == "nominal" and self.branch != 0:
            self.branch = 0
            self.route_event = None
            return self.ramp_action(obs)
        return action

    # ---------- top-level decision ----------
    def choose(self, obs):
        stage, lives, screens, changed, life_changed, screen_changed = self.synchronize(obs)
        self.collect_jev()
        lifecycle = self.lifecycle_action(obs, stage, lives)
        if lifecycle is not None:
            return lifecycle
        if self.stop_requested:
            return "noop"
        if stage not in ("ramp", "rivet"):
            return "noop"

        # A committed jump owns its literal direction.  No inference can
        # reverse FIRE during the airborne part of the transaction.
        ongoing = self.jump_action(obs)
        if ongoing is not None:
            return ongoing

        emergency = self.hazard_action(obs, stage)
        if emergency is not None:
            return emergency

        action = self.ramp_action(obs) if stage == "ramp" else self.rivet_action(obs)

        # The route action can expose a different ramp phase in this very same
        # observation.  Re-run the hard hazard guard after that state change;
        # this closes the one-tick blind spot at the upper connector.
        if stage == "ramp" and self.recheck_hazard:
            self.recheck_hazard = False
            second = self.hazard_action(obs, stage)
            if second is not None:
                action = second
        if self.route_event == "stall":
            self.request_jev(obs, "stall", action)
        action = self.apply_jev(obs, action)

        if changed or life_changed or screen_changed:
            self.log("decision_boundary", stage=stage, phase=self.phase,
                     branch=self.branch, action=action, lives=lives,
                     screens=screens, farm=self.farm_mode)
        if self.tick % 20 == 0:
            x, y = xy_of(obs)
            hazard = self.hazard_facts(obs, stage)
            self.log("snapshot", tick=self.tick, stage=stage, phase=self.phase,
                     branch=self.branch, x=round(x), y=round(y), action=action,
                     climb=self.climb_open, jump=self.jump_direction,
                     jump_lock=self.jump_lock, jump_phase=self.jump_phase,
                     raw_jump=jump_byte(obs), hazard=hazard.get("urgency"),
                     row_error=hazard.get("row_error"), lives=lives,
                     score=number(obs.get("score_total"), 0.0), farm=self.farm_mode)
        return action

    def run(self):
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
            if not self.post("fire"):
                return
        else:
            if not self.post(self.choose(obs)):
                return

        while not self.done():
            try:
                self.ctx.wait_for_tick()
            except Exception:
                break
            if self.done():
                break
            try:
                obs = self.ctx.observe()
            except Exception:
                break
            if not isinstance(obs, dict):
                obs = {}
            self.tick += 1
            action = self.choose(obs)
            if self.stop_requested:
                self.log("final_transition_end", age=self.transition_age,
                         score=number(obs.get("score_total"), 0.0))
                break
            if self.done():
                break
            if not self.post(action):
                break
        self.log("finished", ticks=self.tick, score=getattr(self.game, "score", None),
                 phase=self.phase, jumps=self.jump_count,
                 screens=self.last_screens, farm=self.farm_mode)


class Program:
    def __init__(self, ctx):
        self.ctx = ctx

    def run_episode(self, game) -> None:
        Controller(self.ctx, game).run()


def build(ctx):
    return Program(ctx)
