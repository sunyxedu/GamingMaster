"""Ground-plane hazard classifier for Atari 2600 Donkey Kong.

The controller's main new mechanism is a local geometry ledger.  A ramp is
not a horizontal row: its y coordinate changes with x.  Earlier controllers
classified an obstacle as being on Mario's row with a large absolute-y
window, which makes a barrel on the neighbouring sloped girder look like an
immediate collision.  This controller learns the current ramp's x/y line from
Mario's grounded observations and classifies hazards against that line.

Inference is deliberately advisory and asynchronous.  The four-frame actor
never waits for a model; it owns lifecycle, exact arithmetic, jump leases,
and the raw rivet bytes.
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

# Mario's useful left-edge alignment, not the left edge of the ladder box.
# The graph is only a prior; height observations are the authority for progress.
RAMP_Y = (176.0, 148.0, 120.0, 92.0, 64.0, 40.0)
RAMP_CONNECTORS = (
    (106.0,),
    (46.0, 78.0),
    (106.0, 86.0),
    (46.0, 66.0),
    (106.0,),
    (30.0, 74.0),
)
RIVET_Y = (176.0, 148.0, 120.0, 92.0)
RIVET_X = {"left": 46.0, "right": 106.0}


def num(value, default=0.0):
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


def visible(value):
    return isinstance(value, dict) and value.get("visible", True) is not False


def xy(obs):
    player = player_of(obs)
    return num(player.get("x"), 45.0), num(player.get("y"), 176.0)


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


def raw_byte(obs, address):
    raw = obs.get("raw_ram", []) if isinstance(obs, dict) else []
    if not isinstance(raw, (list, tuple)):
        return 0
    index = int(address) & 0x7F
    return integer(raw[index], 0) if index < len(raw) else 0


def jump_byte(obs):
    # $85 is index 5 in the adaptor's wrapped 128-byte snapshot.
    return raw_byte(obs, 0x85)


def move(direction, fire=False):
    if direction not in ("left", "right"):
        return "fire" if fire else "noop"
    return direction + ("fire" if fire else "")


def probability(answer, option):
    values = getattr(answer, "probabilities", None)
    if not isinstance(values, dict):
        return None
    value = num(values.get(option), -1.0)
    return value if value >= 0.0 else None


def json_object(text):
    """Parse only a JSON object, tolerating fenced or prefixed Astra text."""
    if not isinstance(text, str):
        return None
    candidates = [text.strip()]
    left, right = text.find("{"), text.rfind("}")
    if left >= 0 and right > left:
        candidates.insert(0, text[left:right + 1])
    for candidate in candidates:
        try:
            value = json.loads(candidate)
        except (TypeError, ValueError):
            continue
        if isinstance(value, dict):
            return value
    return None


def remaining_side(mask, preferred):
    """Return a side still containing a rivet for the normal row bytes."""
    if mask == 18:
        return None
    if mask == 6:       # left rivet has been removed
        return "right"
    if mask == 12:      # right rivet has been removed
        return "left"
    return preferred if preferred in ("left", "right") else "left"


class RampPlane:
    """Small online line fit for one grounded sloped walkway.

    It intentionally forgets samples after a jump or route boundary.  A fit is
    considered reliable only when it has both enough samples and meaningful
    horizontal span; otherwise hazard classification falls back to a narrow
    player-y test rather than inventing a slope.
    """

    def __init__(self):
        self.phase = None
        self.samples = []

    def reset(self, phase):
        self.phase = int(phase)
        self.samples = []

    def add(self, phase, x, y):
        if self.phase != int(phase):
            self.reset(phase)
        if not (math.isfinite(x) and math.isfinite(y)):
            return
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
        mx = sum(xs) / len(xs)
        my = sum(ys) / len(ys)
        variance = sum((value - mx) ** 2 for value in xs)
        if variance < 20.0:
            return None
        slope = sum((x - mx) * (y - my) for x, y in self.samples) / variance
        slope = max(-0.75, min(0.75, slope))
        intercept = my - slope * mx
        # A few landing samples can be outliers.  Reject a fit whose residual
        # is implausibly large; the caller then uses the conservative fallback.
        residual = sum(abs(y - (slope * x + intercept)) for x, y in self.samples) / len(self.samples)
        if residual > 8.0:
            return None
        return slope, intercept, residual, len(self.samples), hi - lo

    def expected(self, x):
        fitted = self.fit()
        if fitted is None:
            return None
        return fitted[0] * float(x) + fitted[1]


class Controller:
    def __init__(self, ctx, game):
        self.ctx = ctx
        self.game = game
        self.tick = 0
        self.last_action = "noop"
        self.last_stage = None
        self.last_lives = None
        self.last_screens = 0

        # Ramp route state.  `phase` is observed walkway, `branch` is only the
        # connector being tried on that walkway.
        self.phase = 0
        self.branch = 0
        self.climb_open = False
        self.climb_origin_y = None
        self.climb_best_y = None
        self.climb_age = 0
        self.climb_stall = 0
        self.walk_stall = 0
        self.previous_x = None
        self.plane = RampPlane()
        self.route_failures = 0
        self.branch_hint = None

        # A jump is an action lease.  The initial direction is an escape
        # corridor, then the route is reacquired only after the RAM edge and a
        # short grounded tail.  No model can interrupt this lease.
        self.jump_direction = None
        self.jump_seen = False
        self.jump_guard = 8
        self.landing_ticks = 0
        self.jump_lock = 0
        self.jump_reason = None
        self.jump_count = 0
        self.landing_window = 0

        # Lifecycle: lives_remaining is spare Marios, not a terminal signal.
        self.transition_age = 0
        self.restart_edges = 0
        self.idle_edges = 0

        # Rivet route state.
        self.rivet_row = 0
        self.rivet_first = "left"
        self.previous_rows = None
        self.gap_direction = None
        self.rivet_climb = False

        # Nonblocking model bridges, each stamped with the situation that
        # produced it.  At most one Jev request is outstanding.
        self.jev_future = None
        self.jev_stamp = None
        self.jev_started = -10000
        self.jev_hint = None
        self.jev_hint_stamp = None
        self.astra_future = None
        self.astra_started = False
        self.astra_hint = {}
        self.stop_requested = False

    # ---------------- contract-safe execution ----------------
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

    def budget_allows(self, key):
        try:
            if bool(self.ctx.done):
                return False
            remaining = self.ctx.budget.remaining()
            return integer(remaining.get(key), 0) > 0
        except Exception:
            return False

    # ---------------- lifecycle and observation synchronization ----------------
    def clear_jump(self):
        self.jump_direction = None
        self.jump_seen = False
        self.jump_guard = 8
        self.landing_ticks = 0
        self.jump_lock = 0
        self.jump_reason = None
        self.landing_window = 0

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
        self.plane.reset(0)
        self.route_failures = 0
        self.clear_jump()
        self.rivet_climb = False
        self.gap_direction = None
        if stage == "rivet":
            self.rivet_row = 0
            side = self.astra_hint.get("first_side")
            self.rivet_first = side if side in ("left", "right") else "left"
            self.previous_rows = None

    def synchronize(self, obs):
        stage = stage_of(obs)
        lives = integer(obs.get("lives_remaining"), 0) if isinstance(obs, dict) else 0
        screens = integer(obs.get("screens_completed"), 0) if isinstance(obs, dict) else 0
        changed = stage != self.last_stage
        life_changed = self.last_lives is not None and lives != self.last_lives
        screen_changed = screens != self.last_screens
        if changed or life_changed or screen_changed:
            self.log("boundary", old_stage=self.last_stage, stage=stage,
                     old_lives=self.last_lives, lives=lives,
                     old_screens=self.last_screens, screens=screens,
                     phase=self.phase)
        if stage in ("ramp", "rivet") and (changed or life_changed):
            self.reset_active(stage)
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

    def lifecycle_action(self, obs, stage, lives):
        if stage == "idle":
            if self.idle_edges < 3:
                self.idle_edges += 1
                return "fire"
            return "noop"
        if stage == "game_over" or bool(obs.get("terminated")) or bool(obs.get("truncated")):
            return "noop"
        if stage == "transition":
            # A transition after a death or a completed screen accepts a sparse
            # FIRE edge.  Do this even at zero spare lives: that byte describes
            # the currently visible Mario's *spares*.
            restart_points = (18, 46, 74, 102, 130, 158)
            if self.transition_age in restart_points and self.restart_edges < len(restart_points):
                self.restart_edges += 1
                self.log("restart_edge", age=self.transition_age,
                         lives=lives, edge=self.restart_edges)
                return "fire"
            # A final no-bonus transition can otherwise remain a nonterminal
            # shell indefinitely.  Give it enough time to become game_over.
            if (self.transition_age > 180 and lives <= 0 and
                    num(obs.get("current_bonus"), 1.0) <= 0.0):
                self.stop_requested = True
                self.log("final_transition_end", age=self.transition_age)
                return None
            return "noop"
        if stage not in ("ramp", "rivet"):
            return "noop"
        return None

    # ---------------- local ramp geometry ----------------
    def route_options(self):
        phase = max(0, min(self.phase, len(RAMP_CONNECTORS) - 1))
        return RAMP_CONNECTORS[phase]

    def route_target(self):
        options = self.route_options()
        index = max(0, min(self.branch, len(options) - 1))
        return float(options[index])

    def learn_plane(self, obs):
        if stage_of(obs) != "ramp" or self.jump_direction is not None or self.climb_open:
            return
        if jump_byte(obs) != 0 or self.last_action not in ("left", "right"):
            return
        x, y = xy(obs)
        self.plane.add(self.phase, x, y)

    def hazard_facts(self, obs, force_stage=None):
        """Find a closing hazard on the same *sloped* walkway.

        `row_error` is the distance from the learned local ramp line.  It is
        included in the diagnostic state, but never delegated to Jev.
        """
        p = player_of(obs)
        px, py = xy(obs)
        ph = num(p.get("h"), 15.0)
        pdx = num(p.get("dx"), 0.0)
        stage = force_stage or stage_of(obs)
        fitted = self.plane.fit() if stage == "ramp" else None
        hazards = obs.get("hazards", []) if isinstance(obs, dict) else []
        if not isinstance(hazards, list):
            hazards = []
        best = None
        for item in hazards:
            if not visible(item):
                continue
            hx, hy = num(item.get("x"), 0.0), num(item.get("y"), 0.0)
            hw, hh = num(item.get("w"), 8.0), num(item.get("h"), 8.0)
            dx, dy = num(item.get("dx"), 0.0), num(item.get("dy"), 0.0)
            delta = hx - px
            distance = abs(delta)
            if fitted is not None:
                expected = fitted[0] * hx + fitted[1]
                row_error = abs((hy + hh * 0.5) - (expected + ph * 0.5))
                same_row = row_error <= 7.0
            else:
                expected = py
                row_error = abs((hy + hh * 0.5) - (py + ph * 0.5))
                same_row = row_error <= 7.0
            # A very close sprite can straddle the learned line during a
            # landing; retain it only when its boxes are genuinely adjacent.
            if not same_row and not (
                    distance <= 9.0 and
                    abs((hy + hh * 0.5) - (py + ph * 0.5)) <= 6.0):
                continue
            relative = dx - pdx
            closing = (delta * relative < -0.45) or (distance <= 9.0 and abs(relative) <= 0.5)
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
                "relative_motion": "closing" if closing else "opening",
                "closing": bool(closing),
                "row_error": round(row_error, 1),
                "distance_bucket": (
                    "overlap" if distance <= 10.0 else
                    "near" if distance <= 24.0 else
                    "approaching" if distance <= 42.0 else "far"
                ),
                "urgency": urgency,
                "vertical_motion": "up" if dy < -0.5 else "down" if dy > 0.5 else "steady",
            }
            rank = (
                0 if urgency == "overlap" else
                1 if urgency == "commit" else
                2 if urgency == "soon" else
                3 if urgency == "watch" else 4,
                distance,
            )
            if best is None or rank < best[0]:
                best = (rank, fact)
        if best is not None:
            return best[1]
        return {
            "present": False, "slot": -1, "side": "none", "motion": "none",
            "distance": 999.0, "relative_motion": "none", "closing": False,
            "row_error": 999.0, "distance_bucket": "none", "urgency": "clear",
            "vertical_motion": "none",
        }

    def open_climb(self, y):
        self.climb_open = True
        self.climb_origin_y = y
        self.climb_best_y = y
        self.climb_age = 0
        self.climb_stall = 0
        self.plane.reset(self.phase)

    def switch_branch(self, reason):
        options = self.route_options()
        old = self.branch
        if len(options) > 1:
            wanted = self.branch_hint
            if wanted not in (0, 1):
                wanted = 1 if old == 0 else 0
            self.branch = int(wanted)
        else:
            self.branch = 0
        self.branch_hint = None
        self.route_failures += 1
        self.climb_open = False
        self.climb_origin_y = None
        self.climb_best_y = None
        self.climb_age = 0
        self.climb_stall = 0
        self.walk_stall = 0
        self.previous_x = None
        self.plane.reset(self.phase)
        self.log("branch_switch", phase=self.phase, old=old, new=self.branch,
                 target=self.route_target(), reason=reason,
                 failures=self.route_failures)

    def ramp_action(self, obs):
        self.learn_plane(obs)
        x, y = xy(obs)
        target = self.route_target()
        options = self.route_options()

        if self.climb_open:
            self.climb_age += 1
            if abs(x - target) > 4.0:
                # A jump/landing displaced Mario.  UP at empty x is not a climb;
                # close the transaction and reacquire the same connector.
                self.climb_open = False
                self.climb_origin_y = None
                self.climb_best_y = None
                self.climb_age = 0
                self.climb_stall = 0
                return self.ramp_action(obs)
            if self.climb_best_y is None or y < self.climb_best_y - 1.0:
                self.climb_best_y = y
                self.climb_stall = 0
            elif abs(num(player_of(obs).get("dy"), 0.0)) <= 1.5:
                self.climb_stall += 1
            if self.phase + 1 < len(RAMP_Y):
                next_y = RAMP_Y[self.phase + 1]
                gain = (self.climb_origin_y - y) if self.climb_origin_y is not None else 0.0
                if self.climb_age >= 5 and gain >= 8.0 and y <= next_y + 6.0:
                    old = self.phase
                    self.phase += 1
                    self.branch = 0
                    self.climb_open = False
                    self.climb_origin_y = None
                    self.climb_best_y = None
                    self.climb_age = 0
                    self.climb_stall = 0
                    self.walk_stall = 0
                    self.previous_x = None
                    self.plane.reset(self.phase)
                    self.log("ramp_advance", old_phase=old, phase=self.phase,
                             x=round(x, 1), y=round(y, 1))
                    return self.ramp_action(obs)
            if self.climb_stall >= 30:
                self.switch_branch("vertical_stall")
                return self.ramp_action(obs)
            return "up"

        # After a jump, landing at the alternate fixed connector is evidence
        # worth adopting; unlike a timer this cannot silently change the route.
        if self.landing_window > 0 and len(options) > 1:
            alternate = options[1]
            if abs(x - alternate) <= 7.0 and y <= RAMP_Y[self.phase] + 10.0:
                if self.branch != 1:
                    self.log("landing_connector", phase=self.phase,
                             old=self.branch, new=1, x=round(x, 1), y=round(y, 1))
                self.branch = 1
        if self.landing_window > 0:
            self.landing_window -= 1

        # Count horizontal non-response only while walking toward the selected
        # connector; this is the bounded alternate-connector recovery.
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

        # At the final connector, holding UP lets the cartridge own goal
        # detection; no speculative horizontal move is made at the top.
        if self.phase >= len(RAMP_Y) - 1:
            return "up"
        self.open_climb(y)
        self.log("climb_begin", phase=self.phase, branch=self.branch,
                 target=round(target, 1), x=round(x, 1), y=round(y, 1))
        return "up"

    # ---------------- jump transaction ----------------
    def jump_direction_for(self, obs, hazard, reason="hazard"):
        if reason == "rivet_gap":
            return hazard.get("side") if hazard.get("side") in ("left", "right") else "right"
        # A ramp barrel is safest when Mario and it meet while Mario is
        # rising.  The public traces that earn the jump award use the
        # counter-motion direction: a barrel moving left is met by RIGHT, and
        # a barrel moving right by LEFT.  This is deliberately different from
        # choosing a direction from the object's side (which can steer into a
        # late collision).  The learned plane decides whether this guard fires;
        # this lease decides how to cross once it does.
        motion = hazard.get("motion")
        if motion == "left":
            return "right"
        if motion == "right":
            return "left"
        side = hazard.get("side")
        if side == "left":
            return "right"
        if side == "right":
            return "left"
        x, _ = xy(obs)
        return "right" if self.route_target() > x else "left"

    def begin_jump(self, obs, direction, reason, hazard=None):
        if direction not in ("left", "right"):
            direction = "right"
        self.jump_direction = direction
        self.jump_seen = False
        self.jump_guard = 8
        self.landing_ticks = 0
        self.jump_lock = 22
        self.jump_reason = reason
        self.landing_window = 0
        self.jump_count += 1
        self.climb_open = False
        self.climb_origin_y = None
        self.climb_best_y = None
        self.climb_age = 0
        self.plane.reset(self.phase)
        self.log("jump_begin", tick=self.tick, phase=self.phase,
                 direction=direction, reason=reason,
                 hazard={key: (hazard or {}).get(key) for key in
                         ("slot", "side", "motion", "distance", "row_error", "urgency")})
        return move(direction, True)

    def jump_action(self, obs):
        if self.jump_direction is not None:
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
                if self.landing_ticks < 4:
                    self.landing_ticks += 1
                    return move(direction, False)
                self.log("jump_end", tick=self.tick, phase=self.phase,
                         direction=direction, reason=self.jump_reason,
                         x=round(xy(obs)[0], 1), y=round(xy(obs)[1], 1))
                self.jump_direction = None
                self.jump_seen = False
                self.landing_ticks = 0
                self.jump_guard = 8
                self.jump_lock = 24
                self.landing_window = 8
                self.plane.reset(self.phase)
                return "up"
            # Missing/garbled RAM must not hold FIRE forever.
            self.log("jump_unconfirmed", tick=self.tick, phase=self.phase)
            self.clear_jump()
            self.jump_lock = 8
            return None
        if self.jump_lock > 0:
            self.jump_lock -= 1
        return None

    def hazard_action(self, obs, stage):
        if stage not in ("ramp", "rivet"):
            return None
        if self.jump_direction is not None or self.jump_lock > 0:
            return None
        hazard = self.hazard_facts(obs, stage)
        if hazard.get("urgency") not in ("overlap", "commit"):
            return None
        # Let a climb finish through a projected collision when Mario has
        # already gained height.  An actual overlap still gets the hard jump.
        if self.climb_open and self.climb_age >= 4 and hazard.get("urgency") != "overlap":
            return None
        direction = self.jump_direction_for(obs, hazard, "hazard")
        return self.begin_jump(obs, direction, "hazard", hazard)

    # ---------------- rivet screen ----------------
    def nearest_ladder(self, obs, side):
        desired = 108.0 if side == "right" else 48.0
        _, py = xy(obs)
        ladders = obs.get("ladders", []) if isinstance(obs, dict) else []
        candidates = []
        if isinstance(ladders, list):
            for item in ladders:
                if not visible(item):
                    continue
                lx, ly = num(item.get("x"), desired), num(item.get("y"), py)
                if abs(ly - py) <= 23.0 or abs(ly - (py - 8.0)) <= 23.0:
                    candidates.append((abs(lx - desired) + 0.05 * abs(ly - py), lx - 2.0))
        if candidates:
            candidates.sort()
            return candidates[0][1]
        return desired - 2.0

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
        preferred = self.rivet_first if row % 2 == 0 else (
            "right" if self.rivet_first == "left" else "left")

        if self.gap_direction in ("left", "right") and self.jump_lock == 0:
            direction = self.gap_direction
            self.gap_direction = None
            return self.begin_jump(obs, direction, "rivet_gap",
                                   {"side": direction, "slot": -1,
                                    "motion": "none", "urgency": "gap"})

        x, y = xy(obs)
        side = remaining_side(current[row], preferred)
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

    # ---------------- asynchronous System-1/System-2 bridge ----------------
    def model_state(self, obs, event, fallback):
        p = player_of(obs)
        hazard = self.hazard_facts(obs, stage_of(obs))
        fitted = self.plane.fit()
        x, y = xy(obs)
        if event == "ramp_stall":
            route = {
                "phase": self.phase,
                "branch": self.branch,
                "options": len(self.route_options()),
                "climb_open": self.climb_open,
                "plane": "learned" if fitted is not None else "uncertain",
            }
        else:
            route = {
                "row": self.rivet_row,
                "row_status": rows_of(obs),
                "gap_pending": self.gap_direction in ("left", "right"),
                "climb_open": self.rivet_climb,
            }
        return {
            "game": "Atari 2600 Donkey Kong NTSC retail cartridge",
            "event": event,
            "stage": stage_of(obs),
            "player": {
                "horizontal_band": "left" if x < 52 else "right" if x > 100 else "middle",
                "vertical_band": "bottom" if y > 160 else "lower" if y > 132 else "middle" if y > 78 else "top",
                "motion": "up" if num(p.get("dy"), 0) < -1 else "down" if num(p.get("dy"), 0) > 1 else "steady",
                "jump_committed": self.jump_direction is not None,
            },
            "route": route,
            "hazard": {key: hazard.get(key) for key in
                       ("present", "side", "motion", "distance_bucket", "relative_motion",
                        "closing", "urgency")},
            "fallback_action": fallback,
            "progress": {
                "lives_remaining": integer(obs.get("lives_remaining"), 0),
                "screens_completed": integer(obs.get("screens_completed"), 0),
                "rivets_removed": integer(obs.get("rivets_removed"), 0),
            },
            "last_action": self.last_action,
        }

    def stamp(self, obs, event):
        hazard = self.hazard_facts(obs, stage_of(obs))
        return (event, stage_of(obs), self.phase, self.branch, self.rivet_row,
                tuple(rows_of(obs)), hazard.get("slot", -1),
                hazard.get("urgency", "clear"), self.route_failures)

    def request_jev(self, obs, event, fallback):
        if self.jev_future is not None or not self.budget_allows("system1_calls"):
            return
        if event not in ("ramp_stall", "rivet_route"):
            return
        stamp = self.stamp(obs, event)
        if stamp == self.jev_stamp and self.tick - self.jev_started < 12:
            return
        state = self.model_state(obs, event, fallback)
        if event == "ramp_stall":
            questions = {
                "branch": Choice(
                    "At a verified connector stall, choose only among the explicitly supplied fixed options. Do not invent a coordinate; choose hold if a climb is active or no alternative is listed.",
                    {
                        "hold": "Keep the current connector and let the bounded fallback continue.",
                        "nominal": "Retry the nominal connector already listed.",
                        "alternate": "Try the alternate connector explicitly listed for this phase.",
                    },
                ),
                "row": Noul(
                    "Does the supplied learned-plane state establish that the named hazard is on Mario's current walkway?",
                    {
                        "true": "The hazard is a current-plane closing threat, not merely a neighbouring ramp object.",
                        "false": "The plane is uncertain or the hazard is not established on the current walkway.",
                    },
                ),
            }
        else:
            questions = {
                "side": Choice(
                    "Choose a preference only for the next legal rivet sweep. Use route.row_status; choose hold while a gap or climb transaction is active.",
                    {
                        "left": "Approach the left remaining rivet on the current row.",
                        "right": "Approach the right remaining rivet on the current row.",
                        "hold": "Preserve the deterministic gap, climb, or hazard transaction.",
                    },
                ),
                "danger": Noul(
                    "Is hazard explicitly a closing overlap or commit threat on Mario's current row?",
                    {
                        "true": "hazard.urgency is overlap or commit and hazard.closing is true.",
                        "false": "No such current-row threat is supplied.",
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

    def collect_jev(self, obs):
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
            if answer is not None:
                value = getattr(answer, "value", None)
                p = probability(answer, value)
                if value in ("hold", "nominal", "alternate") and (p is None or p >= 0.60):
                    self.jev_hint = value
                    self.log("jev_branch", value=value, probability=p)
            answer = answers.get("side")
            if answer is not None:
                value = getattr(answer, "value", None)
                p = probability(answer, value)
                if value in ("left", "right") and (p is None or p >= 0.60):
                    self.jev_hint = value
                    self.log("jev_side", value=value, probability=p)
        except Exception as exc:
            self.log("jev_failed", error=type(exc).__name__)

    def apply_jev(self, obs, action, event):
        hint = self.jev_hint
        self.jev_hint = None
        if hint is None:
            return action
        if event == "ramp_stall" and hint in ("nominal", "alternate"):
            if hint == "alternate" and len(self.route_options()) > 1:
                self.branch_hint = 1
                self.switch_branch("jev_alternate")
                return self.ramp_action(obs)
            if hint == "nominal":
                self.branch_hint = 0
                self.switch_branch("jev_nominal")
                return self.ramp_action(obs)
        if event == "rivet_route" and hint in ("left", "right"):
            row = min(max(self.rivet_row, 0), 3)
            side = remaining_side(rows_of(obs)[row], hint)
            if side == hint and self.gap_direction is None and not self.rivet_climb:
                self.rivet_first = hint
        return action

    def start_astra(self, obs, reason):
        if self.astra_started or not self.budget_allows("system2_calls"):
            return
        if stage_of(obs) not in ("ramp", "rivet"):
            return
        self.astra_started = True
        state = self.model_state(obs, reason, self.last_action)
        prompt = (
            "You are Astra, a slow macro reviewer for the Atari 2600 Donkey Kong "
            "NTSC retail cartridge, not the arcade game. The real-time actor owns "
            "exact timing, learned ramp geometry, hazards, raw rivet bytes, and "
            "lifecycle. Return only one compact JSON object with optional labels "
            "branch='nominal'|'alternate'|'adaptive', hazard='escape'|'climb', "
            "first_side='left'|'right'. Invalid or absent labels are ignored. "
            "A branch preference applies only after a measured stall; hazard cannot "
            "override the hard jump/plane classifier. State="
            + json.dumps(state, separators=(",", ":"), allow_nan=False)
        )
        try:
            self.astra_future = self.ctx.spawn(
                lambda: self.ctx.think(prompt, timeout_s=85.0)
            )
        except Exception as exc:
            self.astra_future = None
            self.log("astra_start_failed", error=type(exc).__name__)

    def collect_astra(self):
        future = self.astra_future
        if future is None:
            return
        try:
            if not future.done():
                return
        except Exception:
            self.astra_future = None
            return
        self.astra_future = None
        try:
            result = future.result()
            parsed = json_object(getattr(result, "text", ""))
        except Exception as exc:
            self.log("astra_failed", error=type(exc).__name__)
            return
        if not isinstance(parsed, dict):
            return
        branch = parsed.get("branch")
        hazard = parsed.get("hazard")
        side = parsed.get("first_side")
        if branch in ("nominal", "alternate", "adaptive"):
            self.astra_hint["branch"] = branch
        if hazard in ("escape", "climb"):
            self.astra_hint["hazard"] = hazard
        if side in ("left", "right"):
            self.astra_hint["first_side"] = side
            if stage_of(self._last_obs) == "rivet" and self.rivet_row == 0:
                self.rivet_first = side
        self.log("astra_hint", branch=self.astra_hint.get("branch"),
                 hazard=self.astra_hint.get("hazard"),
                 first_side=self.astra_hint.get("first_side"))

    # ---------------- top-level policy ----------------
    def choose(self, obs):
        self._last_obs = obs
        stage, lives, screens, stage_changed, life_changed, screen_changed = self.synchronize(obs)
        self.collect_jev(obs)
        self.collect_astra()

        lifecycle = self.lifecycle_action(obs, stage, lives)
        if self.stop_requested:
            return None
        if lifecycle is not None:
            return lifecycle
        if stage == "game_over" or bool(obs.get("terminated")) or bool(obs.get("truncated")):
            return "noop"
        if stage not in ("ramp", "rivet"):
            return "noop"

        if stage == "rivet" and stage_changed:
            self.start_astra(obs, "verified rivet entry")
        elif stage == "ramp" and self.route_failures >= 2 and self.astra_future is None and not self.astra_started:
            self.start_astra(obs, "repeated measured connector stalls")

        ongoing = self.jump_action(obs)
        if ongoing is not None:
            return ongoing
        emergency = self.hazard_action(obs, stage)
        if emergency is not None:
            # The code has already committed to the hard safety action.  Jev is
            # queried only as a parallel diagnostic at the next boundary.
            return emergency

        if stage == "ramp":
            action = self.ramp_action(obs)
            event = "ramp_stall" if self.climb_stall >= 24 or self.walk_stall >= 10 else ""
            if event:
                self.request_jev(obs, event, action)
                action = self.apply_jev(obs, action, event)
        else:
            action = self.rivet_action(obs)
            self.request_jev(obs, "rivet_route", action)
            action = self.apply_jev(obs, action, "rivet_route")

        if stage_changed or life_changed or screen_changed:
            self.log("decision_boundary", stage=stage, phase=self.phase,
                     row=self.rivet_row, action=action, lives=lives, screens=screens)
        if self.tick % 20 == 0:
            p = player_of(obs)
            fitted = self.plane.fit()
            hazard = self.hazard_facts(obs, stage)
            self.log("snapshot", tick=self.tick, stage=stage, phase=self.phase,
                     branch=self.branch, row=self.rivet_row,
                     x=round(num(p.get("x"), 0)), y=round(num(p.get("y"), 0)),
                     action=action, climb=self.climb_open,
                     jump=self.jump_direction, raw_jump=jump_byte(obs),
                     plane="learned" if fitted is not None else "fallback",
                     plane_slope=round(fitted[0], 3) if fitted is not None else None,
                     hazard=hazard.get("urgency"), row_error=hazard.get("row_error"),
                     lives=lives, score=num(obs.get("score_total"), 0.0))
        return action

    def run(self):
        try:
            obs = self.ctx.observe()
        except Exception:
            return
        if not isinstance(obs, dict):
            obs = {}
        self._last_obs = obs
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
            # `None` means the final transition was deliberately bounded.
            if action is None:
                break
            if not self.post(action):
                break
        self.log("finished", ticks=self.tick, score=getattr(self.game, "score", None),
                 phase=self.phase, branch=self.branch, jumps=self.jump_count,
                 screens=self.last_screens)


class Program:
    def __init__(self, ctx):
        self.ctx = ctx

    def run_episode(self, game) -> None:
        Controller(self.ctx, game).run()


def build(ctx):
    return Program(ctx)
