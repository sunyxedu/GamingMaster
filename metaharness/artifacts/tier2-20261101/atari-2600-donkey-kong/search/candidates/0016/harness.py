"""Adaptive connector controller for Atari 2600 Donkey Kong.

The actor is a small event-driven machine rather than a model-driven joystick:
walks are checked for actual horizontal motion, ladder connectors are selected
from a finite graph, and a stalled connector is replaced by the other visible
connector on that band.  A jump is a transaction whose FIRE and landing phases
are separated by the observed RAM jump byte.  Jev only arbitrates a live
connector/hazard event; Astra reviews a repeated route deadlock asynchronously.
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

# Values are Mario's observed left edge, not the ladder box's left edge.  The
# second member on a split band is the short connector visible at x=80/88/68.
# Unlike a fixed route, this table is a graph: a connector becomes preferred only
# after an observation proves that its climb advanced.
RAMP_GRAPH = (
    (176, (106,)),
    (148, (46, 78)),
    (120, (106, 86)),
    (92, (46, 66)),
    (64, (106,)),
    (40, (30, 74)),
)
RIVET_ROWS_Y = (176, 148, 120, 92)
RIVET_SIDE_X = {"left": 46, "right": 106}


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


def visible(value):
    return isinstance(value, dict) and value.get("visible", True) is not False


def raw_byte(obs, address):
    raw = obs.get("raw_ram", []) if isinstance(obs, dict) else []
    if not isinstance(raw, (list, tuple)):
        return 0
    index = address & 0x7F
    return integer(raw[index], 0) if index < len(raw) else 0


def player_xy(obs):
    p = player_of(obs)
    return number(p.get("x"), 45.0), number(p.get("y"), 176.0)


def normalized_rows(obs):
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


def json_object(text):
    """Extract one defensive JSON object from arbitrary Astra text."""
    if not isinstance(text, str):
        return None
    stripped = text.strip()
    candidates = []
    left = stripped.find("{")
    right = stripped.rfind("}")
    if left >= 0 and right > left:
        candidates.append(stripped[left:right + 1])
    candidates.append(stripped)
    for candidate in candidates:
        try:
            value = json.loads(candidate)
        except (TypeError, ValueError):
            continue
        if isinstance(value, dict):
            return value
    return None


def answer_probability(answer, option):
    probabilities = getattr(answer, "probabilities", None)
    if not isinstance(probabilities, dict):
        return None
    value = number(probabilities.get(option), -1.0)
    return value if value >= 0.0 else None


def jump_action(direction, fire):
    if direction not in ("left", "right"):
        return "fire" if fire else "noop"
    return direction + ("fire" if fire else "")


def hazard_facts(obs):
    """Find the nearest closing object on Mario's current horizontal band.

    The decoder's top-left y values are used consistently for both sprites.  A
    narrow band avoids the false interrupts seen when an object on the adjacent
    sloping girder was treated as a collision.  Code computes this arithmetic;
    no model is asked to compare coordinates.
    """
    px, py = player_xy(obs)
    hazards = obs.get("hazards", []) if isinstance(obs, dict) else []
    if not isinstance(hazards, list):
        hazards = []
    found = None
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
        closing = ((delta < 0.0 and dx > 0.2) or
                   (delta > 0.0 and dx < -0.2) or
                   (distance <= 15.0 and abs(dx) <= 0.2))
        if not closing and distance > 17.0:
            continue
        if distance <= 14.0:
            urgency = "overlap"
        elif distance <= 22.0:
            urgency = "immediate"
        elif closing and distance <= 39.0:
            urgency = "soon"
        elif closing and distance <= 60.0:
            urgency = "watch"
        else:
            urgency = "near"
        fact = {
            "present": True,
            "slot": integer(item.get("slot"), -1),
            "side": "left" if delta < 0.0 else "right",
            "motion": "left" if dx < -0.2 else "right" if dx > 0.2 else "still",
            "distance": round(distance, 1),
            "distance_bucket": (
                "overlap" if distance <= 14.0 else
                "near" if distance <= 27.0 else
                "approaching" if distance <= 39.0 else "far"
            ),
            "closing": bool(closing),
            "urgency": urgency,
        }
        rank = (0 if urgency == "overlap" else
                1 if urgency == "immediate" else
                2 if urgency == "soon" else
                3 if urgency == "watch" else 4, distance)
        if found is None or rank < found[0]:
            found = (rank, fact)
    if found is not None:
        return found[1]
    return {
        "present": False, "slot": -1, "side": "none", "motion": "none",
        "distance": 999.0, "distance_bucket": "none", "closing": False,
        "urgency": "clear",
    }


class Controller:
    """Owns one episode and keeps model judgments off the tick-critical path."""

    def __init__(self, ctx, game):
        self.ctx = ctx
        self.game = game
        self.tick = 0
        self.last_action = "noop"
        self.last_stage = None
        self.last_lives = None
        self.last_screens = 0

        # Route graph state.  preferred remembers successful connectors across
        # lives, while failed records only a demonstrated no-motion deadlock.
        self.phase = 0
        self.branch = 0
        self.preferred = {}
        self.failed = set()
        self.phase_start_y = None
        self.climb_open = False
        self.climb_age = 0
        self.climb_best_y = None
        self.climb_no_progress = 0
        self.walk_stall = 0
        self.previous_x = None
        self.deadlock_count = 0
        self.deadlock_pending = False
        self.deadlock_signal = False
        self.last_deadlock_phase = -1

        # Jump state has a FIRE edge, a RAM-confirmed airborne section, and a
        # same-direction landing tail.  The tail is intentionally not a second
        # route decision: it gets Mario away from the meeting point.
        self.jump_direction = None
        self.jump_seen = False
        self.jump_guard = 0
        self.jump_tail = 0
        self.jump_lock = 0
        self.jump_slot = -1
        self.jump_count = 0

        # Rivet bookkeeping is a separate graph with verified row bytes.
        self.rivet_row = 0
        self.rivet_preference = "left"
        self.rivet_previous = None
        self.rivet_removed = {}
        self.gap_direction = None

        # Cartridge lifecycle is code-owned.
        self.idle_attempts = 0
        self.transition_age = 0
        self.restart_attempts = 0

        # Jev is an event arbiter.  One batch can answer both a connector and a
        # hazard recognition question, but only if its state signature survives.
        self.jev_future = None
        self.jev_stamp = None
        self.jev_started_tick = -10000
        self.jev_answer = None
        self.jev_answer_stamp = None
        self.jev_answer_tick = -10000

        # Astra is a slow route reviewer, invoked by evidence of a deadlock or
        # entry to the uncalibrated rivet screen.  It never blocks this actor.
        self.astra_future = None
        self.astra_calls = 0
        self.planner_mode = {"connector": "adaptive", "hazard": "jump", "first_side": "left"}

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

    def route_target(self, obs, phase=None, branch=None):
        p = self.phase if phase is None else max(0, min(int(phase), len(RAMP_GRAPH) - 1))
        options = RAMP_GRAPH[p][1]
        b = self.branch if branch is None else max(0, min(int(branch), len(options) - 1))
        desired = options[b]
        # Verify the public prior against the current fixed ladder boxes.  The
        # offset is applied only after matching the intended row and x.
        ladders = obs.get("ladders", []) if isinstance(obs, dict) else []
        if isinstance(ladders, list):
            same = []
            for item in ladders:
                if not visible(item):
                    continue
                lx = number(item.get("x"), 0.0)
                ly = number(item.get("y"), 0.0)
                if abs(ly - RAMP_GRAPH[p][0]) <= 2.0 and abs((lx - 2.0) - desired) <= 4.0:
                    same.append(lx - 2.0)
            if same:
                return min(same, key=lambda x: abs(x - desired))
        return float(desired)

    def _reset_active(self, stage):
        self.phase = 0
        self.branch = 0
        self.phase_start_y = None
        self.climb_open = False
        self.climb_age = 0
        self.climb_best_y = None
        self.climb_no_progress = 0
        self.walk_stall = 0
        self.previous_x = None
        self.deadlock_pending = False
        self.deadlock_signal = False
        self.clear_jump()
        self.rivet_row = 0
        self.rivet_preference = self.planner_mode.get("first_side", "left")
        self.rivet_previous = None
        self.rivet_removed = {}
        self.gap_direction = None
        self._choose_branch(0, reset=True)
        self.log("active_reset", stage=stage, preferred=dict(self.preferred), failed=list(self.failed))

    def _sync(self, obs):
        stage = stage_of(obs)
        lives = integer(obs.get("lives_remaining"), 0) if isinstance(obs, dict) else 0
        screens = integer(obs.get("screens_completed"), 0) if isinstance(obs, dict) else 0
        stage_changed = stage != self.last_stage
        life_changed = self.last_lives is not None and lives != self.last_lives
        screen_changed = screens != self.last_screens
        if stage_changed or life_changed or screen_changed:
            self.log("boundary", old_stage=self.last_stage, stage=stage,
                     old_lives=self.last_lives, lives=lives,
                     old_screens=self.last_screens, screens=screens)
            if stage in ("ramp", "rivet"):
                # Do not erase preferred/failed connectors: a life is an online
                # experiment, and the next Mario should exploit its evidence.
                self._reset_active(stage)
        if stage == "transition":
            self.transition_age = self.transition_age + 1 if self.last_stage == "transition" else 1
        else:
            self.transition_age = 0
        if stage_changed:
            self.idle_attempts = 0 if stage != "idle" else self.idle_attempts
            if stage != "transition":
                self.restart_attempts = 0
        self.last_stage, self.last_lives, self.last_screens = stage, lives, screens
        return stage, lives, screens, stage_changed, life_changed, screen_changed

    def _choose_branch(self, phase, reset=False):
        phase = max(0, min(int(phase), len(RAMP_GRAPH) - 1))
        options = RAMP_GRAPH[phase][1]
        if not options:
            self.branch = 0
            return
        chosen = self.preferred.get(phase, 0)
        if chosen >= len(options) or (phase, chosen) in self.failed:
            chosen = next((i for i in range(len(options)) if (phase, i) not in self.failed), 0)
        self.branch = chosen
        if reset:
            self.log("connector_selected", phase=phase, branch=chosen,
                     target=options[chosen])

    def _try_other_connector(self, reason):
        options = RAMP_GRAPH[self.phase][1]
        old = self.branch
        self.failed.add((self.phase, old))
        candidates = [i for i in range(len(options)) if i != old and (self.phase, i) not in self.failed]
        if not candidates:
            # A single connector gets a fresh attempt after a small lateral
            # reacquisition, rather than oscillating UP/DOWN forever.
            self.failed.discard((self.phase, old))
            candidates = [old]
        self.branch = candidates[0]
        self.climb_open = False
        self.climb_age = 0
        self.phase_start_y = None
        self.climb_best_y = None
        self.climb_no_progress = 0
        self.walk_stall = 0
        self.deadlock_pending = True
        self.deadlock_signal = True
        self.deadlock_count += 1
        self.last_deadlock_phase = self.phase
        self.log("connector_replaced", phase=self.phase, old=old,
                 new=self.branch, reason=reason, target=options[self.branch])

    def _mark_climb_success(self, old_phase, y):
        self.preferred[old_phase] = self.branch
        self.failed.discard((old_phase, self.branch))
        self.phase += 1
        self._choose_branch(self.phase)
        self.climb_open = False
        self.climb_age = 0
        self.phase_start_y = None
        self.climb_best_y = None
        self.climb_no_progress = 0
        self.walk_stall = 0
        self.deadlock_pending = False
        self.log("connector_confirmed", old_phase=old_phase, phase=self.phase,
                 branch=self.preferred.get(old_phase), x=round(player_xy(self._obs)[0], 1),
                 y=round(y, 1))

    def clear_jump(self):
        self.jump_direction = None
        self.jump_seen = False
        self.jump_guard = 0
        self.jump_tail = 0
        self.jump_lock = 0
        self.jump_slot = -1

    def _start_jump(self, direction, reason, hazard=None):
        if direction not in ("left", "right"):
            direction = "right"
        self.jump_direction = direction
        self.jump_seen = False
        self.jump_guard = 4
        self.jump_tail = 0
        self.jump_lock = 14
        self.jump_slot = integer((hazard or {}).get("slot"), -1)
        self.jump_count += 1
        self.log("jump_started", tick=self.tick, direction=direction,
                 reason=reason, slot=self.jump_slot,
                 hazard={k: (hazard or {}).get(k) for k in
                         ("side", "motion", "distance_bucket", "urgency")})
        return jump_action(direction, True)

    def _jump_step(self, obs):
        """Return the committed action, or None once the transaction is done."""
        if self.jump_direction is None:
            if self.jump_lock > 0:
                self.jump_lock -= 1
            return None
        direction = self.jump_direction
        state = raw_byte(obs, 0x85)
        if state > 0:
            self.jump_seen = True
            self.jump_guard = 3
            return jump_action(direction, True)
        if not self.jump_seen and self.jump_guard > 0:
            self.jump_guard -= 1
            return jump_action(direction, True)
        if self.jump_seen and self.jump_tail < 4:
            self.jump_tail += 1
            # Keep the latched direction after FIRE.  Reversing here was the
            # previous controller's unexplained landing failure mode.
            return jump_action(direction, False)
        self.log("jump_finished", tick=self.tick, direction=direction,
                 seen=self.jump_seen, slot=self.jump_slot)
        self.clear_jump()
        # clear_jump is also used for life resets; retain a short post-landing
        # lock here so the same sprite cannot immediately retrigger FIRE.
        self.jump_lock = 14
        return None

    def _jump_direction(self, obs, hazard):
        motion = hazard.get("motion")
        if motion == "left":
            return "right"
        if motion == "right":
            return "left"
        px, _ = player_xy(obs)
        target = self.route_target(obs)
        if abs(target - px) > 3:
            return "right" if target > px else "left"
        if hazard.get("side") == "left":
            return "right"
        if hazard.get("side") == "right":
            return "left"
        return "right"

    def _hazard_jump(self, obs, stage):
        if stage not in ("ramp", "rivet") or self.jump_direction is not None or self.jump_lock > 0:
            return None
        hazard = hazard_facts(obs)
        if not hazard.get("present"):
            return None
        distance = number(hazard.get("distance"), 999.0)
        # On a rung, climbing is the escape: do not throw away a nearly
        # completed connector for a hazard on the old girder.  Only an overlap
        # can interrupt vertical progress.
        climbing_escape = self.climb_open and self.climb_no_progress < 3 and distance > 13.0
        if climbing_escape:
            return None
        if hazard.get("urgency") not in ("overlap", "immediate"):
            return None
        # A warning on the adjacent part of a slope is not yet a crossing.
        # Wait for the verified current-row window; this preserves the close
        # jump timing that earned the parent's 100-point crossings.
        return self._start_jump(self._jump_direction(obs, hazard), "hazard", hazard)

    def _ramp_action(self, obs):
        self._obs = obs
        px, py = player_xy(obs)
        phase = max(0, min(self.phase, len(RAMP_GRAPH) - 1))
        row_y = RAMP_GRAPH[phase][0]
        target = self.route_target(obs, phase, self.branch)

        # A climb remains open while there is vertical response even if a slope
        # changes Mario's x.  This is the key difference from x-locking the
        # transaction to a fixed pixel.
        if self.climb_open:
            self.climb_age += 1
            if self.climb_best_y is None or py < self.climb_best_y - 1.0:
                self.climb_best_y = py
                self.climb_no_progress = 0
            else:
                self.climb_no_progress += 1
            if phase + 1 < len(RAMP_GRAPH):
                next_y = RAMP_GRAPH[phase + 1][0]
                gain = ((self.phase_start_y - py) if self.phase_start_y is not None else 0.0)
                if py <= next_y + 6.0 and gain >= 8.0 and self.climb_age >= 5:
                    old = phase
                    self._mark_climb_success(old, py)
                    return self._ramp_action(obs)
            if self.climb_no_progress >= 18:
                if len(RAMP_GRAPH[phase][1]) > 1:
                    self._try_other_connector("climb_no_progress")
                    return self._ramp_action(obs)
                self.climb_open = False
                self.climb_age = 0
                self.phase_start_y = None
                self.climb_best_y = None
                self.climb_no_progress = 0
                self.walk_stall = 0
            else:
                return "up"

        # A horizontal command that leaves x unchanged is a geometric signal,
        # not a reason to keep holding it.  At phase 2 this recovers the x=87/111
        # deadlock by selecting the x=86 connector at the same band.
        if self.previous_x is not None and self.last_action in ("left", "right"):
            if abs(px - self.previous_x) < 0.6:
                self.walk_stall += 1
            else:
                self.walk_stall = 0
        else:
            self.walk_stall = 0
        self.previous_x = px

        if abs(px - target) > 1.0:
            if self.walk_stall >= 8 and len(RAMP_GRAPH[phase][1]) > 1:
                self._try_other_connector("horizontal_no_motion")
                return self._ramp_action(obs)
            return "right" if px < target else "left"

        # The bottom FIRE start leaves Mario a few pixels above the nominal row;
        # a real ladder response is still safe when the feet are aligned.
        if py <= row_y + 11.0 or abs(px - target) <= 2.0:
            self.climb_open = True
            self.climb_age = 0
            self.phase_start_y = py
            self.climb_best_y = py
            self.climb_no_progress = 0
            self.walk_stall = 0
            self.log("climb_open", phase=phase, branch=self.branch,
                     target=round(target, 1), x=round(px, 1), y=round(py, 1))
            return "up"
        return "up"

    def _rivet_ladder_target(self, obs, side):
        desired = 106 if side == "right" else 46
        px, py = player_xy(obs)
        ladders = obs.get("ladders", []) if isinstance(obs, dict) else []
        candidates = []
        if isinstance(ladders, list):
            for item in ladders:
                if not visible(item):
                    continue
                x = number(item.get("x"), 0.0) - 2.0
                y = number(item.get("y"), 0.0)
                if abs(y - py) <= 22.0 or abs(y - (py - 8.0)) <= 22.0:
                    candidates.append((abs(x - desired), x))
        return min(candidates)[1] if candidates else float(desired)

    def _remaining_side(self, row, mask):
        if mask == 18:
            return None
        if mask == 6:
            return "right"
        if mask == 12:
            return "left"
        if row == 0 and self.rivet_preference in ("left", "right"):
            return self.rivet_preference
        return "left" if row % 2 == 0 else "right"

    def _rivet_action(self, obs):
        rows = normalized_rows(obs)
        if self.rivet_previous is not None:
            for i, (old, new) in enumerate(zip(self.rivet_previous, rows)):
                if old != new and i == self.rivet_row and new in (6, 12, 18):
                    removed = "left" if new == 6 else "right" if new == 12 else None
                    if removed:
                        self.rivet_removed[i] = removed
                        self.gap_direction = "right" if removed == "left" else "left"
                        self.log("rivet_cleared", row=i, removed=removed, new=new)
        self.rivet_previous = list(rows)

        while self.rivet_row < 4 and rows[self.rivet_row] == 18:
            self.rivet_row += 1
            self.gap_direction = None
        if self.rivet_row >= 4:
            return "up"

        row = self.rivet_row
        mask = rows[row]
        side = self._remaining_side(row, mask)
        if self.gap_direction in ("left", "right"):
            direction = self.gap_direction
            self.gap_direction = None
            return self._start_jump(direction, "rivet_gap")

        px, py = player_xy(obs)
        target_y = RIVET_ROWS_Y[row]
        if side is None:
            climb_side = "right" if row % 2 == 0 else "left"
            target = self._rivet_ladder_target(obs, climb_side)
            if abs(px - target) > 2.0:
                return "right" if px < target else "left"
            return "up"
        if py > target_y + 10.0:
            climb_side = "right" if row % 2 == 0 else "left"
            target = self._rivet_ladder_target(obs, climb_side)
            if abs(px - target) > 2.0:
                return "right" if px < target else "left"
            return "up"

        target_x = RIVET_SIDE_X[side]
        removed = self.rivet_removed.get(row)
        if removed and removed != side:
            gap_x = RIVET_SIDE_X[removed]
            if (px < gap_x < target_x) or (px > gap_x > target_x):
                return self._start_jump("right" if target_x > px else "left", "rivet_gap")
        if abs(px - target_x) > 7.0:
            return "right" if px < target_x else "left"
        # Continue through the rivet rather than parking on its sprite.
        return "left" if side == "left" else "right"

    def _signature(self, obs, context):
        hazard = hazard_facts(obs)
        return (
            context, stage_of(obs), integer(obs.get("lives_remaining"), 0),
            self.phase, self.branch, self.rivet_row,
            hazard.get("slot", -1), hazard.get("distance_bucket", "none"),
            hazard.get("motion", "none"), self.deadlock_count,
        )

    def _model_state(self, obs, context, fallback):
        px, py = player_xy(obs)
        hazard = hazard_facts(obs)
        return {
            "game": "Atari 2600 Donkey Kong NTSC retail cartridge",
            "context": context,
            "stage": stage_of(obs),
            "player": {
                "horizontal": "left" if px < 52 else "right" if px > 100 else "middle",
                "vertical": "low" if py > 160 else "middle" if py > 90 else "high",
                "motion": "up" if number(player_of(obs).get("dy"), 0) < -1 else
                          "down" if number(player_of(obs).get("dy"), 0) > 1 else "level",
                "jump_committed": self.jump_direction is not None,
            },
            "route": {
                "fallback_action": fallback,
                "ramp_phase": self.phase,
                "connector": self.branch,
                "connector_options": len(RAMP_GRAPH[min(self.phase, len(RAMP_GRAPH) - 1)][1]),
                "climb_open": self.climb_open,
                "horizontal_stall": self.walk_stall,
                "deadlocks": self.deadlock_count,
                "rivet_row": self.rivet_row,
                "rivet_rows": normalized_rows(obs),
            },
            "hazard": {key: hazard.get(key) for key in
                       ("present", "slot", "side", "motion", "distance_bucket", "closing", "urgency")},
            "progress": {
                "lives": integer(obs.get("lives_remaining"), 0),
                "screens": integer(obs.get("screens_completed"), 0),
                "rivets_removed": integer(obs.get("rivets_removed"), 0),
            },
            "last_action": self.last_action,
        }

    def _request_jev(self, obs, context, fallback):
        if self.jev_future is not None or context == "clear":
            return
        try:
            if self.ctx.done or self.ctx.budget.remaining().get("system1_calls", 0) <= 0:
                return
            if getattr(self.ctx, "system1", True) is None:
                return
        except Exception:
            return
        stamp = self._signature(obs, context)
        # Avoid paying repeatedly for an unchanged event while the action state
        # machine is already doing the safe deterministic thing.
        if stamp == self.jev_stamp and self.tick - self.jev_started_tick < 10:
            return
        state = self._model_state(obs, context, fallback)
        if context == "deadlock":
            tactic_instruction = (
                "This is a connector deadlock. The code has supplied the exact legal "
                "fallback action and connector index. Choose try_other only when the "
                "alternate connector on this band is the sensible upward route; do not "
                "invent coordinates or a joystick sequence."
            )
        else:
            tactic_instruction = (
                "This is a hazard interrupt. Choose a jump direction only if the named "
                "same-row object is soon or overlapping. Otherwise preserve the supplied "
                "fallback route. Do not calculate distances."
            )
        questions = {
            "tactic": Choice(
                tactic_instruction,
                {
                    "keep_route": "Keep the code-computed fallback and current connector.",
                    "try_other": "Use the other connector already listed by route.connector_options.",
                    "jump_left": "Commit to a leftward directional jump for the named hazard.",
                    "jump_right": "Commit to a rightward directional jump for the named hazard.",
                    "pause": "Use neutral only while an explicit transition or unsettled state is shown.",
                },
            ),
            "danger": Noul(
                "Does hazard.urgency explicitly show a same-row closing object that is soon or overlapping?",
                {
                    "true": "hazard.urgency is soon, immediate, or overlap and hazard.closing is true.",
                    "false": "No soon/overlapping same-row closing object is present.",
                },
            ),
        }
        self.jev_stamp = stamp
        self.jev_started_tick = self.tick
        self.jev_future = None
        try:
            self.jev_future = self.ctx.spawn(
                lambda: self.ctx.decide(state, questions, timeout_s=0.38)
            )
        except Exception as exc:
            self.log("jev_spawn_failed", error=type(exc).__name__)

    def _collect_jev(self, obs):
        future = self.jev_future
        if future is None or not future.done():
            return
        self.jev_future = None
        try:
            answers = future.result()
        except Exception as exc:
            self.log("jev_failed", error=type(exc).__name__)
            return
        if not isinstance(answers, dict):
            return
        answer = answers.get("tactic")
        value = getattr(answer, "value", None)
        if value not in ("keep_route", "try_other", "jump_left", "jump_right", "pause"):
            return
        probability = answer_probability(answer, value)
        if probability is not None and probability < 0.55:
            return
        self.jev_answer = value
        self.jev_answer_stamp = self.jev_stamp
        self.jev_answer_tick = self.tick
        self.log("jev_tactic", value=value, probability=probability,
                 context=self.jev_stamp[0] if self.jev_stamp else "unknown")

    def _apply_jev(self, obs, fallback):
        value = self.jev_answer
        if value is None:
            return fallback
        current = self._signature(obs, self.jev_answer_stamp[0] if self.jev_answer_stamp else "clear")
        if current != self.jev_answer_stamp or self.tick - self.jev_answer_tick > 3:
            self.jev_answer = None
            return fallback
        context = self.jev_answer_stamp[0]
        hazard = hazard_facts(obs)
        if value == "try_other" and context == "deadlock":
            self.jev_answer = None
            if len(RAMP_GRAPH[self.phase][1]) > 1:
                self._try_other_connector("jev_deadlock")
                return self._ramp_action(obs)
        if value in ("jump_left", "jump_right") and context == "hazard":
            climbing_escape = self.climb_open and number(hazard.get("distance"), 999.0) > 13.0
            if (not climbing_escape and
                    hazard.get("urgency") in ("overlap", "immediate") and
                    self.jump_direction is None):
                direction = value.rsplit("_", 1)[-1]
                self.jev_answer = None
                return self._start_jump(direction, "jev_hazard", hazard)
        self.jev_answer = None
        return fallback

    def _start_astra(self, obs, reason):
        if self.astra_future is not None or self.astra_calls >= 3:
            return
        try:
            if self.ctx.done or self.ctx.budget.remaining().get("system2_calls", 0) <= 0:
                return
        except Exception:
            return
        self.astra_calls += 1
        context = "route deadlock" if self.deadlock_count else "rivet entry"
        state = self._model_state(obs, context, "code route")
        prompt = (
            "You are Astra, the slow macro reviewer for the NTSC Atari 2600 Donkey Kong "
            "retail port. The real-time actor already owns exact joystick timing, ladder "
            "alignment, raw rivet bookkeeping, hazard arithmetic, and restart edges. Do "
            "not return an action sequence. Use only the supplied fixed ladder graph and "
            "observed facts, not arcade boards. Return one compact JSON object with "
            "connector (nominal, alternate, or adaptive), hazard (climb or jump), and "
            "first_side (left or right), each as a short label. It is advisory and may "
            "be ignored if the next observation contradicts it. Reason: " + str(reason) +
            "\nFacts:\n" + json.dumps(state, separators=(",", ":"), allow_nan=False)
        )
        try:
            self.astra_future = self.ctx.spawn(
                lambda: self.ctx.think(prompt, timeout_s=95.0)
            )
        except Exception as exc:
            self.astra_future = None
            self.log("astra_spawn_failed", error=type(exc).__name__)

    def _collect_astra(self):
        future = self.astra_future
        if future is None or not future.done():
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
        connector = parsed.get("connector")
        hazard = parsed.get("hazard")
        side = parsed.get("first_side")
        if connector in ("nominal", "alternate", "adaptive"):
            self.planner_mode["connector"] = connector
        if hazard in ("climb", "jump"):
            self.planner_mode["hazard"] = hazard
        if side in ("left", "right"):
            self.planner_mode["first_side"] = side
        self.log("astra_policy", connector=self.planner_mode["connector"],
                 hazard=self.planner_mode["hazard"], first_side=self.planner_mode["first_side"])

    def choose(self, obs):
        stage, lives, screens, stage_changed, life_changed, screen_changed = self._sync(obs)
        self._collect_jev(obs)
        self._collect_astra()

        if stage == "idle":
            if self.idle_attempts < 3:
                self.idle_attempts += 1
                return "fire", False
            return "noop", False
        if stage == "game_over" or bool(obs.get("terminated")) or bool(obs.get("truncated")):
            return "noop", True
        if stage == "transition":
            if lives <= 0:
                return "noop", True
            if (self.transition_age >= 18 and self.restart_attempts < 4 and
                    (self.transition_age - 18) % 28 == 0):
                self.restart_attempts += 1
                return "fire", False
            return "noop", False
        if stage not in ("ramp", "rivet"):
            return "noop", False

        if stage_changed and stage == "rivet":
            self._start_astra(obs, "the ramp screen completed")
        elif self.deadlock_signal and self.astra_future is None:
            self._start_astra(obs, "a connector made no horizontal or vertical progress")
            self.deadlock_signal = False

        ongoing = self._jump_step(obs)
        if ongoing is not None:
            return ongoing, False

        # Let the route transaction inspect the current observation before the
        # hazard guard.  If Mario is already aligned and climbing, UP is the
        # escape; otherwise the guard owns the directional jump.
        if stage == "ramp":
            fallback = self._ramp_action(obs)
        else:
            fallback = self._rivet_action(obs)
        hazard = hazard_facts(obs)
        emergency = self._hazard_jump(obs, stage)
        if emergency is not None:
            self._request_jev(obs, "hazard", emergency)
            return emergency, False

        if self.deadlock_pending:
            self._request_jev(obs, "deadlock", fallback)
            self.deadlock_pending = False
        elif hazard.get("urgency") in ("immediate", "overlap"):
            self._request_jev(obs, "hazard", fallback)
        action = self._apply_jev(obs, fallback)

        if stage_changed or life_changed or screen_changed:
            self.log("decision_boundary", stage=stage, phase=self.phase,
                     branch=self.branch, action=action, lives=lives, screens=screens)
        if self.tick % 20 == 0:
            px, py = player_xy(obs)
            self.log("snapshot", tick=self.tick, stage=stage, phase=self.phase,
                     branch=self.branch, x=round(px), y=round(py), action=action,
                     climb=self.climb_open, stall=self.walk_stall,
                     jump=self.jump_direction, raw_jump=raw_byte(obs, 0x85),
                     hazard=hazard.get("urgency"), lives=lives,
                     score=number(obs.get("score_total"), 0.0))
        return action, False

    def run(self):
        try:
            obs = self.ctx.observe()
        except Exception:
            return
        if not isinstance(obs, dict):
            obs = {}
        self._obs = obs
        self.last_stage = stage_of(obs)
        self.last_lives = integer(obs.get("lives_remaining"), 0)
        self.last_screens = integer(obs.get("screens_completed"), 0)
        if self.last_stage == "idle":
            if not self.post("fire"):
                return
        else:
            action, stop = self.choose(obs)
            if stop or not self.post(action):
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
            action, stop = self.choose(obs)
            if stop or self.done():
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
