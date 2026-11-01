"""Adaptive two-lane Donkey Kong controller.

The controller keeps the real-time path deterministic.  A route transaction
chooses a fixed ladder edge, while a separate landing transaction reconciles a
jump with whichever split connector Mario actually reaches.  The important
case is the phase-2 landing near x=87: it is treated as the x=86 connector,
not as a reason to keep walking toward x=106.

Jev and Astra are asynchronous reviewers of route ambiguity.  They never own
a tick, and every answer is checked against the observation signature that
created it.
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

# These are Mario's useful left-edge positions.  The second entry is a real
# short connector on a split row, not a free-form coordinate invented at run
# time.  The y values are the public ladder rows.
RAMP_Y = (176, 148, 120, 92, 64, 40)
CONNECTORS = ((106,), (46, 78), (106, 86), (46, 66), (106,), (30, 74))
RIVET_Y = (176, 148, 120, 92)
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


def stage(obs):
    return obs.get("stage", "unknown") if isinstance(obs, dict) else "unknown"


def player(obs):
    value = obs.get("player", {}) if isinstance(obs, dict) else {}
    return value if isinstance(value, dict) else {}


def xy(obs):
    p = player(obs)
    return num(p.get("x"), 45.0), num(p.get("y"), 176.0)


def visible(value):
    return isinstance(value, dict) and value.get("visible", True) is not False


def rows(obs):
    value = obs.get("rivet_rows", []) if isinstance(obs, dict) else []
    if not isinstance(value, (list, tuple)):
        value = []
    answer = []
    for item in list(value)[:4]:
        n = integer(item, 0)
        answer.append(n if n in (0, 6, 12, 18) else 0)
    while len(answer) < 4:
        answer.append(0)
    return answer


def raw_byte(obs, address):
    raw = obs.get("raw_ram", []) if isinstance(obs, dict) else []
    index = int(address) & 0x7F
    if isinstance(raw, (list, tuple)) and index < len(raw):
        return integer(raw[index], 0)
    return 0


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


def hazard_on_row(obs):
    """Compute a small semantic hazard record; Jev never does this arithmetic."""
    px, py = xy(obs)
    values = obs.get("hazards", []) if isinstance(obs, dict) else []
    if not isinstance(values, list):
        values = []
    best = None
    for item in values:
        if not visible(item):
            continue
        hx, hy = num(item.get("x")), num(item.get("y"))
        if abs(hy - py) > 19.0:
            continue
        dx = num(item.get("dx"))
        delta = hx - px
        distance = abs(delta)
        closing = ((delta < 0 and dx > 0.2) or
                   (delta > 0 and dx < -0.2) or
                   (distance <= 15 and abs(dx) <= 0.2))
        if not closing and distance > 18:
            continue
        if distance <= 15:
            urgency = "overlap"
        elif distance <= 22:
            urgency = "commit"
        elif closing and distance <= 42:
            urgency = "soon"
        elif closing:
            urgency = "watch"
        else:
            urgency = "near"
        fact = {
            "present": True,
            "slot": integer(item.get("slot"), -1),
            "side": "left" if delta < 0 else "right",
            "motion": "left" if dx < -0.2 else "right" if dx > 0.2 else "still",
            "closing": bool(closing),
            "distance": round(distance, 1),
            "distance_bucket": (
                "overlap" if distance <= 15 else
                "near" if distance <= 30 else
                "approaching" if distance <= 48 else "far"
            ),
            "urgency": urgency,
        }
        rank = (0 if urgency == "overlap" else
                1 if urgency == "commit" else
                2 if urgency == "soon" else
                3 if urgency == "watch" else 4, distance)
        if best is None or rank < best[0]:
            best = (rank, fact)
    if best is not None:
        return best[1]
    return {
        "present": False, "slot": -1, "side": "none", "motion": "none",
        "closing": False, "distance": 999.0, "distance_bucket": "none",
        "urgency": "clear",
    }


def remaining_side(mask, preference):
    if mask == 18:
        return None
    if mask == 6:
        return "right"
    if mask == 12:
        return "left"
    return preference if preference in ("left", "right") else "left"


def side_present(mask, side):
    if mask == 0:
        return True
    if mask == 6:
        return side == "right"
    if mask == 12:
        return side == "left"
    return False


class Controller:
    def __init__(self, ctx, game):
        self.ctx = ctx
        self.game = game
        self.tick = 0
        self.last_action = "noop"
        self.last_stage = None
        self.last_lives = None
        self.last_screens = 0

        # Route state.  phase is observed walkway number; branch is a member of
        # the finite connector tuple for that walkway.
        self.phase = 0
        self.branch = 0
        self.preferred = {}
        self.failed_edges = set()
        self.climb_open = False
        self.climb_origin = None
        self.climb_best = None
        self.climb_age = 0
        self.climb_stall = 0
        self.walk_stall = 0
        self.previous_x = None
        self.route_failures = 0
        self.last_active_phase = 0
        self.route_event = None

        # The first Mario gets a genuine upward-route attempt.  After a life
        # boundary the risk budget changes to the durable score lane, unless a
        # rivet screen has already been reached.  A route stall in the same life
        # also falls back rather than spending the entire bonus on a deadlock.
        self.farm_next_life = False
        self.farm_mode = False
        self.farm_target = 106.0
        self.farm_armed = False

        # Jump is a transaction.  Ramp route crossings use horizontal FIRE and
        # an opposite-direction landing tail; the score lane keeps the same
        # direction, the mechanism that made 0018 durable.
        self.jump_direction = None
        self.jump_seen = False
        self.jump_guard = 8
        self.jump_tail = 0
        self.jump_tail_direction = None
        self.jump_lock = 0
        self.post_jump_up = 0
        self.jump_phase = 0
        self.jump_reason = None
        self.jump_count = 0

        # Lifecycle is edge-triggered.  lives_remaining is spare lives, so zero
        # does not suppress the restart edge for the currently visible Mario.
        self.idle_edges = 0
        self.transition_age = 0
        self.restart_edges = 0
        self.final_transition_age = 0

        # Rivet state is reset only at a screen/life boundary and otherwise
        # follows the observed row bytes.
        self.rivet_row = 0
        self.rivet_preference = "left"
        self.previous_rows = None
        self.gap_direction = None
        self.rivet_climb = False

        # One asynchronous Jev request and a small number of slow reviews.
        self.jev_future = None
        self.jev_stamp = None
        self.jev_answer = None
        self.jev_answer_stamp = None
        self.jev_answer_tick = -10000
        self.astra_future = None
        self.astra_calls = 0
        self.astra_hint = {"branch": "adaptive", "first_side": "left", "risk": "route"}

    # ---------------- execution and defensive plumbing ----------------
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

    def clear_jump(self):
        self.jump_direction = None
        self.jump_seen = False
        self.jump_guard = 8
        self.jump_tail = 0
        self.jump_tail_direction = None
        self.jump_lock = 0
        self.post_jump_up = 0
        self.jump_reason = None

    def reset_route(self):
        self.phase = 0
        self.branch = self.preferred.get(0, 0)
        self.branch = 0 if self.branch not in range(len(CONNECTORS[0])) else self.branch
        self.failed_edges.clear()
        self.climb_open = False
        self.climb_origin = None
        self.climb_best = None
        self.climb_age = 0
        self.climb_stall = 0
        self.walk_stall = 0
        self.previous_x = None
        self.route_event = None
        self.clear_jump()
        self.rivet_row = 0
        self.rivet_preference = (
            self.astra_hint.get("first_side")
            if self.astra_hint.get("first_side") in ("left", "right") else "left"
        )
        self.previous_rows = None
        self.gap_direction = None
        self.rivet_climb = False
        self.farm_mode = bool(self.farm_next_life)
        self.farm_armed = False
        self.log("active_reset", stage="ramp", farm=self.farm_mode,
                 preferred=dict(self.preferred), route_failures=self.route_failures)

    def reset_rivet(self):
        self.phase = 0
        self.branch = 0
        self.climb_open = False
        self.climb_origin = None
        self.climb_best = None
        self.climb_age = 0
        self.climb_stall = 0
        self.walk_stall = 0
        self.previous_x = None
        self.clear_jump()
        self.rivet_row = 0
        self.rivet_preference = (
            self.astra_hint.get("first_side")
            if self.astra_hint.get("first_side") in ("left", "right") else "left"
        )
        self.previous_rows = None
        self.gap_direction = None
        self.rivet_climb = False
        self.farm_mode = False
        self.log("active_reset", stage="rivet", first_side=self.rivet_preference)

    def sync(self, obs):
        current = stage(obs)
        lives = integer(obs.get("lives_remaining"), 0) if isinstance(obs, dict) else 0
        screens = integer(obs.get("screens_completed"), 0) if isinstance(obs, dict) else 0
        changed = current != self.last_stage
        life_changed = self.last_lives is not None and lives != self.last_lives
        screen_changed = screens != self.last_screens

        if life_changed and lives < (self.last_lives if self.last_lives is not None else lives):
            # Preserve the learned edge only as a preference for the next life;
            # it is not a seed-specific route.  A first-life upper death buys a
            # conservative score-lane attempt on the remaining Mario(s).
            self.route_failures += 1
            self.farm_next_life = current in ("ramp", "transition") and screens == 0
            if self.last_active_phase >= 1 and self.last_active_phase < len(CONNECTORS):
                options = CONNECTORS[self.last_active_phase]
                if len(options) > 1:
                    self.preferred[self.last_active_phase] = 1
            self.log("life_boundary", lives=lives, failures=self.route_failures,
                     last_phase=self.last_active_phase, farm_next=self.farm_next_life)

        if changed or life_changed or screen_changed:
            self.log("boundary", old_stage=self.last_stage, stage=current,
                     old_lives=self.last_lives, lives=lives,
                     old_screens=self.last_screens, screens=screens)
        if current == "ramp" and (changed or life_changed or screen_changed):
            self.reset_route()
        elif current == "rivet" and (changed or life_changed or screen_changed):
            self.reset_rivet()

        if current == "transition":
            self.transition_age = self.transition_age + 1 if self.last_stage == "transition" else 1
            self.final_transition_age = self.final_transition_age + 1 if lives <= 0 else 0
        else:
            self.transition_age = 0
            self.final_transition_age = 0
        if changed:
            if current == "idle":
                self.idle_edges = 0
            if current != "transition":
                self.restart_edges = 0
        self.last_stage, self.last_lives, self.last_screens = current, lives, screens
        return current, lives, screens, changed, life_changed, screen_changed

    # ---------------- route graph and landing reconciliation ----------------
    def connector_options(self, phase=None):
        p = self.phase if phase is None else phase
        p = min(max(integer(p), 0), len(CONNECTORS) - 1)
        return CONNECTORS[p]

    def select_branch(self, phase):
        options = self.connector_options(phase)
        chosen = self.preferred.get(phase, 0)
        # On the first route attempt, take the short phase-2 connector before
        # the barrel's observed closing window.  This is a route decision, not
        # a seed choice; later lives retain the learned/fallback preference.
        if phase == 2 and not self.farm_mode and self.route_failures == 0 and len(options) > 1:
            chosen = 1
        if self.astra_hint.get("branch") == "alternate" and len(options) > 1:
            chosen = 1
        if chosen >= len(options) or (phase, chosen) in self.failed_edges:
            chosen = next((i for i in range(len(options))
                           if (phase, i) not in self.failed_edges), 0)
        self.branch = chosen
        return chosen

    def target(self):
        options = self.connector_options()
        self.branch = min(max(integer(self.branch), 0), len(options) - 1)
        return float(options[self.branch])

    def begin_climb(self, y):
        self.climb_open = True
        self.climb_origin = y
        self.climb_best = y
        self.climb_age = 0
        self.climb_stall = 0
        self.walk_stall = 0

    def adopt_landing_connector(self, obs):
        """Resolve a jump landing before the steering tail can walk away.

        This is the crossover's new mechanism.  A landing at x=87,y=118 is
        evidence for the phase-2 x=86 short connector.  Earlier controllers
        waited until the jump transaction ended, then continued toward x=106
        and collided with the same obstacle again.
        """
        if stage(obs) != "ramp" or self.jump_phase >= len(CONNECTORS):
            return
        x, y = xy(obs)
        options = self.connector_options(self.jump_phase)
        row = RAMP_Y[self.jump_phase]
        if len(options) < 2 or y < row - 9 or y > row + 10:
            return
        candidates = [(abs(x - value), index)
                      for index, value in enumerate(options)
                      if abs(x - value) <= 8.0]
        if not candidates:
            return
        _, chosen = min(candidates)
        if chosen != self.branch:
            old = self.branch
            self.branch = chosen
            self.preferred[self.jump_phase] = chosen
            self.failed_edges.discard((self.jump_phase, chosen))
            self.climb_open = False
            self.climb_origin = None
            self.climb_best = None
            self.climb_age = 0
            self.climb_stall = 0
            self.walk_stall = 0
            self.log("landing_connector_adopted", phase=self.jump_phase,
                     old_branch=old, branch=chosen, x=round(x, 1), y=round(y, 1))

    def fail_connector(self, reason):
        options = self.connector_options()
        old = self.branch
        self.failed_edges.add((self.phase, old))
        alternatives = [i for i in range(len(options))
                        if i != old and (self.phase, i) not in self.failed_edges]
        if alternatives:
            self.branch = alternatives[0]
            self.preferred[self.phase] = self.branch
            self.route_event = "connector_stall"
            self.log("connector_switch", phase=self.phase, old=old,
                     branch=self.branch, reason=reason)
        else:
            self.route_event = "route_exhausted"
            self.farm_mode = True
            self.farm_target = 106.0
            self.log("route_fallback", phase=self.phase, reason=reason)
        self.climb_open = False
        self.climb_origin = None
        self.climb_best = None
        self.climb_age = 0
        self.climb_stall = 0
        self.walk_stall = 0
        self.previous_x = None

    def ramp_action(self, obs):
        x, y = xy(obs)
        p = player(obs)
        dy = num(p.get("dy"), 0.0)
        self.last_active_phase = self.phase

        # Farm mode intentionally stops at the proven phase-1 score lane.  It
        # still uses the hazard transaction above this method, but never turns
        # repeated jump rewards into a claim of route progress.
        if self.farm_mode:
            if self.phase == 0:
                target = 106.0
                if x < target - 1.5:
                    return "right"
                if x > target + 1.5:
                    return "left"
                if not self.climb_open:
                    self.begin_climb(y)
                # Let the bottom connector be confirmed normally.
            else:
                self.phase = min(self.phase, 1)
                target = self.farm_target
                if abs(x - target) > 5.0:
                    return "right" if x < target else "left"
                self.climb_open = False
                return "up"

        options = self.connector_options()
        self.branch = min(max(integer(self.branch), 0), len(options) - 1)
        target = self.target()
        row = RAMP_Y[self.phase]

        if self.climb_open:
            self.climb_age += 1
            if self.climb_best is None or y < self.climb_best - 1.0:
                self.climb_best = y
                self.climb_stall = 0
            elif abs(dy) <= 1.5:
                self.climb_stall += 1

            if self.phase + 1 < len(RAMP_Y):
                next_y = RAMP_Y[self.phase + 1]
                gain = self.climb_origin - y if self.climb_origin is not None else 0.0
                if gain >= 8.0 and y <= next_y + 6.0 and self.climb_age >= 5:
                    old = self.phase
                    self.phase += 1
                    self.preferred[old] = self.branch
                    self.branch = self.select_branch(self.phase)
                    self.climb_open = False
                    self.climb_origin = None
                    self.climb_best = None
                    self.climb_age = 0
                    self.climb_stall = 0
                    self.walk_stall = 0
                    self.route_event = None
                    if self.farm_mode and self.phase >= 1:
                        self.phase = 1
                    self.log("ramp_advance", old_phase=old, phase=self.phase,
                             branch=self.branch, x=round(x, 1), y=round(y, 1))
                    return self.ramp_action(obs)

            # A jump can have moved Mario laterally while he is still on the
            # ladder.  Do not cancel a visibly progressing climb for x error.
            if self.climb_stall >= 20:
                self.fail_connector("vertical_no_progress")
                if self.route_event == "route_exhausted":
                    return self.ramp_action(obs)
                return self.ramp_action(obs)
            return "up"

        if self.previous_x is not None and self.last_action in ("left", "right"):
            self.walk_stall = self.walk_stall + 1 if abs(x - self.previous_x) < 0.7 else 0
        else:
            self.walk_stall = 0
        self.previous_x = x

        if abs(x - target) > 2.0:
            if self.walk_stall >= 10 and len(options) > 1:
                self.fail_connector("horizontal_no_motion")
                return self.ramp_action(obs)
            return "right" if x < target else "left"

        self.begin_climb(y)
        self.log("climb_begin", phase=self.phase, branch=self.branch,
                 target=round(target, 1), x=round(x, 1), y=round(y, 1))
        return "up"

    # ---------------- jump transaction ----------------
    def jump_fire_input(self, direction):
        # The durable score lane uses the diagonal FIRE form from 0018.  The
        # first-life route experiment uses horizontal FIRE so its landing x is
        # a useful connector observation.
        if self.farm_mode:
            return "uprightfire" if direction == "right" else "upleftfire"
        return move(direction, True)

    def jump_direction_for(self, obs, hazard):
        # Meeting a barrel from the side is the verified public strategy.  The
        # landing reconciler, not this choice, decides which connector is kept.
        side = hazard.get("side")
        if side in ("left", "right"):
            return side
        x, _ = xy(obs)
        target = self.target()
        return "right" if x < target else "left"

    def begin_jump(self, obs, direction, reason, hazard=None):
        if direction not in ("left", "right"):
            direction = "right"
        self.jump_direction = direction
        self.jump_seen = False
        self.jump_guard = 8
        self.jump_tail = 0
        self.jump_phase = self.phase
        self.jump_reason = reason
        self.jump_lock = 18
        self.jump_count += 1
        self.jump_tail_direction = direction
        self.log("jump_begin", tick=self.tick, phase=self.phase,
                 direction=direction, tail=self.jump_tail_direction,
                 reason=reason,
                 hazard={k: (hazard or {}).get(k)
                         for k in ("slot", "side", "motion", "distance_bucket", "urgency")})
        # Horizontal FIRE is deliberate on the route experiment; the durable
        # score lane uses the diagonal form that made 0018's fallback survive.
        return self.jump_fire_input(direction)

    def jump_step(self, obs):
        if self.jump_direction is None:
            if self.post_jump_up > 0:
                self.post_jump_up -= 1
                if self.jump_lock > 0:
                    self.jump_lock -= 1
                return "up"
            if self.jump_lock > 0:
                self.jump_lock -= 1
            return None

        direction = self.jump_direction
        state = raw_byte(obs, 0x85)
        if state > 0:
            self.jump_seen = True
            self.jump_guard = 8
            return self.jump_fire_input(direction)
        if not self.jump_seen and self.jump_guard > 0:
            self.jump_guard -= 1
            return self.jump_fire_input(direction)
        if self.jump_seen:
            if self.jump_tail <= 0:
                self.jump_tail = 5
            self.adopt_landing_connector(obs)
            self.jump_tail -= 1
            if self.jump_tail <= 0:
                self.log("jump_end", tick=self.tick, phase=self.jump_phase,
                         direction=direction, reason=self.jump_reason)
                self.jump_direction = None
                self.post_jump_up = 6
                self.jump_lock = 24
                return "up"
            return move(self.jump_tail_direction or direction, False)

        # If the observation did not expose the RAM jump byte, fail safe rather
        # than latching FIRE indefinitely.
        self.log("jump_unconfirmed", tick=self.tick, phase=self.phase)
        self.clear_jump()
        self.jump_lock = 8
        return None

    def hazard_action(self, obs, current_stage):
        if current_stage not in ("ramp", "rivet"):
            return None
        if self.jump_direction is not None or self.jump_lock > 0:
            return None
        fact = hazard_on_row(obs)
        if self.farm_mode:
            # The proven score lane is deliberately late: the earlier farm
            # experiment died when it jumped at the wider warning band.
            if not fact.get("present") or num(fact.get("distance"), 999.0) > 16.0:
                return None
        elif fact.get("urgency") not in ("commit", "overlap"):
            return None
        if current_stage == "ramp" and self.phase == 2 and self.branch == 1:
            # The short connector is not itself a safe place to idle: once
            # aligned, resume the route-directed jump/escape policy.  Proactive
            # selection changes the landing target, not the hard collision guard.
            pass
        if current_stage == "ramp" and self.climb_open and self.climb_origin is not None:
            _, y = xy(obs)
            if self.climb_origin - y >= 5.0 and fact.get("urgency") != "overlap":
                return None
        return self.begin_jump(obs, self.jump_direction_for(obs, fact),
                               "hazard", fact)

    # ---------------- rivet route ----------------
    def ladder_for_row(self, obs, side):
        desired = 108.0 if side == "right" else 48.0
        _, py = xy(obs)
        candidates = []
        values = obs.get("ladders", []) if isinstance(obs, dict) else []
        if isinstance(values, list):
            for item in values:
                if not visible(item):
                    continue
                lx, ly = num(item.get("x"), desired) - 2.0, num(item.get("y"), py)
                if abs(ly - py) <= 23 or abs(ly - (py - 8)) <= 23:
                    candidates.append((abs(lx - (desired - 2.0)), lx))
        return min(candidates)[1] if candidates else desired - 2.0

    def rivet_action(self, obs):
        current = rows(obs)
        if self.previous_rows is not None:
            old = self.previous_rows
            for index, (before, after) in enumerate(zip(old, current)):
                if index != self.rivet_row or before == after:
                    continue
                removed = [side for side in ("left", "right")
                           if side_present(before, side) and not side_present(after, side)]
                if removed:
                    self.gap_direction = "right" if removed[0] == "left" else "left"
                    self.log("rivet_change", row=index, old=before, new=after,
                             removed=removed[0], gap=self.gap_direction)
        self.previous_rows = list(current)

        while self.rivet_row < 4 and current[self.rivet_row] == 18:
            self.rivet_row += 1
            self.gap_direction = None
            self.rivet_climb = False
        if self.rivet_row >= 4:
            return "up"

        row = self.rivet_row
        mask = current[row]
        preference = self.rivet_preference if row % 2 == 0 else (
            "right" if self.rivet_preference == "left" else "left"
        )
        side = remaining_side(mask, preference)
        if self.gap_direction in ("left", "right"):
            direction = self.gap_direction
            self.gap_direction = None
            return self.begin_jump(obs, direction, "rivet_gap")

        x, y = xy(obs)
        target_y = RIVET_Y[row]
        climb_side = "right" if row % 2 == 0 else "left"
        if side is None or y > target_y + 10:
            target = self.ladder_for_row(obs, climb_side)
            if abs(x - target) > 2:
                return "right" if x < target else "left"
            self.rivet_climb = True
            return "up"
        if y < target_y - 12:
            return "down"
        target_x = RIVET_X[side]
        if abs(x - target_x) > 8:
            return "right" if x < target_x else "left"
        return "right" if side == "left" else "left"

    # ---------------- model bridge ----------------
    def model_state(self, obs, event, fallback):
        x, y = xy(obs)
        p = player(obs)
        fact = hazard_on_row(obs)
        options = self.connector_options()
        return {
            "game": "Atari 2600 Donkey Kong NTSC retail cartridge",
            "event": event,
            "stage": stage(obs),
            "player": {
                "horizontal_band": "left" if x < 52 else "right" if x > 100 else "middle",
                "vertical_band": "bottom" if y > 160 else "lower" if y > 132 else "middle" if y > 78 else "top",
                "motion": "up" if num(p.get("dy")) < -1 else "down" if num(p.get("dy")) > 1 else "steady",
                "jump_committed": self.jump_direction is not None,
            },
            "route": {
                "phase": integer(self.phase),
                "selected_branch": integer(self.branch),
                "branch_count": len(options),
                "alternate_available": len(options) > 1,
                "climb_open": bool(self.climb_open),
                "fallback_action": fallback,
                "row": integer(self.rivet_row),
                "rivet_rows": rows(obs),
                "farm_mode": bool(self.farm_mode),
            },
            "hazard": {key: fact.get(key) for key in
                       ("present", "slot", "side", "motion", "distance_bucket", "closing", "urgency")},
            "progress": {
                "lives_remaining": integer(obs.get("lives_remaining"), 0),
                "screens_completed": integer(obs.get("screens_completed"), 0),
                "rivets_removed": integer(obs.get("rivets_removed"), 0),
            },
            "last_action": self.last_action,
        }

    def budget_allows(self, kind):
        try:
            if bool(self.ctx.done):
                return False
            return integer(self.ctx.budget.remaining().get(kind), 0) > 0
        except Exception:
            return False

    def signature(self, obs, event):
        fact = hazard_on_row(obs)
        return (event, stage(obs), integer(self.phase), integer(self.branch),
                integer(self.rivet_row), integer(fact.get("slot"), -1),
                fact.get("urgency", "clear"), integer(self.route_failures))

    def request_jev(self, obs, event, fallback):
        if self.jev_future is not None or event not in ("connector_stall", "hazard"):
            return
        if not self.budget_allows("system1_calls"):
            return
        stamp = self.signature(obs, event)
        if stamp == self.jev_stamp and self.tick - self.jev_answer_tick < 12:
            return
        state = self.model_state(obs, event, fallback)
        questions = {
            "route": Choice(
                "Use only the finite route choices explicitly described in `route`. "
                "At a connector stall choose alternate only when an alternative is "
                "available and no climb progress was observed. Choose hold during a "
                "jump or a live climb.",
                {
                    "hold": "Keep the deterministic fallback and current connector.",
                    "nominal": "Retry the selected nominal connector.",
                    "alternate": "Use the other fixed connector on this split row.",
                    "farm": "Stop route experimentation and preserve the score lane.",
                },
            ),
            "danger": Noul(
                "Does `hazard` explicitly describe a commit or overlap closing object "
                "on Mario's current walkway?",
                {
                    "true": "hazard.urgency is commit or overlap and hazard.closing is true.",
                    "false": "No commit/overlap current-row hazard is supplied.",
                },
            ),
        }
        self.jev_stamp = stamp
        try:
            self.jev_future = self.ctx.spawn(
                lambda: self.ctx.decide(state, questions, timeout_s=0.45)
            )
        except Exception as exc:
            self.jev_future = None
            self.log("jev_start_failed", error=type(exc).__name__)

    def collect_jev(self):
        future = self.jev_future
        if future is None or not future.done():
            return
        self.jev_future = None
        try:
            answers = future.result()
            answer = answers.get("route") if isinstance(answers, dict) else None
            value = getattr(answer, "value", None)
            if value not in ("hold", "nominal", "alternate", "farm"):
                return
            p = probability(answer, value)
            if p is not None and p < 0.60:
                return
            self.jev_answer = value
            self.jev_answer_stamp = self.jev_stamp
            self.jev_answer_tick = self.tick
            self.log("jev_route", value=value, probability=p)
        except Exception as exc:
            self.log("jev_failed", error=type(exc).__name__)

    def apply_jev(self, obs, fallback):
        value = self.jev_answer
        if value is None:
            return fallback
        stamp = self.jev_answer_stamp
        current = self.signature(obs, stamp[0] if isinstance(stamp, tuple) else "unknown")
        if current != stamp or self.tick - self.jev_answer_tick > 3:
            self.jev_answer = None
            return fallback
        self.jev_answer = None
        if stamp[0] != "connector_stall":
            return fallback
        if value == "alternate" and len(self.connector_options()) > 1:
            self.failed_edges.add((self.phase, self.branch))
            self.branch = next((i for i in range(len(self.connector_options()))
                                if i != self.branch), self.branch)
            self.preferred[self.phase] = self.branch
            self.log("jev_branch", branch=self.branch)
            return self.ramp_action(obs)
        if value == "nominal":
            self.failed_edges.discard((self.phase, 0))
            self.branch = 0
            return self.ramp_action(obs)
        if value == "farm":
            self.farm_mode = True
            self.farm_target = 106.0
            self.log("jev_farm")
        return fallback

    def start_astra(self, obs, reason):
        if self.astra_future is not None or self.astra_calls >= 2:
            return
        if not self.budget_allows("system2_calls"):
            return
        self.astra_calls += 1
        state = self.model_state(obs, reason, self.last_action)
        prompt = (
            "You are Astra, a slow macro reviewer for the Atari 2600 Donkey Kong "
            "NTSC retail cartridge. The actor owns every tick, exact action names, "
            "the fixed ladder graph, jump timing, raw rivet bookkeeping, and life "
            "transitions. Do not return a joystick sequence. Return only one compact "
            "JSON object with optional labels: branch='nominal'|'alternate'|'adaptive', "
            "first_side='left'|'right', risk='route'|'survive'. Apply labels only "
            "at a later observed boundary; invalid text is ignored. Facts=" +
            json.dumps(state, separators=(",", ":"), allow_nan=False)
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
        if future is None or not future.done():
            return
        self.astra_future = None
        try:
            parsed = json_object(getattr(future.result(), "text", ""))
        except Exception as exc:
            self.log("astra_failed", error=type(exc).__name__)
            return
        if not isinstance(parsed, dict):
            return
        changed = {}
        for key, allowed in (
            ("branch", ("nominal", "alternate", "adaptive")),
            ("first_side", ("left", "right")),
            ("risk", ("route", "survive")),
        ):
            value = parsed.get(key)
            if value in allowed:
                self.astra_hint[key] = value
                changed[key] = value
        if changed:
            if changed.get("first_side") in ("left", "right") and stage(self._last_obs) == "rivet":
                if self.rivet_row == 0 and self.previous_rows in (None, [0, 0, 0, 0]):
                    self.rivet_preference = changed["first_side"]
            self.log("astra_hint", **changed)

    # ---------------- top-level policy and loop ----------------
    def choose(self, obs):
        self._last_obs = obs if isinstance(obs, dict) else {}
        current, lives, screens, changed, life_changed, screen_changed = self.sync(obs)
        self.collect_jev()
        self.collect_astra()

        if current == "idle":
            if self.idle_edges < 3:
                self.idle_edges += 1
                return "fire"
            return "noop"
        if current == "game_over" or bool(obs.get("terminated", False)) or bool(obs.get("truncated", False)):
            return "noop"
        if current == "transition":
            if (lives <= 0 and self.final_transition_age >= 180):
                self.log("final_transition_bound", age=self.final_transition_age)
                return "noop"
            points = (18, 46, 74, 102, 130, 158)
            if self.transition_age in points and self.restart_edges < len(points):
                self.restart_edges += 1
                self.log("restart_edge", age=self.transition_age, lives=lives,
                         edge=self.restart_edges)
                return "fire"
            return "noop"
        if current not in ("ramp", "rivet"):
            return "noop"

        if current == "rivet" and changed:
            self.start_astra(obs, "rivet_entry")
        elif self.route_event in ("connector_stall", "route_exhausted"):
            self.start_astra(obs, self.route_event)
            self.route_event = None

        ongoing = self.jump_step(obs)
        if ongoing is not None:
            return ongoing
        emergency = self.hazard_action(obs, current)
        if emergency is not None:
            if current == "ramp" and self.route_event:
                self.request_jev(obs, self.route_event, emergency)
            return emergency

        action = self.ramp_action(obs) if current == "ramp" else self.rivet_action(obs)
        if self.route_event in ("connector_stall", "route_exhausted"):
            self.request_jev(obs, self.route_event, action)
        action = self.apply_jev(obs, action)

        if changed or life_changed or screen_changed:
            self.log("decision_boundary", stage=current, phase=self.phase,
                     branch=self.branch, action=action, lives=lives, screens=screens)
        if self.tick % 20 == 0:
            x, y = xy(obs)
            self.log("snapshot", tick=self.tick, stage=current, phase=self.phase,
                     branch=self.branch, x=round(x), y=round(y), action=action,
                     climb=self.climb_open, farm=self.farm_mode,
                     jump=self.jump_direction, raw_jump=raw_byte(obs, 0x85),
                     hazard=hazard_on_row(obs).get("urgency"), lives=lives,
                     score=num(obs.get("score_total"), 0.0))
        return action

    def run(self):
        try:
            obs = self.ctx.observe()
        except Exception:
            return
        if not isinstance(obs, dict):
            obs = {}
        self._last_obs = obs
        self.last_stage = stage(obs)
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
            if self.done():
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
