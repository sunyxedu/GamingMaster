"""Transactional ladder controller for Atari 2600 Donkey Kong.

This is a code-first controller with two independent observations of progress:
a connector is selected from the fixed ladder graph, but a route level changes
only after Mario's y coordinate proves that the climb happened.  A hazard jump
is a committed same-lane transaction, so a late model answer cannot interrupt
it.  System-1/System-2 are advisory for connector recovery and the later rivet
sweep; they are never on the real-time action path.
"""
from __future__ import annotations

import json
import math

from metaharness.models.jev import Choice, Noul


# Player-left-edge coordinates inferred from the public ladder boxes.  The
# order is bottom to top; split rows retain a nominal connector and an
# explicitly bounded alternative.
RAMP_Y = (176, 148, 120, 92, 64, 40)
RAMP_GROUPS = ((0,), (1, 2), (4, 3), (5, 6), (7,), (9, 8))
RAMP_FALLBACKS = ((106,), (46, 78), (106, 86), (46, 66), (106,), (30, 74))
ALIGNMENT_PROBES = (0, -1, 1)
# A jump can land on the second connector of a split row.  These are the
# player-edge positions of those connectors; adoption requires both a matching
# x and evidence that Mario is above the old walkway.
ALTERNATE_EDGES = {1: 78, 2: 86, 3: 66, 5: 74}

RIVET_Y = (176, 148, 120, 92)
RIVET_EDGE = {"left": 46, "right": 106}

ACTIONS = {
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


def visible(value):
    return isinstance(value, dict) and value.get("visible", True) is not False


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


def raw_jump_state(obs):
    raw = obs.get("raw_ram", []) if isinstance(obs, dict) else []
    if isinstance(raw, list) and len(raw) > 5:
        return integer(raw[5], 0)
    return 0


def directional(direction, fire=False):
    if direction not in ("left", "right"):
        return "fire" if fire else "noop"
    return direction + ("fire" if fire else "")


def remaining_side(mask, preferred):
    """Decode the documented 0/6/12/18 rivet row values."""
    if mask == 18:
        return None
    if mask == 0:
        return preferred
    if mask == 6:
        return "right" if preferred == "left" else "left"
    if mask == 12:
        return "left" if preferred == "right" else "right"
    return preferred


def closing_hazard(obs):
    """Return one derived current-row hazard; all arithmetic stays in code."""
    p = player_of(obs)
    px = number(p.get("x"), 45.0)
    py = number(p.get("y"), 176.0)
    hazards = obs.get("hazards", []) if isinstance(obs, dict) else []
    if not isinstance(hazards, list):
        hazards = []
    best = None
    for raw in hazards:
        if not visible(raw):
            continue
        hx = number(raw.get("x"), 0.0)
        hy = number(raw.get("y"), 0.0)
        # A neighboring girder is not an immediate collision candidate.
        if abs(hy - py) > 20.0:
            continue
        dx = number(raw.get("dx"), 0.0)
        delta = hx - px
        distance = abs(delta)
        closing = ((delta < 0.0 and dx > 0.2)
                   or (delta > 0.0 and dx < -0.2)
                   or (distance <= 16.0 and abs(dx) <= 0.2))
        if not closing:
            continue
        urgency = "overlap" if distance <= 16.0 else "commit" if distance <= 22.0 else "watch"
        fact = {
            "present": True,
            "side": "left" if delta < 0 else "right",
            "motion": "left" if dx < -0.2 else "right" if dx > 0.2 else "still",
            "distance_bucket": (
                "overlapping" if distance <= 16.0 else
                "near" if distance <= 32.0 else "approaching"
            ),
            "closing": True,
            "urgency": urgency,
            "distance": round(distance, 1),
            "slot": integer(raw.get("slot"), -1),
        }
        rank = (0 if urgency == "overlap" else 1 if urgency == "commit" else 2,
                distance)
        if best is None or rank < best[0]:
            best = (rank, fact)
    if best is not None:
        return best[1]
    return {
        "present": False, "side": "none", "motion": "none",
        "distance_bucket": "none", "closing": False, "urgency": "clear",
        "distance": 999.0, "slot": -1,
    }


class Controller:
    def __init__(self, ctx):
        self.ctx = ctx
        self.ticks = 0
        self.last_action = "noop"
        self.last_stage = None
        self.last_lives = None
        self.last_screens = 0

        # Route transaction.  level is the walkway currently occupied.  It is
        # never incremented by an action count; y progress and a short settle
        # interval are required.  connector and probe are bounded recovery
        # choices supplied by the fixed ten-ladder observation.
        self.level = 0
        self.connector = 0
        self.probe = 0
        self.climb = False
        self.climb_age = 0
        self.climb_origin_y = None
        self.climb_best_y = None
        self.climb_progress = False
        self.settle_age = 0
        self.no_progress = 0
        self.last_y = None
        self.recovery_count = 0

        # A jump owns the action until the RAM jump state has been seen and
        # cleared.  The landing tail keeps the same direction, the robust
        # mechanism from 0017, then returns control to the connector route.
        self.jump_direction = None
        self.jump_seen = False
        self.jump_guard = 8
        self.landing_left = 0
        self.jump_lock = 0
        self.post_jump_up = 0
        self.landing_window = 0
        self.jump_count = 0

        # Lifecycle is edge-triggered.  FIRE starts idle and is sparse during a
        # death transition; movement is still allowed for the final Mario.
        self.transition_age = 0
        self.restart_edges = 0
        self.idle_edges = 0
        self.stop_after_transition = False

        # Rivet route state, reset from the observed bytes after each screen or
        # life.  A changed byte creates a gap crossing transaction.
        self.rivet_row = 0
        self.rivet_preference = "left"
        self.previous_rows = None
        self.gap_direction = None

        # Advisory futures.  They are deliberately absent from ramp timing
        # except for a bounded connector-recovery judgment.
        self.jev_future = None
        self.jev_key = None
        self.jev_hint = None
        self.last_jev_tick = -10000
        self.astra_future = None
        self.astra_started = False
        self.astra_hint = None

    def log(self, event, **fields):
        try:
            self.ctx.log(event, **fields)
        except Exception:
            pass

    def done(self, game):
        try:
            if bool(self.ctx.done):
                return True
        except Exception:
            return True
        try:
            return bool(game.done)
        except Exception:
            return False

    def post(self, action):
        if not isinstance(action, str) or action not in ACTIONS:
            action = "noop"
        try:
            self.ctx.act(action)
            self.last_action = action
            return True
        except Exception as exc:
            self.log("action_failed", action=action, error=type(exc).__name__)
            return False

    # -------------------- state synchronization --------------------
    def clear_jump(self):
        self.jump_direction = None
        self.jump_seen = False
        self.jump_guard = 8
        self.landing_left = 0
        self.jump_lock = 0
        self.post_jump_up = 0
        self.landing_window = 0

    def reset_play(self, stage):
        self.level = 0
        self.connector = 0
        self.probe = 0
        self.climb = False
        self.climb_age = 0
        self.climb_origin_y = None
        self.climb_best_y = None
        self.climb_progress = False
        self.settle_age = 0
        self.no_progress = 0
        self.last_y = None
        self.recovery_count = 0
        self.clear_jump()
        self.gap_direction = None
        if stage == "rivet":
            self.rivet_row = 0
            self.rivet_preference = (
                self.astra_hint if self.astra_hint in ("left", "right") else "left"
            )
            self.previous_rows = None

    def sync(self, obs):
        stage = stage_of(obs)
        lives = integer(obs.get("lives_remaining"), 0) if isinstance(obs, dict) else 0
        screens = integer(obs.get("screens_completed"), 0) if isinstance(obs, dict) else 0
        stage_changed = stage != self.last_stage
        life_changed = self.last_lives is not None and lives != self.last_lives
        screen_changed = screens != self.last_screens
        if stage_changed or life_changed:
            self.log("boundary", old_stage=self.last_stage, stage=stage,
                     old_lives=self.last_lives, lives=lives,
                     level=self.level, screens=screens)
        if stage_changed and stage in ("ramp", "rivet"):
            self.reset_play(stage)
        elif life_changed and stage in ("ramp", "rivet"):
            self.reset_play(stage)
        if stage_changed:
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
        return stage, lives, screens, stage_changed, life_changed, screen_changed

    # -------------------- connectors and ramp route --------------------
    def connector_targets(self, obs, level):
        level = min(max(integer(level), 0), len(RAMP_GROUPS) - 1)
        ladders = obs.get("ladders", []) if isinstance(obs, dict) else []
        by_slot = {}
        if isinstance(ladders, list):
            for ladder in ladders:
                if visible(ladder):
                    slot = integer(ladder.get("slot"), -1)
                    if slot >= 0:
                        by_slot[slot] = number(ladder.get("x"), 0.0) - 2.0
        fallback = RAMP_FALLBACKS[level]
        result = []
        for index, slot in enumerate(RAMP_GROUPS[level]):
            base = by_slot.get(slot, float(fallback[min(index, len(fallback) - 1)]))
            result.append((int(round(base)), slot))
        return result

    def active_target(self, obs):
        options = self.connector_targets(obs, self.level)
        if not options:
            return 46, -1
        self.connector = min(max(self.connector, 0), len(options) - 1)
        self.probe = min(max(self.probe, 0), len(ALIGNMENT_PROBES) - 1)
        base, slot = options[self.connector]
        return base + ALIGNMENT_PROBES[self.probe], slot

    def open_climb(self, y):
        self.climb = True
        self.climb_age = 0
        self.climb_origin_y = y
        self.climb_best_y = y
        self.climb_progress = False
        self.settle_age = 0
        self.no_progress = 0
        self.last_y = y

    def request_connector_judgment(self, obs):
        if self.jev_future is not None or stage_of(obs) != "ramp":
            return
        key = (self.level, self.connector, self.probe,
               integer(obs.get("lives_remaining"), 0))
        if key == self.jev_key:
            return
        try:
            if self.ctx.budget.remaining().get("system1_calls", 0) <= 0:
                return
        except Exception:
            return
        p = player_of(obs)
        hazard = closing_hazard(obs)
        state = {
            "game": "Atari 2600 Donkey Kong NTSC retail port",
            "stage": "ramp",
            "player": {
                "horizontal_band": "left" if number(p.get("x"), 45) < 52 else
                                   "right" if number(p.get("x"), 45) > 100 else "middle",
                "vertical_band": "bottom" if number(p.get("y"), 176) > 160 else
                                 "upper" if number(p.get("y"), 176) < 135 else "middle",
                "motion": "up" if number(p.get("dy"), 0) < -1 else
                          "down" if number(p.get("dy"), 0) > 1 else "steady",
            },
            "connector": {
                "level": integer(self.level),
                "candidate": integer(self.connector),
                "candidate_count": len(self.connector_targets(obs, self.level)),
                "alignment_probe": integer(self.probe),
                "climb_age": integer(self.climb_age),
                "vertical_progress_seen": bool(self.climb_progress),
            },
            "hazard": {k: hazard.get(k) for k in
                       ("present", "side", "motion", "distance_bucket", "urgency")},
            "last_action": self.last_action,
            "lives_remaining": integer(obs.get("lives_remaining"), 0),
        }
        questions = {
            "recovery": Choice(
                "Choose one bounded recovery operation for the named connector. "
                "Do not invent coordinates or actions. If progress is absent but "
                "the attempt may still be settling, choose continue briefly.",
                {
                    "retry_alignment": "Retry this fixed connector with a nearby alignment probe.",
                    "alternate_connector": "Probe the other fixed connector on this split row.",
                    "continue": "Continue the current UP transaction for a short bounded interval.",
                },
            ),
            "danger": Noul(
                "Does the supplied hazard explicitly show an overlap or commit "
                "closing object on Mario's current walkway?",
                {
                    "true": "hazard.urgency is overlap or commit and hazard.closing is true.",
                    "false": "No overlap/commit closing current-row object is supplied.",
                },
            ),
        }
        self.jev_key = key
        self.last_jev_tick = self.ticks
        try:
            self.jev_future = self.ctx.spawn(
                lambda: self.ctx.decide(state, questions, timeout_s=0.65)
            )
        except Exception as exc:
            self.jev_future = None
            self.log("jev_start_failed", error=type(exc).__name__)

    def request_rivet_judgment(self, obs):
        if self.jev_future is not None or stage_of(obs) != "rivet":
            return
        hazard = closing_hazard(obs)
        interval = 5 if hazard.get("urgency") in ("overlap", "commit") else 16
        if self.ticks - self.last_jev_tick < interval:
            return
        try:
            if self.ctx.budget.remaining().get("system1_calls", 0) <= 0:
                return
        except Exception:
            return
        p = player_of(obs)
        state = {
            "game": "Atari 2600 Donkey Kong NTSC retail port",
            "stage": "rivet",
            "player": {
                "horizontal_band": "left" if number(p.get("x"), 45) < 52 else
                                   "right" if number(p.get("x"), 45) > 100 else "middle",
                "vertical_band": "bottom" if number(p.get("y"), 176) > 160 else
                                 "upper" if number(p.get("y"), 176) < 135 else "middle",
                "motion": "up" if number(p.get("dy"), 0) < -1 else
                          "down" if number(p.get("dy"), 0) > 1 else "steady",
            },
            "route": {
                "row": integer(self.rivet_row),
                "row_status": rows_of(obs),
                "gap_crossing": self.gap_direction is not None,
                "jump_committed": self.jump_direction is not None,
            },
            "hazard": {k: hazard.get(k) for k in
                       ("present", "side", "motion", "distance_bucket", "urgency")},
            "last_action": self.last_action,
        }
        questions = {
            "sweep": Choice(
                "Choose the already-described legal rivet sweep preference. "
                "Use route.row_status and route.row; do not count or calculate. "
                "Choose hold while a gap, jump, or climb transaction is active.",
                {
                    "left": "Approach the left remaining rivet on this row.",
                    "right": "Approach the right remaining rivet on this row.",
                    "hold": "Preserve the deterministic route transaction.",
                },
            ),
            "danger": Noul(
                "Is hazard an overlap or commit closing object on Mario's current row?",
                {
                    "true": "The named hazard is overlap/commit and closing.",
                    "false": "No overlap/commit current-row hazard is shown.",
                },
            ),
        }
        self.last_jev_tick = self.ticks
        try:
            self.jev_future = self.ctx.spawn(
                lambda: self.ctx.decide(state, questions, timeout_s=0.65)
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
            answer = answers.get("recovery") if isinstance(answers, dict) else None
            if answer is None:
                answer = answers.get("sweep") if isinstance(answers, dict) else None
            value = getattr(answer, "value", None)
            probabilities = getattr(answer, "probabilities", None)
            probability = None
            if isinstance(probabilities, dict):
                probability = number(probabilities.get(value), 0.0)
            if value in ("retry_alignment", "alternate_connector", "continue",
                         "left", "right") and (probability is None or probability >= 0.55):
                self.jev_hint = value
                self.log("jev_hint", value=value, probability=probability)
        except Exception as exc:
            self.log("jev_failed", error=type(exc).__name__)

    def recover_connector(self, obs):
        options = self.connector_targets(obs, self.level)
        hint = self.jev_hint
        self.jev_hint = None
        if hint == "continue" and self.recovery_count < 2:
            # A single extra transaction is useful for a sticky-action delay,
            # but it is bounded so a bad rung cannot consume a whole life.
            self.recovery_count += 1
        elif hint == "alternate_connector" and self.connector + 1 < len(options):
            self.connector += 1
            self.probe = 0
            self.recovery_count += 1
        elif hint == "retry_alignment" and self.probe + 1 < len(ALIGNMENT_PROBES):
            self.probe += 1
            self.recovery_count += 1
        elif self.probe + 1 < len(ALIGNMENT_PROBES):
            self.probe += 1
            self.recovery_count += 1
        elif self.connector + 1 < len(options):
            self.connector += 1
            self.probe = 0
            self.recovery_count += 1
        else:
            self.connector = 0
            self.probe = 0
            self.recovery_count += 1
        self.climb = False
        self.climb_age = 0
        self.climb_origin_y = None
        self.climb_best_y = None
        self.climb_progress = False
        self.settle_age = 0
        self.no_progress = 0
        self.last_y = None
        self.log("connector_recovery", level=self.level, candidate=self.connector,
                 probe=self.probe, hint=hint, attempts=self.recovery_count)

    def ramp_action(self, obs):
        p = player_of(obs)
        x = number(p.get("x"), 45.0)
        y = number(p.get("y"), 176.0)
        dy = number(p.get("dy"), 0.0)
        self.level = min(max(self.level, 0), len(RAMP_Y) - 1)
        target, slot = self.active_target(obs)

        # A committed crossing may land at a valid split-row connector rather
        # than the nominal one.  Adopt it only when Mario is visibly above the
        # old walkway but not already at the next row; this is landing evidence,
        # not a timer-based phase change.
        alternate = ALTERNATE_EDGES.get(self.level)
        next_threshold = (
            RAMP_Y[self.level + 1] + 6.0
            if self.level + 1 < len(RAMP_Y) else -999.0
        )
        if (self.connector == 0 and alternate is not None
                and abs(x - alternate) <= 4.0
                and self.landing_window > 0
                and y <= RAMP_Y[self.level] + 4.0
                and (self.level + 1 >= len(RAMP_Y) or y > next_threshold)):
            self.connector = 1
            self.probe = 0
            target, slot = self.active_target(obs)
            if self.climb:
                self.climb = False
                self.climb_age = 0
                self.climb_origin_y = None
                self.climb_best_y = None
                self.climb_progress = False
                self.settle_age = 0
                self.no_progress = 0
                self.last_y = None
            self.log("landed_split_connector", level=self.level,
                     candidate=self.connector, x=round(x), y=round(y), slot=slot)

        if self.landing_window > 0:
            self.landing_window -= 1

        if self.climb:
            self.climb_age += 1
            # Record height before judging horizontal displacement.  A jump can
            # carry Mario onto the next walkway; that observed height is still
            # valid connector evidence even when x is no longer over the rung.
            if self.last_y is not None and y < self.last_y - 1.0:
                self.climb_progress = True
            if self.climb_origin_y is not None and y < self.climb_origin_y - 2.0:
                self.climb_progress = True
            self.last_y = y
            if self.climb_best_y is None or y < self.climb_best_y - 1.0:
                self.climb_best_y = y
                self.no_progress = 0
            elif abs(dy) <= 1.5:
                self.no_progress += 1

            # This check is intentionally outside any alternate-connector
            # branch.  It is the key repair: every connector can advance after
            # observed height, including the nominal first ladder.
            if self.level + 1 < len(RAMP_Y):
                next_y = RAMP_Y[self.level + 1]
                reached = y <= next_y + 6.0
                enough_motion = (self.climb_origin_y is not None and
                                 self.climb_origin_y - y >= 8.0)
                if reached and enough_motion and self.climb_age >= 5:
                    if abs(dy) <= 2.0:
                        self.settle_age += 1
                    else:
                        self.settle_age = 0
                    if self.settle_age >= 2:
                        old = self.level
                        self.level += 1
                        self.connector = 0
                        self.probe = 0
                        self.climb = False
                        self.climb_age = 0
                        self.climb_origin_y = None
                        self.climb_best_y = None
                        self.climb_progress = False
                        self.settle_age = 0
                        self.no_progress = 0
                        self.last_y = None
                        self.recovery_count = 0
                        self.log("ramp_advance", old_level=old,
                                 level=self.level, x=round(x), y=round(y), slot=slot)
                        return self.ramp_action(obs)

            # If the height test did not yet accept the connector, a displaced
            # Mario must reacquire it horizontally instead of holding UP in
            # empty space.  This comes after the evidence check deliberately.
            if abs(x - target) > 4.0:
                self.climb = False
                self.climb_age = 0
                self.climb_origin_y = None
                self.climb_best_y = None
                self.climb_progress = False
                self.settle_age = 0
                self.no_progress = 0
                self.last_y = None
                return self.ramp_action(obs)
            elif y <= RAMP_Y[-1] + 5.0 and self.climb_progress:
                # At the top, keep UP; the cartridge owns the goal transition.
                return "up"

            if self.no_progress >= 24:
                self.request_connector_judgment(obs)
                self.recover_connector(obs)
                return self.ramp_action(obs)
            return "up"

        if x < target - 1.0:
            return "right"
        if x > target + 1.0:
            return "left"
        self.open_climb(y)
        self.log("climb_open", level=self.level, candidate=self.connector,
                 probe=self.probe, slot=slot, x=round(x), y=round(y))
        return "up"

    # -------------------- committed jumps --------------------
    def jump_direction_for(self, obs, hazard):
        side = hazard.get("side")
        if side in ("left", "right"):
            # The durable parent uses the hazard's side for the initial FIRE
            # edge and keeps that lane through landing; the route controller
            # below then reacquires the observed connector.
            return side
        p = player_of(obs)
        x = number(p.get("x"), 45.0)
        target, _ = self.active_target(obs)
        return "right" if x < target else "left"

    def jump_input(self, fire=False):
        if not fire:
            return directional(self.jump_direction)
        return "uprightfire" if self.jump_direction == "right" else "upleftfire"

    def begin_jump(self, obs, direction, reason):
        self.jump_direction = direction if direction in ("left", "right") else "right"
        self.jump_seen = False
        self.jump_guard = 8
        self.landing_left = 0
        self.jump_lock = 28
        self.post_jump_up = 0
        self.landing_window = 0
        self.jump_count += 1
        self.log("jump_begin", tick=self.ticks, direction=self.jump_direction,
                 reason=reason, level=self.level, hazard=closing_hazard(obs))
        return self.jump_input(True)

    def jump_action(self, obs):
        if self.jump_direction is not None:
            raw = raw_jump_state(obs)
            if raw > 0:
                self.jump_seen = True
                self.jump_guard = 8
                return self.jump_input(True)
            if self.jump_seen:
                if self.landing_left <= 0:
                    self.landing_left = 5
                self.landing_left -= 1
                if self.landing_left <= 0:
                    self.jump_direction = None
                    self.post_jump_up = 6
                    self.landing_window = 8
                    return "up"
                return directional(self.jump_direction)
            if self.jump_guard > 0:
                self.jump_guard -= 1
                return self.jump_input(True)
            self.clear_jump()
            return None
        if self.post_jump_up > 0:
            self.post_jump_up -= 1
            if self.landing_window > 0:
                self.landing_window -= 1
            if self.jump_lock > 0:
                self.jump_lock -= 1
            return "up"
        if self.jump_lock > 0:
            self.jump_lock -= 1
        if self.jump_lock > 0:
            hazard = closing_hazard(obs)
            if not hazard.get("present"):
                self.jump_lock = 0
        return None

    def hazard_jump(self, obs):
        if self.jump_direction is not None or self.jump_lock > 0:
            return None
        hazard = closing_hazard(obs)
        if hazard.get("urgency") not in ("overlap", "commit"):
            return None
        # Do not throw away a visibly progressing climb for a distant warning;
        # an overlap is still a hard escape.
        if self.climb and self.climb_progress and hazard.get("urgency") != "overlap":
            return None
        return self.begin_jump(obs, self.jump_direction_for(obs, hazard), "hazard")

    # -------------------- rivet route --------------------
    def nearest_ladder(self, obs, side, y):
        desired = 108 if side == "right" else 48
        values = obs.get("ladders", []) if isinstance(obs, dict) else []
        candidates = []
        if isinstance(values, list):
            for item in values:
                if not visible(item):
                    continue
                lx = number(item.get("x"), desired)
                ly = number(item.get("y"), y)
                if abs(ly - y) <= 22.0 or abs(ly - (y - 8.0)) <= 22.0:
                    candidates.append((abs(lx - desired), lx - 2.0))
        if candidates:
            candidates.sort()
            return candidates[0][1]
        return float(desired - 2)

    def rivet_action(self, obs):
        rows = rows_of(obs)
        if self.previous_rows is not None:
            for index, (old, new) in enumerate(zip(self.previous_rows, rows)):
                if index == self.rivet_row and old != new and new in (0, 6, 12, 18):
                    self.gap_direction = (
                        "right" if self.rivet_preference == "left" else "left"
                    )
                    self.log("rivet_change", row=index, old=old, new=new,
                             gap=self.gap_direction)
        self.previous_rows = list(rows)

        row = next((i for i, value in enumerate(rows) if value < 18), 4)
        if row >= 4:
            return "up"
        if row != self.rivet_row:
            self.rivet_row = row
            self.rivet_preference = (
                self.astra_hint if row == 0 and self.astra_hint in ("left", "right")
                else ("left" if row % 2 == 0 else "right")
            )
            self.gap_direction = None

        if self.gap_direction in ("left", "right") and self.jump_lock == 0:
            direction = self.gap_direction
            self.gap_direction = None
            return self.begin_jump(obs, direction, "rivet_gap")

        p = player_of(obs)
        x = number(p.get("x"), 45.0)
        y = number(p.get("y"), 176.0)
        preferred_ladder_side = "right" if row % 2 == 0 else "left"
        target_y = RIVET_Y[row]
        if y > target_y + 9.0:
            target = self.nearest_ladder(obs, preferred_ladder_side, y)
            if x < target - 1.0:
                return "right"
            if x > target + 1.0:
                return "left"
            return "up"
        if y < target_y - 12.0:
            return "down"

        side = remaining_side(rows[row], self.rivet_preference)
        if side is None:
            return "up"
        target_x = RIVET_EDGE[side]
        if x < target_x - 8.0:
            return "right"
        if x > target_x + 8.0:
            return "left"
        # Approach a left rivet from left-to-right and a right rivet from
        # right-to-left; this gives the byte-change detector a gap edge.
        return "right" if side == "left" else "left"

    # -------------------- slow Astra bridge --------------------
    def start_astra(self, obs):
        if self.astra_started or stage_of(obs) != "rivet":
            return
        try:
            if self.ctx.budget.remaining().get("system2_calls", 0) <= 0:
                return
        except Exception:
            return
        self.astra_started = True
        state = {
            "game": "Atari 2600 Donkey Kong NTSC retail port",
            "stage": "rivet",
            "rivet_rows": rows_of(obs),
            "player_side": "left" if number(player_of(obs).get("x"), 45) < 80 else "right",
            "goal": "choose only the first bottom-row sweep side; code owns timing and gaps",
        }
        prompt = (
            "You are a slow route reviewer for the Atari 2600 Donkey Kong NTSC "
            "retail port. The actor owns all timing, hazards, ladder alignment, "
            "raw row bookkeeping, and lifecycle. Based only on this JSON, return "
            "exactly {\"first_side\":\"left\"} or "
            "{\"first_side\":\"right\"}. Do not return actions or prose. State="
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
            result = future.result()
            text = getattr(result, "text", "")
            if not isinstance(text, str):
                return
            parsed = None
            try:
                parsed = json.loads(text)
            except (TypeError, ValueError):
                left = text.find("{")
                right = text.rfind("}")
                if left >= 0 and right > left:
                    try:
                        parsed = json.loads(text[left:right + 1])
                    except (TypeError, ValueError):
                        parsed = None
            side = parsed.get("first_side") if isinstance(parsed, dict) else None
            if side in ("left", "right"):
                self.astra_hint = side
                if self.rivet_row == 0 and self.previous_rows in (None, [0, 0, 0, 0]):
                    self.rivet_preference = side
                self.log("astra_hint", side=side)
        except Exception as exc:
            self.log("astra_failed", error=type(exc).__name__)

    # -------------------- policy and loop --------------------
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
            if lives > 0 and self.transition_age >= 18 and self.restart_edges < 4:
                if (self.transition_age - 18) % 28 == 0:
                    self.restart_edges += 1
                    return "fire"
            # Let the emulator reach game_over, but do not leave a no-life
            # transition running for the entire wall deadline.
            if lives <= 0 and self.transition_age >= 90:
                self.stop_after_transition = True
            return "noop"
        if stage not in ("ramp", "rivet"):
            return "noop"

        if stage == "rivet":
            if stage_changed:
                self.start_astra(obs)
            self.request_rivet_judgment(obs)

        ongoing = self.jump_action(obs)
        if ongoing is not None:
            return ongoing
        emergency = self.hazard_jump(obs)
        if emergency is not None:
            return emergency

        action = self.ramp_action(obs) if stage == "ramp" else self.rivet_action(obs)
        if stage == "ramp" and self.climb and self.no_progress >= 18:
            self.request_connector_judgment(obs)
        if stage == "rivet" and self.jev_hint in ("left", "right"):
            if not self.climb and self.gap_direction is None:
                self.rivet_preference = self.jev_hint
            self.jev_hint = None
        if stage_changed or life_changed or screen_changed:
            self.log("decision_boundary", stage=stage, level=self.level,
                     action=action, lives=lives, screens=screens)
        if self.ticks % 20 == 0:
            p = player_of(obs)
            self.log("snapshot", tick=self.ticks, stage=stage, level=self.level,
                     connector=self.connector, probe=self.probe,
                     x=round(number(p.get("x"), 0)), y=round(number(p.get("y"), 0)),
                     action=action, climb=self.climb, jump=self.jump_direction,
                     raw_jump=raw_jump_state(obs), hazard=closing_hazard(obs).get("urgency"),
                     lives=lives, score=number(obs.get("score_total"), 0))
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
        if self.last_stage == "idle" and not self.post("fire"):
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
            self.ticks += 1
            action = self.choose(obs)
            if not self.post(action):
                break
            if self.stop_after_transition:
                break
        self.log("finished", ticks=self.ticks, score=getattr(game, "score", None),
                 level=self.level, jumps=self.jump_count)


class Program:
    def __init__(self, ctx):
        self.ctx = ctx

    def run_episode(self, game) -> None:
        Controller(self.ctx).run(game)


def build(ctx):
    return Program(ctx)
