"""Observation-driven route supervisor for Atari 2600 Donkey Kong.

The actor is deliberately independent of inference on the four-frame control
path.  It executes a small connector graph, confirms every upward transition
from the observed player y, and treats a jump as a transaction.  Jev is an
asynchronous event arbiter for a real connector stall or a dangerous hazard;
Astra is an asynchronous macro reviewer that can change connector preference
or the rivet sweep only at a revalidated route boundary.
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

# Player-left-edge coordinates.  The public ladder boxes are two pixels to the
# right of the useful alignment point in the traces.  Each tuple is one
# walkway, with the second value on a split walkway being the alternate short
# connector.
RAMP_ROWS = (176, 148, 120, 92, 64, 40)
RAMP_CONNECTORS = (
    (106,),
    (46, 78),
    (106, 86),
    (46, 66),
    (106,),
    (30, 74),
)
RIVET_ROWS = (176, 148, 120, 92)
RIVET_X = {"left": 46, "right": 106}


# ---------- defensive observation helpers ----------


def _number(value, default=0.0):
    try:
        result = float(value)
        return result if math.isfinite(result) else default
    except (TypeError, ValueError):
        return default


def _integer(value, default=0):
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _stage(obs):
    return obs.get("stage", "unknown") if isinstance(obs, dict) else "unknown"


def _player(obs):
    value = obs.get("player", {}) if isinstance(obs, dict) else {}
    return value if isinstance(value, dict) else {}


def _xy(obs):
    p = _player(obs)
    return _number(p.get("x"), 45.0), _number(p.get("y"), 176.0)


def _visible(item):
    return isinstance(item, dict) and item.get("visible", True) is not False


def _raw(obs, address):
    values = obs.get("raw_ram", []) if isinstance(obs, dict) else []
    if not isinstance(values, (list, tuple)):
        return 0
    index = int(address) & 0x7F
    if index >= len(values):
        return 0
    return _integer(values[index], 0)


def _rows(obs):
    values = obs.get("rivet_rows", []) if isinstance(obs, dict) else []
    if not isinstance(values, (list, tuple)):
        values = []
    result = []
    for value in list(values)[:4]:
        item = _integer(value, 0)
        result.append(item if item in (0, 6, 12, 18) else 0)
    while len(result) < 4:
        result.append(0)
    return result


def _move(direction, fire=False):
    if direction not in ("left", "right"):
        return "fire" if fire else "noop"
    return direction + ("fire" if fire else "")


def _probability(answer, option):
    probabilities = getattr(answer, "probabilities", None)
    if not isinstance(probabilities, dict):
        return None
    value = _number(probabilities.get(option), -1.0)
    return value if value >= 0.0 else None


def _json_object(text):
    """Extract a JSON object without trusting arbitrary Astra formatting."""
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


# ---------- exact code-owned perception ----------


def _hazard(obs):
    """Return the closest closing obstacle on Mario's current walkway."""
    px, py = _xy(obs)
    hazards = obs.get("hazards", []) if isinstance(obs, dict) else []
    if not isinstance(hazards, list):
        hazards = []
    best = None
    for item in hazards:
        if not _visible(item):
            continue
        hx = _number(item.get("x"), 0.0)
        hy = _number(item.get("y"), 0.0)
        if abs(hy - py) > 18.0:
            continue
        dx = _number(item.get("dx"), 0.0)
        delta = hx - px
        distance = abs(delta)
        closing = (
            (delta < 0.0 and dx > 0.2) or
            (delta > 0.0 and dx < -0.2) or
            (distance <= 15.0 and abs(dx) <= 0.2)
        )
        if not closing and distance > 18.0:
            continue
        if distance <= 14.0:
            urgency = "overlap"
        elif distance <= 23.0:
            urgency = "immediate"
        elif closing and distance <= 38.0:
            urgency = "soon"
        elif closing and distance <= 64.0:
            urgency = "watch"
        else:
            urgency = "near"
        fact = {
            "present": True,
            "slot": _integer(item.get("slot"), -1),
            "side": "left" if delta < 0.0 else "right",
            "motion": "left" if dx < -0.2 else "right" if dx > 0.2 else "still",
            "closing": bool(closing),
            "distance": round(distance, 1),
            "distance_bucket": (
                "overlap" if distance <= 14.0 else
                "near" if distance <= 28.0 else
                "approaching" if distance <= 42.0 else "far"
            ),
            "urgency": urgency,
        }
        rank = (
            0 if urgency == "overlap" else
            1 if urgency == "immediate" else
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
        "closing": False, "distance": 999.0, "distance_bucket": "none",
        "urgency": "clear",
    }


def _remaining_side(mask, preference):
    """Decode the verified normal 0/6/12/18 rivet row statuses."""
    if mask == 18:
        return None
    if mask == 6:
        return "right"
    if mask == 12:
        return "left"
    return preference if preference in ("left", "right") else "left"


class RouteController:
    """Own one episode; inference can advise but never owns a tick."""

    def __init__(self, ctx, game):
        self.ctx = ctx
        self.game = game
        self.tick = 0
        self.last_action = "noop"
        self.last_stage = None
        self.last_lives = None
        self.last_screens = 0

        # Route graph state.  `phase` means the observed walkway currently
        # occupied, not the number of UP commands issued.  `branch` is the
        # selected connector on that walkway.  Successful branch choices are
        # retained within the episode, including across life resets.
        self.phase = 0
        self.branch = 0
        self.preferred = {}
        self.failed = set()
        self.climb = False
        self.climb_origin_y = None
        self.climb_best_y = None
        self.climb_age = 0
        self.climb_stall = 0
        self.walk_stall = 0
        self.previous_x = None
        self.route_event = None
        self.deadlocks = 0
        # Risk schedule: preserve 0018's first-Mario score lane, spend exactly
        # one subsequent Mario on a nominal route probe, then fall back to score.
        # This is episode-local and driven only by observed life boundaries.
        self.farm_mode = False
        self.route_probe_used = False
        self.adopt_pending = False

        # A jump is a transaction: directional FIRE until RAM confirms the
        # airborne state, then a short route-directed landing tail.  This avoids
        # both FIRE spam and the old one-action reversal/phase coupling.
        self.jump_direction = None
        self.jump_seen = False
        self.jump_guard = 0
        self.jump_tail = 0
        self.jump_lock = 0
        self.post_jump_up = 0
        self.jump_reason = None
        self.jump_count = 0

        # Cartridge lifecycle.  lives_remaining is spare lives, so it is never
        # used as a terminal test; only the explicit game-over flags are.
        self.idle_edges = 0
        self.transition_age = 0
        self.restart_edges = 0

        # Rivet screen state.
        self.rivet_row = 0
        self.rivet_preference = "left"
        self.previous_rows = None
        self.rivet_gap_direction = None
        self.rivet_climb = False

        # Nonblocking model bridge.  Answers carry a route signature and expire
        # quickly; an old answer cannot move Mario after a boundary.
        self.jev_future = None
        self.jev_stamp = None
        self.jev_started = -10000
        self.jev_answer = None
        self.jev_answer_stamp = None
        self.jev_answer_tick = -10000

        self.astra_future = None
        self.astra_calls = 0
        self.macro = {
            "connector": "adaptive",
            "hazard": "jump",
            "first_side": "left",
        }
        self.astra_reason = None

    # ---------- execution and lifecycle ----------

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

    def _clear_jump(self):
        self.jump_direction = None
        self.jump_seen = False
        self.jump_guard = 0
        self.jump_tail = 0
        self.post_jump_up = 0
        self.jump_reason = None

    def _reset_active(self, stage):
        self.phase = 0
        self.branch = 0
        self.climb = False
        self.climb_origin_y = None
        self.climb_best_y = None
        self.climb_age = 0
        self.climb_stall = 0
        self.walk_stall = 0
        self.previous_x = None
        self.route_event = None
        self._clear_jump()
        self.jump_lock = 0
        self.adopt_pending = False
        self.rivet_row = 0
        self.rivet_preference = self.macro.get("first_side", "left")
        if self.rivet_preference not in ("left", "right"):
            self.rivet_preference = "left"
        self.previous_rows = None
        self.rivet_gap_direction = None
        self.rivet_climb = False
        self._select_branch(0, reset=True)
        self.log("active_reset", stage=stage, phase=self.phase,
                 farm=self.farm_mode, route_probe_used=self.route_probe_used,
                 preferred=dict(self.preferred), macro=dict(self.macro))

    def _sync(self, obs):
        stage = _stage(obs)
        lives = _integer(obs.get("lives_remaining"), 0) if isinstance(obs, dict) else 0
        screens = _integer(obs.get("screens_completed"), 0) if isinstance(obs, dict) else 0
        changed = stage != self.last_stage
        life_changed = self.last_lives is not None and lives != self.last_lives
        screen_changed = screens != self.last_screens
        if changed or life_changed or screen_changed:
            self.log("boundary", old_stage=self.last_stage, stage=stage,
                     old_lives=self.last_lives, lives=lives,
                     old_screens=self.last_screens, screens=screens)
            if stage in ("ramp", "rivet"):
                if (life_changed and stage == "ramp" and
                        self.last_stage == "transition" and screens == 0):
                    if not self.route_probe_used:
                        # First life boundary: spend the next Mario on the route
                        # graph instead of immediately resuming score farming.
                        self.route_probe_used = True
                        self.farm_mode = False
                        self.log("route_probe_enabled", phase=self.phase, lives=lives,
                                 reason="first life boundary on incomplete first screen")
                    else:
                        # The one-time route attempt has consumed its risk budget.
                        # Remaining play uses the parent's durable score lane.
                        self.farm_mode = True
                        self.log("score_fallback_enabled", phase=self.phase,
                                 reason="route probe life boundary")
                self._reset_active(stage)
        if stage == "transition":
            self.transition_age = self.transition_age + 1 if self.last_stage == "transition" else 1
        else:
            self.transition_age = 0
        if changed:
            if stage == "idle":
                self.idle_edges = 0
            if stage != "transition":
                self.restart_edges = 0
        self.last_stage, self.last_lives, self.last_screens = stage, lives, screens
        return stage, lives, screens, changed, life_changed, screen_changed

    def _lifecycle_action(self, obs, stage, lives):
        if stage == "idle":
            # FIRE is an edge, not a continuously held jump.
            if self.idle_edges < 3:
                self.idle_edges += 1
                return "fire"
            return "noop"
        if stage == "game_over" or bool(obs.get("terminated", False)) or bool(obs.get("truncated", False)):
            return "noop"
        if stage == "transition":
            # The same sparse schedule works for death and screen transitions.
            # In particular, do not suppress it when lives==0: that is the
            # spare-life byte and the current Mario can still restart.
            restart_points = (18, 46, 74, 102, 130, 158)
            if self.transition_age in restart_points and self.restart_edges < len(restart_points):
                self.restart_edges += 1
                self.log("restart_edge", age=self.transition_age, edge=self.restart_edges,
                         lives=lives, game_state=obs.get("game_state"),
                         life_lost_flag=obs.get("life_lost_flag"))
                return "fire"
            return "noop"
        if stage not in ("ramp", "rivet"):
            return "noop"
        return None

    # ---------- connector graph ----------

    def _select_branch(self, phase, reset=False):
        phase = max(0, min(int(phase), len(RAMP_CONNECTORS) - 1))
        options = RAMP_CONNECTORS[phase]
        if len(options) == 1:
            self.branch = 0
            return
        selected = self.preferred.get(phase, 0)
        if self.macro.get("connector") == "alternate" and phase not in self.preferred:
            selected = 1
        if selected >= len(options) or (phase, selected) in self.failed:
            selected = next((i for i in range(len(options))
                             if (phase, i) not in self.failed), 0)
        self.branch = selected
        if reset:
            self.log("connector_select", phase=phase, branch=selected,
                     target=options[selected])

    def _target(self):
        phase = max(0, min(self.phase, len(RAMP_CONNECTORS) - 1))
        options = RAMP_CONNECTORS[phase]
        branch = max(0, min(self.branch, len(options) - 1))
        return float(options[branch])

    def _adopt_observed_connector(self, obs):
        """Reconcile a jump landing with the graph before walking again.

        A known difficult landing is around x=87,y=111 on the phase-2 band.
        The code does not infer a new level from x alone: it only adopts a
        connector that is close to Mario and whose row is the current phase.
        Normal horizontal travel never calls this operation as an implicit
        ladder choice.
        """
        if self.last_stage != "ramp" or not self.adopt_pending:
            return
        # Consume the one-observation permission before examining geometry.
        # If the landing is not near an alternate, ordinary route control
        # resumes rather than carrying a stale guess into later ticks.
        self.adopt_pending = False
        if self.phase >= len(RAMP_CONNECTORS):
            return
        x, y = _xy(obs)
        options = RAMP_CONNECTORS[self.phase]
        if len(options) <= 1:
            return
        row_y = RAMP_ROWS[self.phase]
        # A landing can be a few pixels above the walkway; a very high landing
        # belongs to the next row and is not used to change this connector.
        if y < row_y - 22.0 or y > row_y + 14.0:
            return
        candidates = [(abs(x - target), index)
                      for index, target in enumerate(options)
                      if abs(x - target) <= 6.0]
        if not candidates:
            return
        _, chosen = min(candidates)
        if chosen != self.branch:
            old = self.branch
            self.branch = chosen
            self.preferred[self.phase] = chosen
            self.failed.discard((self.phase, chosen))
            self.climb = False
            self.climb_origin_y = None
            self.climb_best_y = None
            self.climb_age = 0
            self.climb_stall = 0
            self.walk_stall = 0
            self.log("connector_adopted", phase=self.phase, old=old,
                     branch=chosen, x=round(x, 1), y=round(y, 1))

    def _switch_connector(self, reason):
        options = RAMP_CONNECTORS[max(0, min(self.phase, len(RAMP_CONNECTORS) - 1))]
        old = self.branch
        self.failed.add((self.phase, old))
        alternatives = [i for i in range(len(options))
                        if i != old and (self.phase, i) not in self.failed]
        if not alternatives:
            # A one-connector or exhausted band gets a clean retry, rather
            # than an endless left/right oscillation.
            self.failed.discard((self.phase, old))
            alternatives = [old]
        self.branch = alternatives[0]
        self.preferred[self.phase] = self.branch
        self.climb = False
        self.climb_origin_y = None
        self.climb_best_y = None
        self.climb_age = 0
        self.climb_stall = 0
        self.walk_stall = 0
        self.route_event = "connector_stall"
        self.deadlocks += 1
        self.log("connector_switch", phase=self.phase, old=old,
                 branch=self.branch, target=options[self.branch], reason=reason,
                 deadlocks=self.deadlocks)

    def _mark_climb(self, obs, y):
        old = self.phase
        self.preferred[old] = self.branch
        self.failed.discard((old, self.branch))
        self.phase = min(self.phase + 1, len(RAMP_CONNECTORS) - 1)
        # Only the first active Mario is intentionally score-laned.  After
        # route_probe_enabled, retain the connector graph through phase 1 and
        # let the same observed-y transactions test upper-route progress.
        if old == 0 and self.phase == 1 and not self.route_probe_used:
            self.farm_mode = True
            self.log("score_fallback_enabled", phase=self.phase,
                     reason="first-life score reserve")
        self.climb = False
        self.climb_origin_y = None
        self.climb_best_y = None
        self.climb_age = 0
        self.climb_stall = 0
        self.walk_stall = 0
        self.previous_x = None
        self.route_event = "ramp_advance"
        self._select_branch(self.phase)
        self.log("ramp_advance", old_phase=old, phase=self.phase,
                 branch=self.branch, x=round(_xy(obs)[0], 1), y=round(y, 1))

    def _ramp_action(self, obs):
        """Advance the connector graph; all phase changes require y evidence."""
        x, y = _xy(obs)
        if self.farm_mode:
            # Keep Mario at the proven right-side phase-1 ladder.  UP is
            # intentionally not represented as an open climb here: holding it
            # at y≈154 preserves the hazard lane without advancing a phantom
            # route phase.  The hazard transaction still runs before this
            # fallback and returns Mario to this target after each jump.
            target = 106.0
            if x < target - 1.5:
                return "right"
            if x > target + 1.5:
                return "left"
            return "up"

        phase = max(0, min(self.phase, len(RAMP_CONNECTORS) - 1))
        target = self._target()
        row_y = RAMP_ROWS[phase]

        if self.climb:
            self.climb_age += 1
            if self.climb_best_y is None or y < self.climb_best_y - 1.0:
                self.climb_best_y = y
                self.climb_stall = 0
            else:
                self.climb_stall += 1
            if phase + 1 < len(RAMP_ROWS):
                next_y = RAMP_ROWS[phase + 1]
                gain = ((self.climb_origin_y - y)
                        if self.climb_origin_y is not None else 0.0)
                if gain >= 8.0 and y <= next_y + 6.0 and self.climb_age >= 5:
                    self._mark_climb(obs, y)
                    return self._ramp_action(obs)
            # An alternate connector is a real branch, not a timing tweak.  If
            # the current rung has not responded, try it and ask Jev whether
            # that switch is semantically appropriate while the fallback acts.
            if self.climb_stall >= 20:
                if len(RAMP_CONNECTORS[phase]) > 1:
                    self._switch_connector("vertical_no_progress")
                    return self._ramp_action(obs)
                self.climb = False
                self.climb_origin_y = None
                self.climb_best_y = None
                self.climb_age = 0
                self.climb_stall = 0
            else:
                return "up"

        # Reacquire a target after a jump or after a slope moved the sprite.
        # Horizontal no-motion is evidence for a graph branch switch only after
        # several observations; it is never treated as an automatic phase
        # advance.
        if self.previous_x is not None and self.last_action in ("left", "right"):
            if abs(x - self.previous_x) < 0.6:
                self.walk_stall += 1
            else:
                self.walk_stall = 0
        else:
            self.walk_stall = 0
        self.previous_x = x

        if abs(x - target) > 1.5:
            if self.walk_stall >= 8 and len(RAMP_CONNECTORS[phase]) > 1:
                self._switch_connector("horizontal_no_progress")
                return self._ramp_action(obs)
            return "right" if x < target else "left"

        # At the top, reaching the final goal-side x is the useful action; an
        # extra UP at the top is harmless and lets the cartridge close the ramp.
        if phase == len(RAMP_CONNECTORS) - 1:
            return "up" if y <= RAMP_ROWS[phase] + 5.0 else "left"

        self.climb = True
        self.climb_origin_y = y
        self.climb_best_y = y
        self.climb_age = 0
        self.climb_stall = 0
        self.route_event = None
        self.log("climb_begin", phase=phase, branch=self.branch,
                 target=round(target, 1), x=round(x, 1), y=round(y, 1))
        return "up"

    # ---------- jump transaction ----------

    def _jump_direction(self, obs, hazard):
        motion = hazard.get("motion")
        if motion == "left":
            return "right"
        if motion == "right":
            return "left"
        x, _ = _xy(obs)
        target = 106.0 if self.farm_mode else self._target()
        if abs(target - x) > 3.0:
            return "right" if target > x else "left"
        side = hazard.get("side")
        return "right" if side == "left" else "left" if side == "right" else "right"

    def _begin_jump(self, obs, direction, reason, hazard=None):
        if direction not in ("left", "right"):
            direction = "right"
        self.jump_direction = direction
        self.jump_seen = False
        self.jump_guard = 8
        self.jump_tail = 0
        self.jump_reason = reason
        self.jump_lock = 16
        self.climb = False
        self.climb_origin_y = None
        self.climb_best_y = None
        self.climb_age = 0
        self.climb_stall = 0
        self.jump_count += 1
        self.log("jump_begin", tick=self.tick, direction=direction,
                 reason=reason, phase=self.phase,
                 hazard={k: (hazard or {}).get(k) for k in
                         ("slot", "side", "motion", "distance_bucket", "urgency")})
        return _move(direction, True)

    def _landing_direction(self, obs, default):
        if self.farm_mode:
            # The score-preserving lane was empirically the committed FIRE
            # direction through the landing tail; route reacquisition resumes
            # only after the tail and post-jump UP have finished.
            return default
        x, _ = _xy(obs)
        target = self._target()
        if x < target - 2.0:
            return "right"
        if x > target + 2.0:
            return "left"
        return default

    def _jump_action(self, obs):
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
        state = _raw(obs, 0x85)
        if state > 0:
            self.jump_seen = True
            self.jump_guard = 8
            return _move(direction, True)
        if not self.jump_seen and self.jump_guard > 0:
            self.jump_guard -= 1
            return _move(direction, True)
        if self.jump_seen:
            if self.jump_tail < 5:
                self.jump_tail += 1
                return _move(self._landing_direction(obs, direction), False)
            self.log("jump_end", tick=self.tick, direction=direction,
                     reason=self.jump_reason, phase=self.phase)
            self._clear_jump()
            # Keep UP for a short post-landing tail and suppress immediate
            # re-triggering by the same slot.  This is the lane-preserving
            # primitive that made the score fallback durable in the archive.
            self.post_jump_up = 8
            self.jump_lock = 26
            if self.last_stage == "ramp":
                self.adopt_pending = True
                self._adopt_observed_connector(obs)
            return None
        # Missing/garbled RAM must not latch FIRE forever.
        self.log("jump_unconfirmed", tick=self.tick, phase=self.phase)
        self._clear_jump()
        self.jump_lock = 8
        return None

    def _hazard_action(self, obs, stage):
        if stage not in ("ramp", "rivet"):
            return None
        if self.jump_direction is not None or self.jump_lock > 0:
            return None
        fact = _hazard(obs)
        if self.farm_mode:
            # The stable archived lane waits until the sprite is actually
            # overlapping/within the 16-pixel collision window.  Starting at
            # the wider early band is what killed the first fallback trial.
            if not fact.get("present") or _number(fact.get("distance"), 999.0) > 16.0:
                return None
        elif fact.get("urgency") not in ("overlap", "immediate"):
            return None
        # A rung is safer than abandoning a nearly completed climb.  An actual
        # overlap is the exception; it is too late to rely on the ladder.
        if self.climb and self.climb_origin_y is not None:
            _, y = _xy(obs)
            gain = self.climb_origin_y - y
            if gain >= 5.0 and fact.get("urgency") != "overlap":
                return None
        direction = self._jump_direction(obs, fact)
        return self._begin_jump(obs, direction, "hazard", fact)

    # ---------- rivet route ----------

    def _rivet_ladder(self, obs, side):
        desired = 108.0 if side == "right" else 48.0
        _, py = _xy(obs)
        ladders = obs.get("ladders", []) if isinstance(obs, dict) else []
        candidates = []
        if isinstance(ladders, list):
            for item in ladders:
                if not _visible(item):
                    continue
                lx = _number(item.get("x"), desired) - 2.0
                ly = _number(item.get("y"), py)
                if abs(ly - py) <= 23.0 or abs(ly - (py - 8.0)) <= 23.0:
                    candidates.append((abs(lx - (desired - 2.0)) + abs(ly - py) * 0.05, lx))
        return min(candidates)[1] if candidates else desired - 2.0

    def _rivet_action(self, obs):
        current = _rows(obs)
        if self.previous_rows is not None:
            for row, (old, new) in enumerate(zip(self.previous_rows, current)):
                if row != self.rivet_row or old == new:
                    continue
                if new in (6, 12, 18):
                    removed = "left" if new == 6 else "right" if new == 12 else None
                    if removed is not None:
                        other = "right" if removed == "left" else "left"
                        self.rivet_gap_direction = other
                        self.log("rivet_removed", row=row, removed=removed,
                                 next_side=other, new=new)
        self.previous_rows = list(current)

        while self.rivet_row < 4 and current[self.rivet_row] == 18:
            self.rivet_row += 1
            self.rivet_gap_direction = None
            self.rivet_climb = False
        if self.rivet_row >= 4:
            return "up"

        row = self.rivet_row
        mask = current[row]
        preference = self.rivet_preference if row % 2 == 0 else (
            "right" if self.rivet_preference == "left" else "left"
        )
        side = _remaining_side(mask, preference)

        # After a rivet is removed the other target is across the hole.  Use a
        # committed jump, rather than walking through a cleared location.
        if self.rivet_gap_direction in ("left", "right"):
            direction = self.rivet_gap_direction
            self.rivet_gap_direction = None
            return self._begin_jump(obs, direction, "rivet_gap")

        x, y = _xy(obs)
        if side is None:
            ladder_side = "right" if row % 2 == 0 else "left"
            target = self._rivet_ladder(obs, ladder_side)
            if abs(x - target) > 2.0:
                return "right" if x < target else "left"
            self.rivet_climb = True
            return "up"

        target_y = RIVET_ROWS[row]
        if y > target_y + 10.0 or self.rivet_climb:
            ladder_side = "right" if row % 2 == 0 else "left"
            target = self._rivet_ladder(obs, ladder_side)
            if abs(x - target) > 2.0:
                self.rivet_climb = False
                return "right" if x < target else "left"
            if y <= target_y + 8.0:
                self.rivet_climb = False
            return "up"

        target_x = RIVET_X[side]
        if abs(x - target_x) > 7.0:
            return "right" if x < target_x else "left"
        # One directional walk over the remaining rivet; the row-byte change
        # on the next observation causes a gap jump to the other side.
        return "left" if side == "left" else "right"

    # ---------- model bridge ----------

    def _model_state(self, obs, event, fallback):
        x, y = _xy(obs)
        p = _player(obs)
        hazard = _hazard(obs)
        phase = max(0, min(self.phase, len(RAMP_CONNECTORS) - 1))
        options = list(RAMP_CONNECTORS[phase])
        return {
            "game": "Atari 2600 Donkey Kong NTSC retail cartridge",
            "event": event,
            "stage": _stage(obs),
            "player": {
                "horizontal_band": "left" if x < 52 else "right" if x > 100 else "middle",
                "vertical_band": "bottom" if y > 160 else "lower" if y > 132 else "middle" if y > 78 else "top",
                "motion": "up" if _number(p.get("dy"), 0.0) < -1 else
                          "down" if _number(p.get("dy"), 0.0) > 1 else "steady",
                "jump_committed": self.jump_direction is not None,
            },
            "route": {
                "phase": phase,
                "selected_connector": self.branch,
                "connector_options": len(options),
                "alternative_available": len(options) > 1,
                "climb_open": self.climb,
                "fallback_action": fallback,
                "rivet_row": self.rivet_row,
                "rivet_rows": _rows(obs),
                "farm_mode": self.farm_mode,
                "route_probe_used": self.route_probe_used,
            },
            "hazard": {key: hazard.get(key) for key in
                       ("present", "slot", "side", "motion", "distance_bucket", "closing", "urgency")},
            "progress": {
                "lives_remaining": _integer(obs.get("lives_remaining"), 0),
                "screens_completed": _integer(obs.get("screens_completed"), 0),
                "rivets_removed": _integer(obs.get("rivets_removed"), 0),
            },
            "last_action": self.last_action,
            "macro": dict(self.macro),
        }

    def _stamp(self, obs, event):
        hazard = _hazard(obs)
        return (
            event, _stage(obs), self.phase, self.branch, self.rivet_row,
            hazard.get("slot", -1), hazard.get("urgency", "clear"),
            hazard.get("motion", "none"), self.deadlocks,
        )

    def _budget_allows(self, kind):
        try:
            if bool(self.ctx.done):
                return False
            remaining = self.ctx.budget.remaining()
            return _integer(remaining.get(kind), 0) > 0
        except Exception:
            return False

    def _request_jev(self, obs, event, fallback):
        if self.jev_future is not None:
            return
        if event not in ("connector_stall", "hazard"):
            return
        if not self._budget_allows("system1_calls"):
            return
        stamp = self._stamp(obs, event)
        if stamp == self.jev_stamp and self.tick - self.jev_started < 10:
            return
        state = self._model_state(obs, event, fallback)
        questions = {
            "route": Choice(
                "Use the supplied `route` facts. At a connector stall, choose the alternate only if "
                "an alternative is explicitly available and the nominal connector has made no progress. "
                "Choose hold for a live climb or when the event is a hazard.",
                {
                    "hold": "Keep the code fallback and do not change the connector.",
                    "nominal": "Use the already selected nominal connector on this walkway.",
                    "alternate": "Use the other connector explicitly available on this walkway.",
                },
            ),
            "escape": Choice(
                "Judge the immediate tactical label from `hazard`. Choose keep_climbing when a "
                "committed climb is the safer escape; choose a jump direction only for a same-row "
                "overlap or immediate closing object. Do not calculate coordinates.",
                {
                    "keep_climbing": "Continue the current UP transaction and do not jump.",
                    "jump_left": "Start or maintain a leftward directional jump.",
                    "jump_right": "Start or maintain a rightward directional jump.",
                },
            ),
        }
        self.jev_stamp = stamp
        self.jev_started = self.tick
        try:
            self.jev_future = self.ctx.spawn(
                lambda: self.ctx.decide(state, questions, timeout_s=0.65)
            )
        except Exception as exc:
            self.jev_future = None
            self.log("jev_spawn_failed", error=type(exc).__name__)

    def _collect_jev(self):
        future = self.jev_future
        if future is None or not future.done():
            return
        self.jev_future = None
        try:
            answers = future.result()
            if not isinstance(answers, dict):
                return
            route = answers.get("route")
            escape = answers.get("escape")
            route_value = getattr(route, "value", None)
            escape_value = getattr(escape, "value", None)
            route_p = _probability(route, route_value)
            escape_p = _probability(escape, escape_value)
            route_ok = route_value in ("hold", "nominal", "alternate") and (
                route_p is not None and route_p >= 0.60 or
                route_value == "hold" and route_p is None
            )
            escape_ok = escape_value in ("keep_climbing", "jump_left", "jump_right") and (
                escape_p is not None and escape_p >= 0.60 or
                escape_value == "keep_climbing" and escape_p is None
            )
            self.jev_answer = {
                "route": route_value if route_ok else None,
                "escape": escape_value if escape_ok else None,
            }
            self.jev_answer_stamp = self.jev_stamp
            self.jev_answer_tick = self.tick
            self.log("jev_result", route=self.jev_answer["route"],
                     escape=self.jev_answer["escape"], route_probability=route_p,
                     escape_probability=escape_p)
        except Exception as exc:
            self.log("jev_failed", error=type(exc).__name__)

    def _apply_jev(self, obs, fallback):
        answer = self.jev_answer
        if not isinstance(answer, dict):
            return fallback
        current = self._stamp(obs, self.jev_answer_stamp[0]
                              if isinstance(self.jev_answer_stamp, tuple) else "unknown")
        if current != self.jev_answer_stamp or self.tick - self.jev_answer_tick > 3:
            self.jev_answer = None
            return fallback
        event = self.jev_answer_stamp[0]
        route = answer.get("route")
        escape = answer.get("escape")
        self.jev_answer = None
        if event == "connector_stall" and route in ("nominal", "alternate"):
            if route == "alternate" and len(RAMP_CONNECTORS[self.phase]) > 1:
                if self.branch == 0:
                    self._switch_connector("jev_alternate")
                    return self._ramp_action(obs)
            if route == "nominal" and self.branch != 0:
                self.branch = 0
                self.failed.discard((self.phase, 0))
                return self._ramp_action(obs)
        if event == "hazard" and escape in ("jump_left", "jump_right"):
            fact = _hazard(obs)
            if (fact.get("urgency") in ("overlap", "immediate") and
                    self.jump_direction is None and self.jump_lock == 0):
                return self._begin_jump(obs, escape.rsplit("_", 1)[-1], "jev_hazard", fact)
        return fallback

    def _start_astra(self, obs, reason):
        if self.astra_future is not None or self.astra_calls >= 2:
            return
        if not self._budget_allows("system2_calls"):
            return
        self.astra_calls += 1
        self.astra_reason = reason
        state = self._model_state(obs, reason, self.last_action)
        prompt = (
            "You are Astra, a slow macro reviewer for the Atari 2600 Donkey Kong NTSC "
            "retail cartridge. The real-time actor owns exact joystick timing, RAM jump "
            "confirmation, ladder geometry, rivet bytes, and lifecycle. Do not return a "
            "joystick sequence. Use only the supplied two-screen port facts. Return one "
            "compact JSON object with exactly these optional labels: "
            "connector='nominal'|'alternate'|'adaptive', hazard='climb'|'jump', "
            "first_side='left'|'right'. An absent/invalid label is ignored. "
            "The result is advisory and will be applied only after its route boundary "
            "is revalidated. Reason=" + str(reason) + "\nSTATE=" +
            json.dumps(state, separators=(",", ":"), allow_nan=False)
        )
        try:
            self.astra_future = self.ctx.spawn(
                lambda: self.ctx.think(prompt, timeout_s=85.0)
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
            parsed = _json_object(getattr(result, "text", ""))
        except Exception as exc:
            self.log("astra_failed", error=type(exc).__name__)
            return
        if not isinstance(parsed, dict):
            return
        connector = parsed.get("connector")
        hazard = parsed.get("hazard")
        side = parsed.get("first_side")
        if connector in ("nominal", "alternate", "adaptive"):
            self.macro["connector"] = connector
        if hazard in ("climb", "jump"):
            self.macro["hazard"] = hazard
        if side in ("left", "right"):
            self.macro["first_side"] = side
            if self.rivet_row == 0 and self.last_stage == "rivet":
                self.rivet_preference = side
        self.log("astra_policy", connector=self.macro["connector"],
                 hazard=self.macro["hazard"], first_side=self.macro["first_side"])

    # ---------- top-level decision ----------

    def choose(self, obs):
        stage, lives, screens, changed, life_changed, screen_changed = self._sync(obs)
        self._collect_jev()
        self._collect_astra()

        lifecycle = self._lifecycle_action(obs, stage, lives)
        if lifecycle is not None:
            return lifecycle

        # The route may use the explicit alternate only after the current
        # observation has proved that Mario is on that connector.  The model is
        # asked after the fallback is computed, never before the action.
        if stage == "rivet" and changed:
            self._start_astra(obs, "rivet entry")
        elif self.deadlocks >= 2 and self.astra_future is None and self.astra_calls < 1:
            self._start_astra(obs, "repeated connector deadlock")

        ongoing = self._jump_action(obs)
        if ongoing is not None:
            return ongoing

        emergency = self._hazard_action(obs, stage)
        if emergency is not None:
            self._request_jev(obs, "hazard", emergency)
            return emergency

        action = self._ramp_action(obs) if stage == "ramp" else self._rivet_action(obs)
        if self.route_event == "connector_stall":
            self._request_jev(obs, "connector_stall", action)
        action = self._apply_jev(obs, action)

        if changed or life_changed or screen_changed:
            self.log("decision_boundary", stage=stage, phase=self.phase,
                     branch=self.branch, action=action, lives=lives, screens=screens)
        if self.tick % 20 == 0:
            x, y = _xy(obs)
            self.log("snapshot", tick=self.tick, stage=stage, phase=self.phase,
                     branch=self.branch, x=round(x), y=round(y), action=action,
                     climb=self.climb, farm=self.farm_mode,
                     route_probe_used=self.route_probe_used,
                     jump=self.jump_direction, raw_jump=_raw(obs, 0x85),
                     hazard=_hazard(obs).get("urgency"), lives=lives,
                     score=_number(obs.get("score_total"), 0.0))
        return action

    def run(self):
        try:
            obs = self.ctx.observe()
        except Exception:
            return
        if not isinstance(obs, dict):
            obs = {}
        self.last_stage = _stage(obs)
        self.last_lives = _integer(obs.get("lives_remaining"), 0)
        self.last_screens = _integer(obs.get("screens_completed"), 0)

        if self.last_stage == "idle":
            if not self.post("fire"):
                return
        else:
            action = self.choose(obs)
            if not self.post(action):
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
            # Once the bonus is exhausted and the spare-life byte is zero, the
            # cartridge is a nonterminal transition shell with no remaining
            # score opportunity.  Stop there instead of paying the 900-second
            # deadline, matching the best candidate's score-preserving endpoint.
            if (_stage(obs) == "transition" and
                    _integer(obs.get("lives_remaining"), 0) <= 0 and
                    obs.get("current_bonus") is not None and
                    _number(obs.get("current_bonus"), 0.0) <= 0.0):
                self.log("score_shell_end", tick=self.tick,
                         score=_number(obs.get("score_total"), 0.0))
                break
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
        RouteController(self.ctx, game).run()


def build(ctx):
    return Program(ctx)
