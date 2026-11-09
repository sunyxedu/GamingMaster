"""Lane-committed real-time controller for Atari 2600 Donkey Kong.

The controller is deliberately code-first.  It treats a barrel encounter as a
crossing transaction with an explicit landing lane, rather than as a short
FIRE burst followed by a direction reversal.  A route transaction owns each
ladder until the observation proves that Mario reached the next girder.
"""
from __future__ import annotations

import json
import math

from metaharness.models.jev import Choice, Noul


# These are player-left-edge targets.  The public ladder boxes are two pixels
# to the right of the useful Mario edge in the successful traces.
RAMP_TARGETS = (106, 46, 106, 46, 106, 30)
RAMP_Y = (176, 148, 120, 92, 64, 40)
RIVET_Y = (176, 148, 120, 92)
RIVET_X = {"left": 46, "right": 106}

ACTIONS = {
    "noop", "fire", "up", "right", "left", "down", "upright", "upleft",
    "downright", "downleft", "upfire", "rightfire", "leftfire", "downfire",
    "uprightfire", "upleftfire", "downrightfire", "downleftfire",
}


def _num(value, default=0.0):
    try:
        result = float(value)
        return result if math.isfinite(result) else default
    except (TypeError, ValueError):
        return default


def _int(value, default=0):
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _stage(obs):
    return obs.get("stage", "unknown") if isinstance(obs, dict) else "unknown"


def _player(obs):
    value = obs.get("player", {}) if isinstance(obs, dict) else {}
    return value if isinstance(value, dict) else {}


def _rows(obs):
    value = obs.get("rivet_rows", []) if isinstance(obs, dict) else []
    if not isinstance(value, (list, tuple)):
        value = []
    answer = []
    for item in list(value)[:4]:
        n = _int(item, 0)
        answer.append(n if n in (0, 6, 12, 18) else 0)
    while len(answer) < 4:
        answer.append(0)
    return answer


def _visible(item):
    return isinstance(item, dict) and item.get("visible", True) is not False


def _raw_jump(obs):
    raw = obs.get("raw_ram", []) if isinstance(obs, dict) else []
    if isinstance(raw, list) and len(raw) > 5:
        return _int(raw[5], 0)
    return 0


def _move(direction, fire=False):
    if direction not in ("left", "right"):
        return "fire" if fire else "noop"
    return direction + ("fire" if fire else "")


def _same_row_hazard(obs):
    """Return the nearest closing object on Mario's current walkway.

    All numeric comparisons happen here, not in Jev.  A hazard on a neighboring
    girder is excluded by the vertical gate.  The extra projected bucket is
    used only by the deterministic controller to start a crossing early.
    """
    p = _player(obs)
    px = _num(p.get("x"), 45.0)
    py = _num(p.get("y"), 176.0)
    best = None
    hazards = obs.get("hazards", []) if isinstance(obs, dict) else []
    if not isinstance(hazards, list):
        hazards = []
    for item in hazards:
        if not _visible(item):
            continue
        hx = _num(item.get("x"), 0.0)
        hy = _num(item.get("y"), 0.0)
        dx = _num(item.get("dx"), 0.0)
        delta = hx - px
        distance = abs(delta)
        if abs(hy - py) > 18.0:
            continue
        closing = ((delta < 0 and dx > 0.2) or
                   (delta > 0 and dx < -0.2) or
                   (distance <= 16.0 and abs(dx) <= 0.2))
        if not closing:
            continue
        # Keep the trigger close to the successful 0012 crossing.  The
        # controller still labels the larger approach band for Jev/diagnostics,
        # but does not spend a jump at the far edge of it.
        urgency = (
            "overlap" if distance <= 16.0 else
            "commit" if distance <= 22.0 else
            "warning" if distance <= 58.0 else "far"
        )
        item_fact = {
            "present": True,
            "side": "left" if delta < 0 else "right",
            "motion": "left" if dx < -0.2 else "right" if dx > 0.2 else "still",
            "closing": True,
            "distance": round(distance, 1),
            "distance_bucket": (
                "overlapping" if distance <= 16.0 else
                "near" if distance <= 34.0 else
                "approaching" if distance <= 58.0 else "far"
            ),
            "urgency": urgency,
            "slot": _int(item.get("slot"), -1),
        }
        rank = (0 if urgency == "overlap" else
                1 if urgency == "commit" else
                2 if urgency == "warning" else 3, distance)
        if best is None or rank < best[0]:
            best = (rank, item_fact)
    if best is not None:
        return best[1]
    return {
        "present": False, "side": "none", "motion": "none", "closing": False,
        "distance": 999.0, "distance_bucket": "none", "urgency": "clear",
        "slot": -1,
    }


def _remaining_side(mask, preferred):
    """Decode the verified normal 0/6/12/18 rivet statuses."""
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
        self.tick = 0
        self.last_action = "noop"
        self.last_stage = None
        self.last_lives = None
        self.last_screens = 0

        # A route phase is the walkway currently occupied.  It is advanced only
        # by observed y progress, never by the number of UP actions posted.
        self.phase = 0
        self.climb_open = False
        self.climb_origin_y = None
        self.climb_best_y = None
        self.climb_ticks = 0
        self.climb_stall = 0
        self.edge_bias = 0

        # Crossing transaction.  The FIRE direction gets Mario over the
        # object; exit_dir is separately chosen from the current route target,
        # so a crossing at an alternating walkway returns toward its rung.
        self.cross_dir = None
        self.exit_dir = None
        self.cross_active_seen = False
        self.cross_start_guard = 7
        self.cross_landing = 0
        self.post_cross_up = 0
        self.cross_lock = 0
        self.cross_count = 0
        self.last_hazard_slot = -1

        # State machine for idle/death transitions.
        self.transition_age = 0
        self.restart_attempts = 0
        self.idle_attempts = 0

        # Rivet bookkeeping is reset from observations after every life.
        self.rivet_row = 0
        self.rivet_first = "left"
        self.previous_rows = None
        self.gap_dir = None

        # Jev is advisory and is only requested for a verified rivet route or a
        # genuine route stall.  It can never block an action or override a hard
        # crossing/lifecycle guard.
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

    def clear_crossing(self):
        self.cross_dir = None
        self.exit_dir = None
        self.cross_active_seen = False
        self.cross_start_guard = 7
        self.cross_landing = 0
        self.post_cross_up = 0
        self.cross_lock = 0
        self.last_hazard_slot = -1

    def reset_play(self, stage):
        self.phase = 0
        self.climb_open = False
        self.climb_origin_y = None
        self.climb_best_y = None
        self.climb_ticks = 0
        self.climb_stall = 0
        self.edge_bias = 0
        self.clear_crossing()
        self.gap_dir = None
        if stage == "rivet":
            self.rivet_row = 0
            self.rivet_first = "left"
            self.previous_rows = None

    def sync(self, obs):
        stage = _stage(obs)
        lives = _int(obs.get("lives_remaining"), 0) if isinstance(obs, dict) else 0
        screens = _int(obs.get("screens_completed"), 0) if isinstance(obs, dict) else 0
        changed = stage != self.last_stage
        life_changed = self.last_lives is not None and lives != self.last_lives
        screen_changed = screens != self.last_screens
        if changed or life_changed:
            self.log("boundary", old_stage=self.last_stage, stage=stage,
                     old_lives=self.last_lives, lives=lives, phase=self.phase)
            if stage in ("ramp", "rivet"):
                self.reset_play(stage)
        if changed:
            self.transition_age = 0
            self.restart_attempts = 0
            self.idle_attempts = 0
        if stage == "transition":
            self.transition_age += 1
        else:
            self.transition_age = 0
        self.last_stage = stage
        self.last_lives = lives
        self.last_screens = screens
        return stage, lives, screens, changed, life_changed, screen_changed

    # -------------------- hazard crossing --------------------
    def crossing_direction(self, obs, hazard):
        """Select the lane beyond the hazard, with a route-side fallback."""
        side = hazard.get("side")
        if side in ("left", "right"):
            return side
        p = _player(obs)
        x = _num(p.get("x"), 45.0)
        target = RAMP_TARGETS[min(max(self.phase, 0), len(RAMP_TARGETS) - 1)]
        if x < target - 1:
            return "right"
        if x > target + 1:
            return "left"
        return "right"

    def exit_direction(self, obs, fallback):
        p = _player(obs)
        x = _num(p.get("x"), 45.0)
        target = self.route_target()
        if x < target - 2:
            return "right"
        if x > target + 2:
            return "left"
        return fallback

    def begin_crossing(self, obs, hazard, reason="hazard"):
        self.cross_dir = self.crossing_direction(obs, hazard)
        self.exit_dir = self.exit_direction(obs, self.cross_dir)
        self.cross_active_seen = False
        self.cross_start_guard = 7
        self.cross_landing = 0
        self.post_cross_up = 0
        # This lock spans the same hazard's passage.  It is separate from the
        # short landing tail so an overlapping slot cannot retrigger FIRE.
        self.cross_lock = 26
        self.last_hazard_slot = _int(hazard.get("slot"), -1)
        self.cross_count += 1
        self.log("cross_begin", tick=self.tick, direction=self.cross_dir,
                 exit=self.exit_dir, reason=reason, hazard=hazard, phase=self.phase)
        return _move(self.cross_dir, True)

    def crossing_action(self, obs):
        if self.cross_dir is not None:
            state = _raw_jump(obs)
            if state > 0:
                self.cross_active_seen = True
                self.cross_start_guard = 7
                return _move(self.cross_dir, True)
            if self.cross_active_seen:
                if self.cross_landing <= 0:
                    # Exit toward the current rung/row target.  On alternating
                    # walkways this is intentionally not always the FIRE side.
                    self.cross_landing = 5
                self.cross_landing -= 1
                if self.cross_landing <= 0:
                    self.cross_dir = None
                    self.post_cross_up = 8
                    return "up"
                return _move(self.exit_dir or self.cross_dir, False)
            if self.cross_start_guard > 0:
                self.cross_start_guard -= 1
                return _move(self.cross_dir, True)
            # A malformed/missing RAM snapshot must not leave FIRE latched.
            self.cross_dir = None
            return None

        if self.post_cross_up > 0:
            self.post_cross_up -= 1
            return "up"

        if self.cross_lock > 0:
            self.cross_lock -= 1
            # If the current row is visibly clear, permit a new independent
            # object sooner; otherwise let the bounded lock expire.
            hazard = _same_row_hazard(obs)
            if not hazard.get("present"):
                self.cross_lock = 0
        return None

    def last_walk_direction(self, obs):
        p = _player(obs)
        x = _num(p.get("x"), 45.0)
        target = RAMP_TARGETS[min(max(self.phase, 0), len(RAMP_TARGETS) - 1)]
        if x < target - 1:
            return "right"
        if x > target + 1:
            return "left"
        return "right" if self.cross_dir != "left" else "left"

    def hazard_interrupt(self, obs, stage):
        if stage not in ("ramp", "rivet"):
            return None
        if self.cross_dir is not None or self.cross_lock > 0:
            return None
        hazard = _same_row_hazard(obs)
        if hazard.get("urgency") not in ("commit", "overlap"):
            return None
        # Once a climb has visibly gained height, leave it alone unless the
        # object is already overlapping.  This prevents a lower-row warning
        # from throwing Mario off a ladder transaction.
        if self.climb_open and hazard.get("urgency") != "overlap":
            return None
        return self.begin_crossing(obs, hazard)

    # -------------------- ramp route --------------------
    def route_target(self):
        return RAMP_TARGETS[min(max(self.phase, 0), len(RAMP_TARGETS) - 1)] + self.edge_bias

    def open_climb(self, y):
        self.climb_open = True
        self.climb_origin_y = y
        self.climb_best_y = y
        self.climb_ticks = 0
        self.climb_stall = 0

    def ramp_action(self, obs):
        p = _player(obs)
        x = _num(p.get("x"), 45.0)
        y = _num(p.get("y"), 176.0)
        dy = _num(p.get("dy"), 0.0)
        phase = min(max(self.phase, 0), len(RAMP_TARGETS) - 1)
        target = self.route_target()
        # A jump can land on the second usable connector of a split row.  If
        # Mario is already above the nominal walkway and close to that
        # connector, adopt it instead of walking forever toward the other box.
        alternate_edges = {1: 78, 2: 86, 3: 66, 5: 74}
        alternate = alternate_edges.get(phase)
        if (alternate is not None and y < RAMP_Y[phase] - 4 and
                abs(x - alternate) <= 4):
            target = alternate
            self.edge_bias = 0
            self.log("landed_on_alternate", phase=phase, x=round(x), y=round(y))

            # A crossing may have landed away from the rung.  Close this
            # transaction and reacquire rather than holding UP in empty space.
            if abs(x - target) > 2.5:
                self.climb_open = False
                self.climb_origin_y = None
                self.climb_best_y = None
                self.climb_ticks = 0
                self.climb_stall = 0
                return self.ramp_action(obs)
            self.climb_ticks += 1
            if self.climb_best_y is None or y < self.climb_best_y - 1:
                self.climb_best_y = y
                self.climb_stall = 0
            elif abs(dy) <= 1:
                self.climb_stall += 1
            if phase + 1 < len(RAMP_Y):
                next_y = RAMP_Y[phase + 1]
                moved = (self.climb_origin_y is not None and
                         self.climb_origin_y - y >= 8)
                reached = y <= next_y + 6
                if moved and reached and self.climb_ticks >= 5:
                    self.phase += 1
                    self.climb_open = False
                    self.climb_origin_y = None
                    self.climb_best_y = None
                    self.climb_ticks = 0
                    self.climb_stall = 0
                    self.edge_bias = 0
                    self.log("ramp_advance", phase=self.phase, x=round(x), y=round(y))
                    return self.ramp_action(obs)
            if self.climb_stall >= 30:
                # Only a one-pixel calibration probe; the route phase remains
                # unchanged until y confirms a climb.
                self.edge_bias = -1 if self.edge_bias == 0 else 0
                self.climb_open = False
                self.climb_origin_y = None
                self.climb_best_y = None
                self.climb_ticks = 0
                self.climb_stall = 0
                self.log("rung_probe", phase=phase, bias=self.edge_bias,
                         x=round(x), y=round(y))
                return self.ramp_action(obs)
            return "up"

        if x < target - 1:
            return "right"
        if x > target + 1:
            return "left"
        self.open_climb(y)
        return "up"

    # -------------------- rivet route --------------------
    def ladder_for_side(self, obs, side):
        desired = 108 if side == "right" else 48
        p = _player(obs)
        py = _num(p.get("y"), 176.0)
        ladders = obs.get("ladders", []) if isinstance(obs, dict) else []
        options = []
        if isinstance(ladders, list):
            for item in ladders:
                if not _visible(item):
                    continue
                lx = _num(item.get("x"), desired)
                ly = _num(item.get("y"), py)
                if abs(ly - py) <= 22 or abs(ly - (py - 8)) <= 22:
                    options.append((abs(lx - desired) + abs(ly - py) * 0.05, lx - 2))
        if options:
            options.sort()
            return options[0][1]
        return desired - 2

    def rivet_action(self, obs):
        rows = _rows(obs)
        if self.previous_rows is not None:
            for i, (old, new) in enumerate(zip(self.previous_rows, rows)):
                if i == self.rivet_row and old != new and new in (0, 6, 12, 18):
                    self.gap_dir = "right" if self.rivet_first == "left" else "left"
                    self.log("rivet_change", row=i, old=old, new=new,
                             gap=self.gap_dir)
        self.previous_rows = list(rows)

        row = next((i for i, value in enumerate(rows) if value < 18), 4)
        if row >= 4:
            return "up"
        if row != self.rivet_row:
            self.rivet_row = row
            self.gap_dir = None
        preferred = self.rivet_first if row % 2 == 0 else (
            "right" if self.rivet_first == "left" else "left")

        if self.gap_dir in ("left", "right"):
            direction = self.gap_dir
            self.gap_dir = None
            if self.cross_dir is None and self.cross_lock == 0:
                return self.begin_crossing(obs, {"side": direction,
                                                 "motion": "none",
                                                 "distance": 0,
                                                 "slot": -1}, "rivet_gap")

        p = _player(obs)
        x = _num(p.get("x"), 45.0)
        y = _num(p.get("y"), 176.0)
        side = _remaining_side(rows[row], preferred)
        if side is None:
            climb_side = "right" if row % 2 == 0 else "left"
            target = self.ladder_for_side(obs, climb_side)
            if x < target - 1:
                return "right"
            if x > target + 1:
                return "left"
            return "up"
        target_y = RIVET_Y[row]
        if y > target_y + 9:
            climb_side = "right" if row % 2 == 0 else "left"
            target = self.ladder_for_side(obs, climb_side)
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

    # -------------------- advisory models --------------------
    def model_state(self, obs):
        p = _player(obs)
        hazard = _same_row_hazard(obs)
        return {
            "game": "Atari 2600 Donkey Kong NTSC retail port",
            "stage": _stage(obs),
            "player": {
                "horizontal_band": "left" if _num(p.get("x"), 45) < 52 else
                                   "right" if _num(p.get("x"), 45) > 100 else "middle",
                "vertical_band": "bottom" if _num(p.get("y"), 176) > 160 else
                                 "lower" if _num(p.get("y"), 176) > 132 else
                                 "middle" if _num(p.get("y"), 176) > 78 else "top",
                "motion": "up" if _num(p.get("dy"), 0) < -1 else
                          "down" if _num(p.get("dy"), 0) > 1 else "steady",
            },
            "route": {"ramp_phase": self.phase, "climb_open": self.climb_open,
                      "rivet_row": self.rivet_row, "rivet_rows": _rows(obs),
                      "crossing": self.cross_dir is not None},
            "hazard": {key: hazard.get(key) for key in
                       ("present", "side", "motion", "distance_bucket", "closing", "urgency")},
            "progress": {"lives_remaining": _int(obs.get("lives_remaining"), 0),
                         "screens_completed": _int(obs.get("screens_completed"), 0),
                         "rivets_removed": _int(obs.get("rivets_removed"), 0)},
            "last_action": self.last_action,
        }

    def maybe_ask_jev(self, obs):
        if self.jev_future is not None:
            return
        stage = _stage(obs)
        hazard = _same_row_hazard(obs)
        interval = 5 if hazard.get("urgency") in ("commit", "overlap") else 16
        if stage != "rivet" or self.tick - self.jev_requested < interval:
            return
        try:
            if self.ctx.budget.remaining().get("system1_calls", 0) <= 0:
                return
        except Exception:
            return
        state = self.model_state(obs)
        questions = {
            "sweep": Choice(
                "Choose only the already-described rivet-route preference. Use route.rivet_rows and route.rivet_row; do not count, calculate, or invent coordinates. Choose hold while route.crossing or route.climb_open is true.",
                {
                    "left": "The next legal sweep should approach the left remaining rivet.",
                    "right": "The next legal sweep should approach the right remaining rivet.",
                    "hold": "Keep the deterministic route transaction unchanged.",
                },
            ),
            "danger": Noul(
                "Does hazard explicitly describe a commit or overlap closing object on Mario's current row?",
                {
                    "true": "hazard.urgency is commit or overlap and hazard.closing is true.",
                    "false": "No commit/overlap closing current-row object is described.",
                },
            ),
        }
        self.jev_requested = self.tick
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
                probability = _num(probabilities.get(value), 0.0)
            if value in ("left", "right") and (probability is None or probability >= 0.55):
                self.jev_hint = value
                self.log("jev_sweep", side=value, probability=probability)
        except Exception as exc:
            self.log("jev_failed", error=type(exc).__name__)

    def maybe_start_astra(self, obs):
        # A slow call is useful only after a verified rivet screen.  It is
        # asynchronous and supplies a preference, never an action sequence.
        if self.astra_started or _stage(obs) != "rivet":
            return
        try:
            if self.ctx.budget.remaining().get("system2_calls", 0) <= 0:
                return
        except Exception:
            return
        self.astra_started = True
        state = self.model_state(obs)
        prompt = (
            "Review this Atari 2600 Donkey Kong NTSC rivet state. The code owns "
            "timing, hazards, ladder alignment, gaps, and lifecycle. Return only "
            "JSON {\"first_side\":\"left\"} or {\"first_side\":\"right\"}; "
            "never return actions or prose. State=" +
            json.dumps(state, separators=(",", ":"), allow_nan=False)
        )
        try:
            self.astra_future = self.ctx.spawn(
                lambda: self.ctx.think(prompt, timeout_s=80.0)
            )
        except Exception as exc:
            self.log("astra_start_failed", error=type(exc).__name__)
            self.astra_future = None

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
                left, right = text.find("{"), text.rfind("}")
                if left >= 0 and right > left:
                    try:
                        parsed = json.loads(text[left:right + 1])
                    except (TypeError, ValueError):
                        parsed = None
            value = parsed.get("first_side") if isinstance(parsed, dict) else None
            if value in ("left", "right"):
                self.astra_hint = value
                self.rivet_first = value
                self.log("astra_side", side=value)
        except Exception as exc:
            self.log("astra_failed", error=type(exc).__name__)

    # -------------------- policy and loop --------------------
    def choose(self, obs):
        stage, lives, screens, stage_changed, life_changed, screen_changed = self.sync(obs)
        self.collect_jev()
        self.collect_astra()

        if stage == "idle":
            if self.idle_attempts < 2:
                self.idle_attempts += 1
                return "fire"
            return "noop"
        if stage == "game_over" or bool(obs.get("terminated", False)):
            return "noop"
        if stage == "transition":
            if lives <= 0:
                # No remaining Mario can be restarted.  run() exits before
                # posting another noop, avoiding the idle parent's 900-second
                # shell after the final death.
                return "noop"
            if self.transition_age >= 18 and self.restart_attempts < 4:
                if (self.transition_age - 18) % 28 == 0:
                    self.restart_attempts += 1
                    return "fire"
            return "noop"
        if stage not in ("ramp", "rivet"):
            return "noop"

        if stage_changed and stage == "rivet":
            self.maybe_start_astra(obs)
        elif stage == "rivet" and not self.astra_started:
            self.maybe_start_astra(obs)

        ongoing = self.crossing_action(obs)
        if ongoing is not None:
            return ongoing

        interrupt = self.hazard_interrupt(obs, stage)
        if interrupt is not None:
            return interrupt

        action = self.ramp_action(obs) if stage == "ramp" else self.rivet_action(obs)
        self.maybe_ask_jev(obs)
        if stage == "rivet" and self.jev_hint in ("left", "right") and not self.climb_open:
            row = min(max(self.rivet_row, 0), 3)
            side = _remaining_side(_rows(obs)[row], self.jev_hint)
            if side == self.jev_hint:
                self.rivet_first = self.jev_hint
            self.jev_hint = None

        if stage_changed or life_changed or screen_changed:
            self.log("decision_boundary", stage=stage, phase=self.phase,
                     action=action, lives=lives, screens=screens)
        if self.tick % 20 == 0:
            p = _player(obs)
            self.log("snapshot", tick=self.tick, stage=stage, phase=self.phase,
                     x=round(_num(p.get("x"), 0)), y=round(_num(p.get("y"), 0)),
                     action=action, climb=self.climb_open, cross=self.cross_dir,
                     raw_jump=_raw_jump(obs), hazard=_same_row_hazard(obs).get("urgency"),
                     lives=lives, score=_num(obs.get("score_total"), 0))
        return action

    def run(self, game):
        try:
            obs = self.ctx.observe()
        except Exception:
            return
        if not isinstance(obs, dict):
            obs = {}
        self.last_stage = _stage(obs)
        self.last_lives = _int(obs.get("lives_remaining"), 0)
        self.last_screens = _int(obs.get("screens_completed"), 0)
        if self.last_stage == "idle":
            if not self.post("fire"):
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
            stage = _stage(obs)
            lives = _int(obs.get("lives_remaining"), 0)
            if stage == "transition" and lives <= 0:
                self.log("final_transition", tick=self.tick, score=_num(obs.get("score_total"), 0))
                break
            action = self.choose(obs)
            if not self.post(action):
                break
        self.log("finished", ticks=self.tick, score=getattr(game, "score", None),
                 phase=self.phase, crossings=self.cross_count)


class Program:
    def __init__(self, ctx):
        self.ctx = ctx

    def run_episode(self, game) -> None:
        Controller(self.ctx).run(game)


def build(ctx):
    return Program(ctx)
