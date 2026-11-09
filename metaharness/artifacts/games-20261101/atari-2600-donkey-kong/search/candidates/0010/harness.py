"""A feedback controller built around ladder-settling and crossing jumps.

The controller treats a climb as a two-part transaction: press UP until the
new walkway is reached, then keep the ladder input for several observations
while the sprite settles before changing horizontal direction.  Obstacles are
handled by a separate crossing primitive.  Its direction opposes measured
hazard motion (with the route target as a still-hazard fallback), and a latch
prevents a still-overlapping obstacle from retriggering FIRE on every quantum.
"""
from __future__ import annotations

import json
import math

from metaharness.models.jev import Choice, Noul


# Public decoder coordinates.  The first five entries are the nominal ladder
# route; the last entry is the top approach.  A route entry is the horizontal
# side at which UP is attempted on that walkway.
RAMP_Y = (176, 148, 120, 92, 64, 40)
RAMP_ROUTE_X = (108, 48, 108, 48, 108, 32)
RIVET_Y = (176, 148, 120, 92)
RIVET_X = {"left": 48, "right": 108}

# The game advances four NTSC frames per tick.  A jump lasts about 31 frames;
# eight FIRE quanta plus a non-FIRE landing tail is deliberately treated as one
# indivisible crossing primitive.
JUMP_QUANTA = 8
LANDING_QUANTA = 10
SETTLE_QUANTA = 8
MIN_CLIMB_QUANTA = 12


_ACTIONS = {
    "noop", "fire", "up", "right", "left", "down", "upright", "upleft",
    "downright", "downleft", "upfire", "rightfire", "leftfire",
    "downfire", "uprightfire", "upleftfire", "downrightfire", "downleftfire",
}


def _number(value, default=0.0):
    try:
        result = float(value)
        if math.isfinite(result):
            return result
    except (TypeError, ValueError):
        pass
    return default


def _integer(value, default=0):
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _stage(obs):
    if isinstance(obs, dict) and isinstance(obs.get("stage"), str):
        return obs["stage"]
    return "unknown"


def _visible(item):
    return isinstance(item, dict) and item.get("visible", True) is not False


def _player(obs):
    item = obs.get("player", {}) if isinstance(obs, dict) else {}
    return item if isinstance(item, dict) else {}


def _rows(obs):
    values = obs.get("rivet_rows", []) if isinstance(obs, dict) else []
    if not isinstance(values, (list, tuple)):
        values = []
    result = []
    for value in list(values)[:4]:
        n = _integer(value, 0)
        result.append(n if n in (0, 6, 12, 18) else 0)
    while len(result) < 4:
        result.append(0)
    return result


def _band(value, low, high):
    return "left" if value < low else "right" if value > high else "middle"


def _motion(item):
    dx = _number(item.get("dx", 0)) if isinstance(item, dict) else 0.0
    dy = _number(item.get("dy", 0)) if isinstance(item, dict) else 0.0
    if abs(dx) <= 1 and abs(dy) <= 1:
        return "still"
    if abs(dx) >= abs(dy):
        return "right" if dx > 0 else "left"
    return "down" if dy > 0 else "up"


def _hazard_facts(obs):
    """Return one small, named description of the nearest closing obstacle."""
    p = _player(obs)
    px = _number(p.get("x", 45))
    py = _number(p.get("y", 176))
    found = None
    hazards = obs.get("hazards", []) if isinstance(obs, dict) else []
    if not isinstance(hazards, list):
        hazards = []
    for raw in hazards:
        if not _visible(raw):
            continue
        hx = _number(raw.get("x", 0))
        hy = _number(raw.get("y", 0))
        # Hazards on another girder are irrelevant to the current crossing.
        # Keep this band narrow: the old parent reacted to objects one ramp
        # edge away and spent its jump before the actual crossing.
        if abs(hy - py) > 18:
            continue
        dx = _number(raw.get("dx", 0))
        horizontal = hx - px
        distance = abs(horizontal)
        closing = (
            (horizontal < 0 and dx > 0.2)
            or (horizontal > 0 and dx < -0.2)
            or (distance <= 16 and abs(dx) <= 0.2)
        )
        if not closing and distance > 18:
            continue
        if distance <= 23:
            urgency = "immediate"
        elif closing and distance <= 34:
            urgency = "soon"
        elif closing and distance <= 70:
            urgency = "watch"
        else:
            urgency = "near"
        item = {
            "present": True,
            "side": "left" if horizontal < 0 else "right",
            "motion": "left" if dx < -0.2 else "right" if dx > 0.2 else "still",
            "distance_bucket": (
                "overlapping" if distance <= 16 else
                "near" if distance <= 30 else
                "approaching" if distance <= 54 else "far"
            ),
            "closing": bool(closing),
            "urgency": urgency,
            "distance": round(distance, 1),
        }
        rank = (0 if urgency == "immediate" else 1 if urgency == "soon" else 2, distance)
        if found is None or rank < found[0]:
            found = (rank, item)
    if found is None:
        return {
            "present": False,
            "side": "none",
            "motion": "none",
            "distance_bucket": "none",
            "closing": False,
            "urgency": "clear",
        }
    return found[1]


def _row_side(row, first):
    return first if row % 2 == 0 else ("right" if first == "left" else "left")


def _action_for(direction, fire=False):
    if direction not in ("left", "right"):
        return "fire" if fire else "noop"
    return direction + ("fire" if fire else "")


class Controller:
    def __init__(self, ctx):
        self.ctx = ctx
        self.tick = 0
        self.last_action = "noop"
        self.last_stage = None
        self.last_lives = None
        self.last_screens = 0

        # Ramp transaction state.
        self.ramp_phase = 0
        self.climbing = False
        self.climb_ticks = 0
        self.settle_ticks = 0
        self.last_y = None
        self.last_walk_direction = "right"
        self.ramp_stall = 0

        # Rivet state is derived again after every life or stage transition.
        self.rivet_first = "left"
        self.rivet_row = 0
        self.rivet_target = "left"
        self.rivet_last_rows = None
        self.rivet_gap_direction = None
        self.rivet_climb_side = "right"
        self.rivet_climbing = False

        # A crossing is committed.  hazard_lock outlives the FIRE part so a
        # stationary sprite cannot cause a new jump every eight ticks.
        self.jump_direction = "right"
        self.jump_quanta = 0
        self.landing_quanta = 0
        self.hazard_lock = 0
        self.jump_cooldown = 0

        # Lifecycle state.
        self.transition_age = 0
        self.transition_attempts = 0
        self.idle_attempts = 0

        # Model calls are advisory and never sit on the actor's critical path.
        self.jev_future = None
        self.jev_requested_tick = -10000
        self.jev_hint = None
        self.astra_future = None
        self.astra_calls = 0

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

    def valid_action(self, game, action):
        if not isinstance(action, str) or action not in _ACTIONS:
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
        return action

    def post(self, game, action):
        action = self.valid_action(game, action)
        try:
            self.ctx.act(action)
            self.last_action = action
            return True
        except Exception as exc:
            self.log("act_failed", action=action, error=type(exc).__name__)
            return False

    def reset_route(self, stage):
        self.climbing = False
        self.climb_ticks = 0
        self.settle_ticks = 0
        self.last_y = None
        self.ramp_stall = 0
        self.jump_quanta = 0
        self.landing_quanta = 0
        self.hazard_lock = 0
        self.jump_cooldown = 0
        if stage == "ramp":
            self.ramp_phase = 0
            self.last_walk_direction = "right"
        elif stage == "rivet":
            self.rivet_row = 0
            self.rivet_target = self.rivet_first
            self.rivet_last_rows = None
            self.rivet_gap_direction = None
            self.rivet_climbing = False
            self.rivet_climb_side = "right" if self.rivet_first == "left" else "left"

    def sync(self, obs):
        stage = _stage(obs)
        lives = _integer(obs.get("lives_remaining", 0), 0) if isinstance(obs, dict) else 0
        screens = _integer(obs.get("screens_completed", 0), 0) if isinstance(obs, dict) else 0
        stage_changed = stage != self.last_stage
        life_changed = self.last_lives is not None and lives != self.last_lives
        screen_changed = screens != self.last_screens
        if stage_changed:
            self.log("stage_change", old=self.last_stage, new=stage,
                     lives=lives, screens=screens)
            self.transition_age = 0
            self.transition_attempts = 0
            self.idle_attempts = 0
            if stage in ("ramp", "rivet"):
                self.reset_route(stage)
        if life_changed:
            self.log("life_change", old=self.last_lives, new=lives)
            if stage in ("ramp", "rivet"):
                self.reset_route(stage)
        if screen_changed:
            self.log("screen_complete", screens=screens)
        if stage == "transition":
            self.transition_age += 1
        else:
            self.transition_age = 0
        self.last_stage = stage
        self.last_lives = lives
        self.last_screens = screens
        return stage, lives, screens, stage_changed, life_changed, screen_changed

    # ---------- Small model bridge ----------

    def model_state(self, obs):
        p = _player(obs)
        h = _hazard_facts(obs)
        stage = _stage(obs)
        px = _number(p.get("x", 45))
        py = _number(p.get("y", 176))
        if stage == "ramp":
            phase = min(max(self.ramp_phase, 0), len(RAMP_ROUTE_X) - 1)
            target = RAMP_ROUTE_X[phase]
            target_side = "left" if target < px - 2 else "right" if target > px + 2 else "aligned"
            route = {
                "kind": "ladder",
                "phase": phase,
                "target_side": target_side,
                "climb_active": self.climbing,
                "climb_settling": self.climbing and self.settle_ticks > 0,
            }
        elif stage == "rivet":
            route = {
                "kind": "rivet_or_gap",
                "row": min(max(self.rivet_row, 0), 3),
                "target_side": self.rivet_target,
                "climb_active": self.rivet_climbing,
                "climb_settling": False,
                "rivet_rows": _rows(obs),
            }
        else:
            route = {"kind": "none", "target_side": "none"}
        return {
            "game": "Atari 2600 Donkey Kong NTSC retail port",
            "stage": stage,
            "player": {
                "horizontal_band": _band(px, 52, 100),
                "vertical_band": "bottom" if py > 160 else "lower" if py > 132 else "middle" if py > 78 else "top",
                "motion": _motion(p),
                "jump_committed": self.jump_quanta > 0 or self.landing_quanta > 0,
            },
            "route": route,
            "hazard": h,
            "progress": {
                "lives_remaining": _integer(obs.get("lives_remaining", 0), 0),
                "screens_completed": _integer(obs.get("screens_completed", 0), 0),
                "rivets_removed": _integer(obs.get("rivets_removed", 0), 0),
            },
            "last_action": self.last_action,
        }

    def start_astra(self, obs, reason):
        # A slow reviewer can select only the first rivet side.  It is started
        # asynchronously; the verified ramp route and crossing direction are
        # not delegated to a call that may take tens of seconds.
        if self.astra_future is not None or self.astra_calls >= 3 or self.done_dummy():
            return
        try:
            if self.ctx.budget.remaining().get("system2_calls", 0) <= 0:
                return
        except Exception:
            return
        state = self.model_state(obs)
        prompt = (
            "You are a slow reviewer for the Atari 2600 NTSC Donkey Kong retail "
            "port, not the arcade game. The actor already owns timing, ladder "
            "alignment, hazard collision guards, and restart handling. Review "
            "only the future rivet screen. Return exactly compact JSON, no prose: "
            '{"rivet_first":"left|right"}. Choose a side only if the supplied '
            "observations justify it; otherwise return left. Never return an "
            "action sequence. Reason=" + str(reason)[:80] + " State=" +
            json.dumps(state, separators=(",", ":"), allow_nan=False)
        )
        self.astra_calls += 1

        def work():
            return self.ctx.think(prompt, timeout_s=85.0)

        try:
            self.astra_future = self.ctx.spawn(work)
        except Exception as exc:
            self.astra_future = None
            self.log("astra_spawn_failed", error=type(exc).__name__)

    def done_dummy(self):
        try:
            return bool(self.ctx.done)
        except Exception:
            return True

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
            if isinstance(parsed, dict) and parsed.get("rivet_first") in ("left", "right"):
                self.rivet_first = parsed["rivet_first"]
                self.log("astra_rivet_preference", value=self.rivet_first)
        except Exception as exc:
            self.log("astra_failed", error=type(exc).__name__)

    def start_jev(self, obs, hazard):
        if self.jev_future is not None or self.done_dummy():
            return
        # One request can serve the normal route and a hazard interrupt.  The
        # hard controller still acts if this 0.45s call is late.
        urgent = hazard.get("urgency") in ("soon", "immediate")
        if self.tick - self.jev_requested_tick < (4 if urgent else 12):
            return
        try:
            if self.ctx.budget.remaining().get("system1_calls", 0) <= 0:
                return
        except Exception:
            return
        state = self.model_state(obs)
        questions = {
            "tactic": Choice(
                "From the named facts, choose the immediate tactical label. "
                "Choose cross_left or cross_right only for a soon/immediate "
                "same-row closing obstacle; otherwise choose route. Do not do "
                "arithmetic and do not invent a coordinate.",
                {
                    "route": "Continue the code-computed ladder/rivet route; no urgent crossing is shown.",
                    "cross_left": "Commit to a leftward crossing jump for the current closing obstacle.",
                    "cross_right": "Commit to a rightward crossing jump for the current closing obstacle.",
                    "pause": "Use a short neutral hold only for transition or an explicitly unsettled state.",
                },
            ),
            "closing": Noul(
                "Do the normalized hazard facts explicitly show a same-row obstacle "
                "that is soon or immediate and needs a crossing jump?",
                {
                    "true": "hazard.urgency is soon or immediate and hazard.closing is true.",
                    "false": "No such urgent same-row closing obstacle is present.",
                },
            ),
        }
        self.jev_requested_tick = self.tick
        stage = _stage(obs)

        def work():
            return self.ctx.decide(state, questions, timeout_s=0.45)

        try:
            self.jev_future = self.ctx.spawn(work)
            self.jev_hint = (None, stage, self.tick)
        except Exception as exc:
            self.jev_future = None
            self.log("jev_spawn_failed", error=type(exc).__name__)

    def collect_jev(self, obs):
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
                self.jev_hint = (value, _stage(obs), self.tick)
                self.log("jev_tactic", tactic=value,
                         confidence=getattr(answer, "confidence", None))
        except Exception as exc:
            self.log("jev_failed", error=type(exc).__name__)

    # ---------- Route and crossing controller ----------

    def route_direction(self, px):
        if _stage(self.current_obs) == "ramp":
            target = RAMP_ROUTE_X[min(self.ramp_phase, len(RAMP_ROUTE_X) - 1)]
        else:
            target = RIVET_X.get(self.rivet_target, px)
        if target > px:
            return "right"
        if target < px:
            return "left"
        return None

    def choose_jump_direction(self, obs, hazard):
        # A credited jump is a crossing, not an evasive walk.  If the obstacle
        # has a measured horizontal velocity, steer against that velocity so
        # Mario and the obstacle meet in the air.  This is deliberately more
        # specific than choosing the side of the next ladder: the parent trace
        # repeatedly sent rightfire while an obstacle moved right into Mario.
        motion = hazard.get("motion")
        if motion == "left":
            return "right"
        if motion == "right":
            return "left"
        px = _number(_player(obs).get("x", 45))
        preferred = self.route_direction(px)
        if preferred in ("left", "right"):
            return preferred
        if hazard.get("side") == "left":
            return "left"
        if hazard.get("side") == "right":
            return "right"
        return self.last_walk_direction if self.last_walk_direction in ("left", "right") else "right"

    def start_crossing(self, obs, hazard, reason="hazard"):
        direction = self.choose_jump_direction(obs, hazard)
        self.jump_direction = direction
        self.jump_quanta = JUMP_QUANTA
        self.landing_quanta = 0
        # A lock is released by disappearance or by this bounded number of
        # quanta, not merely by the end of FIRE.  This is the anti-retrigger
        # part of the experiment.  Thirty-six quanta is long enough for the
        # same sprite to finish crossing the walkway, while a vanished sprite
        # releases it immediately.
        self.hazard_lock = 36
        self.jump_cooldown = 0
        self.log("crossing_started", direction=direction, reason=reason,
                 urgency=hazard.get("urgency"))
        return _action_for(direction, fire=True)

    def crossing_action(self, obs):
        hazard = _hazard_facts(obs)
        if self.jump_quanta > 0:
            self.jump_quanta -= 1
            if self.jump_quanta == 0:
                self.landing_quanta = LANDING_QUANTA
            return _action_for(self.jump_direction, fire=True), hazard
        if self.landing_quanta > 0:
            self.landing_quanta -= 1
            if self.landing_quanta == 0:
                self.jump_cooldown = 6
            return _action_for(self.jump_direction, fire=False), hazard
        if self.jump_cooldown > 0:
            self.jump_cooldown -= 1
        if self.hazard_lock > 0:
            self.hazard_lock -= 1
            if not hazard.get("present") or hazard.get("urgency") == "clear":
                self.hazard_lock = 0

        # A rivet byte change is a verified hole, not a visual suggestion.  It
        # gets the same committed crossing treatment even when no fireball is
        # close enough to trigger the hazard guard.
        gap = self.rivet_gap_direction if _stage(self.current_obs) == "rivet" else None
        if gap in ("left", "right") and self.jump_cooldown == 0 and self.hazard_lock == 0:
            self.rivet_gap_direction = None
            self.jump_direction = gap
            self.jump_quanta = JUMP_QUANTA
            self.landing_quanta = 0
            self.hazard_lock = 10
            self.log("crossing_started", direction=gap, reason="cleared_gap",
                     urgency=hazard.get("urgency"))
            return _action_for(gap, fire=True), hazard
        if (
            self.hazard_lock == 0
            and self.jump_cooldown == 0
            and hazard.get("urgency") in ("soon", "immediate")
        ):
            return self.start_crossing(obs, hazard), hazard
        return None, hazard

    def _begin_climb(self):
        self.climbing = True
        self.climb_ticks = 0
        self.settle_ticks = 0
        self.last_y = None
        self.ramp_stall = 0

    def ramp_action(self, obs):
        p = _player(obs)
        px = _number(p.get("x", 45))
        py = _number(p.get("y", 176))
        dy = _number(p.get("dy", 0))
        phase = min(max(self.ramp_phase, 0), len(RAMP_ROUTE_X) - 1)
        target_x = RAMP_ROUTE_X[phase]
        current_y = RAMP_Y[phase]

        if self.climbing:
            # A jump can land several pixels away from the ladder.  Continuing
            # to hold UP in that state is not a climb attempt at all; close the
            # transaction and reacquire the exact rung before trying again.
            if abs(px - target_x) > 2:
                self.climbing = False
                self.climb_ticks = 0
                self.settle_ticks = 0
                self.last_y = None
                self.ramp_stall = 0
                return self.ramp_action(obs)
            self.climb_ticks += 1
            if self.last_y is not None and abs(py - self.last_y) <= 1 and abs(dy) <= 1:
                self.settle_ticks += 1
            else:
                self.settle_ticks = 0
            self.last_y = py
            if phase < len(RAMP_ROUTE_X) - 1:
                next_y = RAMP_Y[phase + 1]
                # Do not turn at the first visually-near y.  The extra stable
                # observations let the cartridge finish its ladder animation.
                at_new_walkway = py <= next_y + 10
                if (
                    at_new_walkway
                    and self.climb_ticks >= MIN_CLIMB_QUANTA
                    and self.settle_ticks >= SETTLE_QUANTA
                ):
                    self.ramp_phase += 1
                    self.climbing = False
                    self.climb_ticks = 0
                    self.settle_ticks = 0
                    self.last_y = None
                    self.ramp_stall = 0
                    self.log("ladder_settled", phase=self.ramp_phase, y=round(py, 1))
                    return self.ramp_action(obs)
            # UP remains the only action while the transaction is open.  This
            # is intentionally different from switching to the next horizontal
            # target as soon as y is within a loose tolerance.
            if self.climb_ticks > 40 and self.settle_ticks == 0:
                self.ramp_stall += 1
                # A diagonal nudge is a recovery for a one-pixel alignment
                # miss, but it does not commit the phase or press FIRE.
                return "upright" if target_x >= px else "upleft"
            return "up"

        # The port's ladder collision window is narrower than the walk
        # controller's useful tolerance.  Reaching x=51 for a rung at x=48
        # can look aligned but still ignores UP, so walk to a one-pixel edge
        # before opening the climb transaction.
        if abs(px - target_x) > 1:
            direction = "right" if px < target_x else "left"
            self.last_walk_direction = direction
            return direction

        # A small vertical mismatch is normal after a death restart.  Starting
        # the transaction at the actual reported x is safer than walking away.
        if py > current_y + 12 or abs(px - target_x) <= 1:
            self._begin_climb()
            return "up"
        return "up"

    def ladder_x(self, obs, preferred):
        p = _player(obs)
        py = _number(p.get("y", 176))
        desired = 108 if preferred == "right" else 48
        values = obs.get("ladders", []) if isinstance(obs, dict) else []
        candidates = []
        if isinstance(values, list):
            for item in values:
                if not _visible(item):
                    continue
                x = _number(item.get("x", 0))
                y = _number(item.get("y", 0))
                if abs(y - py) <= 20 or abs(y - (py - 8)) <= 20:
                    candidates.append((abs(x - desired) + abs(y - py) * 0.1, x))
        if candidates:
            candidates.sort()
            return candidates[0][1]
        return desired

    def _rivet_remaining_side(self, mask, preferred):
        # The normal bytes are 0, 6, 12, 18.  The first sweep is chosen by the
        # serpentine route.  Once one byte is observed, use the side that was
        # not swept; this avoids trusting a decoder's prose about bit labels.
        if mask == 18:
            return None
        if mask == 0:
            return preferred
        if self.rivet_gap_direction in ("left", "right"):
            return self.rivet_gap_direction
        # A conservative fallback for a state entered mid-row.
        return "right" if preferred == "left" else "left"

    def rivet_action(self, obs):
        rows = _rows(obs)
        row = next((i for i, value in enumerate(rows) if value < 18), 4)
        if row >= 4:
            return "up"
        if row != self.rivet_row:
            self.rivet_row = row
            self.rivet_target = _row_side(row, self.rivet_first)
            self.rivet_climbing = False
            self.rivet_gap_direction = None
            self.rivet_climb_side = "right" if self.rivet_target == "left" else "left"
        target = self._rivet_remaining_side(rows[row], self.rivet_target)
        if target is None:
            return "up"
        self.rivet_target = target
        p = _player(obs)
        px = _number(p.get("x", 45))
        py = _number(p.get("y", 176))
        target_y = RIVET_Y[row]
        if py > target_y + 11:
            ladder = self.ladder_x(obs, self.rivet_climb_side)
            if abs(px - ladder) > 3:
                self.rivet_climbing = False
                direction = "right" if px < ladder else "left"
                self.last_walk_direction = direction
                return direction
            self.rivet_climbing = True
            if py > target_y + 4:
                return "up"
            self.rivet_climbing = False
        if py < target_y - 12:
            return "down"
        target_x = RIVET_X[target]
        if target == "left":
            if px > target_x + 8:
                self.last_walk_direction = "left"
                return "left"
            if px < target_x - 8:
                self.last_walk_direction = "right"
                return "right"
            self.last_walk_direction = "right"
            return "right"
        if px < target_x - 8:
            self.last_walk_direction = "right"
            return "right"
        if px > target_x + 8:
            self.last_walk_direction = "left"
            return "left"
        self.last_walk_direction = "left"
        return "left"

    def update_rivet_byte(self, obs):
        current = _rows(obs)
        if self.rivet_last_rows is not None:
            for index, (old, new) in enumerate(zip(self.rivet_last_rows, current)):
                if old != new and new in (0, 6, 12, 18) and index == self.rivet_row:
                    # Continue across the cleared location in the direction of
                    # the current sweep; FIRE is supplied by crossing_action.
                    self.rivet_gap_direction = "right" if self.rivet_target == "left" else "left"
                    self.log("rivet_byte_change", row=index, old=old, new=new,
                             cross=self.rivet_gap_direction)
        self.rivet_last_rows = list(current)

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
            # The port accepts a single start edge after the death animation;
            # sparse retries avoid turning every transition into a jump.
            if self.transition_age >= 18 and self.transition_attempts < 3:
                if (self.transition_age - 18) % 28 == 0:
                    self.transition_attempts += 1
                    return "fire"
            return "noop"
        if stage == "ramp":
            return self.ramp_action(obs)
        if stage == "rivet":
            return self.rivet_action(obs)
        return "noop"

    def act_for_observation(self, obs):
        stage, lives, screens, stage_changed, life_changed, screen_changed = self.sync(obs)
        self.current_obs = obs
        self.collect_astra()
        self.collect_jev(obs)
        self.update_rivet_byte(obs)
        if stage_changed or life_changed or screen_changed:
            self.start_astra(obs, "stage/life/screen change")

        fallback = self.fallback(obs, stage, lives)
        hazard = _hazard_facts(obs)
        crossing = None
        if stage in ("ramp", "rivet"):
            crossing, hazard = self.crossing_action(obs)
        action = crossing if crossing is not None else fallback

        if stage in ("ramp", "rivet"):
            self.start_jev(obs, hazard)
        if self.tick % 10 == 0:
            p = _player(obs)
            self.log("control_snapshot", tick=self.tick, stage=stage,
                     phase=self.ramp_phase, x=_integer(p.get("x")),
                     y=_integer(p.get("y")), action=action,
                     climbing=self.climbing, settle=self.settle_ticks,
                     jump=self.jump_quanta, lock=self.hazard_lock,
                     hazard=hazard.get("urgency"))
        if stage == "transition" and lives <= 0:
            return "noop", True
        return action, False

    def run(self, game):
        self.log("controller_started", tick_s=getattr(game, "tick_s", None))
        try:
            obs = self.ctx.observe()
        except Exception:
            return
        if not isinstance(obs, dict):
            obs = {}
        self.current_obs = obs
        self.sync(obs)
        if _stage(obs) == "idle":
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
            action, stop = self.act_for_observation(obs)
            if stop or self.done(game):
                break
            if not self.post(game, action):
                break
        self.log("controller_finished", ticks=self.tick,
                 score=getattr(game, "score", None), action=self.last_action)


class Program:
    def __init__(self, ctx):
        self.ctx = ctx

    def run_episode(self, game) -> None:
        Controller(self.ctx).run(game)


def build(ctx):
    return Program(ctx)
