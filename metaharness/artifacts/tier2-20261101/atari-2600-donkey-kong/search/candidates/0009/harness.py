"""Feedback route controller for the Atari 2600 Donkey Kong cartridge.

The controller is deliberately code-first: the public ladder topology is a small
finite route, while the observation closes the loop around each walk, climb, jump,
and life transition.  Model advice, when available, is only collected in the
rivet stage; it cannot replace a fresh geometric action or a death/restart rule.
"""
from __future__ import annotations

import json
import math

from metaharness.models.jev import Choice, Noul


# These are the decoder's top-left coordinates.  The first parent demonstrated
# that this order reaches the second upper walkway; the second parent supplied
# the measured-progress discipline.  Keep both possible ladders at split rows.
RAMP_ROUTE = (
    (176, (107,)),
    (148, (47, 79)),
    (120, (107, 87)),
    (92, (47, 67)),
    (64, (107,)),
    (40, (31, 75)),
)
RIVET_Y = (176, 148, 120, 92)
RIVET_X = {"left": 48, "right": 108}


def _num(v, default=0.0):
    try:
        x = float(v)
        return x if math.isfinite(x) else default
    except (TypeError, ValueError):
        return default


def _int(v, default=0):
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return default


def _stage(obs):
    if isinstance(obs, dict) and isinstance(obs.get("stage"), str):
        return obs["stage"]
    return "unknown"


def _player(obs):
    p = obs.get("player", {}) if isinstance(obs, dict) else {}
    if not isinstance(p, dict):
        p = {}
    return p


def _rows(obs):
    values = obs.get("rivet_rows", []) if isinstance(obs, dict) else []
    if not isinstance(values, (list, tuple)):
        values = []
    out = []
    for v in list(values)[:4]:
        n = _int(v, 0)
        out.append(n if n in (0, 6, 12, 18) else 0)
    while len(out) < 4:
        out.append(0)
    return out


def _visible(o):
    return isinstance(o, dict) and o.get("visible", True) is not False


def _box(o):
    if not isinstance(o, dict):
        return 0.0, 0.0, 0.0, 0.0
    x = _num(o.get("x", o.get("left", 0)))
    y = _num(o.get("y", o.get("top", 0)))
    w = _num(o.get("w", o.get("width", 0)))
    h = _num(o.get("h", o.get("height", 0)))
    return x, y, w, h


def _hazard(obs):
    """Return one normalized same-walkway closing obstacle, if any."""
    p = _player(obs)
    px, py, pw, ph = _box(p)
    pc = px + pw / 2.0
    best = None
    hazards = obs.get("hazards", []) if isinstance(obs, dict) else []
    if not isinstance(hazards, list):
        hazards = []
    for h in hazards:
        if not _visible(h):
            continue
        hx, hy, hw, hh = _box(h)
        hc = hx + hw / 2.0
        # Obstacle boxes sit a few pixels below Mario on the same girder.
        # Obstacles enter a ramp with a different sprite y before the
        # collision row is visually level.  A wider vertical gate is an early
        # warning, not a claim that every visible slot shares Mario's girder.
        if abs((hy + hh / 2.0) - (py + ph / 2.0)) > 40:
            continue
        dx = _num(h.get("dx", 0))
        delta = hc - pc
        distance = abs(delta)
        closing = ((delta < 0 and dx > 0.2) or
                   (delta > 0 and dx < -0.2) or
                   (distance <= 15 and abs(dx) <= 0.2))
        if not closing:
            continue
        urgency = "immediate" if distance <= 38 else "soon" if distance <= 66 else "watch"
        item = {
            "side": "left" if delta < 0 else "right",
            "motion": "left" if dx < -0.2 else "right" if dx > 0.2 else "still",
            "distance": round(distance, 1),
            "vertical_delta": round((hy + hh / 2.0) - (py + ph / 2.0), 1),
            "hazard_y": round(hy, 1),
            "player_y": round(py, 1),
            "distance_bucket": "overlapping" if distance <= 18 else "near" if distance <= 38 else "approaching" if distance <= 66 else "far",
            "closing": True,
            "urgency": urgency,
        }
        rank = (0 if urgency == "immediate" else 1 if urgency == "soon" else 2, distance)
        if best is None or rank < best[0]:
            best = (rank, item)
    if best is None:
        return {"present": False, "side": "none", "motion": "none",
                "distance_bucket": "none", "closing": False, "urgency": "clear"}
    result = best[1]
    result["present"] = True
    return result


def _remaining(mask, preferred):
    """The cartridge's 6/12 values mean left/right cleared, respectively."""
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


def _action_for(direction, fire=False):
    if fire:
        return "leftfire" if direction == "left" else "rightfire"
    return direction


class Controller:
    def __init__(self, ctx):
        self.ctx = ctx
        self.tick = 0
        self.last_action = "noop"
        self.last_stage = None
        self.last_lives = None
        self.last_screen = 0

        self.ramp_level = 0
        self.ramp_choice = 0
        self.climbing = False
        self.climb_ticks = 0
        self.climb_last_y = None
        self.climb_progress = False
        self.stuck_ticks = 0
        self.waypoint_last_y = None
        self.waypoint_still = 0

        self.jump_left = 0
        self.jump_cooldown = 0
        self.escape_left = 0
        self.jump_dir = "right"
        self.initial_jump_seen = False

        self.transition_age = 0
        self.restart_attempts = 0
        self.idle_attempts = 0

        self.rivet_preference = "left"
        self.rivet_row = 0
        self.previous_rows = None
        self.gap_jump = None

        # A typed Jev review is useful only after the deterministic controller
        # has actually reached the less-calibrated rivet screen.  Astra is not
        # allowed to block a real-time actor and no call is needed to run the
        # verified ramp route.
        self.jev_future = None
        self.jev_requested = -10000
        self.jev_hint = None

    def log(self, event, **fields):
        try:
            self.ctx.log(event, **fields)
        except Exception:
            pass

    def done(self, game):
        try:
            if self.ctx.done or game.done:
                return True
        except Exception:
            return True
        return False

    def post(self, game, action):
        allowed = {
            "noop", "fire", "up", "right", "left", "down", "upright", "upleft",
            "downright", "downleft", "upfire", "rightfire", "leftfire", "downfire",
            "uprightfire", "upleftfire", "downrightfire", "downleftfire",
        }
        if not isinstance(action, str) or action not in allowed:
            action = "noop"
        try:
            self.ctx.act(action)
            self.last_action = action
            return True
        except Exception as exc:
            self.log("action_error", action=action, error=type(exc).__name__)
            return False

    def reset_route(self, stage):
        self.climbing = False
        self.climb_ticks = 0
        self.climb_last_y = None
        self.climb_progress = False
        self.stuck_ticks = 0
        self.waypoint_last_y = None
        self.waypoint_still = 0
        self.jump_left = 0
        self.jump_cooldown = 0
        self.escape_left = 0
        self.gap_jump = None
        if stage == "ramp":
            self.ramp_level = 0
            self.ramp_choice = 0
        elif stage == "rivet":
            self.rivet_row = 0
            self.previous_rows = None
            self.rivet_preference = "left"

    def sync(self, obs):
        stage = _stage(obs)
        lives = _int(obs.get("lives_remaining", 0), 0) if isinstance(obs, dict) else 0
        screen = _int(obs.get("screens_completed", 0), 0) if isinstance(obs, dict) else 0
        stage_changed = stage != self.last_stage
        life_changed = self.last_lives is not None and lives != self.last_lives
        if stage_changed or life_changed:
            self.log("state_reset", old_stage=self.last_stage, stage=stage,
                     old_lives=self.last_lives, lives=lives)
            if stage in ("ramp", "rivet"):
                self.reset_route(stage)
        if stage == "transition":
            self.transition_age += 1
        else:
            self.transition_age = 0
        if stage_changed:
            self.restart_attempts = 0
            self.idle_attempts = 0
        self.last_stage, self.last_lives, self.last_screen = stage, lives, screen
        return stage, lives, screen, stage_changed, life_changed

    def jump_active_from_ram(self, obs):
        raw = obs.get("raw_ram", []) if isinstance(obs, dict) else []
        # The adaptor's 128-byte view wraps the documented $85 state at index 5.
        return isinstance(raw, list) and len(raw) > 5 and _int(raw[5], 0) > 0

    def start_jump(self, obs, direction, reason):
        self.jump_dir = direction if direction in ("left", "right") else "right"
        self.jump_left = 8
        # Do not retrigger FIRE on the same sprite as it crosses the lane.
        # The jump itself lasts about eight four-frame quanta; the cooldown
        # covers the landing/edge of that primitive.
        self.jump_cooldown = 22
        self.escape_left = 0
        hz = _hazard(obs)
        self.log("jump", direction=self.jump_dir, reason=reason,
                 stage=_stage(obs), tick=self.tick, hazard=hz)
        return _action_for(self.jump_dir, True)

    def hazard_action(self, obs):
        if self.jump_left > 0:
            self.jump_left -= 1
            return _action_for(self.jump_dir, True)
        if self.escape_left > 0:
            self.escape_left -= 1
            return _action_for(self.jump_dir, False)
        if self.jump_cooldown > 0:
            self.jump_cooldown -= 1
            return None
        hz = _hazard(obs)
        if not hz.get("present"):
            return None
        # Start early and latch the direction of the approaching sprite.  The
        # cartridge's barrel crossing is most reliable when Mario and the
        # obstacle meet while Mario is rising; the cooldown prevents repeated
        # FIRE presses on the same crossing.
        p = _player(obs)
        px = _num(p.get("x"), 45)
        direction = "left" if hz.get("side") == "left" else "right"
        if direction == "left" and px < 28:
            direction = "right"
        if direction == "right" and px > 116:
            direction = "left"
        return self.start_jump(obs, direction, "hazard")

    def ramp_action(self, obs):
        p = _player(obs)
        x, y = _num(p.get("x"), 45), _num(p.get("y"), 176)
        level = min(max(self.ramp_level, 0), len(RAMP_ROUTE) - 1)
        row_y, choices = RAMP_ROUTE[level]
        self.ramp_choice = min(self.ramp_choice, len(choices) - 1)
        target = choices[self.ramp_choice]

        if self.climbing:
            self.climb_ticks += 1
            if self.climb_last_y is not None and abs(y - self.climb_last_y) >= 2:
                self.climb_progress = True
                self.stuck_ticks = 0
            else:
                self.stuck_ticks += 1
            self.climb_last_y = y
            if level + 1 < len(RAMP_ROUTE):
                next_y = RAMP_ROUTE[level + 1][0]
                # Advance only after the first climb has settled near the next
                # walkway; this rejects 0008's premature y=156 phase change.
                settled = y <= next_y + 6 and self.climb_ticks >= 4
                if settled and self.climb_progress:
                    self.ramp_level += 1
                    self.ramp_choice = 0
                    self.climbing = False
                    self.climb_ticks = 0
                    self.climb_last_y = None
                    self.climb_progress = False
                    self.stuck_ticks = 0
                    self.log("ramp_advance", level=self.ramp_level, x=round(x), y=round(y))
                    return self.ramp_action(obs)
            if self.stuck_ticks >= 18 and len(choices) > 1:
                self.ramp_choice = (self.ramp_choice + 1) % len(choices)
                self.climbing = False
                self.climb_ticks = 0
                self.climb_last_y = None
                self.climb_progress = False
                self.stuck_ticks = 0
                self.log("ladder_fallback", level=level, choice=self.ramp_choice)
                return self.ramp_action(obs)
            return "up"

        if abs(x - target) > 2:
            self.stuck_ticks = 0
            return "right" if x < target else "left"
        if y > row_y + 8:
            return "up"
        self.climbing = True
        self.climb_ticks = 0
        self.climb_last_y = y
        self.climb_progress = False
        self.stuck_ticks = 0
        return "up"

    def nearest_ladder(self, obs, y, side):
        ladders = obs.get("ladders", []) if isinstance(obs, dict) else []
        if not isinstance(ladders, list):
            ladders = []
        preferred = 108 if side == "right" else 48
        choices = []
        for ladder in ladders:
            if not _visible(ladder):
                continue
            lx, ly, _, _ = _box(ladder)
            if abs(ly - y) <= 20 or abs(ly - (y - 8)) <= 20:
                choices.append((abs(lx - preferred), lx))
        choices.sort()
        return choices[0][1] if choices else preferred

    def rivet_action(self, obs):
        rows = _rows(obs)
        if self.previous_rows is not None:
            for i, (old, new) in enumerate(zip(self.previous_rows, rows)):
                if new != old and new in (0, 6, 12, 18) and i == self.rivet_row:
                    # A cleared rivet is now a hole; cross it with a committed
                    # jump before any later route action.
                    self.gap_jump = "right" if new == 6 else "left" if new == 12 else None
                    if self.gap_jump:
                        self.log("rivet_cleared", row=i, old=old, new=new)
        self.previous_rows = list(rows)

        row = next((i for i, value in enumerate(rows) if value < 18), 4)
        if row >= 4:
            return "up"
        if row != self.rivet_row:
            self.rivet_row = row
            self.climbing = False
            self.stuck_ticks = 0
            self.rivet_preference = "left" if row % 2 == 0 else "right"
        if self.gap_jump:
            direction = self.gap_jump
            self.gap_jump = None
            return self.start_jump(obs, direction, "rivet_gap")

        p = _player(obs)
        x, y = _num(p.get("x"), 45), _num(p.get("y"), 176)
        mask = rows[row]
        side = _remaining(mask, self.rivet_preference)
        if side is None:
            # Row is complete: use the opposite side as the next ladder side.
            target = self.nearest_ladder(obs, y, "right" if row % 2 == 0 else "left")
            if abs(x - target) > 3:
                return "right" if x < target else "left"
            return "up"
        target_x = RIVET_X[side]
        # Get onto the row first.  The finishing side of the preceding sweep is
        # the useful side for the next ladder, but the raw row mask decides the
        # actual rivet target.
        if y > RIVET_Y[row] + 8:
            ladder_side = "right" if row % 2 == 0 else "left"
            lx = self.nearest_ladder(obs, y, ladder_side)
            if abs(x - lx) > 3:
                return "right" if x < lx else "left"
            return "up"
        if side == "left":
            if x > target_x + 9:
                return "left"
            return "right"
        if x < target_x - 9:
            return "right"
        return "left"

    def model_state(self, obs):
        p = _player(obs)
        hz = _hazard(obs)
        return {
            "game": "Atari 2600 Donkey Kong NTSC",
            "stage": _stage(obs),
            "player_band": "left" if _num(p.get("x"), 0) < 52 else "right" if _num(p.get("x"), 0) > 100 else "middle",
            "route": {"ramp_level": self.ramp_level, "rivet_row": self.rivet_row,
                      "rivet_rows": _rows(obs), "jump_committed": self.jump_left > 0},
            "hazard": {k: hz.get(k) for k in ("present", "side", "distance_bucket", "closing", "urgency")},
            "last_action": self.last_action,
            "lives_remaining": _int(obs.get("lives_remaining", 0)),
            "screens_completed": _int(obs.get("screens_completed", 0)),
        }

    def ask_jev(self, obs):
        if self.jev_future is not None or _stage(obs) != "rivet":
            return
        if self.tick - self.jev_requested < 12:
            return
        try:
            if self.ctx.budget.remaining().get("system1_calls", 0) <= 0:
                return
        except Exception:
            return
        state = self.model_state(obs)
        questions = {
            "side": Choice(
                "For the current rivet row, select the tactical side already supported by the raw row mask. Do not count or calculate; choose only the side whose rivet remains and the serpentine route can reach.",
                {"left": "Approach or sweep the left remaining rivet.",
                 "right": "Approach or sweep the right remaining rivet.",
                 "wait": "The row/phase is unsettled; do not change the deterministic route."},
            ),
            "danger": Noul(
                "Does the normalized state show an immediate closing same-row hazard that should take priority over rivet collection?",
                {"true": "The hazard is immediate or soon and a jump guard should win.",
                 "false": "There is no immediate same-row closing hazard."},
            ),
        }
        self.jev_requested = self.tick
        try:
            self.jev_future = self.ctx.spawn(
                lambda: self.ctx.decide(state, questions, timeout_s=0.75)
            )
        except Exception as exc:
            self.jev_future = None
            self.log("jev_spawn_error", error=type(exc).__name__)

    def collect_jev(self):
        f = self.jev_future
        if f is None or not f.done():
            return
        self.jev_future = None
        try:
            result = f.result()
            answer = result.get("side") if isinstance(result, dict) else None
            value = getattr(answer, "value", None)
            if value in ("left", "right"):
                self.jev_hint = value
                self.log("jev_hint", side=value)
        except Exception as exc:
            # Timeout, garbage stand-in output, and a closed episode are all
            # ordinary advisory failures; the code route remains executable.
            self.log("jev_failed", error=type(exc).__name__)

    def choose(self, obs):
        stage, lives, _, stage_changed, life_changed = self.sync(obs)
        self.collect_jev()
        if stage == "idle":
            if self.idle_attempts < 2:
                self.idle_attempts += 1
                return "fire"
            return "noop"
        if stage == "game_over" or bool(obs.get("terminated", False)):
            return "noop"
        if stage == "transition":
            if lives <= 0:
                return "noop"
            # One FIRE edge after the death/bonus animation, with sparse retries.
            if self.transition_age >= 18 and self.restart_attempts < 4 and (self.transition_age - 18) % 30 == 0:
                self.restart_attempts += 1
                return "fire"
            return "noop"
        if stage not in ("ramp", "rivet"):
            return "noop"

        # A current jump owns the action for its full calibrated duration.
        emergency = self.hazard_action(obs)
        if emergency is not None:
            return emergency
        action = self.ramp_action(obs) if stage == "ramp" else self.rivet_action(obs)
        self.ask_jev(obs)
        # Jev is a low-authority side hint: use it only to select an already
        # legal rivet sweep direction, never to override a route or hazard.
        if stage == "rivet" and self.jev_hint in ("left", "right"):
            row = min(max(self.rivet_row, 0), 3)
            side = _remaining(_rows(obs)[row], self.jev_hint)
            if side == self.jev_hint:
                self.jev_hint = None
                target_x = RIVET_X[side]
                x = _num(_player(obs).get("x"), 45)
                if abs(x - target_x) > 9:
                    action = "left" if x > target_x else "right"
        if stage_changed or life_changed:
            self.log("stage_action", stage=stage, action=action, lives=lives)
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
            if (_stage(obs) == "transition" and
                    _int(obs.get("lives_remaining", 0), 0) <= 0):
                # On this adaptor the last death can remain in a non-terminal
                # transition shell for a long time.  No Mario can be restarted;
                # returning preserves the reached score instead of burning the
                # wall deadline on noop ticks.
                self.log("last_life_transition", tick=self.tick)
                break
            action = self.choose(obs)
            if self.tick % 20 == 0:
                p = _player(obs)
                raw = obs.get("raw_ram", []) if isinstance(obs, dict) else []
                self.log("snapshot", tick=self.tick, stage=_stage(obs), x=round(_num(p.get("x"))),
                         y=round(_num(p.get("y"))), dy=p.get("dy"), level=self.ramp_level,
                         action=action, jump=self.jump_left,
                         raw_jump=_int(raw[5], 0) if isinstance(raw, list) and len(raw) > 5 else 0,
                         lives=_int(obs.get("lives_remaining", 0)), score=obs.get("score_total"))
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
