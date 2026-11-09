"""Risk-aware, progress-seeking controller for Atari 2600 Donkey Kong.

Unlike the earlier score-lane controllers, this policy treats route progress as
an observed geometric fact and jumps away from a closing hazard by default.
Slow and fast models may advise at boundaries, but never own the real-time loop.
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
RAMP_Y = (176, 148, 120, 92, 64, 40)
# Useful Mario left-edge coordinates.  First entry is the empirically supported
# nominal zig-zag; alternatives are used only after a measured connector stall.
RAMP_OPTIONS = ((106,), (46, 78), (106, 86), (46, 66), (106,), (30, 74))
RIVET_Y = (176, 148, 120, 92)
RIVET_X = {"left": 46.0, "right": 106.0}


def _num(value, default=0.0):
    try:
        out = float(value)
        return out if math.isfinite(out) else default
    except (TypeError, ValueError, OverflowError):
        return default


def _int(value, default=0):
    try:
        return int(float(value))
    except (TypeError, ValueError, OverflowError):
        return default


def _stage(obs):
    return obs.get("stage", "unknown") if isinstance(obs, dict) else "unknown"


def _player(obs):
    value = obs.get("player", {}) if isinstance(obs, dict) else {}
    return value if isinstance(value, dict) else {}


def _xy(obs):
    p = _player(obs)
    return _num(p.get("x"), 45.0), _num(p.get("y"), 176.0)


def _raw(obs, address):
    values = obs.get("raw_ram", []) if isinstance(obs, dict) else []
    index = int(address) & 0x7f
    if isinstance(values, (list, tuple)) and index < len(values):
        return _int(values[index], 0)
    return 0


def _rows(obs):
    source = obs.get("rivet_rows", []) if isinstance(obs, dict) else []
    if not isinstance(source, (list, tuple)):
        source = []
    out = []
    for value in list(source)[:4]:
        n = _int(value, 0)
        out.append(n if n in (0, 6, 12, 18) else 0)
    return out + [0] * (4 - len(out))


def _visible(item):
    return isinstance(item, dict) and item.get("visible", True) is not False


def _prob(answer, key):
    probs = getattr(answer, "probabilities", None)
    if not isinstance(probs, dict):
        return None
    value = _num(probs.get(key), -1.0)
    return value if 0.0 <= value <= 1.0 else None


def _move(direction, fire=False):
    if direction not in ("left", "right"):
        return "fire" if fire else "noop"
    return direction + ("fire" if fire else "")


def _json_object(text):
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


def _hazard(obs):
    """Nearest visible, closing obstacle on Mario's current walkway."""
    px, py = _xy(obs)
    values = obs.get("hazards", []) if isinstance(obs, dict) else []
    if not isinstance(values, list):
        values = []
    best = None
    for item in values:
        if not _visible(item):
            continue
        hx, hy = _num(item.get("x")), _num(item.get("y"))
        if abs(hy - py) > 19.0:
            continue
        dx = _num(item.get("dx"))
        delta = hx - px
        distance = abs(delta)
        closing = (delta * dx < -0.2) or (distance <= 15.0 and abs(dx) <= 0.2)
        if not closing or distance > 66.0:
            continue
        urgency = ("overlap" if distance <= 14.0 else
                   "commit" if distance <= 25.0 else
                   "soon" if distance <= 38.0 else "watch")
        fact = {
            "present": True,
            "slot": _int(item.get("slot"), -1),
            "side": "left" if delta < 0 else "right",
            "motion": "left" if dx < -0.2 else "right" if dx > 0.2 else "still",
            "closing": True,
            "distance": round(distance, 1),
            "distance_bucket": ("overlap" if distance <= 14 else
                                "near" if distance <= 28 else
                                "approaching" if distance <= 44 else "far"),
            "urgency": urgency,
        }
        rank = (distance, 0 if urgency == "overlap" else 1)
        if best is None or rank < best[0]:
            best = (rank, fact)
    if best:
        return best[1]
    return {"present": False, "slot": -1, "side": "none", "motion": "none",
            "closing": False, "distance": 999.0, "distance_bucket": "none",
            "urgency": "clear"}


class Controller:
    def __init__(self, ctx, game):
        self.ctx, self.game = ctx, game
        self.tick = 0
        self.last_action = "noop"
        self.last_stage = None
        self.last_lives = None
        self.last_screens = 0
        self.transition_age = 0
        self.restart_edges = 0
        self.idle_edges = 0

        # Route is a geometric ledger, not an action count.  A rung is credited
        # only after the observed player y enters the next walkway band.
        self.phase = 0
        self.branch = 0
        self.climbing = False
        self.climb_origin = None
        self.climb_best = None
        self.climb_stall = 0
        self.walk_stall = 0
        self.previous_x = None
        self.attempted_branch = set()
        self.alignment_nudge = 0
        self.phase_gain = 0
        self.height_ticks = 0
        self.jump_route_pending = False
        self.jump_route_age = 0

        # Jump transaction uses the documented wrapped RAM byte at $85 (index 5).
        self.jump_direction = None
        self.jump_reason = None
        self.jump_seen = False
        self.jump_guard = 7
        self.jump_tail = 0
        self.jump_lock = 0
        self.post_jump_up = 0
        self.last_threat = None
        self.jump_count = 0

        # Fast asynchronous tactical judgment and slow, once-per-episode macro
        # review.  Futures are polled, never awaited by the actor.
        self.jev_future = None
        self.jev_stamp = None
        self.jev_started = -10000
        self.jev_answer = None
        self.jev_answer_stamp = None
        self.jev_answer_tick = -10000
        self.astra_future = None
        self.astra_started = False
        self.macro_pending = None
        self.macro_active = {"route": "nominal", "hazard": "retreat",
                             "first_rivet": "left"}
        self.last_death_phase = 0

        self.rivet_preference = "left"
        self.previous_rows = None
        self.gap_direction = None

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
        self.ctx.act(action)
        self.last_action = action

    def _budget(self, key):
        try:
            if bool(self.ctx.done):
                return False
            return _int(self.ctx.budget.remaining().get(key), 0) > 0
        except Exception:
            return False

    def _clear_jump(self):
        self.jump_direction = None
        self.jump_reason = None
        self.jump_seen = False
        self.jump_guard = 7
        self.jump_tail = 0
        self.last_threat = None

    def _reset_route(self, use_pending=False):
        self.phase = 0
        self.branch = 0
        self.climbing = False
        self.climb_origin = self.climb_best = None
        self.climb_stall = self.walk_stall = 0
        self.previous_x = None
        self.attempted_branch.clear()
        self.alignment_nudge = 0
        self.phase_gain = 0
        self.height_ticks = 0
        self.jump_route_pending = False
        self.jump_route_age = 0
        self._clear_jump()
        self.jump_lock = self.post_jump_up = 0
        self.rivet_preference = self.macro_active.get("first_rivet", "left")
        self.previous_rows = None
        self.gap_direction = None
        if use_pending and isinstance(self.macro_pending, dict):
            self.macro_active.update(self.macro_pending)
            self.rivet_preference = self.macro_active.get("first_rivet", "left")
            self.log("macro_applied", macro=dict(self.macro_active))
            self.macro_pending = None
        self.log("route_reset", stage=self.last_stage, macro=dict(self.macro_active))

    def _sync(self, obs):
        stage = _stage(obs)
        lives = _int(obs.get("lives_remaining"), 0) if isinstance(obs, dict) else 0
        screens = _int(obs.get("screens_completed"), 0) if isinstance(obs, dict) else 0
        changed = stage != self.last_stage
        life_changed = self.last_lives is not None and lives != self.last_lives
        screen_changed = screens != self.last_screens
        life_lost = (self.last_lives is not None and lives < self.last_lives)
        if changed or life_changed or screen_changed:
            self.log("boundary", old_stage=self.last_stage, stage=stage,
                     old_lives=self.last_lives, lives=lives, screens=screens)
        if life_lost:
            self.last_death_phase = self.phase
            self._start_astra(obs)
        if stage == "transition":
            self.transition_age = self.transition_age + 1 if self.last_stage == "transition" else 1
        else:
            self.transition_age = 0
        if changed:
            if stage == "idle":
                self.idle_edges = 0
            if stage != "transition":
                self.restart_edges = 0
        if stage in ("ramp", "rivet") and (changed or life_changed or screen_changed):
            if stage == "ramp":
                self._reset_route(use_pending=life_lost or self.last_stage == "transition")
            else:
                self._clear_jump()
                self.rivet_preference = self.macro_active.get("first_rivet", "left")
                self.previous_rows = None
                self.gap_direction = None
        self.last_stage, self.last_lives, self.last_screens = stage, lives, screens
        return stage, lives, changed, life_changed, screen_changed, life_lost

    def _lifecycle(self, obs, stage, lives):
        if stage == "idle":
            if self.idle_edges < 1:
                self.idle_edges += 1
                return "fire"
            return "noop"
        if stage == "game_over" or bool(obs.get("terminated", False)) or bool(obs.get("truncated", False)):
            return "noop"
        if stage == "transition":
            # Spare-Mario count is not a terminal signal; pulse FIRE after a
            # release interval, including the last active Mario's transition.
            points = (18, 46, 74, 102, 130, 158)
            if self.transition_age in points and self.restart_edges < len(points):
                self.restart_edges += 1
                self.log("restart_edge", age=self.transition_age, edge=self.restart_edges,
                         lives=lives)
                return "fire"
            return "noop"
        if stage not in ("ramp", "rivet"):
            return "noop"
        return None

    # ---------- route and measured progress ----------
    def _target(self):
        row = max(0, min(self.phase, len(RAMP_OPTIONS) - 1))
        options = RAMP_OPTIONS[row]
        self.branch = max(0, min(self.branch, len(options) - 1))
        if self.macro_active.get("route") == "alternate" and row == 2 and len(options) > 1:
            if (row, 0) in self.attempted_branch:
                return float(options[1])
        return float(options[self.branch] + self.alignment_nudge)

    def _advance_from_y(self, obs, x, y):
        """Confirm a new row from a climb or a settled post-jump landing."""
        if self.phase + 1 >= len(RAMP_Y):
            return False
        threshold = RAMP_Y[self.phase + 1] + 6.0
        in_upper_band = y <= threshold and _raw(obs, 0x85) == 0
        climb_evidence = (
            self.climbing and self.climb_origin is not None and
            self.climb_origin - y >= 8.0
        )
        jump_evidence = self.jump_route_pending and self.jump_route_age <= 28
        if in_upper_band and (climb_evidence or jump_evidence):
            self.height_ticks += 1
        else:
            self.height_ticks = 0
        # Two grounded observations prevent the jump apex alone from being
        # mistaken for a reached walkway, while still crediting a safe landing
        # that clears the ramp connector without a conventional ladder climb.
        if self.height_ticks < 2:
            return False
        old = self.phase
        self.phase += 1
        self.branch = 0
        self.climbing = False
        self.climb_origin = self.climb_best = None
        self.climb_stall = self.walk_stall = 0
        self.previous_x = None
        self.alignment_nudge = 0
        self.phase_gain += 1
        self.height_ticks = 0
        self.jump_route_pending = False
        self.log("ramp_advance", old_phase=old, phase=self.phase,
                 x=round(x, 1), y=round(y, 1),
                 evidence="two grounded upper-band observations")
        return True

    def _retry_connector(self, x, y):
        options = RAMP_OPTIONS[self.phase]
        if len(options) > 1 and (self.phase, self.branch) not in self.attempted_branch:
            self.attempted_branch.add((self.phase, self.branch))
            old = self.branch
            self.branch = 1 if old == 0 else 0
            self.climbing = False
            self.climb_origin = self.climb_best = None
            self.climb_stall = self.walk_stall = 0
            self.previous_x = None
            self.log("connector_retry", phase=self.phase, old=old,
                     branch=self.branch, from_x=round(x), y=round(y))
            return
        self.alignment_nudge = -1 if self.alignment_nudge == 0 else 0
        self.climbing = False
        self.climb_origin = self.climb_best = None
        self.climb_stall = self.walk_stall = 0
        self.log("alignment_retry", phase=self.phase, nudge=self.alignment_nudge,
                 x=round(x), y=round(y))

    def _ramp_action(self, obs):
        x, y = _xy(obs)
        self._advance_from_y(obs, x, y)
        if self.post_jump_up > 0:
            self.post_jump_up -= 1
            return "up"

        target = self._target()
        if self.climbing:
            if self.climb_best is None or y < self.climb_best - 1.0:
                self.climb_best = y
                self.climb_stall = 0
            else:
                self.climb_stall += 1
            # A displacement from the rung during a jump invalidates the climb
            # lease; walking reacquires the same geometric target.
            if abs(x - target) > 5.0:
                self.climbing = False
                self.climb_origin = self.climb_best = None
                return "right" if x < target else "left"
            if self.phase >= len(RAMP_Y) - 1:
                return "up"
            if self.climb_stall >= 24:
                self._retry_connector(x, y)
                return "right" if x < self._target() else "left"
            return "up"

        if self.previous_x is not None and self.last_action in ("left", "right"):
            self.walk_stall = self.walk_stall + 1 if abs(x - self.previous_x) < 0.6 else 0
        else:
            self.walk_stall = 0
        self.previous_x = x
        target = self._target()
        if abs(x - target) > 1.5:
            if self.walk_stall >= 12:
                self._retry_connector(x, y)
                target = self._target()
            return "right" if x < target else "left"

        self.climbing = True
        self.climb_origin = y
        self.climb_best = y
        self.climb_stall = 0
        self.log("climb_begin", phase=self.phase, branch=self.branch,
                 target=round(target, 1), x=round(x, 1), y=round(y, 1))
        return "up"

    # ---------- hazard escape and jump transaction ----------
    def _retreat_direction(self, hazard):
        side = hazard.get("side")
        away = "right" if side == "left" else "left"
        if self.macro_active.get("hazard") == "cross":
            motion = hazard.get("motion")
            away = "right" if motion == "left" else "left" if motion == "right" else away
        x, _ = _xy(self._last_obs)
        # Preserve room to land: do not steer through the screen edge.
        if away == "left" and x < 34:
            away = "right"
        elif away == "right" and x > 106:
            away = "left"
        return away

    def _begin_jump(self, direction, reason, hazard=None):
        if direction not in ("left", "right"):
            direction = "right"
        self.jump_direction = direction
        self.jump_reason = reason
        self.jump_seen = False
        self.jump_guard = 7
        self.jump_tail = 0
        self.jump_lock = 16
        if _stage(self._last_obs) == "ramp":
            self.jump_route_pending = True
            self.jump_route_age = 0
        self.climbing = False
        self.climb_origin = self.climb_best = None
        self.climb_stall = 0
        self.jump_count += 1
        self.last_threat = dict(hazard or {})
        self.log("jump_begin", tick=self.tick, direction=direction, reason=reason,
                 phase=self.phase,
                 hazard={k: (hazard or {}).get(k) for k in
                         ("slot", "side", "motion", "distance_bucket", "urgency")})
        return _move(direction, True)

    def _jump_action(self, obs):
        if self.jump_direction is None:
            return None
        direction = self.jump_direction
        if _raw(obs, 0x85) > 0:
            self.jump_seen = True
            self.jump_guard = 7
            return _move(direction, True)
        if self.jump_seen:
            if self.jump_tail < 4:
                self.jump_tail += 1
                return _move(direction, False)
            self.log("jump_end", tick=self.tick, direction=direction,
                     reason=self.jump_reason, phase=self.phase)
            self._clear_jump()
            self.post_jump_up = 4
            return "up"
        if self.jump_guard > 0:
            self.jump_guard -= 1
            return _move(direction, True)
        self.log("jump_unconfirmed", tick=self.tick, phase=self.phase)
        self._clear_jump()
        self.jump_lock = 8
        return None

    def _hazard_action(self, obs, stage):
        if stage not in ("ramp", "rivet") or self.jump_direction is not None:
            return None
        if self.jump_lock > 0:
            self.jump_lock -= 1
            return None
        hazard = _hazard(obs)
        if not hazard.get("present"):
            return None
        self._request_jev(obs, hazard)
        x, y = _xy(obs)
        # Give a committed, visibly progressing ladder climb priority unless
        # the collision box is already overlapping the threat.
        if self.climbing and self.climb_origin is not None and hazard.get("urgency") != "overlap":
            if self.climb_origin - y >= 4.0:
                return None
        # A matching fresh Jev vote can start the escape in the wider warning
        # band; hard code still starts a jump at the deterministic contact guard.
        vote = self._usable_jev_vote(obs, hazard)
        if vote in ("jump_left", "jump_right") and hazard["distance"] <= 54.0:
            direction = "left" if vote == "jump_left" else "right"
            return self._begin_jump(direction, "jev_early_escape", hazard)
        if hazard["distance"] <= 33.0 or hazard.get("urgency") == "overlap":
            return self._begin_jump(self._retreat_direction(hazard), "hard_escape", hazard)
        return None

    # ---------- System-1: duplicated, reordered tactical choice ----------
    def _jev_signature(self, obs, hazard):
        # Urgency changes as the same object approaches. Keep the vote valid
        # across that expected change, but bind it to identity, row and motion.
        return (_stage(obs), self.phase, hazard.get("slot"), hazard.get("side"),
                hazard.get("motion"))

    def _request_jev(self, obs, hazard):
        if self.jev_future is not None or not self._budget("system1_calls"):
            return
        if getattr(self.ctx, "system1", True) is None:
            return
        if hazard.get("urgency") not in ("watch", "soon", "commit", "overlap"):
            return
        if self.tick - self.jev_started < 12:
            return
        x, y = _xy(obs)
        fact = {k: hazard.get(k) for k in
                ("present", "slot", "side", "motion", "closing", "distance_bucket", "urgency")}
        state = {
            "game": "Atari 2600 Donkey Kong, two-screen NTSC cartridge",
            "event": "same_walkway_hazard",
            "stage": _stage(obs),
            "player": {"horizontal_band": "left" if x < 50 else "right" if x > 100 else "middle",
                       "vertical_band": "bottom" if y > 160 else "lower" if y > 130 else "upper",
                       "climb_committed": bool(self.climbing),
                       "route_progress": self.phase},
            "hazard": fact,
            "escape_rule": "A directional jump is latched; moving away from the hazard's current side is the conservative landing choice.",
            "code_fallback": self._retreat_direction(hazard),
        }
        left_first = {
            "jump_left": "Start a leftward jump, only if the left side is the safer open landing side.",
            "jump_right": "Start a rightward jump, only if the right side is the safer open landing side.",
            "hold": "Do not start a jump; code fallback keeps climbing or approaching its route target.",
        }
        hold_first = {
            "hold": "Do not start a jump; code fallback keeps climbing or approaching its route target.",
            "jump_right": "Start a rightward jump, only if the right side is the safer open landing side.",
            "jump_left": "Start a leftward jump, only if the left side is the safer open landing side.",
        }
        questions = {
            "escape_a": Choice(
                "From the supplied `hazard` on Mario's current walkway, which immediate label best avoids a collision while preserving the route? Do not compute from raw coordinates. `hold` is appropriate only when climbing is already visibly progressing or the obstacle is not closing.",
                left_first),
            "escape_b": Choice(
                "Independently judge the same immediate choice using `hazard` and `player`. Prefer a safe open landing side; choose `hold` only if a jump is not warranted.",
                hold_first),
            "closing": Noul(
                "Does `hazard` explicitly describe a closing obstacle on Mario's current walkway?",
                {"true": "hazard.closing is true and the object is on the same walkway.",
                 "false": "The object is not a closing same-walkway threat."}),
        }
        stamp = self._jev_signature(obs, hazard)
        self.jev_stamp = stamp
        self.jev_started = self.tick
        try:
            self.jev_future = self.ctx.spawn(
                lambda: self.ctx.decide(state, questions, timeout_s=0.65)
            )
            self.log("jev_requested", phase=self.phase, hazard=fact)
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
            a, b = answers.get("escape_a"), answers.get("escape_b")
            va, vb = getattr(a, "value", None), getattr(b, "value", None)
            pa, pb = _prob(a, va), _prob(b, vb)
            if va == vb and va in ("jump_left", "jump_right") and pa is not None and pb is not None and min(pa, pb) >= 0.60:
                self.jev_answer = va
                self.jev_answer_stamp = self.jev_stamp
                self.jev_answer_tick = self.tick
                self.log("jev_escape_vote", value=va, p=min(pa, pb))
            else:
                self.jev_answer = None
        except Exception as exc:
            self.log("jev_failed", error=type(exc).__name__)

    def _usable_jev_vote(self, obs, hazard):
        if self.jev_answer not in ("jump_left", "jump_right"):
            return None
        if self.tick - self.jev_answer_tick > 5:
            self.jev_answer = None
            return None
        if self.jev_answer_stamp != self._jev_signature(obs, hazard):
            self.jev_answer = None
            return None
        result = self.jev_answer
        self.jev_answer = None
        return result

    # ---------- System-2: post-death macro review ----------
    def _start_astra(self, obs):
        if self.astra_started or self.astra_future is not None or not self._budget("system2_calls"):
            return
        self.astra_started = True
        hazard = _hazard(obs)
        x, y = _xy(obs)
        facts = {
            "game": "Atari 2600 Donkey Kong NTSC two-screen cartridge",
            "event": "life_lost_before_screen_completion",
            "last_confirmed_walkway": self.phase,
            "player_band": {"x": "left" if x < 50 else "right" if x > 100 else "middle",
                            "y": "bottom" if y > 160 else "lower" if y > 130 else "upper"},
            "hazard": {k: hazard.get(k) for k in
                       ("present", "side", "motion", "distance_bucket", "urgency")},
            "score": _num(obs.get("score_total"), 0.0),
            "screens_completed": _int(obs.get("screens_completed"), 0),
            "rivet_rows": _rows(obs),
            "current_macro": dict(self.macro_active),
            "lesson": "The actor resets route bookkeeping from observations; it must not farm repeated jumps in place.",
        }
        prompt = (
            "You are the slow macro planner for the Atari 2600 Donkey Kong cartridge. "
            "The controller owns exact tick timing, ladder coordinates, RAM jump state, "
            "hazard arithmetic, rivet bytes, and transitions. Do not give a joystick "
            "sequence. Based only on these public observations, return one compact JSON "
            "object with optional labels: route='nominal'|'alternate', "
            "hazard='retreat'|'cross', first_rivet='left'|'right'. 'alternate' means "
            "try the documented short x=86 connector only after the nominal x=106 "
            "connector stalls; it does not override observed progress. "
            "Invalid/missing labels will be ignored. FACTS=" +
            json.dumps(facts, separators=(",", ":"), allow_nan=False)
        )
        try:
            self.astra_future = self.ctx.spawn(
                lambda: self.ctx.think(prompt, timeout_s=90.0)
            )
            self.log("astra_requested", death_phase=self.last_death_phase, facts=facts)
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
            self.log("astra_unparsed")
            return
        policy = {}
        if parsed.get("route") in ("nominal", "alternate"):
            policy["route"] = parsed["route"]
        if parsed.get("hazard") in ("retreat", "cross"):
            policy["hazard"] = parsed["hazard"]
        if parsed.get("first_rivet") in ("left", "right"):
            policy["first_rivet"] = parsed["first_rivet"]
        if policy:
            self.macro_pending = policy
            self.log("astra_policy_pending", policy=policy)

    # ---------- rivet sweep from raw status bytes ----------
    @staticmethod
    def _remaining_side(mask, preference):
        if mask == 18:
            return None
        if mask == 6:
            return "right"
        if mask == 12:
            return "left"
        return preference

    def _rivet_action(self, obs):
        rows = _rows(obs)
        row = next((i for i, value in enumerate(rows) if value != 18), 4)
        if self.previous_rows is not None and row < 4:
            old = self.previous_rows[row]
            new = rows[row]
            if old != new:
                if old in (0, 12) and new == 6:
                    self.gap_direction = "right"
                elif old in (0, 6) and new == 12:
                    self.gap_direction = "left"
                self.log("rivet_byte_change", row=row, old=old, new=new,
                         gap=self.gap_direction)
        self.previous_rows = list(rows)
        if row >= 4:
            return "up"
        if self.gap_direction in ("left", "right"):
            direction = self.gap_direction
            self.gap_direction = None
            return self._begin_jump(direction, "cleared_rivet_gap")

        x, y = _xy(obs)
        mask = rows[row]
        preference = self.rivet_preference if row % 2 == 0 else (
            "right" if self.rivet_preference == "left" else "left")
        side = self._remaining_side(mask, preference)
        climb_side = "right" if row % 2 == 0 else "left"
        target_y = RIVET_Y[row]
        ladder_x = RIVET_X[climb_side]
        if side is None or y > target_y + 9.0:
            if abs(x - ladder_x) > 2.0:
                return "right" if x < ladder_x else "left"
            return "up"
        target_x = RIVET_X[side]
        if abs(x - target_x) > 8.0:
            return "right" if x < target_x else "left"
        return "right" if side == "left" else "left"

    # ---------- one observed state -> one posted action ----------
    def choose(self, obs):
        self._last_obs = obs if isinstance(obs, dict) else {}
        stage, lives, changed, life_changed, screen_changed, life_lost = self._sync(obs)
        self._collect_jev()
        self._collect_astra()
        lifecycle = self._lifecycle(obs, stage, lives)
        if lifecycle is not None:
            return lifecycle

        if stage == "transition" and life_lost:
            # Astra is already in flight from _sync; transition stays code-owned.
            pass
        if stage == "ramp":
            if self.jump_route_pending:
                self.jump_route_age += 1
                if self.jump_route_age > 28:
                    self.jump_route_pending = False
                    self.height_ticks = 0
            ongoing = self._jump_action(obs)
            if ongoing is not None:
                return ongoing
            emergency = self._hazard_action(obs, stage)
            if emergency is not None:
                return emergency
            action = self._ramp_action(obs)
        else:
            ongoing = self._jump_action(obs)
            if ongoing is not None:
                return ongoing
            emergency = self._hazard_action(obs, stage)
            if emergency is not None:
                return emergency
            action = self._rivet_action(obs)

        if changed or life_changed or screen_changed:
            self.log("decision", stage=stage, phase=self.phase, action=action,
                     lives=lives, score=_num(obs.get("score_total"), 0.0))
        if self.tick % 30 == 0:
            x, y = _xy(obs)
            self.log("snapshot", tick=self.tick, stage=stage, phase=self.phase,
                     branch=self.branch, x=round(x), y=round(y), action=action,
                     climb=self.climbing, jump=self.jump_direction,
                     hazard=_hazard(obs).get("urgency"), lives=lives,
                     screens=_int(obs.get("screens_completed"), 0),
                     rivets=_int(obs.get("rivets_removed"), 0),
                     score=_num(obs.get("score_total"), 0.0))
        return action

    def run(self):
        obs = self.ctx.observe()
        if not isinstance(obs, dict):
            obs = {}
        self.last_stage = _stage(obs)
        self.last_lives = _int(obs.get("lives_remaining"), 0)
        self.last_screens = _int(obs.get("screens_completed"), 0)
        self._last_obs = obs
        if self.last_stage == "idle":
            self.idle_edges = 1
            self.post("fire")
        else:
            self.post(self.choose(obs))

        while not self.done():
            # Never spin: each iteration is synchronized to the real-time actor.
            try:
                self.ctx.wait_for_tick()
            except Exception:
                if self.done():
                    break
                raise
            if self.done():
                break
            try:
                obs = self.ctx.observe()
            except Exception:
                if self.done():
                    break
                raise
            if not isinstance(obs, dict):
                obs = {}
            self.tick += 1
            action = self.choose(obs)
            if self.done():
                break
            self.post(action)
        self.log("finished", ticks=self.tick, score=getattr(self.game, "score", None),
                 phase=self.phase, jumps=self.jump_count,
                 screens=self.last_screens)


class Program:
    def __init__(self, ctx):
        self.ctx = ctx

    def run_episode(self, game) -> None:
        Controller(self.ctx, game).run()


def build(ctx):
    return Program(ctx)
