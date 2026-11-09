"""Observation-driven Donkey Kong controller.

The earlier controllers either delegated the joystick to a slow/ambiguous model or
selected the geometrically nearest ladder.  This controller treats the cartridge as
a small feedback machine: it walks to a candidate ladder, probes UP, and only
advances the route after the observed player height changes.  Models judge
interrupts and route ambiguity, but never own the real-time actor loop.
"""
from __future__ import annotations

import json
import math

from metaharness.models.jev import Choice, Noul
from metaharness.core.contract import BudgetExceeded, EpisodeOver, ModelUnavailable


# Coordinates are the public decoder's screen coordinates.  Each tuple is the
# set of possible ladders on that horizontal band, from the bottom upwards.
# The first member is the alternating far-side prior; the other member is a
# recovery choice when the cartridge does not accept the first ladder.
RAMP_CANDIDATES = (
    ((108, 176),),
    ((48, 148), (80, 148)),
    ((88, 120), (108, 120)),
    ((48, 92), (68, 92)),
    ((108, 64),),
    ((32, 40), (76, 40)),
)

# The four rivet rows use the lower four walkway heights.  The decoder's rivet
# bytes are authoritative; these coordinates are only movement waypoints.
RIVET_Y = (176, 148, 120, 92)
RIVET_X = {"left": 48, "right": 108}

HORIZONTAL_ACTION = {"left": "left", "right": "right"}
JUMP_ACTION = {"left": "leftfire", "right": "rightfire"}


def _finite(value, default=0.0):
    try:
        x = float(value)
        return x if math.isfinite(x) else default
    except (TypeError, ValueError):
        return default


def _integer(value, default=0):
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _visible(obj):
    return isinstance(obj, dict) and obj.get("visible", True) is not False


def _center(obj):
    if not isinstance(obj, dict):
        return (0.0, 0.0)
    x = _finite(obj.get("x", obj.get("left", 0)))
    y = _finite(obj.get("y", obj.get("top", 0)))
    w = _finite(obj.get("w", obj.get("width", 0)))
    h = _finite(obj.get("h", obj.get("height", 0)))
    return x + w / 2.0, y + h / 2.0


def _raw_byte(obs, address):
    raw = obs.get("raw_ram") if isinstance(obs, dict) else None
    if not isinstance(raw, list) or not raw:
        return None
    # The public map uses the mirrored $80-$FF RIOT addresses while the JSON
    # snapshot is 128 bytes wide.
    index = address & 0x7F
    if index >= len(raw):
        return None
    return _integer(raw[index], 0)


def _prob(answer, key):
    probs = getattr(answer, "probabilities", None)
    if not isinstance(probs, dict):
        return 0.0
    return _finite(probs.get(key, 0.0), 0.0)


def _parse_json_object(text):
    if not isinstance(text, str):
        return None
    text = text.strip()
    for start in range(len(text)):
        if text[start] != "{":
            continue
        end = text.rfind("}")
        if end <= start:
            break
        try:
            value = json.loads(text[start:end + 1])
        except (TypeError, ValueError):
            continue
        if isinstance(value, dict):
            return value
        break
    return None


class Controller:
    def __init__(self, ctx, game):
        self.ctx = ctx
        self.game = game
        self.last_action = "noop"
        self.last_stage = None
        self.last_lives = None
        self.last_screens = 0
        self.local_tick = 0

        # Route state is deliberately an empirical state machine, not a timer.
        self.ramp_level = 0
        self.ramp_choice = 0
        self.up_probes = 0
        self.up_start_y = None
        self.best_y = None
        self.last_player_y = None
        self.last_player_x = None
        self.last_jumping = False
        self.route_failures = 0

        self.transition_tick = None
        self.restart_sent = False
        self.start_sent = False

        # FIRE is a commitment.  Keep the directional jump through its hang
        # time and suppress another jump until a later safe window.
        self.jump_hold = 0
        self.jump_direction = None
        self.jump_cooldown = 0

        # Rivet bookkeeping is based on raw row bytes, with a side inference
        # from the action that caused a byte to change.
        self.rivet_level = 0
        self.rivet_side = None
        self.rivet_expected_side = None
        self.rivet_previous_rows = None
        self.rivet_removed_sides = {}

        self.plan = {"risk": "conservative", "ramp": "feedback", "rivet": "serpentine"}
        self.plan_version = 0
        self.astra_future = None
        self.astra_calls = 0
        self.jev_future = None
        self.jev_capture_tick = -1
        self.jev_capture_stage = None
        self.jev_answer = None
        self.jev_answer_tick = -1
        self.jev_answer_stage = None
        self.last_jev_start = -1000

    def log(self, event, **fields):
        try:
            self.ctx.log(event, **fields)
        except Exception:
            # Trace logging is diagnostic and must never break the game loop.
            pass

    def _tick(self, obs):
        value = _integer(obs.get("tick"), self.local_tick)
        self.local_tick = max(self.local_tick + 1, value)
        return value

    def _stage(self, obs):
        return str(obs.get("stage", "unknown")) if isinstance(obs, dict) else "unknown"

    def _player(self, obs):
        p = obs.get("player", {}) if isinstance(obs, dict) else {}
        x, y = _center(p)
        return {
            "x": x,
            "y": y,
            "dx": _finite(p.get("dx")) if isinstance(p, dict) else 0.0,
            "dy": _finite(p.get("dy")) if isinstance(p, dict) else 0.0,
            "w": _finite(p.get("w", p.get("width", 8))) if isinstance(p, dict) else 8.0,
            "h": _finite(p.get("h", p.get("height", 16))) if isinstance(p, dict) else 16.0,
            "visible": _visible(p),
        }

    def _ladders(self, obs):
        result = []
        raw = obs.get("ladders", []) if isinstance(obs, dict) else []
        if not isinstance(raw, list):
            raw = []
        for i, item in enumerate(raw):
            if not _visible(item):
                continue
            x, y = _center(item)
            result.append({
                "x": x,
                "y": y,
                "slot": _integer(item.get("slot", i)) if isinstance(item, dict) else i,
            })
        return result

    def _jumping(self, obs, player=None):
        state = _raw_byte(obs, 0x85)
        if state is not None:
            return state != 0
        p = player or self._player(obs)
        return abs(p["dy"]) > 1.0

    def _rows(self, obs):
        raw = obs.get("rivet_rows", []) if isinstance(obs, dict) else []
        if not isinstance(raw, list):
            raw = []
        values = []
        for item in raw[:4]:
            value = _integer(item, 0)
            values.append(value if value in (0, 6, 12, 18) else 0)
        while len(values) < 4:
            values.append(18)
        return values

    def _phase_reset(self, obs, stage, tick):
        lives = _integer(obs.get("lives_remaining"), 0)
        screens = _integer(obs.get("screens_completed"), 0)
        changed = stage != self.last_stage or (
            self.last_lives is not None and lives != self.last_lives
        ) or screens != self.last_screens
        if not changed:
            return

        self.jev_answer = None
        self.ramp_level = 0
        self.ramp_choice = 0
        self.up_probes = 0
        self.up_start_y = None
        self.best_y = None
        self.last_player_y = None
        self.last_player_x = None
        self.route_failures = 0
        self.jump_hold = 0
        self.jump_direction = None
        self.jump_cooldown = 0
        self.rivet_level = 0
        self.rivet_side = None
        self.rivet_expected_side = None
        self.rivet_previous_rows = None
        self.rivet_removed_sides = {}

        if stage == "transition":
            self.transition_tick = tick
            self.restart_sent = False
        elif stage == "idle":
            self.start_sent = False
        else:
            self.transition_tick = None
            self.restart_sent = False

        self.log("phase_reset", stage=stage, lives=lives, screens=screens)

    def _ladder_for(self, obs, desired):
        """Return an observed ladder close to (x,y), or the public prior."""
        dx, dy = desired
        ladders = self._ladders(obs)
        same = [item for item in ladders if abs(item["y"] - (dy + 8.5)) <= 12]
        if not same:
            same = [item for item in ladders if abs(item["y"] - dy) <= 12]
        if same:
            chosen = min(same, key=lambda item: abs(item["x"] - (dx + 2)))
            # Observation x is the box center; the Atari alignment variable is
            # much closer to Mario's center against the ladder's left edge.
            return {"x": chosen["x"] - 2.0, "y": chosen["y"],
                    "slot": chosen["slot"]}
        return {"x": dx, "y": dy + 8.5, "slot": -1}

    def _route_candidates(self):
        level = max(0, min(self.ramp_level, len(RAMP_CANDIDATES) - 1))
        return list(RAMP_CANDIDATES[level])

    def _current_level_from_y(self, obs, player, jumping):
        if jumping:
            return
        # Compare player and ladder centers.  The exact sprite height changes
        # during a jump, so centers are more stable than raw top-left values.
        ladders = self._ladders(obs)
        ys = []
        for _, y in sum((list(row) for row in RAMP_CANDIDATES), []):
            ys.append(y + 8.5)
        if ladders:
            observed_ys = sorted(set(round(item["y"], 1) for item in ladders), reverse=True)
            if observed_ys:
                ys = observed_ys
        nearest = min(range(len(ys)), key=lambda i: abs(player["y"] - ys[i]))
        distance = abs(player["y"] - ys[nearest])
        if distance <= 13:
            # A fall after death is a reset; upward movement is progress.
            if nearest > self.ramp_level and nearest <= len(RAMP_CANDIDATES) - 1:
                self.ramp_level = nearest
                self.ramp_choice = 0
                self.up_probes = 0
                self.up_start_y = None
                self.route_failures = 0
                self.log("climb_confirmed", level=nearest)
            elif self.last_stage == "ramp" and nearest < self.ramp_level and player["y"] > 150:
                self.ramp_level = nearest
                self.ramp_choice = 0
                self.up_probes = 0

    def _hazard(self, obs, player):
        hazards = obs.get("hazards", []) if isinstance(obs, dict) else []
        if not isinstance(hazards, list):
            hazards = []
        candidates = []
        for raw in hazards:
            if not _visible(raw):
                continue
            hx, hy = _center(raw)
            dx = _finite(raw.get("dx")) if isinstance(raw, dict) else 0.0
            dy = _finite(raw.get("dy")) if isinstance(raw, dict) else 0.0
            horizontal = hx - player["x"]
            distance = abs(horizontal)
            same_row = abs(hy - player["y"]) <= 24
            closing = same_row and (
                (horizontal < 0 and dx > 0.5)
                or (horizontal > 0 and dx < -0.5)
                or (distance <= 18 and abs(dx) <= 0.5)
            )
            # A descending object entering the current band is also a threat.
            entering = abs(horizontal) <= 52 and abs(hy - player["y"]) <= 32 and dy > 1
            if not (closing or entering):
                continue
            if distance <= 22 and (closing or entering):
                urgency = "immediate"
            elif distance <= 52:
                urgency = "soon"
            else:
                urgency = "watch"
            candidates.append((0 if urgency == "immediate" else 1 if urgency == "soon" else 2,
                               distance, hx, dx, urgency, dy))
        if not candidates:
            return {
                "present": False, "urgency": "none", "side": "none",
                "closing": False, "jump_direction": "none", "count": 0,
            }
        candidates.sort(key=lambda item: (item[0], item[1]))
        _, distance, hx, dx, urgency, dy = candidates[0]
        side = "left" if hx < player["x"] else "right"
        # Jump away from the object, except at a screen edge where the open
        # side is safer.  The visible 160-wide playfield has a practical
        # Mario range around 32..112.
        left_room = player["x"] - 32
        right_room = 112 - player["x"]
        if player["x"] < 42:
            jump_direction = "right"
        elif player["x"] > 102:
            jump_direction = "left"
        else:
            jump_direction = "right" if side == "left" else "left"
            if right_room < 18 and left_room > right_room:
                jump_direction = "left"
            if left_room < 18 and right_room > left_room:
                jump_direction = "right"
        return {
            "present": True, "urgency": urgency, "side": side,
            "closing": True, "jump_direction": jump_direction,
            "count": len(candidates),
        }

    def _start_jump(self, direction, reason):
        if direction not in ("left", "right"):
            direction = "right"
        self.jump_direction = direction
        self.jump_hold = max(self.jump_hold, 7)
        self.jump_cooldown = 11
        self.log("jump_guard", direction=direction, reason=reason)
        return JUMP_ACTION[direction]

    def _ramp_action(self, obs, player, jumping):
        self._current_level_from_y(obs, player, jumping)
        candidates = self._route_candidates()
        if self.ramp_choice >= len(candidates):
            self.ramp_choice = 0
        desired = candidates[self.ramp_choice]
        ladder = self._ladder_for(obs, desired)
        target_x, target_y = ladder["x"], ladder["y"]
        dx = target_x - player["x"]

        # A FIRE-start jump should be steered toward the first ladder rather
        # than interrupted by UP.  For later jumps, preserve the latched side.
        if jumping:
            if self.jump_direction:
                return JUMP_ACTION[self.jump_direction]
            return "right" if dx > 0 else "left" if dx < 0 else "noop"

        # Detect an actual height response to the current UP probe.  This is
        # the important difference from holding UP forever at a bad ladder.
        if self.up_start_y is not None and player["y"] < self.up_start_y - 7:
            self.ramp_level = min(self.ramp_level + 1, len(RAMP_CANDIDATES) - 1)
            self.ramp_choice = 0
            self.up_probes = 0
            self.up_start_y = None
            self.route_failures = 0
            self.log("climb_height_response", level=self.ramp_level)
            return self._ramp_action(obs, player, False)

        if abs(dx) > 0:
            self.up_probes = 0
            self.up_start_y = None
            return "right" if dx > 0 else "left"

        # Aligned: probe UP for a bounded number of decisions.  The game can
        # reject a visually present ladder, hence the alternate candidate.
        if self.up_start_y is None:
            self.up_start_y = player["y"]
        self.up_probes += 1
        if self.up_probes in (1, 9):
            self.log("ladder_probe", level=self.ramp_level,
                     choice=self.ramp_choice, player_x=round(player["x"], 1),
                     player_y=round(player["y"], 1),
                     target_x=round(target_x, 1), target_y=round(target_y, 1),
                     ram_x=_raw_byte(obs, 0x93),
                     ladder_ram=[_raw_byte(obs, a) for a in (0xC9, 0xCA, 0xCB, 0xCC)],
                     probe=self.up_probes)
        if self.up_probes > 9:
            self.route_failures += 1
            self.up_probes = 0
            self.up_start_y = None
            if self.ramp_choice + 1 < len(candidates):
                self.ramp_choice += 1
                self.log("ladder_alternate", level=self.ramp_level,
                         choice=self.ramp_choice)
                return self._ramp_action(obs, player, False)
            # No second ladder on this band: take a small alignment sweep and
            # retry instead of oscillating UP/DOWN.
            self.ramp_choice = 0
            nudge = -8 if self.route_failures % 2 else 8
            return "left" if nudge < 0 else "right"
        return "upright" if self.ramp_level == 0 else "up"

    def _rivet_row_index(self, obs, player):
        # Use actual ladder centers where possible, otherwise the public row
        # values.  Bottom-to-top is the only route invariant assumed here.
        ladders = self._ladders(obs)
        ys = sorted(set(round(item["y"], 1) for item in ladders), reverse=True)
        if len(ys) >= 4:
            # Lower four bands in screen order.  The ramp has six bands; rivet
            # rows are the bottom four.
            ys = ys[:4]
        else:
            ys = [y + 8.5 for y in RIVET_Y]
        idx = min(range(4), key=lambda i: abs(player["y"] - ys[min(i, len(ys)-1)]))
        if abs(player["y"] - ys[min(idx, len(ys)-1)]) <= 15:
            self.rivet_level = max(self.rivet_level, idx)
        return min(self.rivet_level, 3)

    def _remaining_side(self, row, mask):
        if row in self.rivet_removed_sides:
            removed = self.rivet_removed_sides[row]
            return "right" if removed == "left" else "left"
        if mask == 6:
            return "right"
        if mask == 12:
            return "left"
        # Serpentine: bottom row starts on the left; next row reverses.
        return "left" if row % 2 == 0 else "right"

    def _rivet_action(self, obs, player, jumping):
        rows = self._rows(obs)
        if self.rivet_previous_rows is not None:
            for i, (old, new) in enumerate(zip(self.rivet_previous_rows, rows)):
                if new != old and new in (6, 12, 18) and self.rivet_expected_side:
                    self.rivet_removed_sides[i] = self.rivet_expected_side
        self.rivet_previous_rows = list(rows)

        while self.rivet_level < 4 and rows[self.rivet_level] == 18:
            self.rivet_level += 1
            self.rivet_side = None
            self.rivet_expected_side = None
        if self.rivet_level >= 4:
            return "up"

        row = self.rivet_level
        mask = rows[row]
        side = self._remaining_side(row, mask)
        if self.rivet_side != side:
            self.rivet_side = side
            self.rivet_expected_side = side
        target_x = RIVET_X[side] + 4.0

        if jumping:
            return JUMP_ACTION[self.jump_direction] if self.jump_direction else self.last_action

        # A row with a removed rivet has a gap.  If walking toward the other
        # rivet would cross that known location, use a deliberate jump.
        removed = self.rivet_removed_sides.get(row)
        if removed and removed != side:
            gap_x = RIVET_X[removed] + 4.0
            crosses_gap = ((player["x"] < gap_x < target_x) or
                           (player["x"] > gap_x > target_x))
            if crosses_gap:
                return self._start_jump("right" if target_x > player["x"] else "left", "rivet_gap")

        if abs(player["x"] - target_x) > 7:
            self.rivet_expected_side = side
            return "right" if target_x > player["x"] else "left"

        # Keep moving through the rivet.  If the byte does not change after a
        # short contact window, continue toward the far edge; this avoids
        # parking on the sprite and lets the cartridge register the crossing.
        self.rivet_expected_side = side
        if side == "left":
            return "left" if player["x"] > target_x - 2 else "right"
        return "right" if player["x"] < target_x + 2 else "left"

    def _fallback(self, obs, tick):
        stage = self._stage(obs)
        player = self._player(obs)
        jumping = self._jumping(obs, player)
        self._phase_reset(obs, stage, tick)

        if bool(obs.get("terminated")) or bool(obs.get("truncated")) or stage == "game_over":
            return "noop"
        if stage == "idle":
            if not self.start_sent:
                self.start_sent = True
                return "fire"
            return "noop"
        if stage == "transition":
            if self.transition_tick is None:
                self.transition_tick = tick
            lives = _integer(obs.get("lives_remaining"), 0)
            # The observed cartridge needs a release interval after death;
            # restart once, then let the new stage tell us when play began.
            if lives > 0 and not self.restart_sent and tick - self.transition_tick >= 38:
                self.restart_sent = True
                return "fire"
            return "noop"
        if stage not in ("ramp", "rivet"):
            return "noop"

        if self.jump_cooldown > 0:
            self.jump_cooldown -= 1
        if self.jump_hold > 0:
            self.jump_hold -= 1
        hazard = self._hazard(obs, player)

        if hazard["urgency"] in ("immediate", "soon") and not jumping and self.jump_cooldown <= 0:
            return self._start_jump(hazard["jump_direction"], "hazard")
        if jumping and self.jump_direction and self.jump_hold > 0:
            return JUMP_ACTION[self.jump_direction]

        if stage == "ramp":
            return self._ramp_action(obs, player, jumping)
        return self._rivet_action(obs, player, jumping)

    def _jev_state(self, obs, fallback, tick):
        stage = self._stage(obs)
        player = self._player(obs)
        hazard = self._hazard(obs, player)
        rows = self._rows(obs)
        return {
            "game": "Atari 2600 Donkey Kong NTSC retail port",
            "tick": tick,
            "stage": stage,
            "player": {
                "side": "left" if player["x"] < 55 else "right" if player["x"] > 95 else "middle",
                "band": "low" if player["y"] > 160 else "middle" if player["y"] > 90 else "high",
                "motion": "up" if player["dy"] < -1 else "down" if player["dy"] > 1 else "level",
                "jumping": self._jumping(obs, player),
            },
            "route": {
                "fallback": fallback,
                "ramp_level": self.ramp_level,
                "ramp_choice": self.ramp_choice,
                "up_probe_count": self.up_probes,
                "rivet_level": self.rivet_level,
                "rivet_rows": rows,
            },
            "hazard": hazard,
            "progress": {
                "lives": _integer(obs.get("lives_remaining"), 0),
                "screens": _integer(obs.get("screens_completed"), 0),
                "rivets_removed": _integer(obs.get("rivets_removed"), 0),
                "bonus": _integer(obs.get("current_bonus"), 0),
            },
            "last_posted_action": self.last_action,
        }

    def _jev_questions(self, fallback, hazard):
        options = {
            "continue_route": "Use the deterministic route action supplied in route.fallback; do not press FIRE.",
            "jump_left": "Start an early directional jump to the left because a closing hazard or gap requires it.",
            "jump_right": "Start an early directional jump to the right because a closing hazard or gap requires it.",
            "hold": "Release movement briefly while the current transition or alignment is unsafe.",
        }
        if hazard.get("urgency") in ("immediate", "soon"):
            preferred = "jump_" + str(hazard.get("jump_direction", "right"))
            instruction = (
                "Identify the immediate tactical mode. The code has already computed exact geometry. "
                "If hazard.urgency is immediate or soon, choose the matching directional jump; "
                "otherwise choose continue_route. Choose only a listed option."
            )
        else:
            preferred = "continue_route"
            instruction = (
                "Recognize whether the normalized state requires an escape. There is no closing hazard; "
                "preserve the deterministic route and do not invent a jump. Choose only a listed option."
            )
        # Put the code's candidate first for safe stand-in behavior, while the
        # bridge still requires probability/confidence before accepting a model
        # jump that code did not already require.
        ordered = [preferred] + [key for key in options if key != preferred]
        return Choice(instruction, {key: options[key] for key in ordered})

    def _start_jev(self, obs, fallback, tick):
        if self.jev_future is not None or self.ctx.done:
            return
        try:
            if self.ctx.budget.remaining().get("system1_calls", 0) <= 0:
                return
        except Exception:
            return
        if tick - self.last_jev_start < 4:
            return
        if getattr(self.ctx, "system1", True) is None:
            return
        player = self._player(obs)
        hazard = self._hazard(obs, player)
        state = self._jev_state(obs, fallback, tick)
        questions = {
            "tactic": self._jev_questions(fallback, hazard),
            "danger": Noul(
                "Is hazard.present true because a visible object is closing on Mario's current walkway?",
                {"true": "A visible object is closing on the current walkway.",
                 "false": "No visible object is an immediate closing threat."},
            ),
        }
        self.jev_capture_tick = tick
        self.jev_capture_stage = self._stage(obs)
        self.last_jev_start = tick

        def work():
            return self.ctx.decide(state, questions, timeout_s=0.34)

        self.jev_future = self.ctx.spawn(work)

    def _collect_jev(self, obs, tick):
        future = self.jev_future
        if future is None or not future.done():
            return
        self.jev_future = None
        try:
            answers = future.result()
        except (TimeoutError, RuntimeError):
            self.log("jev_failed")
            return
        except (BudgetExceeded, EpisodeOver, ModelUnavailable):
            # These are normal lifecycle signals when a background request
            # crosses the episode boundary; no stale answer is executable.
            return
        except Exception as exc:
            self.log("jev_failed", error=type(exc).__name__)
            return
        if not isinstance(answers, dict):
            return
        answer = answers.get("tactic")
        value = getattr(answer, "value", None)
        if not isinstance(value, str):
            return
        self.jev_answer = value
        self.jev_answer_tick = tick
        self.jev_answer_stage = self._stage(obs)
        self.log("jev_tactic", value=value, probability=_prob(answer, value))

    def _apply_jev(self, obs, fallback, tick):
        value = self.jev_answer
        if value is None or self.jev_answer_stage != self._stage(obs):
            return fallback
        if tick - self.jev_answer_tick > 3:
            self.jev_answer = None
            return fallback
        player = self._player(obs)
        hazard = self._hazard(obs, player)
        if value in ("jump_left", "jump_right"):
            # A model may reinforce a detected threat, but cannot create a
            # jump in a clear route unless it is very confident and a jump is
            # not already committed.
            answer_probability = 0.0
            # The probability is logged by the answer object, but the answer
            # itself need not be retained for the fallback path. Hard guard is
            # the authority for all lethal situations.
            if hazard["urgency"] in ("immediate", "soon"):
                direction = value.rsplit("_", 1)[-1]
                if direction == hazard["jump_direction"]:
                    return self._start_jump(direction, "jev_confirmed")
            return fallback
        return fallback

    def _start_astra(self, obs, reason):
        if self.astra_future is not None or self.ctx.done or self.astra_calls >= 3:
            return
        try:
            if self.ctx.budget.remaining().get("system2_calls", 0) <= 0:
                return
        except Exception:
            return
        self.astra_calls += 1
        snapshot = self._jev_state(obs, "feedback", _integer(obs.get("tick"), self.local_tick))
        prompt = (
            "You are the slow strategic planner for the Atari 2600 Donkey Kong retail port. "
            "Do not give an action sequence. Code already executes a feedback controller: it "
            "walks to public ladder candidates, probes UP, and abandons a candidate after no "
            "height response. The cartridge alternates ramp/barrel and rivet/fireball screens. "
            "Choose only macro preferences that can transfer across RNG seeds. Return a compact "
            "JSON object with risk (conservative or balanced), ramp (one short route/recovery "
            "rule), rivet (one short serpentine/gap rule), and replan (one trigger). Under 100 "
            "words. Current facts and reason follow.\nReason: " + str(reason) +
            "\nFacts:\n" + json.dumps(snapshot, separators=(",", ":"))
        )

        def work():
            return self.ctx.think(prompt, timeout_s=110.0)

        self.astra_future = self.ctx.spawn(work)

    def _collect_astra(self):
        future = self.astra_future
        if future is None or not future.done():
            return
        self.astra_future = None
        try:
            result = future.result()
        except (TimeoutError, RuntimeError, BudgetExceeded, EpisodeOver, ModelUnavailable):
            self.log("astra_failed")
            return
        except Exception as exc:
            self.log("astra_failed", error=type(exc).__name__)
            return
        parsed = _parse_json_object(getattr(result, "text", ""))
        if not isinstance(parsed, dict):
            return
        risk = str(parsed.get("risk", "conservative"))
        self.plan = {
            "risk": risk if risk in ("conservative", "balanced") else "conservative",
            "ramp": str(parsed.get("ramp", "feedback"))[:180],
            "rivet": str(parsed.get("rivet", "serpentine"))[:180],
        }
        self.plan_version += 1
        self.log("astra_plan", version=self.plan_version, risk=self.plan["risk"])

    def _valid_action(self, action):
        try:
            names = set(self.game.action_spec.names)
            noop = self.game.action_spec.noop or "noop"
        except Exception:
            names, noop = set(), "noop"
        return action if isinstance(action, str) and action in names else noop

    def run(self):
        ctx = self.ctx
        try:
            obs = ctx.observe()
        except Exception:
            obs = {}
        if not isinstance(obs, dict):
            obs = {}
        tick = _integer(obs.get("tick"), 0)
        stage = self._stage(obs)
        self.last_stage = None
        self._start_astra(obs, "initial route and interface review")

        # FIRE is posted before the first game quantum.  It is never used as a
        # generic model fallback during active play.
        if stage == "idle":
            self.start_sent = True
            first = self._valid_action("fire")
        else:
            first = self._valid_action(self._fallback(obs, tick))
        ctx.act(first)
        self.last_action = first
        self.last_lives = _integer(obs.get("lives_remaining"), 0)
        self.last_screens = _integer(obs.get("screens_completed"), 0)
        self.log("controller_started", first_action=first)

        while not ctx.done and not getattr(self.game, "done", False):
            try:
                ctx.wait_for_tick()
            except (BudgetExceeded, EpisodeOver):
                break
            if ctx.done or getattr(self.game, "done", False):
                break
            try:
                obs = ctx.observe()
            except (BudgetExceeded, EpisodeOver):
                break
            except Exception:
                obs = {}
            if not isinstance(obs, dict):
                obs = {}
            tick = self._tick(obs)
            stage = self._stage(obs)
            lives = _integer(obs.get("lives_remaining"), 0)
            screens = _integer(obs.get("screens_completed"), 0)

            if bool(obs.get("terminated")) or bool(obs.get("truncated")) or stage == "game_over":
                break

            if (stage != self.last_stage or
                    (self.last_lives is not None and lives != self.last_lives) or
                    screens != self.last_screens):
                self._start_astra(obs, "stage, life, or screen progress changed")

            self._collect_astra()
            self._collect_jev(obs, tick)
            fallback = self._fallback(obs, tick)

            if stage in ("ramp", "rivet") and tick % 3 == 0:
                self._start_jev(obs, fallback, tick)
            action = self._apply_jev(obs, fallback, tick)
            action = self._valid_action(action)
            if ctx.done or getattr(self.game, "done", False):
                break
            try:
                ctx.act(action)
            except (BudgetExceeded, EpisodeOver):
                break
            self.last_action = action
            self.last_stage = stage
            self.last_lives = lives
            self.last_screens = screens

            if tick % 100 == 0:
                self.log("progress", tick=tick, stage=stage,
                         score=_finite(obs.get("score_total")),
                         lives=lives,
                         screens=screens,
                         rivets=_integer(obs.get("rivets_removed"), 0),
                         level=self.ramp_level)

        self.log("controller_finished", ticks=self.local_tick,
                 score=_finite(getattr(self.game, "score", 0)),
                 plan_version=self.plan_version)


class Program:
    def __init__(self, ctx):
        self.ctx = ctx

    def run_episode(self, game):
        Controller(self.ctx, game).run()


def build(ctx):
    return Program(ctx)
