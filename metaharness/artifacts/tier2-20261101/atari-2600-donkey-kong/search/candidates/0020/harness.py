"""Observation-ledger controller for Atari 2600 Donkey Kong.

The controller deliberately separates two facts that the earlier ramp agents
conflated: (1) which walkway Mario has actually reached, and (2) which ladder
connector he is currently trying to use.  The walkway is committed from
repeated observed y positions, while the connector is a small online choice
among the visible fixed ladder boxes.  A jump is an action lease synchronized
to the documented jump byte, not a collection of independent per-tick guesses.

Astra and Jev are advisory only on the rivet screen.  The ramp actor never
waits for inference, because the useful ramp facts are arithmetic and the
model latency is longer than a safe four-frame hazard decision.
"""
from __future__ import annotations

import json
import math

from metaharness.models.jev import Choice, Noul


ACTIONS = frozenset(
    {
        "noop", "fire", "up", "right", "left", "down", "upright", "upleft",
        "downright", "downleft", "upfire", "rightfire", "leftfire", "downfire",
        "uprightfire", "upleftfire", "downrightfire", "downleftfire",
    }
)

# Values are Mario's left edge.  The observation's ladder boxes are two pixels
# farther right.  The second values are the short alternative connectors on
# split bands; they are tried only after observed horizontal non-response.
RAMP_BANDS = (
    (176, (106,)),
    (148, (46, 78)),
    (120, (106, 86)),
    (92, (46, 66)),
    (64, (106,)),
    (40, (30, 74)),
)
RIVET_Y = (176, 148, 120, 92)
RIVET_X = {"left": 46, "right": 106}


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


def player_xy(obs):
    player = player_of(obs)
    return number(player.get("x"), 45.0), number(player.get("y"), 176.0)


def visible(item):
    return isinstance(item, dict) and item.get("visible", True) is not False


def raw_byte(obs, address):
    raw = obs.get("raw_ram", []) if isinstance(obs, dict) else []
    if not isinstance(raw, (list, tuple)):
        return 0
    index = address & 0x7F
    if index >= len(raw):
        return 0
    return integer(raw[index], 0)


def jump_state(obs):
    # The adaptor's 128-byte snapshot is wrapped at the documented RAM base;
    # $85 is therefore index 5 in raw_ram.
    return raw_byte(obs, 0x85)


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


def directional(direction, fire=False):
    if direction not in ("left", "right"):
        return "fire" if fire else "noop"
    return direction + ("fire" if fire else "")


def extract_json_object(text):
    """Accept only a JSON object, while tolerating fenced Astra output."""
    if not isinstance(text, str):
        return None
    text = text.strip()
    pieces = [text]
    left, right = text.find("{"), text.rfind("}")
    if left >= 0 and right > left:
        pieces.insert(0, text[left : right + 1])
    for piece in pieces:
        try:
            value = json.loads(piece)
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


def normalized_ramp_row(y):
    """Map a y observation to the highest band it could have reached.

    The current band is deliberately a semantic observation, not an action
    count.  The seven-pixel overlap admits the settled positions seen in the
    public traces (154, 126, 98, ...), while the caller requires repeated
    non-jump observations before committing a new band.
    """
    value = number(y, 176.0)
    for index in range(len(RAMP_BANDS) - 1, -1, -1):
        if value <= RAMP_BANDS[index][0] + 7.0:
            return index
    return 0


def remaining_side(mask, preferred):
    """Decode the verified 0/6/12/18 rivet-row statuses."""
    if mask == 18:
        return None
    if mask == 6:  # left rivet cleared
        return "right"
    if mask == 12:  # right rivet cleared
        return "left"
    return preferred if preferred in ("left", "right") else "left"


def same_row_hazard(obs):
    """Return the most urgent closing obstacle on Mario's current band.

    Numeric distance and closing tests are performed here in code.  Jev never
    receives raw coordinates and is not put in charge of collision arithmetic.
    """
    px, py = player_xy(obs)
    hazards = obs.get("hazards", []) if isinstance(obs, dict) else []
    if not isinstance(hazards, list):
        hazards = []
    best = None
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
        closing = (delta * dx < -0.2) or (distance <= 14.0 and abs(dx) <= 0.2)
        if not closing and distance > 18.0:
            continue
        if distance <= 16.0:
            urgency = "overlap"
        elif distance <= 22.0 and closing:
            urgency = "commit"
        elif distance <= 38.0 and closing:
            urgency = "soon"
        elif distance <= 62.0 and closing:
            urgency = "watch"
        else:
            urgency = "near"
        fact = {
            "present": True,
            "slot": integer(item.get("slot"), -1),
            "side": "left" if delta < 0.0 else "right",
            "motion": "left" if dx < -0.2 else "right" if dx > 0.2 else "still",
            "closing": bool(closing),
            "distance": round(distance, 1),
            "distance_bucket": (
                "overlap" if distance <= 16.0 else
                "near" if distance <= 28.0 else
                "approaching" if distance <= 62.0 else "far"
            ),
            "urgency": urgency,
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
        "present": False,
        "slot": -1,
        "side": "none",
        "motion": "none",
        "closing": False,
        "distance": 999.0,
        "distance_bucket": "none",
        "urgency": "clear",
    }


class Controller:
    """One episode's observation-ledger and real-time action leases."""

    def __init__(self, ctx, game):
        self.ctx = ctx
        self.game = game
        self.tick = 0
        self.last_action = "noop"
        self.last_stage = None
        self.last_lives = None
        self.last_screens = 0

        # Ramp ledger.  `band` is the last committed observed walkway.  A
        # pending band is promoted only after three consecutive non-jump
        # observations, preventing a high jump arc from masquerading as a climb.
        self.band = 0
        self.pending_band = None
        self.pending_count = 0
        self.connector = 0
        self.climb_open = False
        self.climb_origin_y = None
        self.climb_age = 0
        self.horizontal_stall = 0
        self.previous_x = None
        self.connector_attempts = {}
        # After a real upward attempt consumes a life, preserve the remaining
        # lives with the known safe scoring lane instead of replaying the same
        # upper-band collision.
        self.farm_mode = False
        self.jump_direction = None
        self.jump_seen = False
        self.jump_guard = 6
        self.jump_tail = 0
        self.jump_lock = 0
        self.jump_slot = -1
        self.jump_count = 0
        self.post_jump_up = 0

        # Cartridge lifecycle is independent of the spare-life byte.  The byte
        # counts spare Marios; a nonterminal zero-spare transition is playable.
        self.transition_age = 0
        self.restart_edges = 0
        self.idle_edges = 0

        # Rivet route bookkeeping.
        self.rivet_row = 0
        self.rivet_preference = "left"
        self.previous_rows = None
        self.gap_direction = None

        # Optional asynchronous model bridge.  It is dormant until rivet play.
        self.jev_future = None
        self.jev_stamp = None
        self.jev_started = -10000
        self.jev_hint = None
        self.astra_future = None
        self.astra_started = False
        self.astra_hint = None

    # ---------- small contract-safe helpers ----------
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
        if action not in ACTIONS:
            action = "noop"
        try:
            self.ctx.act(action)
        except Exception as exc:
            self.log("act_failed", action=action, error=type(exc).__name__)
            return False
        self.last_action = action
        return True

    def reset_active(self, stage):
        self.band = 0
        self.pending_band = None
        self.pending_count = 0
        self.connector = 0
        self.climb_open = False
        self.climb_origin_y = None
        self.climb_age = 0
        self.horizontal_stall = 0
        self.previous_x = None
        self.connector_attempts = {}
        self.clear_jump()
        self.gap_direction = None
        if stage == "rivet":
            self.rivet_row = 0
            self.rivet_preference = self.astra_hint if self.astra_hint in ("left", "right") else "left"
            self.previous_rows = None

    def clear_jump(self):
        self.jump_direction = None
        self.jump_seen = False
        self.jump_guard = 6
        self.jump_tail = 0
        self.jump_lock = 0
        self.jump_slot = -1
        self.post_jump_up = 0

    def synchronize(self, obs):
        stage = stage_of(obs)
        lives = integer(obs.get("lives_remaining"), 0) if isinstance(obs, dict) else 0
        screens = integer(obs.get("screens_completed"), 0) if isinstance(obs, dict) else 0
        changed = stage != self.last_stage
        life_changed = self.last_lives is not None and lives != self.last_lives
        screen_changed = screens != self.last_screens
        if changed or life_changed or screen_changed:
            self.log(
                "boundary",
                old_stage=self.last_stage,
                stage=stage,
                old_lives=self.last_lives,
                lives=lives,
                old_screens=self.last_screens,
                screens=screens,
                band=self.band,
            )
            if life_changed and stage in ("ramp", "transition") and self.band >= 1 and not self.farm_mode:
                self.farm_mode = True
                self.log("risk_fallback", failed_band=self.band, lives=lives)
            if stage in ("ramp", "rivet"):
                self.reset_active(stage)
        if stage == "transition":
            self.transition_age = self.transition_age + 1 if self.last_stage == "transition" else 1
        else:
            self.transition_age = 0
        if changed:
            self.restart_edges = 0
            if stage == "idle":
                self.idle_edges = 0
        self.last_stage = stage
        self.last_lives = lives
        self.last_screens = screens
        return stage, lives, screens, changed, life_changed, screen_changed

    # ---------- observed height ledger ----------
    def observe_height(self, obs):
        """Commit one upward band only from repeated grounded observations."""
        if jump_state(obs) > 0:
            self.pending_band = None
            self.pending_count = 0
            return
        observed = normalized_ramp_row(player_of(obs).get("y"))
        if observed <= self.band:
            self.pending_band = None
            self.pending_count = 0
            return
        # A normal climb advances one band.  A landed jump may appear one band
        # higher, but never allow a multi-band inference in one observation.
        observed = min(observed, self.band + 1)
        if self.pending_band == observed:
            self.pending_count += 1
        else:
            self.pending_band = observed
            self.pending_count = 1
        if self.pending_count < 3:
            return
        old = self.band
        self.band = observed
        self.pending_band = None
        self.pending_count = 0
        self.connector = 0
        self.climb_open = False
        self.climb_origin_y = None
        self.climb_age = 0
        self.horizontal_stall = 0
        self.connector_attempts.pop(old, None)
        self.log(
            "band_confirmed",
            old_band=old,
            band=self.band,
            x=round(player_xy(obs)[0], 1),
            y=round(player_xy(obs)[1], 1),
        )

    def route_options(self):
        index = min(max(self.band, 0), len(RAMP_BANDS) - 1)
        return RAMP_BANDS[index][1]

    def route_target(self):
        options = self.route_options()
        index = min(max(self.connector, 0), len(options) - 1)
        return float(options[index])

    def route_direction(self, obs):
        x, _ = player_xy(obs)
        target = self.route_target()
        if x < target - 1.0:
            return "right"
        if x > target + 1.0:
            return "left"
        return None

    def try_alternate(self, reason):
        options = self.route_options()
        if len(options) < 2:
            self.horizontal_stall = 0
            return False
        old = self.connector
        attempts = self.connector_attempts.get(self.band, 0) + 1
        self.connector_attempts[self.band] = attempts
        if attempts > 2:
            # Both connectors have been observed without progress.  Reacquire
            # the nominal edge instead of oscillating indefinitely.
            self.connector = 0
            self.horizontal_stall = 0
            self.climb_open = False
            self.climb_age = 0
            self.log("connector_reacquire", band=self.band, reason=reason)
            return False
        self.connector = 1 if old == 0 else 0
        self.horizontal_stall = 0
        self.climb_open = False
        self.climb_origin_y = None
        self.climb_age = 0
        self.log(
            "connector_switch",
            band=self.band,
            old=old,
            new=self.connector,
            target=options[self.connector],
            reason=reason,
        )
        return True

    def farm_action(self, obs):
        """Hold the proven lower-ramp lane without promoting a route band."""
        x, y = player_xy(obs)
        target = 106.0
        if self.climb_open:
            if abs(x - target) > 3.5:
                self.climb_open = False
                self.climb_origin_y = None
                self.climb_age = 0
            else:
                self.climb_age += 1
                return "up"
        if x < target - 1.0:
            return "right"
        if x > target + 1.0:
            return "left"
        self.climb_open = True
        self.climb_origin_y = y
        self.climb_age = 0
        return "up"

    def ramp_action(self, obs):
        if self.farm_mode:
            return self.farm_action(obs)
        self.observe_height(obs)
        x, y = player_xy(obs)
        target = self.route_target()
        options = self.route_options()

        # Track actual horizontal response to the previous walk command.  This
        # is the only trigger for the alternate connector; elapsed time alone
        # cannot silently change the route.
        if self.previous_x is not None and self.last_action in ("left", "right"):
            if abs(x - self.previous_x) < 0.6:
                self.horizontal_stall += 1
            else:
                self.horizontal_stall = 0
        else:
            self.horizontal_stall = 0
        self.previous_x = x

        if self.climb_open:
            # Jump landing can displace Mario from the rung.  Reacquire rather
            # than holding UP in empty space.
            if abs(x - target) > 3.5:
                self.climb_open = False
                self.climb_origin_y = None
                self.climb_age = 0
                return self.ramp_action(obs)
            self.climb_age += 1
            if self.band >= len(RAMP_BANDS) - 1:
                return "up"
            if self.horizontal_stall >= 30:
                self.try_alternate("climb_no_response")
                return self.ramp_action(obs)
            return "up"

        if abs(x - target) > 1.0:
            if self.horizontal_stall >= 10 and len(options) > 1:
                self.try_alternate("walk_no_response")
                return self.ramp_action(obs)
            return "right" if x < target else "left"

        # Reaching the edge opens a real climb transaction.  The height ledger
        # above will close it when the next band's y is repeatedly visible.
        self.climb_open = True
        self.climb_origin_y = y
        self.climb_age = 0
        self.horizontal_stall = 0
        self.log(
            "climb_begin",
            band=self.band,
            connector=self.connector,
            target=round(target, 1),
            x=round(x, 1),
            y=round(y, 1),
        )
        return "up"

    # ---------- jump lease and hazard policy ----------
    def jump_direction_for(self, obs, hazard):
        # If the named object is closing from the right, Mario's successful
        # trace jumps right into its path; if it is closing from the left, jump
        # left.  This is the documented counter-motion primitive, not an escape
        # to the opposite screen edge.
        if hazard.get("side") == "left":
            return "left"
        if hazard.get("side") == "right":
            return "right"
        route = self.route_direction(obs)
        return route if route in ("left", "right") else "right"

    def begin_jump(self, obs, direction, reason, hazard=None):
        if direction not in ("left", "right"):
            direction = "right"
        self.jump_direction = direction
        self.jump_seen = False
        self.jump_guard = 6
        self.jump_tail = 0
        self.jump_lock = 18
        self.jump_slot = integer((hazard or {}).get("slot"), -1)
        self.jump_count += 1
        self.log(
            "jump_begin",
            tick=self.tick,
            direction=direction,
            reason=reason,
            slot=self.jump_slot,
            hazard={
                key: (hazard or {}).get(key)
                for key in ("side", "motion", "distance_bucket", "urgency")
            },
            band=self.band,
        )
        return directional(direction, True)

    def jump_action(self, obs):
        if self.jump_direction is not None:
            state = jump_state(obs)
            if state > 0:
                self.jump_seen = True
                self.jump_guard = 6
                return directional(self.jump_direction, True)
            if self.jump_seen:
                if self.jump_tail < 4:
                    self.jump_tail += 1
                    # The landing lane is the same lane selected at the FIRE
                    # edge.  Do not reverse at the collision point.
                    return directional(self.jump_direction, False)
                direction = self.jump_direction
                self.clear_jump()
                self.jump_lock = 14
                self.post_jump_up = 6
                self.log("jump_finished", tick=self.tick, direction=direction)
                return "up"
            if self.jump_guard > 0:
                self.jump_guard -= 1
                return directional(self.jump_direction, True)
            self.log("jump_unconfirmed", tick=self.tick)
            self.clear_jump()
            return None
        if self.post_jump_up > 0:
            self.post_jump_up -= 1
            return "up"
        if self.jump_lock > 0:
            self.jump_lock -= 1
        return None

    def hazard_action(self, obs, stage):
        if stage not in ("ramp", "rivet"):
            return None
        if self.jump_direction is not None or self.jump_lock > 0:
            return None
        hazard = same_row_hazard(obs)
        allowed_urgency = ("overlap",) if self.farm_mode else ("overlap", "commit")
        if hazard.get("urgency") not in allowed_urgency:
            return None
        # An overlap is allowed to interrupt a climb; earlier warnings are
        # filtered above, so a usable ladder is not abandoned prematurely.
        direction = self.jump_direction_for(obs, hazard)
        return self.begin_jump(obs, direction, "hazard", hazard)

    # ---------- rivet route ----------
    def nearest_ladder(self, obs, side):
        desired = 108 if side == "right" else 48
        _, py = player_xy(obs)
        ladders = obs.get("ladders", []) if isinstance(obs, dict) else []
        candidates = []
        if isinstance(ladders, list):
            for item in ladders:
                if not visible(item):
                    continue
                lx = number(item.get("x"), desired)
                ly = number(item.get("y"), py)
                if abs(ly - py) <= 22.0 or abs(ly - (py - 8.0)) <= 22.0:
                    candidates.append((abs(lx - desired) + abs(ly - py) * 0.05, lx - 2.0))
        if candidates:
            candidates.sort()
            return candidates[0][1]
        return float(desired - 2)

    def rivet_action(self, obs):
        rows = rows_of(obs)
        if self.previous_rows is not None:
            for index, (old, new) in enumerate(zip(self.previous_rows, rows)):
                if index != self.rivet_row or old == new:
                    continue
                removed = None
                if new == 6:
                    removed = "left"
                elif new == 12:
                    removed = "right"
                elif new == 18:
                    removed = "right" if old == 6 else "left" if old == 12 else None
                if removed is not None:
                    self.gap_direction = "right" if removed == "left" else "left"
                    self.log("rivet_changed", row=index, old=old, new=new, removed=removed)
        self.previous_rows = list(rows)

        first = next((index for index, value in enumerate(rows) if value < 18), 4)
        if first >= 4:
            return "up"
        if first != self.rivet_row:
            self.rivet_row = first
            self.gap_direction = None

        if self.gap_direction in ("left", "right") and self.jump_direction is None:
            direction = self.gap_direction
            self.gap_direction = None
            return self.begin_jump(obs, direction, "rivet_gap")

        row = self.rivet_row
        preferred = self.rivet_preference if row == 0 else ("left" if row % 2 == 0 else "right")
        side = remaining_side(rows[row], preferred)
        x, y = player_xy(obs)
        if side is None:
            ladder_side = "right" if row % 2 == 0 else "left"
            target = self.nearest_ladder(obs, ladder_side)
            if x < target - 2.0:
                return "right"
            if x > target + 2.0:
                return "left"
            return "up"
        if y > RIVET_Y[row] + 10.0:
            ladder_side = "right" if row % 2 == 0 else "left"
            target = self.nearest_ladder(obs, ladder_side)
            if x < target - 2.0:
                return "right"
            if x > target + 2.0:
                return "left"
            return "up"
        target_x = RIVET_X[side]
        if x < target_x - 8.0:
            return "right"
        if x > target_x + 8.0:
            return "left"
        return "left" if side == "left" else "right"

    # ---------- advisory model bridge ----------
    def model_state(self, obs, context):
        x, y = player_xy(obs)
        hazard = same_row_hazard(obs)
        return {
            "game": "Atari 2600 Donkey Kong NTSC retail port",
            "context": context,
            "stage": stage_of(obs),
            "player": {
                "horizontal_band": "left" if x < 52 else "right" if x > 100 else "middle",
                "vertical_band": "bottom" if y > 160 else "lower" if y > 132 else "middle" if y > 78 else "top",
                "motion": "up" if number(player_of(obs).get("dy"), 0) < -1 else "down" if number(player_of(obs).get("dy"), 0) > 1 else "steady",
                "jump_committed": self.jump_direction is not None,
            },
            "route": {
                "ramp_band": self.band,
                "rivet_row": self.rivet_row,
                "rivet_rows": rows_of(obs),
                "gap_jump_pending": self.gap_direction in ("left", "right"),
            },
            "hazard": {
                key: hazard.get(key)
                for key in ("present", "side", "motion", "distance_bucket", "closing", "urgency")
            },
            "progress": {
                "lives_remaining": integer(obs.get("lives_remaining"), 0),
                "screens_completed": integer(obs.get("screens_completed"), 0),
                "rivets_removed": integer(obs.get("rivets_removed"), 0),
            },
            "last_action": self.last_action,
        }

    def maybe_start_jev(self, obs):
        if stage_of(obs) != "rivet" or self.jev_future is not None:
            return
        interval = 5 if same_row_hazard(obs).get("urgency") in ("overlap", "commit", "soon") else 14
        if self.tick - self.jev_started < interval:
            return
        try:
            if self.ctx.done or self.ctx.budget.remaining().get("system1_calls", 0) <= 0:
                return
            if getattr(self.ctx, "system1", True) is None:
                return
        except Exception:
            return
        state = self.model_state(obs, "rivet sweep")
        questions = {
            "side": Choice(
                "Choose a side only for the next legal rivet sweep. Use route.rivet_rows and the named row; choose hold while a gap jump or climb is active. Do not count or calculate.",
                {
                    "left": "Approach the left remaining rivet on the current row.",
                    "right": "Approach the right remaining rivet on the current row.",
                    "hold": "Preserve the deterministic route transaction.",
                },
            ),
            "hazard": Noul(
                "Does hazard explicitly describe an overlap or commit closing object on Mario's current row?",
                {
                    "true": "The current-row hazard urgency is overlap or commit and it is closing.",
                    "false": "No overlap or commit closing current-row hazard is described.",
                },
            ),
        }
        self.jev_started = self.tick
        self.jev_stamp = (stage_of(obs), self.rivet_row, tuple(rows_of(obs)), self.last_lives)
        try:
            self.jev_future = self.ctx.spawn(
                lambda: self.ctx.decide(state, questions, timeout_s=0.55)
            )
        except Exception as exc:
            self.log("jev_start_failed", error=type(exc).__name__)
            self.jev_future = None

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
            answer = answers.get("side") if isinstance(answers, dict) else None
            value = getattr(answer, "value", None)
            if value not in ("left", "right"):
                return
            probability = answer_probability(answer, value)
            if probability is not None and probability < 0.55:
                return
            stamp = (stage_of(obs), self.rivet_row, tuple(rows_of(obs)), self.last_lives)
            if stamp == self.jev_stamp:
                self.jev_hint = value
                self.log("jev_side", side=value, probability=probability)
        except Exception as exc:
            self.log("jev_failed", error=type(exc).__name__)

    def maybe_start_astra(self, obs):
        if self.astra_started or stage_of(obs) != "rivet":
            return
        try:
            if self.ctx.done or self.ctx.budget.remaining().get("system2_calls", 0) <= 0:
                return
        except Exception:
            return
        self.astra_started = True
        state = self.model_state(obs, "rivet route review")
        prompt = (
            "Review this Atari 2600 Donkey Kong NTSC rivet state. The actor owns "
            "timing, hazards, ladder alignment, gap jumps, and lifecycle. Return "
            "only JSON {\"first_side\":\"left\"} or {\"first_side\":\"right\"}; "
            "never return actions or prose. Use only the supplied row statuses.\n" +
            json.dumps(state, separators=(",", ":"), allow_nan=False)
        )
        try:
            self.astra_future = self.ctx.spawn(
                lambda: self.ctx.think(prompt, timeout_s=90.0)
            )
        except Exception as exc:
            self.log("astra_start_failed", error=type(exc).__name__)
            self.astra_future = None

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
            parsed = extract_json_object(getattr(result, "text", ""))
            side = parsed.get("first_side") if isinstance(parsed, dict) else None
            if side in ("left", "right"):
                self.astra_hint = side
                self.rivet_preference = side
                self.log("astra_side", side=side)
        except Exception as exc:
            self.log("astra_failed", error=type(exc).__name__)

    # ---------- policy and actor loop ----------
    def choose(self, obs):
        stage, lives, screens, stage_changed, life_changed, screen_changed = self.synchronize(obs)
        self.collect_jev(obs)
        self.collect_astra()

        if stage == "idle":
            if self.idle_edges < 3:
                self.idle_edges += 1
                return "fire", False
            return "noop", False
        if stage == "game_over" or bool(obs.get("terminated")) or bool(obs.get("truncated")):
            return "noop", True
        if stage == "transition":
            # Do not interpret lives_remaining==0 as terminal; the adaptor
            # explicitly reports a playable final Mario in this transition.
            if self.transition_age >= 18 and self.restart_edges < 4 and (self.transition_age - 18) % 28 == 0:
                self.restart_edges += 1
                self.log("restart_edge", tick=self.tick, age=self.transition_age, lives=lives)
                return "fire", False
            return "noop", False
        if stage not in ("ramp", "rivet"):
            return "noop", False

        if stage == "rivet":
            if stage_changed or not self.astra_started:
                self.maybe_start_astra(obs)
        if stage == "ramp" and not self.farm_mode:
            self.observe_height(obs)

        ongoing = self.jump_action(obs)
        if ongoing is not None:
            return ongoing, False

        # A cleared rivet gap is an explicit route transaction and outranks a
        # generic hazard test.  rivet_action starts the lease when necessary.
        if stage == "rivet" and self.gap_direction in ("left", "right"):
            action = self.rivet_action(obs)
            if self.jump_direction is not None:
                return action, False
        emergency = self.hazard_action(obs, stage)
        if emergency is not None:
            return emergency, False

        action = self.ramp_action(obs) if stage == "ramp" else self.rivet_action(obs)
        if stage == "rivet":
            self.maybe_start_jev(obs)
            if self.jev_hint in ("left", "right") and self.jump_direction is None:
                row = min(max(self.rivet_row, 0), 3)
                if remaining_side(rows_of(obs)[row], self.jev_hint) == self.jev_hint:
                    self.rivet_preference = self.jev_hint
                self.jev_hint = None

        if stage_changed or life_changed or screen_changed:
            self.log("decision_boundary", stage=stage, band=self.band, action=action, lives=lives, screens=screens)
        if self.tick % 20 == 0:
            x, y = player_xy(obs)
            self.log(
                "snapshot",
                tick=self.tick,
                stage=stage,
                band=self.band,
                connector=self.connector,
                x=round(x),
                y=round(y),
                action=action,
                climb=self.climb_open,
                jump=self.jump_direction,
                raw_jump=jump_state(obs),
                hazard=same_row_hazard(obs).get("urgency"),
                lives=lives,
                score=number(obs.get("score_total"), 0.0),
            )
        return action, False

    def run(self):
        try:
            obs = self.ctx.observe()
        except Exception:
            return
        if not isinstance(obs, dict):
            obs = {}
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
            if stop:
                break
            if not self.post(action):
                break
        self.log(
            "finished",
            ticks=self.tick,
            score=getattr(self.game, "score", None),
            band=self.band,
            jumps=self.jump_count,
            screens=self.last_screens,
        )


class Program:
    def __init__(self, ctx):
        self.ctx = ctx

    def run_episode(self, game) -> None:
        Controller(self.ctx, game).run()


def build(ctx):
    return Program(ctx)
