"""Jev-only Donkey Kong controller with a route-aware action bridge.

The parent controller gave Jev the complete observation and let it choose one
of all 18 joystick actions.  This version keeps that synchronous one-request
loop, but turns the study's ladder route and hazard guard into a small typed
tactical interface.  Jev chooses a tactic; code translates it to a legal
movement toward the current waypoint or an early directional jump.
"""
from __future__ import annotations

from metaharness.core.contract import BudgetExceeded, ModelUnavailable
from metaharness.models.jev import Choice


# The decoder's fixed ladder boxes, grouped by the height of their lower end.
# The first item in each row is the conservative alternating-route prior.  The
# second item is a same-row escape if the first ladder does not accept UP.
_RAMP_ROWS = (
    (176, (108,)),
    (148, (48, 80)),
    (120, (108, 88)),
    (92, (48, 68)),
    (64, (108,)),
    (40, (32, 76)),
)

_DIRECTIONAL_FIRE = {
    "left": "leftfire",
    "right": "rightfire",
}


class Harness:
    def __init__(self, ctx):
        self.ctx = ctx
        self.last_stage = None
        self.transition_started = None
        self.restart_sent = False
        self.start_attempts = 0
        self.ramp_token = None
        self.ramp_level = 0
        self.ramp_choice = 0
        self.ramp_aligned_calls = 0
        self.previous_y = None
        self.last_posted = "noop"

    @staticmethod
    def _number(value, default=0):
        try:
            if isinstance(value, bool):
                return int(value)
            return int(value)
        except (TypeError, ValueError):
            return default

    @classmethod
    def _point(cls, obj):
        if not isinstance(obj, dict):
            return {"x": 0, "y": 0, "dx": 0, "dy": 0, "visible": False}
        return {
            "x": cls._number(obj.get("x")),
            "y": cls._number(obj.get("y")),
            "dx": cls._number(obj.get("dx")),
            "dy": cls._number(obj.get("dy")),
            "visible": bool(obj.get("visible", True)),
        }

    @staticmethod
    def _side(delta, tolerance=6):
        if delta < -tolerance:
            return "left"
        if delta > tolerance:
            return "right"
        return "aligned"

    @staticmethod
    def _vertical_band(y):
        if y >= 164:
            return "bottom"
        if y >= 134:
            return "lower_middle"
        if y >= 106:
            return "middle"
        if y >= 78:
            return "upper_middle"
        return "top"

    @staticmethod
    def _motion(dx, dy):
        if abs(dx) <= 1 and abs(dy) <= 1:
            return "still"
        if abs(dx) >= abs(dy):
            return "moving_right" if dx > 0 else "moving_left"
        return "moving_down" if dy > 0 else "moving_up"

    def _sync_phase(self, obs):
        stage = str(obs.get("stage", "unknown")) if isinstance(obs, dict) else "unknown"
        tick = self._number(obs.get("tick")) if isinstance(obs, dict) else 0
        if stage != self.last_stage:
            if stage == "idle":
                self.start_attempts = 0
            if stage == "ramp":
                self.ramp_token = None
                self.ramp_level = 0
                self.ramp_choice = 0
                self.ramp_aligned_calls = 0
                self.previous_y = None
            if stage == "transition":
                self.transition_started = tick
                self.restart_sent = False
            self.last_stage = stage

    def _ramp_plan(self, obs):
        player = self._point(obs.get("player", {}))
        px, py = player["x"], player["y"]
        lives = self._number(obs.get("lives_remaining"))
        completed = self._number(obs.get("screens_completed"))
        token = (lives, completed, self._number(obs.get("screen_index")))
        if token != self.ramp_token:
            self.ramp_token = token
            self.ramp_level = 0
            self.ramp_choice = 0
            self.ramp_aligned_calls = 0
            self.previous_y = py

        # Advance the waypoint only after Mario is close to the next fixed
        # height.  This avoids abandoning a climb while still on its ladder.
        nearest = min(
            range(len(_RAMP_ROWS)),
            key=lambda i: abs(py - _RAMP_ROWS[i][0]),
        )
        if nearest > self.ramp_level and abs(py - _RAMP_ROWS[nearest][0]) <= 7:
            self.ramp_level = nearest
            self.ramp_choice = 0
            self.ramp_aligned_calls = 0
        elif nearest < self.ramp_level and abs(py - _RAMP_ROWS[nearest][0]) <= 7:
            # A fresh life can briefly report the previous position before the
            # stage label changes.  Treat that as a route reset, not descent.
            self.ramp_level = nearest
            self.ramp_choice = 0
            self.ramp_aligned_calls = 0

        row_y, choices = _RAMP_ROWS[min(self.ramp_level, len(_RAMP_ROWS) - 1)]
        target_x = choices[min(self.ramp_choice, len(choices) - 1)]
        delta = target_x - px

        # A vertical displacement means that the current climb still owns the
        # joystick.  Do not steer sideways until the landing height is stable.
        if abs(py - row_y) > 7:
            command = "climb_target" if py > row_y else "descend_target"
            action = "up" if py > row_y else "down"
            self.previous_y = py
            return {
                "command": command,
                "action": action,
                "target_side": "above" if py > row_y else "below",
                "target_band": self._vertical_band(row_y),
                "alignment": "vertical_approach",
                "target_index": self.ramp_level,
                "target_x": target_x,
            }

        if self.previous_y is not None and py < self.previous_y - 4:
            self.ramp_aligned_calls = 0
        self.previous_y = py

        if abs(delta) > 6:
            self.ramp_aligned_calls = 0
            action = "right" if delta > 0 else "left"
            return {
                "command": "approach_target",
                "action": action,
                "target_side": "right" if delta > 0 else "left",
                "target_band": self._vertical_band(row_y),
                "alignment": self._side(delta),
                "target_index": self.ramp_level,
                "target_x": target_x,
            }

        self.ramp_aligned_calls += 1
        # UP is the normal command.  If it produces no y change for several
        # fresh decisions, briefly add the direction of the ladder; this is a
        # bounded recovery from a sticky-action or exact-alignment miss.
        if self.ramp_aligned_calls <= 4:
            action = "up"
        elif self.ramp_aligned_calls <= 8:
            action = "upright" if target_x >= px else "upleft"
        elif self.ramp_aligned_calls <= 12:
            action = "up"
        elif len(choices) > 1 and self.ramp_choice + 1 < len(choices):
            self.ramp_choice += 1
            self.ramp_aligned_calls = 0
            alternate = choices[self.ramp_choice]
            action = "right" if alternate > px else "left"
            target_x = alternate
            return {
                "command": "try_alternate_ladder",
                "action": action,
                "target_side": "right" if alternate > px else "left",
                "target_band": self._vertical_band(row_y),
                "alignment": "alternate_target",
                "target_index": self.ramp_level,
                "target_x": target_x,
            }
        else:
            phase = (self.ramp_aligned_calls - 12) // 4
            action = "upleft" if phase % 2 == 0 else "upright"

        return {
            "command": "climb_target",
            "action": action,
            "target_side": "aligned",
            "target_band": self._vertical_band(row_y),
            "alignment": "on_target",
            "target_index": self.ramp_level,
            "target_x": target_x,
        }

    def _rivet_plan(self, obs):
        player = self._point(obs.get("player", {}))
        px, py = player["x"], player["y"]
        rows = obs.get("rivet_rows", [])
        if not isinstance(rows, list):
            rows = []
        normalized = [self._number(v) for v in rows[:4]]
        while len(normalized) < 4:
            normalized.append(18)
        row = next((i for i, value in enumerate(normalized) if value < 18), 3)
        target_y = (176, 148, 120, 92)[row]
        side = "right" if row % 2 == 0 else "left"
        target_x = 104 if side == "right" else 40
        if abs(py - target_y) > 8:
            action = "up" if py > target_y else "down"
            return {
                "command": "reach_rivet_row",
                "action": action,
                "target_side": side,
                "target_band": self._vertical_band(target_y),
                "alignment": "vertical_approach",
                "target_index": row,
                "target_x": target_x,
            }
        if abs(px - target_x) > 7:
            action = "right" if target_x > px else "left"
            return {
                "command": "clear_rivets",
                "action": action,
                "target_side": side,
                "target_band": self._vertical_band(target_y),
                "alignment": self._side(target_x - px),
                "target_index": row,
                "target_x": target_x,
            }
        return {
            "command": "clear_rivets",
            "action": "right" if side == "right" else "left",
            "target_side": side,
            "target_band": self._vertical_band(target_y),
            "alignment": "on_rivet_side",
            "target_index": row,
            "target_x": target_x,
        }

    def _hazard_facts(self, obs):
        player = self._point(obs.get("player", {}))
        px, py = player["x"], player["y"]
        candidates = []
        for raw in obs.get("hazards", []) if isinstance(obs.get("hazards", []), list) else []:
            hazard = self._point(raw)
            if not hazard["visible"]:
                continue
            hx, hy = hazard["x"], hazard["y"]
            dx, dy = hazard["dx"], hazard["dy"]
            horizontal = hx - px
            same_band = abs(hy - py) <= 26
            horizontal_close = abs(horizontal) <= 62
            closing = same_band and horizontal_close and (
                (horizontal < 0 and dx > 0)
                or (horizontal > 0 and dx < 0)
                or (abs(dx) <= 1 and dy > 0 and hy <= py + 10)
            )
            if not closing:
                continue
            distance = abs(horizontal)
            urgency = "immediate" if distance <= 44 else "soon" if distance <= 62 else "watch"
            candidates.append((urgency, distance, hx, hy, dx, dy))
        candidates.sort(key=lambda item: item[1])
        if not candidates:
            return {
                "present": False,
                "urgency": "none",
                "row_relation": "different_band",
                "closing": False,
                "safe_jump_direction": "none",
                "visible_count": 0,
                "items": [],
            }

        urgency, distance, hx, hy, dx, dy = candidates[0]
        # Prefer the side with more landing room.  At the right edge this
        # deliberately jumps left into the open screen rather than running
        # farther into the wall.
        left_room = px
        right_room = 112 - px
        if left_room > right_room + 8:
            jump_side = "left"
        elif right_room > left_room + 8:
            jump_side = "right"
        else:
            jump_side = "right" if hx < px else "left"
        return {
            "present": True,
            "urgency": urgency,
            "row_relation": "same_band",
            "closing": True,
            "safe_jump_direction": jump_side,
            "visible_count": len(candidates),
            "items": [
                {
                    "position": "left" if hx < px else "right" if hx > px else "overlapping",
                    "vertical_motion": "down" if dy > 0 else "up" if dy < 0 else "level",
                    "horizontal_motion": "right" if dx > 0 else "left" if dx < 0 else "still",
                    "urgency": urgency,
                }
            ],
        }

    def _facts(self, obs):
        self._sync_phase(obs)
        stage = str(obs.get("stage", "unknown"))
        player = self._point(obs.get("player", {}))
        if stage == "ramp":
            route = self._ramp_plan(obs)
        elif stage == "rivet":
            route = self._rivet_plan(obs)
        else:
            route = {
                "command": "wait_for_stage",
                "action": "noop",
                "target_side": "none",
                "target_band": "none",
                "alignment": "none",
                "target_index": -1,
                "target_x": 0,
            }
        hazard = self._hazard_facts(obs)
        raw = obs.get("raw_ram", [])
        jump_active = isinstance(raw, list) and len(raw) > 5 and self._number(raw[5]) != 0
        return {
            "stage": stage,
            "player": player,
            "route": route,
            "hazard": hazard,
            "jump_active": jump_active,
            "lives": self._number(obs.get("lives_remaining")),
            "screens_completed": self._number(obs.get("screens_completed")),
            "rivets_removed": self._number(obs.get("rivets_removed")),
            "bonus": self._number(obs.get("current_bonus")),
        }

    def _fallback_action(self, obs, facts):
        stage = facts["stage"]
        if bool(obs.get("terminated")) or bool(obs.get("truncated")) or stage == "game_over":
            return "noop"

        if stage == "idle":
            if self.start_attempts < 3:
                self.start_attempts += 1
                return "fire"
            return "noop"

        if stage == "transition":
            if facts["lives"] <= 0:
                return "noop"
            tick = self._number(obs.get("tick"))
            started = self.transition_started if self.transition_started is not None else tick
            if not self.restart_sent and tick - started >= 45:
                self.restart_sent = True
                return "fire"
            return "noop"

        hazard = facts["hazard"]
        if hazard["urgency"] in ("immediate", "soon") and not facts["jump_active"]:
            return _DIRECTIONAL_FIRE.get(hazard["safe_jump_direction"], "fire")
        if facts["jump_active"] and self.last_posted in ("leftfire", "rightfire"):
            return self.last_posted
        return facts["route"]["action"]

    @staticmethod
    def _probability(answer, option):
        probabilities = getattr(answer, "probabilities", None)
        if not isinstance(probabilities, dict):
            return 0.0
        try:
            return float(probabilities.get(option, 0.0))
        except (TypeError, ValueError):
            return 0.0

    def _jev_state(self, facts):
        p = facts["player"]
        return {
            "game": "Atari 2600 Donkey Kong NTSC Atari 2600 port",
            "stage": facts["stage"],
            "player": {
                "horizontal_position": (
                    "left_edge" if p["x"] < 48 else "right_edge" if p["x"] > 96 else "middle"
                ),
                "vertical_band": self._vertical_band(p["y"]),
                "motion": self._motion(p["dx"], p["dy"]),
                "jump_active": facts["jump_active"],
            },
            "route": {
                "command": facts["route"]["command"],
                "target_band": facts["route"]["target_band"],
                "target_side": facts["route"]["target_side"],
                "alignment": facts["route"]["alignment"],
                "alternate_ladder_available": facts["stage"] == "ramp" and facts["route"]["target_index"] in (1, 2, 3, 5),
            },
            "hazard": facts["hazard"],
            "progress": {
                "lives_remaining": facts["lives"],
                "screens_completed": facts["screens_completed"],
                "rivets_removed": facts["rivets_removed"],
                "bonus_points_remaining": facts["bonus"],
            },
            "last_posted_action": self.last_posted,
        }

    @staticmethod
    def _question(facts):
        stage = facts["stage"]
        hazard = facts["hazard"]
        if stage == "idle":
            options = [
                ("start", "Press FIRE to leave the initial idle screen."),
                ("wait", "Keep the joystick neutral; do not start yet."),
            ]
        elif stage == "transition":
            options = [
                ("restart", "If a spare Mario remains and the transition is ready, press FIRE once."),
                ("wait", "Release the joystick during the death or screen transition."),
            ]
        else:
            nominal = "climb_target" if facts["route"]["command"] in ("climb_target", "reach_rivet_row") else "follow_route"
            if hazard["urgency"] in ("immediate", "soon"):
                preferred = "jump_" + hazard["safe_jump_direction"]
                options = [
                    (preferred, "Press the directional FIRE action toward the safe landing side now."),
                    ("jump_left", "Press LEFTFIRE early to clear the closing hazard."),
                    ("jump_right", "Press RIGHTFIRE early to clear the closing hazard."),
                    (nominal, "Continue the computed ladder or rivet route without FIRE."),
                ]
            else:
                options = [
                    (nominal, "Follow the computed target; code supplies the exact legal joystick direction."),
                    ("jump_left", "Use LEFTFIRE only for a real closing hazard or a known gap."),
                    ("jump_right", "Use RIGHTFIRE only for a real closing hazard or a known gap."),
                    ("hold_safe", "Release movement briefly while the route remains safe."),
                ]
        return Choice(
            instructions=(
                "Choose exactly one tactic for the supplied normalized observation. "
                "The route target, hazard urgency, and legal action mapping were computed by code. "
                "Prefer an early directional jump only when the hazard facts say it is closing; "
                "otherwise preserve upward progress. Return only one listed option."
            ),
            criteria={key: description for key, description in options},
        )

    def _bridge(self, facts, answer, fallback):
        value = getattr(answer, "value", None)
        choice = value if isinstance(value, str) else ""
        hazard = facts["hazard"]
        probability = self._probability(answer, choice)

        # This is a hard safety gate, not a model preference: the raw Jev
        # choice cannot suppress an imminent jump or send FIRE into a gap.
        if hazard["urgency"] in ("immediate", "soon") and not facts["jump_active"]:
            if choice in ("jump_left", "jump_right"):
                chosen = choice.rsplit("_", 1)[-1]
                if chosen == hazard["safe_jump_direction"] or probability >= 0.72:
                    return _DIRECTIONAL_FIRE[chosen]
            return fallback

        if choice in ("jump_left", "jump_right") and probability >= 0.66:
            # A high-confidence unsolicited jump is allowed only in a state
            # where code already sees a tactical reason to escape.
            if hazard["urgency"] == "watch":
                chosen = choice.rsplit("_", 1)[-1]
                return _DIRECTIONAL_FIRE[chosen]
        # Route tactics deliberately collapse to the deterministic bridge.
        # Jev recognizes the situation; code owns geometry and exact timing.
        return fallback

    @staticmethod
    def _valid_action(game, action):
        try:
            names = set(game.action_spec.names)
        except Exception:
            names = set()
        if action in names:
            return action
        try:
            return game.action_spec.noop
        except Exception:
            return "noop"

    def run_episode(self, game):
        ctx = self.ctx
        use_jev = getattr(ctx, "system1", True) is not None
        while not ctx.done and not getattr(game, "done", False):
            obs = game.observe()
            if not isinstance(obs, dict):
                obs = {}
            facts = self._facts(obs)
            fallback = self._fallback_action(obs, facts)
            action = fallback

            if use_jev:
                state = self._jev_state(facts)
                question = self._question(facts)
                try:
                    answers = ctx.decide(state, {"tactic": question}, timeout_s=0.5)
                    answer = answers.get("tactic") if isinstance(answers, dict) else None
                    if answer is not None:
                        action = self._bridge(facts, answer, fallback)
                except (BudgetExceeded, ModelUnavailable):
                    raise
                except (TimeoutError, RuntimeError):
                    action = fallback
                    if not ctx.done and not getattr(game, "done", False):
                        ctx.act(self._valid_action(game, action))
                        self.last_posted = self._valid_action(game, action)
                        ctx.wait_for_tick()
                    continue

            if ctx.done or getattr(game, "done", False):
                break
            action = self._valid_action(game, action)
            ctx.act(action)
            self.last_posted = action
            if not use_jev and not ctx.done and not getattr(game, "done", False):
                ctx.wait_for_tick()


def build(ctx):
    return Harness(ctx)
