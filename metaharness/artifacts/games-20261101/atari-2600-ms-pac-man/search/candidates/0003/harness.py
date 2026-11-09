"""Jev-only Ms. Pac-Man controller with explicit motion-progress feedback.

The parent controller asked Jev to choose from the complete ALE action set while
passing the very large raw observation.  This version keeps the same fast,
per-decision architecture, but exposes a small navigation state and removes a
direction after repeated no-progress observations.
"""
from __future__ import annotations

from metaharness.core.contract import BudgetExceeded, ModelUnavailable
from metaharness.models.jev import Choice


CARDINALS = ("up", "right", "left", "down")


def _as_dict(value):
    return value if isinstance(value, dict) else {}


def _number(value, default=0):
    # Keep the state strictly JSON-friendly even if a stand-in observation is
    # incomplete or contains an unexpected value.
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return value
    return default


def _text(value, default=""):
    return value if isinstance(value, str) else default


def _xy(obj):
    obj = _as_dict(obj)
    xy = obj.get("xy")
    if isinstance(xy, (list, tuple)) and len(xy) >= 2:
        return [_number(xy[0]), _number(xy[1])]
    return [_number(obj.get("x")), _number(obj.get("y"))]


def _visible(obj):
    obj = _as_dict(obj)
    return bool(obj.get("visible", True))


def _direction(obj):
    direction = _text(_as_dict(obj).get("direction"), "")
    return direction if direction in CARDINALS else "unknown"


def _distance_band(player_xy, ghost_xy):
    """Translate coordinates into a small semantic feature for Jev."""
    dx = abs(_number(ghost_xy[0]) - _number(player_xy[0]))
    dy = abs(_number(ghost_xy[1]) - _number(player_xy[1]))
    d = dx + dy
    if d <= 14:
        return "contact_range"
    if d <= 36:
        return "near"
    if d <= 80:
        return "medium"
    return "far"


def _relation(player_xy, ghost_xy):
    dx = _number(ghost_xy[0]) - _number(player_xy[0])
    dy = _number(ghost_xy[1]) - _number(player_xy[1])
    if abs(dx) <= 10 and abs(dy) <= 10:
        return "overlapping"
    if abs(dx) <= 10:
        return "same_column"
    if abs(dy) <= 10:
        return "same_row"
    if abs(dx) > abs(dy):
        return "left_of_player" if dx < 0 else "right_of_player"
    return "above_player" if dy < 0 else "below_player"


def _approach(player_xy, ghost):
    """Use supplied velocity as a coarse label; never ask Jev to calculate it."""
    ghost_xy = _xy(ghost)
    gx = _number(ghost_xy[0]) - _number(player_xy[0])
    gy = _number(ghost_xy[1]) - _number(player_xy[1])
    vx = _number(_as_dict(ghost).get("dx"))
    vy = _number(_as_dict(ghost).get("dy"))
    dot = gx * vx + gy * vy
    if dot > 0:
        return "moving_toward_player"
    if dot < 0:
        return "moving_away_from_player"
    return "unclear"


def _actor_summary(obj, include_colour=False):
    obj = _as_dict(obj)
    xy = _xy(obj)
    result = {
        "visible": _visible(obj),
        "xy": xy,
        "dx": _number(obj.get("dx")),
        "dy": _number(obj.get("dy")),
    }
    if include_colour:
        result["colour"] = _text(obj.get("colour"), "unknown")
    return result


class Harness:
    def __init__(self, ctx):
        self.ctx = ctx
        self.previous_xy = None
        self.previous_lives = None
        self.stalled_decisions = 0
        self.started_moving = False
        self.last_action = "left"

    def _navigation_state(self, obs):
        obs = _as_dict(obs)
        player = _as_dict(obs.get("player"))
        xy = _xy(player)
        lives = _number(obs.get("lives"), 0)
        current_direction = _direction(player)

        life_changed = (
            self.previous_lives is not None and lives != self.previous_lives
        )
        moved_since_decision = (
            self.previous_xy is not None and xy != self.previous_xy
        )

        # The ROM has an automatic opening sequence.  Do not call the opening
        # stationary state a wall until a collectible has actually been eaten
        # or the player has visibly moved.
        if _number(obs.get("score"), 0) > 0 or moved_since_decision:
            self.started_moving = True

        if life_changed or not self.started_moving or moved_since_decision:
            self.stalled_decisions = 0
        elif self.last_action in CARDINALS:
            self.stalled_decisions += 1

        self.previous_xy = xy
        self.previous_lives = lives

        blocked_action = None
        if self.stalled_decisions >= 2 and self.last_action in CARDINALS:
            blocked_action = self.last_action

        return {
            "last_posted_action": self.last_action,
            "player_direction": current_direction,
            "motion": "moving" if moved_since_decision else "stationary",
            "stalled_decisions": int(self.stalled_decisions),
            "blocked_action": blocked_action,
            "opening_sequence": not self.started_moving,
            "life_changed": life_changed,
        }

    def _state(self, obs, navigation):
        obs = _as_dict(obs)
        player = _as_dict(obs.get("player"))
        player_xy = _xy(player)
        current_direction = _direction(player)

        ghosts = []
        for ghost in obs.get("ghosts", []) if isinstance(obs.get("ghosts"), list) else []:
            if not isinstance(ghost, dict):
                continue
            summary = _actor_summary(ghost, include_colour=True)
            if summary["visible"]:
                summary["distance_band"] = _distance_band(player_xy, summary["xy"])
                summary["relation"] = _relation(player_xy, summary["xy"])
                summary["approach"] = _approach(player_xy, ghost)
                ghosts.append(summary)

        power_pills = []
        raw_power_pills = obs.get("power_pills", [])
        if isinstance(raw_power_pills, list):
            for pill in raw_power_pills:
                if isinstance(pill, dict) and _visible(pill):
                    power_pills.append(_xy(pill))

        fruit = _as_dict(obs.get("fruit"))
        fruit_summary = {
            "visible": _visible(fruit),
            "xy": _xy(fruit),
            "kind": _text(fruit.get("kind"), "unknown"),
        }

        visible_collectibles = obs.get("visible_collectibles")
        if isinstance(visible_collectibles, list):
            remaining_pills = len(visible_collectibles)
        else:
            remaining_pills = int(_number(visible_collectibles, 0))

        # Deliberately omit raw RAM and the full pill list: they are useful for
        # code, but the parent trace shows that this large undigested state did
        # not give Jev a usable local movement interface.
        return {
            "game": "Atari 2600 Ms. Pac-Man; cardinal joystick control",
            "objective": "eat dots while surviving; keep moving and prefer a safe escape over fruit or ghost chasing",
            "episode": {
                "frame": int(_number(obs.get("frame"), 0)),
                "score": _number(obs.get("score"), 0),
                "reward_delta": _number(obs.get("reward_delta"), 0),
                "lives": int(_number(obs.get("lives"), 0)),
                "deaths": int(_number(obs.get("deaths"), 0)),
                "mode": int(_number(obs.get("mode"), 0)),
                "game_over": bool(obs.get("game_over", False)),
                "remaining_pills": remaining_pills,
                "mazes_cleared": int(_number(obs.get("mazes_cleared"), 0)),
            },
            "player": {
                "xy": player_xy,
                "direction": current_direction,
                "dx": _number(player.get("dx")),
                "dy": _number(player.get("dy")),
            },
            "ghosts": ghosts,
            "power_pills": power_pills,
            "fruit": fruit_summary,
            "navigation": navigation,
        }

    def _choices(self, navigation, player_direction):
        blocked = navigation.get("blocked_action")
        # Preserve momentum when it is making progress.  Once the same action
        # has failed to move twice, remove it so Jev must test another corridor.
        ordered = []
        if player_direction in CARDINALS:
            ordered.append(player_direction)
        if self.last_action in CARDINALS and self.last_action not in ordered:
            ordered.append(self.last_action)
        ordered.extend(action for action in CARDINALS if action not in ordered)
        if blocked in ordered:
            ordered.remove(blocked)
        if not ordered:
            ordered = ["up", "right", "left", "down"]
        return ordered

    def _question(self, game, choices):
        criteria = {
            "up": "Hold the joystick up; choose only when it is a plausible open turn or escape.",
            "right": "Hold the joystick right; choose only when it is a plausible open turn or escape.",
            "left": "Hold the joystick left; choose only when it is a plausible open turn or escape.",
            "down": "Hold the joystick down; choose only when it is a plausible open turn or escape.",
        }
        criteria = {name: criteria[name] for name in choices}
        return Choice(
            instructions=(
                f"{game.describe()}\n"
                "Choose exactly one of the supplied cardinal joystick actions for the next short quantum. "
                "This is continuous Ms. Pac-Man: the current action continues while Jev is thinking. "
                "Never choose noop or a diagonal. Use `navigation.blocked_action` as a hard exclusion: "
                "it is the last direction that produced no position change for at least two decisions. "
                "Keep clearing dots, avoid a visible normal ghost at contact_range or near while it approaches, "
                "and use a power pill or turn away when survival requires it. Do not infer walls from the raw "
                "coordinates; use the motion feedback and the supplied candidate set."
            ),
            criteria=criteria,
        )

    def run_episode(self, game):
        ctx = self.ctx
        if ctx.system1 is None:
            return

        while not ctx.done and not game.done:
            obs = game.observe()
            navigation = self._navigation_state(obs)
            choices = self._choices(navigation, navigation.get("player_direction"))
            question = {"act": self._question(game, choices)}
            try:
                answers = ctx.decide(
                    self._state(obs, navigation),
                    question,
                    timeout_s=5.0,
                )
            except (BudgetExceeded, ModelUnavailable):
                raise
            except (TimeoutError, RuntimeError):
                # The emulator keeps the previous action.  Waiting for one
                # tick avoids a hot retry loop and lets the next state refresh.
                if not ctx.done and not game.done:
                    ctx.wait_for_tick()
                continue

            value = None
            try:
                value = answers["act"].value
            except (AttributeError, KeyError, TypeError):
                value = None

            if value not in choices:
                # Garbage/late stand-in answers cannot crash pre-flight or send
                # a direction that the current question explicitly excluded.
                value = self.last_action if self.last_action in choices else choices[0]
            self.last_action = value
            if not ctx.done and not game.done:
                ctx.act(value)


def build(ctx):
    return Harness(ctx)
