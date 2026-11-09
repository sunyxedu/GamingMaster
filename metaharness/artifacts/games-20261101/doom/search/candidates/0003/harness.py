"""Tier-2 Jev controller with a deterministic anti-stall escape watchdog.

This keeps the parent's one-choice Jev loop intact.  The only game-specific
addition is a small execution guard: if a translational action has made no
3-D progress for several observations, the player is probably pushing against
a wall or a closed route.  A short retreat/turn/diagonal maneuver is then
allowed to override the next few Jev choices so that the policy can see a new
view instead of spending the call budget on the same action.
"""
from __future__ import annotations

import math

from metaharness.core.contract import BudgetExceeded, ModelUnavailable
from metaharness.models.jev import Choice


# A normal Doom movement action advances much farther than this between two
# Jev responses.  Including z avoids calling a lift/elevator stationary.
_PROGRESS_EPS = 8.0
_STALL_LIMIT = 3


class Harness:
    def __init__(self, ctx):
        self.ctx = ctx

    @staticmethod
    def _position(observation):
        """Return a finite player position, or None for an incomplete state."""
        if not isinstance(observation, dict):
            return None
        player = observation.get("player")
        if not isinstance(player, dict):
            return None
        values = []
        for key in ("x", "y", "z"):
            value = player.get(key)
            if isinstance(value, bool):
                return None
            try:
                value = float(value)
            except (TypeError, ValueError):
                return None
            if not math.isfinite(value):
                return None
            values.append(value)
        return tuple(values)

    @staticmethod
    def _moved(previous, current):
        if previous is None or current is None:
            return False
        dx = current[0] - previous[0]
        dy = current[1] - previous[1]
        dz = current[2] - previous[2]
        return math.sqrt(dx * dx + dy * dy + dz * dz) >= _PROGRESS_EPS

    @staticmethod
    def _is_translational(action):
        return action in {
            "forward",
            "backward",
            "strafe_left",
            "strafe_right",
            "forward_left",
            "forward_right",
            "backward_left",
            "backward_right",
            "fire_forward",
            "fire_strafe_left",
            "fire_strafe_right",
        }

    @staticmethod
    def _recovery_sequence(names, side):
        """Build a legal, moving escape sequence for one side.

        Retreat first, rotate the view, then move diagonally.  The fallbacks
        keep the guard useful if a reduced action set is supplied in a test.
        """
        if side > 0:
            diagonal_back = "backward_right"
            turn = "turn_right"
            diagonal_forward = "forward_right"
            strafe = "strafe_right"
        else:
            diagonal_back = "backward_left"
            turn = "turn_left"
            diagonal_forward = "forward_left"
            strafe = "strafe_left"

        candidates = [
            [diagonal_back, turn, diagonal_forward],
            ["backward", turn, strafe],
            ["backward", turn, "forward"],
            [strafe, turn, "forward"],
            [turn, "forward"],
            ["backward"],
            [turn],
        ]
        legal = set(names)
        for candidate in candidates:
            sequence = [action for action in candidate if action in legal]
            if sequence:
                return sequence
        return []

    @staticmethod
    def _answer_action(answers, names, default):
        """Extract a typed choice defensively, including stand-in models."""
        if not isinstance(answers, dict):
            return default
        try:
            answer = answers.get("act")
            value = getattr(answer, "value", None)
        except Exception:
            return default
        return value if isinstance(value, str) and value in names else default

    def run_episode(self, game):
        ctx = self.ctx
        if ctx.system1 is None:  # evaluated without System-1: do no model work
            return

        spec = game.action_spec
        names = list(getattr(spec, "names", ()) or ())
        if not names:
            return
        names_set = set(names)
        noop = getattr(spec, "noop", None)
        if noop not in names_set:
            noop = names[0]
        descriptions = getattr(spec, "descriptions", {}) or {}
        q = {
            "act": Choice(
                instructions=(
                    f"{game.describe()} Which legal action is best right now? "
                    "Prefer an action that makes immediate progress while "
                    "preserving a retreat path."
                ),
                criteria={
                    name: descriptions.get(name, name) for name in names
                },
            )
        }

        last_action = noop
        previous_position = None
        stalled = 0
        recovery_side = 1
        recovery_queue = []

        while not ctx.done:
            observation = game.observe()
            position = self._position(observation)
            moved = self._moved(previous_position, position)

            # During a forced maneuver, do not let the detector immediately
            # cancel the sequence; otherwise the first retreat step would
            # repeatedly restart the same recovery.
            if recovery_queue:
                stalled = 0
            elif self._is_translational(last_action):
                if moved:
                    stalled = 0
                else:
                    stalled += 1
            else:
                stalled = 0

            try:
                answers = ctx.decide(observation, q)
            except (BudgetExceeded, ModelUnavailable):
                raise
            except (TimeoutError, RuntimeError):
                # The current action remains held by ASYNC_PLAYER.  Waiting
                # avoids a busy loop and lets the watchdog see the next state.
                ctx.wait_for_tick()
                continue

            action = self._answer_action(answers, names_set, noop)

            if recovery_queue:
                action = recovery_queue.pop(0)
            elif stalled >= _STALL_LIMIT:
                sequence = self._recovery_sequence(names, recovery_side)
                recovery_side *= -1
                if sequence:
                    action = sequence[0]
                    recovery_queue = sequence[1:]
                    stalled = 0
                    ctx.log(
                        "stuck_recovery",
                        action=action,
                        sequence=sequence,
                        position=position,
                        previous_action=last_action,
                    )

            # Every posted value is drawn from the advertised action set.
            if action not in names_set:
                action = noop
            ctx.act(action)
            last_action = action
            previous_position = position


def build(ctx):
    return Harness(ctx)
