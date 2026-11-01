"""Jev-only Asteroids controller with a deterministic fire pulse bridge.

The question remains the parent's one-choice-per-observation controller.  The
only policy change is at the action boundary: a manoeuvre with a legal FIRE
variant gets one frame of FIRE, followed by the same manoeuvre without FIRE.
That makes repeated answers genuine presses instead of a held FIRE level.
"""
from __future__ import annotations

from metaharness.core.contract import BudgetExceeded, ModelUnavailable
from metaharness.models.jev import Choice


# These are deliberately a bridge, not a second policy.  DOWN is excluded:
# repeating mode-0 hyperspace merely because a model selected a defensive
# action is not a useful interpretation of "add fire".
_FIRE_VARIANT = {
    "UP": "UPFIRE",
    "RIGHT": "RIGHTFIRE",
    "LEFT": "LEFTFIRE",
    "UPRIGHT": "UPRIGHTFIRE",
    "UPLEFT": "UPLEFTFIRE",
}
_RELEASE_VARIANT = {
    "FIRE": "NOOP",
    "UPFIRE": "UP",
    "RIGHTFIRE": "RIGHT",
    "LEFTFIRE": "LEFT",
    "UPRIGHTFIRE": "UPRIGHT",
    "UPLEFTFIRE": "UPLEFT",
}


class Harness:
    def __init__(self, ctx):
        self.ctx = ctx

    @staticmethod
    def _choice_value(answer, names, noop):
        """Turn even a garbage stand-in answer into one legal action name."""
        value = getattr(answer, "value", None)
        if not isinstance(value, str) or value not in names:
            return noop if isinstance(noop, str) and noop in names else names[0]
        return value

    @staticmethod
    def _pulse_actions(selected, names, noop):
        """Return (pressed, released); only add the one tested change."""
        if selected in _FIRE_VARIANT:
            pressed = _FIRE_VARIANT[selected]
            if pressed in names:
                return pressed, selected
        if selected in _RELEASE_VARIANT:
            pressed = selected
            released = _RELEASE_VARIANT[selected]
            if pressed in names:
                if released not in names:
                    released = noop
                return pressed, released
        # NOOP, DOWN, and any unknown action retain the parent's semantics.
        return selected, selected

    def run_episode(self, game):
        ctx = self.ctx
        if ctx.system1 is None:
            return

        spec = game.action_spec
        names = tuple(spec.names)
        noop = spec.noop if isinstance(spec.noop, str) else "NOOP"
        if noop not in names:
            noop = names[0]
        descriptions = getattr(spec, "descriptions", {}) or {}
        question = {
            "act": Choice(
                instructions=(
                    game.describe()
                    + "\nChoose the best legal action for the current observation. "
                      "The controller will, for ordinary movement actions, press "
                      "the matching FIRE variant for exactly one native frame and "
                      "then release FIRE; DOWN is not augmented."
                ),
                criteria={
                    name: descriptions.get(name, name)
                    for name in names
                },
            )
        }

        while not ctx.done:
            try:
                answer = ctx.decide(game.observe(), question)
            except (BudgetExceeded, ModelUnavailable):
                raise
            except (TimeoutError, RuntimeError):
                # The preceding pulse has already been released.  Let the game
                # advance under that safe current action while retrying.
                try:
                    ctx.wait_for_tick()
                except (BudgetExceeded, ModelUnavailable):
                    raise
                continue

            raw_answer = answer.get("act") if isinstance(answer, dict) else None
            selected = self._choice_value(raw_answer, names, noop)
            pressed, released = self._pulse_actions(selected, names, noop)
            try:
                ctx.act(pressed)
                if pressed != released:
                    # A real tick is the release boundary.  Without it, two
                    # adjacent ctx.act calls could leave FIRE continuously
                    # held and would not test the cartridge's press semantics.
                    ctx.wait_for_tick()
                    if ctx.done or game.done:
                        break
                    ctx.act(released)
                else:
                    # Keep the same fallback behavior as the parent for
                    # NOOP/defense selections.
                    pass
            except (BudgetExceeded, ModelUnavailable):
                raise


def build(ctx):
    return Harness(ctx)
