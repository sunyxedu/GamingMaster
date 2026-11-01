"""Jev controller with an explicit one-frame joystick debounce.

The parent asks Jev for the next action on every available decision.  This
version changes only the action bridge: a directional choice is allowed to
reach exactly one emulator tick, then NOOP is posted before the next slow
judgment.  That makes the literal one-hop semantics true even while Jev is
thinking.
"""
from __future__ import annotations

from metaharness.core.contract import BudgetExceeded, ModelUnavailable
from metaharness.models.jev import Choice


class Harness:
    def __init__(self, ctx):
        self.ctx = ctx

    def _action_from_answer(self, answer, names, noop):
        """Extract only a legal action; stand-in or malformed answers wait."""
        value = getattr(answer, "value", None)
        if isinstance(value, str) and value in names:
            return value
        return noop

    def _post_one_hop(self, action, names, noop):
        """Post one action frame and release the joystick before reasoning."""
        ctx = self.ctx
        if action not in names:
            action = noop
        ctx.act(action)
        if action == noop or ctx.done:
            return

        # A real-time tick consumes the directional press.  Releasing only
        # after that tick avoids replacing the press before the emulator sees
        # it, while preventing the next Jev latency interval from holding it.
        ctx.wait_for_tick()
        if not ctx.done:
            ctx.act(noop)

    def run_episode(self, game):
        ctx = self.ctx
        if ctx.system1 is None:
            return

        spec = game.action_spec
        names = tuple(spec.names)
        noop = getattr(spec, "noop", "NOOP")
        if not isinstance(noop, str) or noop not in names:
            noop = "NOOP" if "NOOP" in names else (names[0] if names else "NOOP")
        criteria = {
            name: spec.descriptions.get(name, name) for name in names
        }
        question = {
            "act": Choice(
                instructions=(
                    f"{game.describe()} Which single action should be applied "
                    "on the next Atari frame? Choose a directional hop only "
                    "when it is preferable to waiting; actions are literal "
                    "strings and the controller releases a hop after one "
                    "frame."
                ),
                criteria=criteria,
            )
        }

        while not ctx.done:
            try:
                answers = ctx.decide(game.observe(), question)
            except (BudgetExceeded, ModelUnavailable):
                raise
            except (TimeoutError, RuntimeError):
                # The debounce bridge leaves NOOP posted after every
                # directional answer, so a failed/late judgment has a safe
                # executable fallback.  Let one live frame pass and retry.
                if not ctx.done:
                    ctx.wait_for_tick()
                continue

            answer = answers.get("act") if isinstance(answers, dict) else None
            action = self._action_from_answer(answer, names, noop)
            # Keep game errors from being mistaken for model failures.
            self._post_one_hop(action, names, noop)


def build(ctx):
    return Harness(ctx)
