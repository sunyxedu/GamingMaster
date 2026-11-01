"""Tier-2 refinement of the Jev-only controller.

The parent asks Jev for the next action from the raw observation.  The only
additional mechanism here is an executor guard: a FLAP answer is posted for
one simulation tick and then the held action is released to the game's noop.
This prevents real-time latency from turning a one-frame impulse into a
multi-frame held flap.
"""
from __future__ import annotations

from metaharness.core.contract import BudgetExceeded, ModelUnavailable
from metaharness.models.jev import Choice


class Harness:
    def __init__(self, ctx):
        self.ctx = ctx

    @staticmethod
    def _legal_answer(answer, spec):
        """Map an arbitrary typed-answer value to a legal action or noop."""
        names = {str(name).upper(): name for name in spec.names}
        raw = getattr(answer, "value", None)
        if isinstance(raw, str):
            return names.get(raw.upper(), spec.noop)
        return spec.noop

    def run_episode(self, game):
        ctx = self.ctx
        if ctx.system1 is None:
            return

        spec = game.action_spec
        question = {
            "act": Choice(
                instructions=f"{game.describe()} Which action is best right now?",
                criteria={
                    name: spec.descriptions.get(name, name)
                    for name in spec.names
                },
            )
        }

        while not ctx.done:
            try:
                answers = ctx.decide(game.observe(), question)
            except (BudgetExceeded, ModelUnavailable):
                raise
            except (TimeoutError, RuntimeError):
                # The held action is already bounded: after every FLAP below,
                # it is released before another request is started.
                if not ctx.done:
                    ctx.wait_for_tick()
                continue

            if ctx.done:
                break

            answer = answers.get("act") if isinstance(answers, dict) else None
            action = self._legal_answer(answer, spec)
            ctx.act(action)

            if action == spec.noop:
                continue

            # A model FLAP is a one-frame command.  Waiting for that frame and
            # immediately posting noop is the sole refinement over the parent;
            # otherwise realtime ticks repeat FLAP during the next Jev call.
            if ctx.done:
                break
            ctx.wait_for_tick()
            if ctx.done:
                break
            ctx.act(spec.noop)


def build(ctx):
    return Harness(ctx)
