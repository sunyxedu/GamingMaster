"""Refine the Jev-only parent with a one-frame, neutral-rearmed actuator.

The model, raw observation, and single action question are unchanged. A chosen
movement is a joystick pulse, not a direction to hold during the next request.
"""
from __future__ import annotations

from metaharness.core.contract import BudgetExceeded, ModelUnavailable
from metaharness.models.jev import Choice


class Harness:
    def __init__(self, ctx):
        self.ctx = ctx

    def run_episode(self, game):
        ctx = self.ctx
        if ctx.system1 is None:
            return
        spec = game.action_spec
        names = tuple(spec.names)
        noop = spec.noop or "NOOP"
        q = {"act": Choice(
            instructions=f"{game.describe()} Which action is best right now?",
            criteria={n: spec.descriptions.get(n, n) for n in names})}

        while not ctx.done:
            try:
                answers = ctx.decide(game.observe(), q)
            except (BudgetExceeded, ModelUnavailable):
                raise
            except (TimeoutError, RuntimeError):
                # Every previous pulse was released before entering inference.
                ctx.wait_for_tick()
                continue

            answer = answers.get("act") if isinstance(answers, dict) else None
            action = getattr(answer, "value", None)
            if not isinstance(action, str) or action not in names:
                action = noop

            ctx.act(action)
            if action != noop:
                # Let the runner consume exactly one directional frame, then
                # consume a centered frame before requesting another judgment.
                ctx.wait_for_tick()
                if ctx.done:
                    return
                ctx.act(noop)
                ctx.wait_for_tick()
                if not ctx.done:
                    ctx.log("hop_pulse", action=action, release=noop)


def build(ctx):
    return Harness(ctx)
