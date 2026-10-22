"""Tier-2 seed #2: Astra alone, one call per decision. Demonstrates why a slow
model cannot drive a real-time game: most ticks run on a stale action.
Game-agnostic (JSON observation, action names from the spec)."""
from __future__ import annotations

from metaharness.baselines._generic import obs_text, parse_action
from metaharness.core.contract import BudgetExceeded


class Harness:
    def __init__(self, ctx):
        self.ctx = ctx

    def run_episode(self, game):
        ctx = self.ctx
        if ctx.system2 is None:              # evaluated without that model: nothing to do
            return
        session = None
        names = list(game.action_spec.names)
        while not ctx.done:
            prompt = (f"{game.describe()}\nActions:\n{game.action_spec.describe()}\n\n"
                      f"Observation:\n{obs_text(game.observe())}\n"
                      "Reply with exactly one action name and nothing else.")
            try:
                res = ctx.think(prompt, session_id=session)
            except BudgetExceeded:
                break                      # budget or episode over
            except (TimeoutError, RuntimeError):
                continue                   # a failed call: ask again
            session = res.session_id
            ctx.act(parse_action(res.text, names, default=game.action_spec.noop or names[0]))


def build(ctx):
    return Harness(ctx)
