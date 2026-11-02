"""Seed: Astra sees the current observation and picks one action per call. No memory, no
hypothesis tracking — the floor the meta-harness must beat. On a real-time game it also shows why a
slow model cannot drive the game alone: most ticks run on a stale action. Game-agnostic on purpose:
only game.describe(), game.action_spec and a JSON rendering of the observation are used."""
from __future__ import annotations

from metaharness.baselines._generic import obs_text, parse_action
from metaharness.core.contract import BudgetExceeded, ModelUnavailable


class Harness:
    def __init__(self, ctx):
        self.ctx = ctx

    def run_episode(self, game):
        ctx = self.ctx
        if ctx.system2 is None:              # evaluated without that model: nothing to do
            return
        session = None
        spec = game.action_spec
        names = list(spec.names)
        while not ctx.done:
            prompt = (f"{game.describe()}\nActions:\n{spec.describe()}\n\n"
                      f"Observation:\n{obs_text(game.observe())}\n\n"
                      "Reply with exactly one action (name, plus arguments if it needs them) and nothing else.")
            try:
                res = ctx.think(prompt, session_id=session)
            except (BudgetExceeded, ModelUnavailable):
                break                      # budget or episode over / backend gone: the run stops
            except (TimeoutError, RuntimeError):
                continue                   # a failed call: ask again
            session = res.session_id
            ctx.act(parse_action(res.text, names, default=spec.noop or names[0]))


def build(ctx):
    return Harness(ctx)
