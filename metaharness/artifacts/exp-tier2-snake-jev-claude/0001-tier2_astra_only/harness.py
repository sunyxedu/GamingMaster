"""Tier-2 seed #2: Astra alone, one call per decision. Demonstrates why a slow
model cannot drive a real-time game: most ticks run on a stale action.
"""
from __future__ import annotations


class Harness:
    def __init__(self, ctx):
        self.ctx = ctx

    def run_episode(self, game):
        ctx = self.ctx
        session = None
        while not ctx.done:
            obs = game.observe()
            prompt = (f"{game.describe()}\nActions:\n{game.action_spec.describe()}\n\n"
                      f"Observation:\n{obs['grid']}\nhead={obs['head']} heading={obs['heading']} "
                      f"food={obs['food']} danger={obs['danger']}\n"
                      "Reply with exactly one action name and nothing else.")
            res = ctx.think(prompt, session_id=session)
            session = res.session_id
            a = res.text.strip().split()[0].strip("`'\".").lower() if res.text.strip() else "keep"
            if a not in game.action_spec.names:
                a = "keep"
            ctx.act(a)


def build(ctx):
    return Harness(ctx)
