"""Tier-2 seed #1: Jev alone, one choice question per tick, raw observation as state.

No Astra at all. Establishes the System-1 floor and the latency profile.
"""
from __future__ import annotations

from metaharness.models.jev import Choice


class Harness:
    def __init__(self, ctx):
        self.ctx = ctx

    def run_episode(self, game):
        ctx = self.ctx
        names = [n for n in game.action_spec.names if n != "keep"]
        q = {"move": Choice(
            instructions="Which heading should the snake take now to eat the food without dying?",
            criteria={n: game.action_spec.descriptions.get(n, n) for n in names})}
        while not ctx.done:
            obs = game.observe()
            ans = ctx.decide(obs, q)
            ctx.act(ans["move"].value)


def build(ctx):
    return Harness(ctx)
