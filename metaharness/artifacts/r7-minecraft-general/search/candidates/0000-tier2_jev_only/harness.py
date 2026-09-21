"""Tier-2 seed #1: Jev alone, one choice question per tick, raw observation as state.

No Astra at all. Establishes the System-1 floor and the latency profile.
Game-agnostic: the options are game.action_spec.names with their descriptions.
"""
from __future__ import annotations

from metaharness.models.jev import Choice


class Harness:
    def __init__(self, ctx):
        self.ctx = ctx

    def run_episode(self, game):
        ctx = self.ctx
        spec = game.action_spec
        q = {"act": Choice(
            instructions=f"{game.describe()} Which action is best right now?",
            criteria={n: spec.descriptions.get(n, n) for n in spec.names})}
        while not ctx.done:
            ans = ctx.decide(game.observe(), q)
            ctx.act(ans["act"].value)


def build(ctx):
    return Harness(ctx)
