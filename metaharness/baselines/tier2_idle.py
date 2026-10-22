"""Tier-2 seed #3: do nothing. The game runs on its noop for the whole episode.

Not a contender: the reference row for what idling scores (survival credit, passive
points), so the leaderboard shows whether a harness earns anything beyond it."""
from __future__ import annotations


class Harness:
    def __init__(self, ctx):
        self.ctx = ctx

    def run_episode(self, game):
        while not self.ctx.done:
            self.ctx.wait_for_tick()


def build(ctx):
    return Harness(ctx)
