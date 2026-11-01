"""Tier-2 seed #3: do nothing. The game runs on its noop for the whole episode.

The reference row for what idling scores (survival credit, passive points), so the
leaderboard shows whether a harness earns anything beyond it. It is ranked like any
candidate: if it wins, no proposal has beaten doing nothing."""
from __future__ import annotations


class Harness:
    def __init__(self, ctx):
        self.ctx = ctx

    def run_episode(self, game):
        if not getattr(game, "tick_s", None):       # not a real-time game: nothing to wait for
            return
        while not self.ctx.done:
            self.ctx.wait_for_tick()


def build(ctx):
    return Harness(ctx)
