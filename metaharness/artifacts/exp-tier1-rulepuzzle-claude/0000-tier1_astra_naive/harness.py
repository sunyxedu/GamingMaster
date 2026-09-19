"""Tier-1 seed: Astra sees the current grid and picks one action per call.
No memory, no hypothesis tracking — the floor the meta-harness must beat."""
from __future__ import annotations

import re


class Harness:
    def __init__(self, ctx):
        self.ctx = ctx

    def run_episode(self, game):
        ctx = self.ctx
        session = None
        obs = game.observe()
        while not ctx.done:
            grid = "\n".join(" ".join(str(c) for c in row) for row in obs["grid"])
            prompt = (f"{game.describe()}\nLegend (digit=colour): {obs['legend']}\n"
                      f"Level {obs['level']}/{obs['levels_total']}, actions left this level: "
                      f"{obs['actions_left_this_level']}\nGrid:\n{grid}\n\n"
                      "Reply with exactly one action name (ACTION1..ACTION6).")
            try:
                res = ctx.think(prompt, session_id=session)
            except RuntimeError:
                break
            session = res.session_id
            m = re.search(r"ACTION[1-6]", res.text or "")
            obs = ctx.act(m.group(0) if m else "ACTION1")


def build(ctx):
    return Harness(ctx)
