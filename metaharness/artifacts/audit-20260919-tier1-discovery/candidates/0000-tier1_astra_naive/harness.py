"""Tier-1 seed: Astra sees the current observation and picks one action per call.
No memory, no hypothesis tracking — the floor the meta-harness must beat.
Game-agnostic on purpose: only game.describe(), game.action_spec and a JSON
rendering of the observation are used."""
from __future__ import annotations

from metaharness.baselines._generic import obs_text, parse_action


class Harness:
    def __init__(self, ctx):
        self.ctx = ctx

    def run_episode(self, game):
        ctx = self.ctx
        session = None
        obs = game.observe()
        names = list(game.action_spec.names)
        while not ctx.done:
            prompt = (f"{game.describe()}\nActions:\n{game.action_spec.describe()}\n\n"
                      f"Observation:\n{obs_text(obs)}\n\n"
                      "Reply with exactly one action (name, plus arguments if it needs them) and nothing else.")
            try:
                res = ctx.think(prompt, session_id=session)
            except RuntimeError:
                break
            session = res.session_id
            obs = ctx.act(parse_action(res.text, names, default=names[0]))


def build(ctx):
    return Harness(ctx)
