"""Reference tier-1 harness (test fixture): hypothesis-tracking exploration.

  * Phase A (scripted, no model calls): try each of ACTION1..6 once from the
    start position, record (action -> displacement) to learn the movement map.
  * Phase B: show Astra an ASCII map + the learned action table + the level
    history, ask for a *sequence* of up to 8 actions; execute until an action
    has no effect twice in a row (hypothesis refuted) or the level changes.
  * ``ctx.memory['discovery_notes']`` accumulates what worked across seeds.
"""
from __future__ import annotations

import re

ACTIONS = ["ACTION1", "ACTION2", "ACTION3", "ACTION4", "ACTION5", "ACTION6"]


def _agent_pos(obs):
    for y, row in enumerate(obs["grid"]):
        for x, c in enumerate(row):
            if c == 9:
                return (x, y)
    return None


def _ascii(obs):
    return "\n".join("".join(str(c) for c in row) for row in obs["grid"])


class Harness:
    def __init__(self, ctx):
        self.ctx = ctx
        self.effects: dict[str, str] = {}
        self.session = None

    def _probe(self, game, obs):
        for a in ACTIONS:
            if self.ctx.done:
                break
            p0 = _agent_pos(obs)
            lvl = obs["level"]
            obs = self.ctx.act(a)
            p1 = _agent_pos(obs)
            if obs["level"] != lvl:
                self.effects[a] = "reached exit"
                break
            if p0 and p1 and p1 != p0:
                dx, dy = p1[0] - p0[0], p1[1] - p0[1]
                self.effects[a] = {(0, -1): "up", (0, 1): "down", (-1, 0): "left", (1, 0): "right"}.get((dx, dy), f"jump {dx},{dy}")
            else:
                self.effects.setdefault(a, "no visible effect (blocked or non-move)")
        return obs

    def run_episode(self, game):
        ctx = self.ctx
        obs = game.observe()
        obs = self._probe(game, obs)
        notes = ctx.memory.setdefault("discovery_notes", [])
        while not ctx.done:
            table = "\n".join(f"  {a}: {e}" for a, e in self.effects.items())
            prompt = (
                f"{game.describe()}\nLegend: {obs['legend']}\n"
                f"Known action effects so far (may be wrong if the agent was blocked):\n{table}\n"
                f"Notes from earlier episodes: {notes[-3:]}\n"
                f"Level {obs['level']}/{obs['levels_total']}, actions left: {obs['actions_left_this_level']}\n"
                f"Map (9 = you, 4 = exit, 5 = wall, other digits = unknown colours):\n{_ascii(obs)}\n\n"
                "Give a sequence of up to 8 actions to make progress (reach the exit, or test one unknown colour "
                "if the direct route is blocked). Reply ONLY with action names separated by spaces."
            )
            try:
                res = ctx.think(prompt, session_id=self.session)
            except RuntimeError:
                break
            self.session = res.session_id
            seq = re.findall(r"ACTION[1-6]", res.text or "")[:8] or ["ACTION1"]
            lvl = obs["level"]
            stuck = 0
            for a in seq:
                if ctx.done:
                    break
                p0 = _agent_pos(obs)
                obs = ctx.act(a)
                if obs["level"] != lvl:
                    notes.append(f"level {lvl} solved with {seq}")
                    break
                stuck = stuck + 1 if _agent_pos(obs) == p0 else 0
                if stuck >= 2:
                    break


def build(ctx):
    return Harness(ctx)
