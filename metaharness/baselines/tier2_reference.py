"""Reference two-timescale harness (test fixture / what a good proposal looks like).

Structure — the "bridge" between Astra and Jev:
  * Astra is asked ONCE at the start (and re-asked in the background every
    ``replan_every`` ticks or after a death-danger spike) for a *policy sketch*:
    a short ordered list of rules in plain language plus a target cell.
  * Every tick, Jev gets a compact state: the policy sketch, the local danger
    map, the food delta, and the *previous decision* — and answers one
    Choice question whose criteria are rewritten each tick with the concrete
    consequence of each move (``"right: moves toward food, safe"``), which is
    what a zero-shot classifier can actually use.
  * Late-decision policy: if Jev is slower than a tick, the mailbox already
    holds the last safe action, and the harness re-checks danger before
    posting so a stale-but-fatal action is overridden by a local safety rule.
"""
from __future__ import annotations

from metaharness.models.jev import Choice

OPP = {"up": "down", "down": "up", "left": "right", "right": "left"}
DIRS = ["up", "down", "left", "right"]


class Harness:
    def __init__(self, ctx, replan_every: int = 40):
        self.ctx = ctx
        self.replan_every = replan_every
        self.plan = "Move toward the food; never enter a dangerous cell; never reverse."
        self.session = None
        self.pending = None

    # ---- Astra (slow) --------------------------------------------------
    def _ask_astra(self, game, obs):
        prompt = (
            f"{game.describe()}\nActions:\n{game.action_spec.describe()}\n\n"
            f"Current board:\n{obs['grid']}\nhead={obs['head']} heading={obs['heading']} food={obs['food']} "
            f"length={obs['length']}\n\n"
            "You are the slow planner for a fast reflex controller that picks a heading each tick from a "
            "short state summary. Write at most 4 short imperative rules (one per line, most important first) "
            "the reflex controller should follow for the next ~40 ticks, e.g. how to approach the food, which "
            "side of the body to keep free, when to prefer open space over food. No prose."
        )
        try:
            res = self.ctx.think(prompt, session_id=self.session)
            self.session = res.session_id
            text = (res.text or "").strip()
            if text:
                self.plan = text[:600]
                self.ctx.log(event="plan_update", plan=self.plan)
        except RuntimeError as e:  # budget exhausted — keep the old plan
            self.ctx.log(event="plan_update_failed", error=str(e))

    # ---- Jev (fast) ----------------------------------------------------
    @staticmethod
    def _consequence(d, obs):
        dx, dy = obs["food_delta"]
        towards = {"up": dy < 0, "down": dy > 0, "left": dx < 0, "right": dx > 0}[d]
        if obs["danger"].get(d):
            return f"{d}: DEATH (wall or body)"
        if d == OPP[obs["heading"]]:
            return f"{d}: ignored (reverse)"
        return f"{d}: safe, {'toward' if towards else 'away from'} food"

    def _jev_state(self, obs, last):
        return {
            "planner_rules": self.plan,
            "head": obs["head"], "heading": obs["heading"], "food_delta": obs["food_delta"],
            "danger": obs["danger"], "length": obs["length"], "last_action": last,
            "local_view": obs["grid"],
        }

    def _decide(self, obs, last):
        safe = [d for d in DIRS if not obs["danger"].get(d) and d != OPP[obs["heading"]]]
        if not safe:
            return obs["heading"]
        if len(safe) == 1:
            return safe[0]
        q = {"move": Choice(
            instructions="Following planner_rules, which safe heading is best now?",
            criteria={d: self._consequence(d, obs) for d in safe})}
        ans = self.ctx.decide(self._jev_state(obs, last), q)
        a = ans["move"].value
        return a if a in safe else safe[0]

    # ---- loop ----------------------------------------------------------
    def run_episode(self, game):
        ctx = self.ctx
        obs = game.observe()
        # First plan is requested in the background too: with a 0.25 s tick a
        # synchronous 2-8 s Astra call would let the snake run into the wall.
        self.pending = ctx.spawn(lambda o=obs: self._ask_astra(game, o))
        last = obs["heading"]
        last_plan_tick = obs["tick"]
        while not ctx.done:
            obs = game.observe()
            if self.pending is None and obs["tick"] - last_plan_tick >= self.replan_every:
                self.pending = ctx.spawn(lambda o=obs: self._ask_astra(game, o))
                last_plan_tick = obs["tick"]
            if self.pending is not None and self.pending.done():
                self.pending = None
            a = self._decide(obs, last)
            ctx.act(a)
            last = a
            ctx.wait_for_tick()   # one decision per tick; never spin


def build(ctx):
    return Harness(ctx)
