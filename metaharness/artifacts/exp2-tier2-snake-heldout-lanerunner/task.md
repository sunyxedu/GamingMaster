Design a harness for a **real-time** game (tier 2).

Game: `snake` — Snake on a 12x12 grid, one tick every 0.500s, max 300 ticks. +1 per food eaten; episode ends on wall/self collision. The snake advances one cell per tick in its heading regardless of whether you decided. Observation: dict with head, heading, body, food, food_delta, danger{up,down,left,right} (True = that cell kills), and an ASCII grid (head=arrow, body=o, food=*).

Actions:
- up: face up (y-1)
- down: face down (y+1)
- left: face left (x-1)
- right: face right (x+1)
- keep: keep current heading
(no decision in time => 'keep')
Reversing into yourself is ignored (heading unchanged). Walls kill. Grid coordinates: x to the right, y downward, origin top-left.

Models available through `ctx`:
* `ctx.think(prompt, session_id=...)` — Astra (System-2, slow: seconds). Deep reasoning, can plan,
  can explain rules. Returns `.text` and `.session_id` (pass it back to keep its reasoning state).
  Budget: 20 calls per episode. Astra never sees the game unless you show it.
* `ctx.decide(state, questions)` — Jev (System-1, fast: ~0.1–0.7 s). Typed questions
  (`metaharness.models.jev.Choice/Score/Noul`), calibrated probabilities. Budget: 4000 calls per
  episode. Jev is a zero-shot classifier: it does not reason, so what you put in `state` and how you
  phrase `instructions`/`criteria` is the whole game.
* `ctx.spawn(fn)` — background thread, so Astra can think while Jev keeps acting.

The game ticks every 0.5s **on its own clock**; `ctx.act(a)` posts the current action (instant).
A tick with no fresh action re-applies the last one. Evaluation metric: mean score over seeds
(primary), then fraction of stale ticks (lower is better), then System-2 calls (fewer is better).
Clock mode: wall.

What the harness must design (this is the whole point): how Astra's slow output (a plan, a rule
summary, a policy sketch, a subgoal) is turned into *state + questions* for Jev at every tick, how
Jev's typed answer is turned into a game action, when to go back to Astra, and what to do when a
decision is late.

**Generality requirement.** The best harness is re-evaluated on a *held-out game* with the same
interface (`game.describe()`, `game.action_spec`, dict observations, `ctx`) but different rules and
different observation keys. Do not hard-code this game's rules or observation fields; discover them
from `game.describe()` / the observation, or ask the System-2 model. Game-specific solvers score 0
on the held-out game.