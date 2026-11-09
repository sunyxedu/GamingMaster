# Candidate 0007 proposal — asynchronous planning plus delay-safe execution

## Parents and concrete diagnosis

This is a crossover of `candidates/0005/` and `candidates/0003/`.

The useful mechanism in 0005 is its exact, body-aware transition model and
post-growth food-route filter.  It raised the mean score to 6.0 and never had a
self-collision.  The useful mechanism in 0003 is its explicit short-horizon
shield: a direction is not offered as the only choice when repeating it for
several ticks would hit a wall, and the latest observation is checked before
acting.

The 0005 traces expose the remaining failure more precisely than a generic
"Jev was stale" explanation.  In seed 0, the snake eats at tick 30 at
`[5,11]`, with `safe_directions_now` equal to `UP,DOWN`, but the next record
is tick 31 with `action=RIGHT`, `fresh=false`, and `attempted_head=[5,12]`.
The same terminal pattern occurs at ticks 40, 57, 73, and 48 in the other
training seeds.  The synchronous `_food_plan` is called after the food changes;
when that search runs longer than the 0.10 s cadence, the controller posts
nothing before the next tick, so the old heading is reapplied.  Replaying that
parent planner on the archived post-food states takes about 0.095 s at seed 0
tick 30 and 0.27--0.36 s on several later states, before loop overhead.  This
is a foreground-computation deadline failure, not evidence that the route
safety filter is useless.

## Hypothesis

A safe planner can improve food score without sacrificing real-time survival if
all potentially expensive search is moved off the tick path and the foreground
has an exact repeated-action shield.  In particular:

* a bounded worker can search the full body state for a food route and, if no
  safe route is found, a moving-tail route;
* while that worker is running, a deterministic one-step/three-repeat filter
  can post a useful action immediately; and
* a completed worker result is usable only after matching the current body,
  heading, and food (or an exact simulated prefix), so asynchronous planning
  cannot introduce stale geometry.

The hypothesis is supported if the candidate posts fresh actions through the
post-food junctions that killed 0005, has fewer stale ticks, and retains or
increases food count without self-collisions.  It is refuted if moving search
off-thread still leaves the fallback unable to reach food, or if the
three-tick filter rejects too many useful turns and lowers score despite fresh
execution.  If it survives but plateaus, the next experiment should target
cycle entry or a stronger time-expanded safety proof, not larger model
prompts.

## What changed

`harness.py` makes one structural change and one careful safety change:

1. `_planner_task` is launched through `ctx.spawn` with a hard 55 ms
   wall-time bound.  It searches exact dynamic-body transitions for a
   post-growth-safe
   food route, then a route toward the moving tail.  The foreground never
   waits for it.  A completed route carries its origin state; the controller
   accepts it only at the origin or at an exactly simulated non-food prefix.
2. `_safe_sets` implements the 0003-style delay shield, but with the 0005
   transition semantics.  It tests one-step collision safety and whether the
   same absolute request can be repeated for three ticks, retaining growth
   cells while treating the next private food position as unknown.  If no
   three-repeat candidate exists, the exact one-step set is still used.
3. The action loop always posts an action before `wait_for_tick`; no
   unbounded A*/BFS search, Astra call, or Jev call can block the clock (only
   small bounded masks and flood fills run in the foreground).  A route action
   is rechecked against the current one-step transition.  An order-validated Hamiltonian
   step remains a late, conservative fallback.
4. Jev is advisory only at a genuine junction when no route worker is
   pending.  Its answer can replace the heuristic fallback only when two
   reordered Choices agree, both selected probabilities are at least 0.34, the
   answer is at most two ticks old, and the answer is still in the current
   shield-safe set.  A timeout, garbage stand-in answer, disagreement, or stale
   answer leaves the deterministic action in place.

Astra is intentionally not called in this candidate.  Both parents establish
that the immediate bottleneck is the 100 ms foreground deadline, while the
measured Astra latency is tens of seconds and the training episodes currently
end in tens of ticks.  Calling Astra cannot supervise those post-food turns.
This isolates the structural hypothesis; a future Astra experiment should
review route-worker logs after long survival and set a risk threshold, rather
than issue a cell-by-cell action.

## Jev state schema and bridge

Every Jev request is strict JSON with this shape:

```json
{
  "schema": "snake-junction-v2",
  "rule": "Choose exactly one listed action; code checked legality",
  "observation": {
    "tick": 24, "head": [0, 10], "tail": [3, 9],
    "direction": "RIGHT", "food": [5, 11], "length": 5, "score": 2
  },
  "candidates": [
    {"action":"RIGHT", "safe_for_repeated_ticks":false,
     "eats_now":false, "post_growth_safe":true,
     "tail_access":true, "space":"open", "keeps_heading":true}
  ],
  "fallback": "DOWN",
  "safety_pool": "one-step-safe"
}
```

Code computes all coordinates, collision outcomes, areas, tail access,
food-growth safety, and the candidate list.  Jev is asked only to recognize
which of those named alternatives is the safest useful one.  `pick_a` and
`pick_b` are identical `Choice` questions with reversed option order; `urgent`
is a batched `Noul` diagnostic and never overrides code.  The bridge uses the
agreement/probability/age gates above and revalidates the answer against the
latest observation before posting it.  The fallback ranking is repeated-tick
safe pool first, then tail access, connected area, food distance, and heading
continuity; if all masks are empty, the current heading is posted so the game
can report its own terminal state.

## Astra revision trigger

There is no Astra output in this isolated test.  Evidence for adding Astra in
the next iteration would be a repeated trace class in which the worker returns
routes that pass static tail access but consistently reduce area or stall at
high occupancy.  Astra should then choose among named policies such as
`conservative-cycle`, `safe-food`, and `tail-unwind`, with code still enforcing
all transitions and deadlines.  Evidence against adding it is a continued
short-episode regime where every failure is a missed foreground deadline;
slow supervision would only consume the secondary call budget.
