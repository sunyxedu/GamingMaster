# Candidate 0020 — two-food value search on a persistent certificate

## Parent and diagnosis

Parent: `candidates/0016/`.  I also compared the best safe line, `0013`, and
its worst episode (`seed 5`).

The current ceiling is not a collision problem.  `0013` survives all 4,000
ticks with zero stale ticks and zero collisions, but seed 5 finishes at score
119 with 22 free cells.  Its end-of-run trace repeatedly has the same pattern:

```text
3988: action LEFT, score 119
3988: food_plan_started, food [2, 1]
3989: action LEFT
3989: food_plan_discarded, not_shorter_than_cycle
...
3992: action UP
3993: action RIGHT
3993: food_plan_discarded, not_shorter_than_cycle
...
4000: action RIGHT, score 119, food [2, 1]
```

Thus the safe planner is usually proving that the current cycle route is not
beaten, one tick at a time; it never uses the choice among equally safe food
routes to improve the *next* target.  The one-food A* objective is locally
reasonable but globally myopic.  The parent `0016` did attempt a forecast, but
it coupled a two-food route to frequent tour changes.  That experiment failed
structurally: seed 4 reached score 47 and then had `tick 767, action DOWN,
fresh=false`, followed by self-collision at tick 768.  Candidate `0017` also
shows that a broader approach search can starve the real-time foreground:
seed 1 has a stale tick at 2495 immediately after a long route-search regime.
Those results argue for changing the objective without relaxing the per-step
certificate or the one-tick scheduler.

## Hypothesis

When the seed/RNG stream is verified from the current observation, a route to
the current food should be ranked by

```
(time to current food) + 0.35 * (cycle distance from the grown head to next food)
```

rather than by time to the current food alone.  The current food move changes
the body and therefore the safe starting position for the next target.  A
slightly longer current route can be worthwhile if it puts the head much closer
to the next sampled apple.  This is different from `0016`: it searches only
one food at a time, keeps one cycle for the whole episode, and uses the
forecast only as an endpoint value, not as permission to switch certificates
or to execute an unchecked multi-food route.

The prediction is more accepted routes with fewer wasted cycle turns and a
mean above `0013`'s 125.000, while retaining 0 stale ticks and 0 collisions.
It is refuted if the endpoint value does not improve food throughput, if it
reduces the number of food plans that are actually installed, or if any
accepted route loses cycle order or causes a collision.  A verified forecast
is optional: when it cannot be reconstructed or disagrees with an
observation, the controller falls back to strict one-food improvement.

## What changed

* Build and validate a small family of rectangular Hamiltonian cycles.
* Commit one cycle at the beginning using the current-food route and, when
  available, the next-food cycle-distance as a tie-break.  Do not switch the
  certificate merely because a new apple appears.
* Reconstruct the documented `random.Random(seed)` food-choice suffix only
  after checking the observed initial food and checking every later spawn.
* Run a bounded background A* from the exact predicted state after the action
  being posted.  Every intermediate transition is simulated with exact
  tail-vacating/growth semantics and must preserve head-to-tail cycle order.
* Keep searching after a current-food goal and compare candidate endpoints by
  the two-food objective.  Install a route only when it strictly improves the
  cycle fallback's corresponding objective (or, without a forecast, its
  current-food length).
* Keep the worker below one tick and revalidate every route action against the
  live body, heading, food, and cycle.  A state mismatch discards the route;
  the cycle action or exact reachable-area fallback is posted immediately.

No seed-specific board or food is embedded.  The forecast is used only after
matching public observations; it is disabled on any mismatch.

## System-1 / System-2 interface and timing

This candidate intentionally makes no Astra or Jev calls.  The game exposes
the complete symbolic body and the exact transition rules, so arithmetic,
collision checking, food bookkeeping, and the safety proof are better done in
code.  Jev's measured ~0.14–0.20 s response spans more than one 0.10 s tick,
and the prior model-assisted candidates did not improve the safe cycle line.
The System-1 schema and question set are therefore empty; there is no answer
parser, confidence threshold, or model freshness cache.  Astra is not in the
control path and would only be justified by a future trace showing that the
public rule/forecast check is wrong or that a verified endpoint objective
systematically chooses poor routes.

The deterministic fallback is:

1. use a route only if its origin is an exact live prefix, its next transition
   is legal, and the resulting body remains ordered on the committed cycle;
2. otherwise use the committed cycle's checked one-step action;
3. if no cycle is available (for an odd-by-odd generalisation or malformed
   state), choose a legal one-step move maximizing tail reachability and
   connected free area.

The worker's deadline is 38 ms, it checks `ctx.done`, and it is launched from
the predicted post-action state.  This preserves the 100 ms clock margin and
avoids the stale-tick failure seen in `0017`.
