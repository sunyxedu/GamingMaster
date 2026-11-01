# Candidate 0019 — confirmed-start conveyor plus progress-gated takeover

## Parents and hypothesis

Parents: `candidates/0015/` and `candidates/0017/`.

The useful mechanism in 0015 is bounded ingredient staging: a free counter can
turn the cooking interval into work on the next soup.  The useful mechanism in
0017 is the progress watchdog: a partner-held dish is only a real handoff while
its route is improving, and a bounded escape prevents a blocked transaction from
becoming an episode-long wait.  I combine both, but change their scheduling
boundary rather than adding another traffic lease.

The specific hypothesis is that the low tail is caused by **starting/staging and
handoff decisions being made from ownership rather than confirmed physical
phases**.  In the 0015 seed-0 trace, the controller deposits ingredients at ticks
23 and 26 and then parks at tick 29; the first explicit start does not occur
until tick 62.  Thus its new buffer is useful eventually but delays the first
soup.  In the 0017 seed-0 trace, the first start is earlier (tick 25), but the
last scored delivery is tick 371 and the final state at tick 400 has the
controlled chef holding a dish, the partner holding an onion, and a soup cooking
at tick 15.  Seeds 3 and 6 show the same 9-delivery tail.  A dish in the partner's
hand is not evidence that the handoff will complete before the horizon.

The candidate therefore enforces two invariants:

1. A stage transaction is generated only after an observation contains a pot
   whose phase is actually `cooking`; full-idle pots always get an immediate
   explicit `start` before staging is considered.
2. A partner-held dish suppresses duplicate dish travel only while code-observed
   route distance is improving or the partner is at the pot.  For a ready pot,
   or the final two cook ticks, eight ticks without route improvement (or ten
   stationary ticks, unless the partner is already adjacent) authorize one
   code-legal backup dish trip.  This is a bounded takeover, not a claim or
   indefinite yield.
3. When the partner is carrying finished soup and the controlled chef is actually
   in the first few cells of the computed serving route, the controlled chef
   yields outside all interaction approaches.  It does not reserve the whole
   kitchen when an open alternate route exists.

The motor is receding-horizon: it recomputes a legal BFS interaction pose from
latest state, predicts the partner's next cell, waits for at most two blocked
steps, then uses a two-step escape and refreshes the transaction.  It does not
repeat a stale directional action while Jev or Astra is thinking.

## What changed

`harness.py` is a new `ConveyorPilot` implementation, not a copied task executor.

* `pot_facts` derives active recipes, ingredient deficits, explicit full-idle
  versus cooking phases, and cooking remainder from the JSON state.
* `cooking_buffer` is a post-start-only, recipe-aware counter buffer.  It stages
  at most two ingredients and keeps one free counter as a shared handoff reserve.
  It is not allowed to delay the first start or to pre-fill a wrong recipe.
* `update_partner` measures the partner's route to a ready/cooking pot or serving
  window using the current terrain.  `takeover_needed` requires urgency plus
  physical non-progress; a partner merely carrying a dish is not treated as a
  reservation.
* `candidates` uses hard priorities: carried soup delivery, route-conflict
  yielding for a partner-carried soup, ready pickup, explicit start, compatible
  fill, urgent/timed or backup dish, post-start staging, then safe parking.  In
  the final 36 ticks it stops launching empty-handed builds unless a pot is
  already cooking, protecting a soup that can still be delivered.
* The motor emits only advertised primitives after checking the current facing,
  adjacent interaction pose, object, pot phase, and counter occupancy.  A late,
  malformed, or garbage model result leaves the deterministic plan unchanged.
* One Astra review is launched off-thread for broad tempo/lane/takeover advice.
  Jev is launched at transaction boundaries, blocked routes, and a heartbeat;
  its answer can only select a current code-generated option within one urgency
  band.  Neither model can demote delivery or invent a target.

Confirmation would be a faster first start than 0015, fewer `park`/`hold_dish`
tails after a cooking start, and at least one additional delivery on cramped
seeds 0/3/6 without losing 0017's clean 16–19-delivery asymmetric episodes.
The hypothesis is refuted if post-start staging consumes the only useful traffic
counter, if backup dishes contest a partner whose distance is improving, or if
an escape causes repeated route resets without new potting/delivery events.
The decisive held-out test is a mixed/tomato or reversed-agent layout: phase and
recipe-derived targets should continue to work without onion or controlled-index
assumptions.

## Tier-2 Jev interface

Every request is strict JSON of this shape:

```json
{
  "game": "Overcooked-AI symbolic real-time kitchen",
  "observation": {
    "tick": 0,
    "layout": "...",
    "self": {"position": [x,y], "facing": [dx,dy], "held": ""},
    "partner": {"position": [x,y], "held": "dish", "still": 0,
                 "route_not_improving": 0, "last_action": "..."},
    "pots": [{"position": [x,y], "phase": "empty|partial|full_idle|cooking|ready",
               "missing": ["..."], "remaining": 7}],
    "buffer": {"staged": [["onion", [x,y]]], "free_sites": 2},
    "handoff": {"takeover_needed": false, "partner_goal": ["ready", [[x,y]]]},
    "plan": {"kind": "fetch_ingredient", "target": [x,y], "age": 3,
             "blocked": 0},
    "candidates": [{"id": "...", "kind": "fill", "priority": 2,
                    "description": "..."}],
    "astra": {"tempo": "stage_after_start"}
  }
}
```

The batched `transaction` Choice uses the exact candidate IDs generated by
code.  A selected ID is accepted only if its listed probability is at least
0.56 (the stand-in/preflight case with no probability is treated as advisory),
and its priority is no more than one band worse than the deterministic best.
The `takeover` Noul is accepted only at true probability at least 0.80 and only
when a current `backup` dish candidate exists; it cannot create a dish target.
Requests use a 1.05-second timeout and carry the plan epoch.  A response from an
old epoch, an unknown ID, absent probabilities, timeout, exception, or exhausted
budget is ignored.  At most one Jev request is outstanding.

The fallback is always executable without a model: current observation facts
choose the transaction, BFS chooses the pose, facing is corrected with a
one-tile directional action, and `stay` is used when no legal progress is
available.  Three predicted blockages trigger a short detour; six measured
blocked movement ticks trigger a bounded escape and replan.  A carried soup is
never abandoned for recovery.  `ctx.act` is wrapped with a legal noop retry,
and every iteration posts an action before `ctx.wait_for_tick`, preserving the
real-time contract and avoiding stale ticks.

## Astra boundary

Astra receives one compact initial snapshot off-thread and may return only
`tempo`, `lane`, and `takeover` labels.  It is deliberately not asked for
routes, arithmetic, primitive actions, or layout-specific scripts.  Its advice
is logged and included in the Jev state but cannot override the confirmed-start,
ready-soup, recipe, or legality invariants.  This tests whether slow reasoning
is more useful as a phase/coordination prior than as a delayed plan.
