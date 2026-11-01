# Candidate 0023 — activate the pre-death edge memory

## Parent and diagnosis

Parent: `candidates/0016/`. I keep its observation-built corridor graph, weighted
route-goal council, ghost interception guard, asynchronous Jev/Astra calls, and
continuous cardinal executor. This is one targeted change: make the parent's
already-written pre-death-edge mechanism live, rather than adding another route
rubric or another power-pill schedule.

The worst parent trace is concrete. In `0016/trace_seed3.jsonl`, all four
`power_window` events occur by frames **436, 1220, 1612, and 2004**, then the
controller repeatedly leases retreat/dot routes as the danger estimate worsens:
frames **4164** (`risk=1, margin=9.77`), **4188** (`risk=1, margin=8.85`),
**4200** (`risk=2, margin=-3.69`), and **4208** (`risk=2, margin=-4.10`), before
`life_reset` at frame **4340**. A second `life_reset` occurs at **6892**; the
episode ends at frame **7601**, score **1790**, with **one visible dot** and a
red ghost at `[12,51]` beside the player at `[10,51]`. The same trace contains
no `predeath_edge` or `edge_taboo` event despite those two life transitions.
The other low parent run, seed 2, shows the same late churn: alternating
`code_guard` leases around `[140,103]`/`[148,103]` at frames **4172–4216**,
including risk 2 and margin **-3.69**, followed by a `jev_council` route with
risk 3 and margin **-54.92** at frame **4232**.

This is not evidence that the respawn coordinate is the collision site. It is
evidence that the controller has a useful pre-death context but never records or
uses it. In fact, `0016` contains `_remember_control_edge`, `_edge_blocked`, and
`edge_failures`, but no call site populates `last_control_node`,
`last_control_next`, or `last_control_action`; the weighted searches never ask
`_edge_blocked`. Thus the intended life-local trap avoidance was not tested.

## Hypothesis

A life loss is more reliable evidence about the **directed edge being attempted
immediately before it** than the following respawn position. If the next life
reuses the same local edge because the route graph and ghost projection call it
safe, the controller can repeat the same trap. Activating a bounded directed
edge taboo should turn the next-life replanning away from that edge while leaving
ordinary routes, pill goals, timing, and the Jev/Astra interface unchanged. This
is deliberately narrower than changing pill reservation: prior candidates changed
pill timing and still had three deaths; this tests the parent's explicitly
unexercised edge-memory hypothesis.

## What changed

1. Before every posted cardinal action, save the current graph node, the matching
   next graph node (if one exists), action, and death count. This is the
   pre-tick context, not the post-respawn coordinate.
2. When the observation's death count increments, call the parent's
   `_remember_control_edge` before normal life-reset processing. It logs
   `predeath_edge`/`edge_taboo`, clears the stale route, and records a bounded
   directed failure for the edge/action.
3. Make both route searches (`_path_to_goals` and weighted search) skip a recorded
   directed taboo. The existing fallback still chooses a fresh legal local
   branch when no goal path remains, so a malformed or disconnected learned
   edge cannot crash the episode. The taboo is reset with a newly detected maze
   epoch; it is retained across lives in the same observed resource epoch.

No seed-specific coordinates, action sequences, or ROM geometry are added.

## Jev state and bridge

The parent Jev state remains compact JSON and contains only derived facts:

- `rules`: score/progress objective, normal-ghost risk, uncertain recent pill
  protection, and the rule that Jev chooses among code-generated route goals,
  never raw joystick directions;
- `plan`: `risk_mode`, `goal_kind`, epoch, sweep/cleanup stage, last-life flag,
  legal route policy version, and current fallback;
- `options`: currently executable `continue`, dot, power/escape, or branch route
  candidates, each labelled with purpose, risk bucket (`clear`, `watch`,
  `interception_possible`, `interception_likely`), exit availability, coarse dot
  yield, reversal flag, resource flag, and whether it continues the current goal;
- `threats`: nearest-distance bucket, closing same-lane count, immediate flag;
- `recent`: score-progress, recent power event, and recent guard intervention.

Each request batches three independent typed judgments:

- `route_a`: `Choice` over the code-generated route keys plus `unknown`;
- `route_b`: the same `Choice` with reversed option order, countering Jev's
  documented first-option bias;
- `renew`: `Noul` asking whether stalled score plus repeated guard intervention
  warrants an interface audit.

Both choices must agree, select an existing option, have minimum selected-option
probability **0.70**, remain within **48 native frames**, and pass fresh graph
risk/legality checks. Jev's `renew` value is used only as evidence for the
background Astra audit when its probability is at least **0.80** and code also
confirms stalled progress. Unknown, disagreement, missing probabilities,
malformed stand-in output, timeout, stale epoch/life/policy, or a newly worse
risk leaves the deterministic route executor in charge. Jev is one asynchronous
request at a time, interval-gated, with the parent's **2.5 s** timeout; the
executor posts every tick and waits with `ctx.wait_for_tick()`.

Astra remains a background rubric author. Its short output is accepted only as an
explicit `RUBRIC=balanced|exits|cleanup`; malformed or late text has no effect.
It never posts an action. Model calls are all through `ctx.think`/`ctx.decide`.

## Confirmation and refutation

Confirmation would be `predeath_edge` and `edge_taboo` events at life losses,
subsequent route leases avoiding the recorded directed edge, fewer repeated
risk-2/3 oscillations, and improved last-life survival or maze/resource renewal
without stale ticks or episode errors. A score gain is strongest if it comes
with fewer deaths or more remaining lives, not merely more dots before game over.

Refutation would be zero edge events (the hook is still not exercised), repeated
reuse of the same recorded edge, or unchanged three-death termination while
route coverage falls because useful corridors were over-tabooed. If a taboo
removes the only route, the trace should show fallback behavior rather than a
contract error; that is part of the safety test.
