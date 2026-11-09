# Candidate 0022 — dephasing a timed route ensemble

**Parent:** `candidates/0019/` (which combines the best `0016` motor/corridor baseline with `0015` rolling-memory and next-junction Jev).  The strongest reference is `0016`.

## Hypothesis

The remaining failure is not simply an illegal edge, a stale Jev answer, or a missing
one-hop landing check.  It is **temporal phase-lock**: the controller repeatedly
traverses a safe-looking corridor at nearly the same game/scheduler phase, while the
unseen rolling-object state is different between observations.  A route change that
only avoids the last spatial edge can still arrive at the same dangerous phase.  A
small, bounded change of route cohort and/or one neutral tick should decorrelate the
arrival from that latent schedule without throwing away the proven coverage motor.

This is a deliberately new mechanism rather than another longer hazard shadow.  Code
keeps a phase ledger: the tick modulo a coarse 64-tick period, the object-scheduler
band, the purple-state band, the edge used, and the edge/phase at life loss.  It
softly penalizes repeating a failed edge near its recorded phase, penalizes its reverse
for a short phase window, and offers at most two short waits in an 80-tick window when
there is an open escape.  After a loss it switches to a different coverage ordering
for the next route cohort.  It does not claim that `purple_ball_state` identifies an
enemy or that a phase is deterministic; the timing evidence is only a conservative
correlation cue.

## Evidence and diagnosis

The parent did reach substantially later boards, so the one-hop protocol is retained.
However, `0019/trace_seed0.jsonl` records losses at ticks 1326, 2160, 2311 and 2363.
The first loss is observed at node 15 during a hop with `hazards=[]`, `purple_ball_state`
36, and only `UP` reported as safe.  The later losses are likewise empty-hazard
observations at nodes 17 and 20.  Immediately before the third loss the controller
repeated the bottom-corner pattern `14 -> 20` (with a retry after a no-start), then
returned through `20 -> 14`; its corridor certificate reported risk 0 and zero exits
for `14 -> 20`.  A spatial memory and next-junction Jev answer did not make that
repeated timetable safe.

The held-out-looking variation in `0019` points away from one permanently bad edge:
seed 1 loses at 0, 18, 1 and 4 (ticks 2411, 2828, 3451, 3927), while seeds 4/5
share near-identical late losses around 2432, 2820/2817, 3201/3066 and 3436/3533.
Thus the same policy can encounter a failure at different nodes/times, and the same
opening can be followed by a repeated late timing pattern.  In `0016`, seed 0 and
seed 5 share the first loss exactly at tick 1722 (node 15 with a visible Purple Ball)
and the second exactly at tick 2148 (node 20 with empty hazards and state 36), despite
the different sticky seeds.  Their later losses are also on the outer return (`15`
and `20`).  This is consistent with a deterministic controller's arrival schedule
being more important than the current one-frame hazard list.

The lessons explicitly refute treating state 36 as a reliable hidden position, broad
radar waiting, timer-only directional guards, and unconditional corner/endcap bans.
They also say that immediate `safe_moves`, verified landings, and exact endpoint
memory are not a safe future horizon.  0022 therefore uses no fabricated hazard
location, no hard ban on a corner, no long radar wait, and no additional enemy shadow.
It changes the *time and route cohort* only when code has an actual loss-phase record,
while preserving the parent's early primary order and neutralized motor.

## What changed

* **`TemporalLedger`** records each edge's coarse clock phase and the bounded
  scheduler/purple-state bands, plus the exact edge at a life-loss observation.  A
  same-edge repeat near the loss phase receives a large but soft cost; the reverse
  edge and tip involvement receive smaller costs.  The ledger forgets old evidence
  through finite deques and phase distances, so it cannot permanently poison the
  graph.  Immediately after a life loss, code enters `dephase` for at most 150 ticks;
  the override expires automatically unless a newer loss renews it.
* **Two coverage cohorts** use distinct, deterministic 21-node orders.  The primary
  order and persistent-target rule preserve the parent's strong coverage behavior;
  speculative Jev planning must not replace the current target merely because the
  projected landing equals it.  A life loss selects a different cohort and
  invalidates target/advice state.  Astra can select primary/alternate, but cannot
  supply a route or action.
* **Bounded dephasing** may insert one neutral tick only if the current best edge has
  a timing penalty, the current node has at least two computed exits, and fewer than
  two such waits occurred recently.  This is not idle play: it is a short shift,
  followed by fresh route computation, and is never used for concrete visible danger.
* **Swept-edge risk** adds a small segment-distance penalty for a rendered hazard
  crossing the animated edge.  The existing short public-object cone remains finite;
  stale RGB-validated objects are not promoted to live enemies.
* **Jev** is moved no further into control: it judges supplied next-junction edges,
  with duplicated reordered `Choice`s, plus a `Noul` about whether a bounded dephase
  wait is explicitly supported.  Code checks the selected edge's current risk,
  temporal cost, exit count, target usefulness, policy version, node, life epoch and
  age before accepting it.
* **Astra** receives a compact loss/timing summary and returns only
  `{"timing":"steady|dephase|diverse", "coverage":"primary|alternate"}`.
  It is asynchronous and capped at two calls.  Invalid or prose output leaves the
  deterministic posture intact.

## Confirmation and refutation

Support requires that 0022 retain the parent's first-board/early-clear behavior,
while the traces show `temporal_edge_failure` records followed by a cohort change or
an accepted phase penalty, fewer repeated losses near the same edge phase, and more
fifth/sixth or later board clears.  A useful causal trace should show an edge rejected
because its phase is near a recorded loss, or a bounded `phase_dephase_wait`, and then
show the subsequent route/score rather than merely reporting a Jev answer.

The hypothesis is refuted if the new primary opening loses substantially earlier, if
waits stall coverage without lowering loss rate, or if post-loss failures occur at the
same phase despite a changed cohort.  If timing costs change decisions but score and
loss timing remain unchanged, the unrendered object is not schedule-correlated (or is
not the cause).  If the score improves while no timing event is used, the gain is
ordinary route variation and not evidence for phase-lock.  A result near 0016/0019
with all later losses still occurring in empty-hazard observations would point instead
to a missing disc/Coily escape or a decoder limitation.

## Tier-2 Jev contract and bridge

### State schema

Every request sends strict JSON containing only derived facts:

* `current_hop`: the already-posted action and projected landing;
* `position`: projected standing node and the currently observed directional safe
  set (not used to authorize the future action);
* `next_options`: up to four code-generated options, each with a legal action key,
  unfinished/transit landing, immediate danger bucket, escape bucket, timing bucket
  (`ordinary_phase` or `repeat_failure_phase`), and deterministic rank;
* `timing`: current clock bucket, finite recent-failure phase relations, scheduler
  band and route posture;
* `progress`, `hazards`, and the survival-first rule.

Jev is never asked to count nodes, decode RAM, infer the graph, estimate a numeric
collision time, or generate an action.  `keep_code` is explicit so uncertainty has a
safe answer.

### Questions

One asynchronous request batches:

1. `timed_edge_a`: `Choice` over supplied edges plus `keep_code`;
2. `timed_edge_b`: the same `Choice` with reversed edge order to counter first-option
   bias;
3. `dephase_needed`: `Noul`, true only when the supplied facts explicitly show a
   repeat-phase conflict and an open escape for a one-tick shift.

The current hop is always completed by the motor; Jev can only influence the next
standing junction.  A response is dropped if the node, board epoch, life epoch,
policy version, hazard/scheduler signature, or age (12 ticks to collect, 3 ticks to
apply) is wrong.  The two choices must agree, and the minimum selected-option
probability must be at least **0.60**.  Code then requires the chosen edge to have
risk no more than 0.45 above fallback, temporal cost no more than 1.0 above fallback,
route cost no more than 1.8 above fallback, and no more than one fewer exit.  It must
also remain useful when the fallback is useful.  Any malformed stand-in answer,
missing probability, disagreement, timeout, or failed call falls back to the fresh
code-ranked edge.

The `dephase_needed` judgment is logged for diagnosis but does not itself authorize a
wait: code alone decides whether the current node has an escape and whether the
finite two-waits-per-80-ticks budget permits one.  This avoids outsourcing an
irreversible timing decision to a low-confidence model answer.

### Freshness and fallback policy

There is at most one outstanding Jev request, started no more often than every eight
standing ticks with a 1.0-second timeout.  While it is late, the current hop proceeds
with neutral pulses and the deterministic fallback.  If no safe edge is supplied,
post a named `NOOP`; never use `FIRE`.  The motor posts a direction only while
standing and only if it is in the current `safe_moves`, retries an ignored edge once,
neutralizes after each landing, verifies the actual node, and enters a bounded
neutral recovery after a loss/mismatch.  The loop stops only at `game.done` or
`ctx.done`, not when the reported lives field reaches zero.

Astra runs in a background future with a 105-second timeout, at most twice.  Its
answer is parsed as a whitelist; all other output is ignored.  Astra is requested at
startup, after a life loss, and for a late-board stall subject to the call cap; it
never blocks a tick.  This makes the experiment about whether timing dephasing helps,
not whether a slow System-2 answer can control a real-time game.
