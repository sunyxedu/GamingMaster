# 0024 — Latched carrier-edge recovery

**Parent:** 0015, as directed; the route-search and asynchronous one-hop plumbing are retained, informed by the stronger 0017/0023 traces. This tests one execution-layer hypothesis; it does not claim to replace the hazard portfolio.

## Hypothesis and evidence

A local edge correction can be undone by the next route choice before the frog is clear of the carrier edge. Candidate 0023's best trace `trace_seed127.jsonl` gives a particularly testable sequence on river row 8: at tick **19922** the frog is at x135 and the 18-frame projection is x144; at **19925**, the lookahead veto chooses LEFT from x136; at **19931**, the normal route selector chooses RIGHT from x131; by **19937**, the frog is at x142 and its next projection is x151. The unresolved shield then has no checked replacement. The same sequence repeats at ticks **21201–21216**. This is a concrete policy-oscillation gap: a valid local recovery action is not a durable commitment, and the route portfolio can immediately steer back toward the threatened edge.

Other 0023 failures at rows 9/10 without a tracked auxiliary actor show that the improvement should not depend on identifying a snake/bonus actor. The test is carrier-edge control only. Neither a projected margin nor a successful counter-hop is treated as a survival certificate.

## Change

Keep the 0023 time-layered route portfolio and its hazard layers, and add a **latched edge-recovery state** in the action bridge:

1. Estimate a rider-specific drift from consecutive public frog x positions only when the frog stays on the same river row, the adaptor says it is moving with a platform, and the preceding posted action was NOOP. Retain a short robust history and clear it after a life/home/level epoch change; use the row-object average if there is no recent rider sample. This tests whether direct rider motion is a better predictor than averaging two carriers.
2. Project both the neutral ride and selected route action over the existing 18-frame recovery horizon. If a route action or neutral ride points beyond the edge warning band (x>142 or x<18), latch a recovery direction for that river row and life/home/level epoch.
3. While latched, do not let a subsequent route choice countermand the correction toward the interior. Prefer a geometrically checked UP/DOWN exit when its future support is valid; otherwise use the checked lateral counter-drift action. Candidate exits must pass both current `Field.advance` checks and a long-horizon support check. If no exit is verified, do not fabricate one: preserve the current safe fallback and log the unresolved latch.
4. Hysteresis clears only after the rider's projected neutral position returns to a safer interior band (x<=132 on the right, x>=28 on the left), or when the frog changes row/epoch. This prevents one apparently improved sample from immediately reopening the outward route. It is not a permanent ban on either side or a blanket NOOP veto.

The existing route search, home selection, local actor/home/bank layers, one-hop release logic, Jev protocol questions, confidence gates, and Astra review remain unchanged. No model response can override the local checked action bridge.

## System-1 state and bridge

The inherited request sends a compact JSON state with phase, policy, ordered open-home facts (frontier, nearest, commitment, failure/alert status and complete-route availability), and named recent-failure, actor, bank and timer facts. Batched questions are: reversed-order `Choice`s for a durable bay itinerary and the TRANSIT/STAGE/RECOVER/SCREEN protocol, plus a `Noul` for whether concrete repeated-entry/bank evidence merits Astra review. It does not ask Jev to calculate motion or emit joystick strings.

Choices must agree under reversed order and each selected option must have probability >=0.70; missing probability data fails closed. A result must match the current epoch and policy version and be at most 90 ticks old. Astra's JSON enums are validated and malformed text leaves the policy unchanged. Model latency never blocks the motor; the code-owned route and latched checked recovery are the fallback. The edge latch is updated from current observations, not model answers, so a slow or stale answer cannot cancel it.

## Confirmation / refutation

Confirm if 0024 repeats the tick-19925 recovery situation without selecting the outward RIGHT at the following decision, reduces row-8/9/10 carrier losses per life on training episodes, and preserves homes/score without a major increase in unresolved waits. Direct rider drift should agree with later observed no-input displacement more often than the row-average estimate. Refute if the latch fails to activate before outward reversals, frequently forces unsupported lateral movement, increases no-route stalls/deaths, or lowers score despite fewer edge witnesses. Compare life-normalized carrier losses and resolved/unresolved latches, not intervention counts alone.

## Delivery

This is the final generation's cautious, falsifiable action-bridge change. It uses only public observations and no seed-specific routes.