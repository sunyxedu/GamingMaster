# 0023 — Long-horizon veto for repeated carrier waits

**Parent:** 0017, retained as the architecture and route planner. This is a targeted refine, not a rewrite.

## Hypothesis and trace evidence

The portfolio planner handles broad route choice well, but its one-action fallback can keep posting `NOOP` on a moving carrier when every immediately certified alternative is unavailable. A `NOOP` is not a stationary safe hold in Frogger: each real-time frame continues to carry the frog, and a locally safe point can become a carrier-edge or moving-actor contact before another useful route opens.

The concrete evidence is in the parent’s low-scoring episodes:

- In `candidates/0017/trace_seed127.jsonl`, at tick **16280**, the preceding state is row 8 / x100, `moving_with_platform=true`, with the tracked auxiliary actor on the same row at x108. The last plan was a one-step `NOOP` from x93 (tick 16265; reported margin 3.5); the life is lost while the frog remains on the turtle. A second matching row-8 loss occurs at tick **26514**, frog x100 and actor x108, again after a `NOOP` fallback (plan tick 26499, reported margin 6.5).
- The parent’s lower-scoring `trace_seed109.jsonl` shows the same decision class at the edge: row 8 frog x146 with same-row actor x130 at tick **36457**, after a `NOOP` tick (preceding one-step plan chose LEFT); and frog x152 / actor x136 at tick **37749**, following a `NOOP` fallback with reported margin 26.75. These are not evidence that the margin itself is a reliable safety certificate; they show that its short horizon does not capture the cumulative consequences of continuing to ride.

This is consistent with the 0018/0020 lesson: a contact shield, scalar margin, or drift emergency is not automatically a save, and an escape must retain carrier support. I therefore do not widen all collision margins, revise the route portfolio, or add unconditional emergency movement.

## One change

Keep 0017’s separated-copy route search, through-bank itinerary, auxiliary/home/bank hazard layers, Jev/Astra supervision, and neutral-rearmed motor unchanged. Add one **carrier-wait lookahead veto** in the final action bridge:

1. Only when the selected action is `NOOP` while the frog is explicitly `moving_with_platform` on a river row, build the existing hazard/support field for four six-frame layers (24 frames).
2. Project the frog along that row’s existing measured carrier drift for 18 frames. Treat a projection beyond x18/x142 as an **early edge-recovery warning**, even if the projected point has not yet gone off-board; the lateral hop needs that lead time. A predicted negative support margin alone triggers an override only when the same-row auxiliary actor is also projected within 28 pixels.
3. Search only immediate non-NOOP exits. An exit is eligible only if the existing `Field.advance` verifies its landing and the destination carrier/support remains valid at the same 18-frame lookahead. Prefer forward progress, then the safest supported option. If no exit passes, retain the parent fallback `NOOP` and log that no certified escape exists; do not jump into water or invent support.
4. This rule does not trigger for ordinary road/median waits, unsupported frogs, or river NOOPs whose longer lookahead remains safe and away from an edge/actor contact. It uses no additional model calls and does not turn a forecast margin into a general survival guarantee.

The change is meant to prevent a sequence of locally acceptable waits from accumulating into a same-row actor contact or the screen edge, while preserving the parent’s route portfolio and route safety checks.

## System-1 schema and bridge (retained from 0017)

Jev receives the parent’s compact JSON: phase; validated Astra policy; ordered open homes with upstream-frontier, nearest-now, current-commitment, prior-entry-failure, matched-home-alert, and complete-route-now booleans; plus named facts for bank threat, complete launch availability, auxiliary actor/spawn status, timer urgency, and recent entry/bank/carrier failures. It is not asked to infer raw RAM arithmetic or choose frame actions.

One batched request asks for two order-reversed `Choice`s for whole-crossing bay itinerary, two order-reversed `Choice`s for TRANSIT/STAGE/RECOVER/SCREEN protocol, and a `Noul` for whether repeated-entry/bank evidence merits Astra review. A choice is accepted only when the reversed answers agree and each selected-option probability is at least 0.70 (missing probabilities count as zero). Advice must match the same life/home/level epoch and policy version and be at most 90 ticks old; route goals are revalidated and expire after 180 ticks. One request is outstanding, with a 3-second timeout and an executable code-owned fallback. Astra remains an asynchronous protocol reviewer; malformed output preserves policy and cannot disable hard local exclusions. Malformed/late model judgments never become joystick actions.

## Confirmation and refutation

Confirm if the new `carrier_wait_lookahead_veto` event replaces waits in the cited danger class with exits that stay supported, and later-level row-8 actor/edge deaths decrease without a material loss in homes or increased no-route stalls. Refute if the veto rarely triggers, if it selects unsupported landings, if deaths persist after certified long-horizon exits, or if conservative exits reduce route throughput enough to lower score. Record per-episode score, homes, deaths, vetoes, supported exits, unresolved cases, and contact outcomes; intervention counts alone are not success evidence.

## Initial harness checks (not comparative evaluation)

The module compiles. An offline replay of 165 public observations passed under first-option, garbage-answer, and timeout stand-ins; it produced only legal actions and exercised both the veto and unresolved branches. A live-Jev diagnostic run on seed127 reached score **2827** at the five-minute try cap; it emitted 19 lookahead vetoes and 54 unresolved cases. A shorter seed109 run reached **1081** at its two-minute cap and logged 26 vetoes / 46 unresolved cases; examples include a supported LEFT correction from row10/x144 with projected x148.5, and a supported RIGHT correction from row10/x10 with projected x14.5. These capped, single-seed tries are not comparable to full parent episodes and do not establish a score gain or life-saving effect. The unresolved count is deliberately visible: when the route certificate cannot name a supported exit, this change preserves the parent fallback rather than forcing a jump.
