# Candidate 0015 — 0007 with a longer rolling-ball branch shadow

**Mode:** refine. **Parent:** `candidates/0007/` (the current best harness).

## Hypothesis

0007's first late-game lives are sometimes lost after a lethal rolling ball disappears from the validated rendered-object list, and its four-tick hazard memory then treats the lower pyramid as clear. A bounded, downward-only branch projection from the last *rendered* red/purple ball position may preserve enough risk information to choose a different legal route through that short observation gap.

The shadow is activated only after **four inferred clears**, matching the parent traces where the repeated unobserved-ball losses began. It is anchored to an observed ball node, follows only the known DOWN/RIGHT descent edges, lasts at most 60 ticks, and is a soft routing cost. It does not infer an enemy's exact current node from a timer, prohibit a move, or introduce a wait. The parent motor, palette epochs, route objective, model scheduling, and policy are otherwise retained.

## Trace evidence and prior lesson

The best parent's lowest score was seed 0 (15,700). In `candidates/0007/trace_seed0.jsonl`, after four inferred clears, the episode first loses a life at tick **2259** with Q*bert at node **19**, `hazards=[]`, and `purple_ball_state=36`; the next losses occur at ticks **2382** (node 9) and **2454** (node 1), again with no rendered hazard and state 36. The trace reports a mismatched hop at tick **2260** (`origin=14`, `expected=9`, `node=19`), evidence that a route can end somewhere other than its planned node while the controller's current-object view is empty.

A repeated earlier pattern appears in seeds 1, 4, and 5: the last useful purple-ball sighting is around node 8 at tick **1964**; at tick **2019**, after roughly 55 ticks without that object, the first life loss is observed at node 13 with `hazards=[]` and `purple_ball_state=36`. At tick **1964** the controller is routing from node 13 toward the lower board. 0007's immediate threat cache expires after four ticks, so it has discarded the only localized rolling-ball evidence long before that loss. The adaptor intentionally suppresses stale sprite placeholders, so a missing object must not be treated as proof that the branch is empty.

The prior experiments constrain this hypothesis. 0013's timer-only/state-36 guard and 4–10-tick shadow did not stop its later losses; lessons explicitly say timer-only uncertainty was not a reliable forecast, and that experiment's `occlusion_decision` records often had `shadow=false`. Candidate 0015 does **not** repeat that guard. Its one test is whether retaining an observed moving ball's directional descendants for a longer but finite horizon helps. No trace proves that the ball caused every cited loss; this is a falsifiable route-risk hypothesis, not a claim of decoded collision identity.

## What changed

`Controller` now retains the last rendered node for each visible `purple_ball` or `red_ball` separately from 0007's short immediate-threat memory. For ages 5–60 ticks after the fourth inferred clear, code propagates possible positions down the public 21-node graph along DOWN/RIGHT edges, one additional row per approximate 15-tick band, capped at four rows. If a candidate landing is in that set, code adds a **2.5 danger cost** (12× in the existing route objective). The shadow influences both candidate selection and the escape-space tie calculation, but never turns a graph-legal action into an illegal one. It is cleared on a life reset or a new palette epoch and expires after 60 ticks.

Jev gets a compact per-candidate `rolling_branch` label (`none` or `recent_hidden_descendant`) and the relevant source age bucket. Its paired Choice may prefer an unshadowed option only when the code's existing cost/danger acceptance checks allow it; the Noul still asks about *confirmed immediate* danger and must not equate a projected branch with occupancy. The deterministic soft cost is the mechanism under test, not a model-generated forecast.

## Jev schema, thresholds, freshness, and fallback

The inherited state retains policy, progress epoch, position/motion, candidate actions, route-usefulness, danger, escape-space, and recent visible threats. Candidate objects add `rolling_branch`; the threat list may include a recently hidden rolling hazard with its proximity bucket and `one_to_two_hops` / `three_to_four_hops` age bucket. The only typed questions remain the reordered-choice pair and immediate-danger Noul. Both choices must agree with selected-option probability at least **0.60**; a danger veto remains at **0.82**. Answers must match the current node, life/board/policy epochs, hazard signature, and age 0–8 ticks; the action is always rechecked against current legal safe moves and current code costs. If Jev is late, malformed, disagrees, or selects an out-of-policy candidate, the existing deterministic route proceeds. The existing one-hop neutralized motor remains the fallback while a Jev request runs.

Astra remains asynchronous and unchanged. Its replies do not author trajectory maps or direct actions. This isolates the route-shadow change from an Astra-policy change.

## Confirmation / refutation

Confirm if the official evaluation retains the parent's early palette-clear performance, `rolling_shadow_decision` events show alternate legal edges marked with a projected source/age, and fewer life losses occur after the fourth clear without substantially increasing missed progress or action staleness. Refute if the relevant losses occur with no recently rendered rolling ball in the preceding 60 ticks, if route choices are unchanged because every viable edge is shadowed, if the projection steers Q*bert into Coily/other danger, or if mean score and post-fourth-clear survival regress against 0007. The key diagnostic is event-level: compare each logged selected action, its shadow source/age/descendant status, the subsequent verified landing, and any life change—not just the candidate mean.
