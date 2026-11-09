# Candidate 0025 — soft two-hop margin for a freshly rendered Coily

**Mode:** refine. **Parent:** `candidates/0016/`. The parent's corridor planner, palette epochs, sticky-action handshake, Jev bridge, and asynchronous Astra posture remain intact.

## Hypothesis and trace evidence

The parent already penalizes Coily on the takeoff cube and inside a one-hop reachability neighborhood. In a worst seed, however, the public trace shows a pursuer closing across a short sequence of individually verified hops:

- In `candidates/0016/trace_seed5.jsonl`, the `corridor_status` record at tick **2069** has Q*bert at node **13**, action **RIGHT** toward node **19**, and a **rendered Coily marked adjacent**.
- The controller then logs verified landings **13→19** at tick **2078**, **19→14** at **2088**, and **14→20** at **2098**. At tick **2099**, the status reports Coily **near** at node 20; the sole onward corridor is LEFT. The controller records `handshake_recovery` at tick **2109** with lives dropping **3→2**.

This does not prove that every edge in the chain was unsafe, or that a particular hidden node was Coily's exact trajectory. It does show a rendered nearby pursuer, a short verified corridor toward the pyramid edge, and a loss immediately after the pursuer is again near. The testable hypothesis is narrower: the parent's one-hop Coily reachability set omits enough of the *second* pursuit ring that a soft margin on those reachable landings may select an equally useful corridor with more distance before Q*bert is forced to the edge.

Other late parent losses still occur with `hazards=[]` and `purple_ball_state=36` (for example seed 5's recovery events at ticks **2492** and **2933**). This proposal does not infer a hidden enemy position from that byte and does not claim to repair those separate failures.

## Single change

`EnemyField.edge_risk` retains the parent's 8-tick Coily arrival calculation and all its existing takeoff/near-landing penalties. For **currently rendered Coily no more than 3 ticks old, within two graph edges of the takeoff node, and after four inferred clears**, code additionally checks the existing graph pursuit set at a 20-tick horizon (the second reachability ring under this planner's Coily timing buckets). If the candidate landing is in that expanded set but not in the parent's 8-tick set, code adds only **0.25 risk** (`coily_two_hop_window`). It is a soft route cost, not a hard block or a claim of certain collision. Stale Coily memory, rolling-ball behavior, state-36-only observations, all other risk weights, target ordering, and the motor are unchanged.

When the term applies, the harness logs `two_hop_risk_applied` with the source, destination, action, and risk reasons. This provides evidence that the new branch ran; the actual action and subsequent loss/clear outcomes must still be inspected to establish usefulness.

## Jev state, questions, freshness, and fallback

The inherited compact Jev JSON contains `position` (node, standing state, `safe_moves`, last posted action), `progress` (inferred epoch/clears, unfinished count, target and posture), supplied `corridors` (action, landing usefulness, danger, escape width and route length), semantic visible/recent hazards, and the code-ranked fallback. Each asynchronous request batches two order-reversed `Choice` questions over the top supplied corridors plus `wait`, alongside `proposed_safe` and `keep_escape` `Noul` questions. Jev judges those options; it cannot create an edge, calculate the board, or return FIRE. Code requires agreeing Choices at probability ≥0.60, the parent's ≥0.62/≥0.55 Noul gates, current candidate membership, and the existing risk/cost tolerance; otherwise code uses the latest deterministic ranking.

At most one Jev call is outstanding, with the inherited 0.9-second timeout. Advice is tagged with node, life epoch, palette epoch, policy version, and hazard/scheduler signature, then rejected if the signature changes, it is over 8 ticks old at collection, or over 3 ticks old at execution. A hop, flash, fall, recovery, timeout, malformed/late answer, or missing safe edge uses the parent's NOOP/handshake fallback. Astra remains asynchronous and may only adjust its durable posture. The controller runs until actual game/context termination, never merely until the lives counter reaches zero.

## Confirmation and refutation

The hypothesis is supported if `two_hop_risk_applied` occurs while Coily is rendered and close; at least one reachable corridor ranking changes because of the added second-ring cost; the motor chooses a supplied safe alternative rather than continuing the node-13-to-edge route; and matched encounters are followed by fewer Coily-proximate losses or more scored clears without a broad early-game score regression. The event log alone is not success: compare chosen edges, verified landings, recovery events, and epoch tallies.

A preliminary live try of the final narrow rule on seed 0 reached **23,125** and six inferred clears, matching the parent seed-0 score. The new risk event fired at tick **2069** on the same node-13 choice, but the controller still selected RIGHT and later recorded the same tick-**2109** recovery loss. This is a concrete local refutation of the hoped-for intervention on that encounter, not a six-seed result; Jev's live answers and sticky-action realization make one try insufficient to assess the broader policy. A final-rule seed-5 try ended after three inferred clears, before the four-clear gate could activate, so it does not test the proposed late-board change.

The hypothesis is refuted if the term rarely intersects any supplied corridor, if it fires but never changes a ranked/action choice, if the same near-Coily edge sequence and loss recur, or if avoiding second-ring nodes reduces coverage without improving life outcomes. Losses in empty-hazard/state-36 windows persisting unchanged are expected and delimit this specific hypothesis; they must not be counted as confirmation.

