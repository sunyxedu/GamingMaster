# Candidate 0021 — certify the forced return before taking a tip

**Mode:** crossover. **Parents:** `candidates/0016/` and `candidates/0019/`.

## Hypothesis and concrete trace evidence

The controller can choose a legal, locally low-risk hop and still enter a place whose only continuation is a lethal return. The strongest parent, 0016, already has the right motor foundation: one-hop directional posts, neutral input during animation, landing verification, and a short corridor estimate. 0019 preserves that asynchronous motor and adds bounded rolling-ball memory plus next-junction Jev advice; it reached seven scored clears in seed 1. I keep both mechanisms, but change what the corridor certifies.

In `0016/trace_seed0.jsonl`, the tick-1592 observation has Q*bert moving onto node 20 and a rendered Coily estimated at node 9. Q*bert lands on node 20 at tick 1593, posts `LEFT` at ticks 1595–1596 to return toward node 14, and loses a life at tick 1603. The trace does not prove Coily caused the loss, but the timing and geometry make this a concrete missed warning: node 20 is a tip with only its incoming edge as a way out, and Coily was adjacent to the forced-return destination. 0016's exit counter intentionally excluded the incoming edge; a useful tip therefore escaped the no-exit penalty, and its future return risk was not evaluated as part of the decision to enter. In the same seed, the later losses at ticks 2148, 2972, and 3122 occur in observations reporting `purple_ball_state=36` and no visible hazards, showing that current-sprite legality alone also leaves an occlusion gap.

0019 supplied two useful components but did not solve those cases. Its 60-tick rolling descent shadow was activated only after four inferred clears, so it could not inform the repeatable 0016 third-clear loss. Its next-junction Jev pipeline produced 39 changed routes across 1,631 requests, while 1,103 answers were rejected by code. In 0019 seed 1, a tick-2778 decision entered the node-13 corridor at risk 6.2 with zero certified non-return exits; a visible Coily was near the branch and another life was lost at tick 2790. The experiment supports retaining prediction while a hop animates, but suggests the decision gate should be asked only when alternatives actually differ and should include a *forced return* fact.

This is not another blanket corner ban or an assertion that `purple_ball_state=36` locates an enemy. Lessons already show that strict endcap deferral and a soft rolling shadow by themselves did not improve the mean. The test is whether explicitly certifying the only retreat edge, using short-lived public-object reachability and a low-weight descendant cone from the first rendered ball sighting, prevents a subset of tip/branch losses without stalling coverage.

## What changed

### 1. Retain 0016's one-hop transaction and coverage base

The 21-node triangular graph is encoded directly. A candidate edge must be in the current `safe_moves` (or the legal geometry fallback). The motor posts at most two directional pulses while still standing, then holds `NOOP` until a standing landing is observed. It resynchronizes from the observed node after a mismatch and uses a bounded recovery interval after a life loss. It never uses FIRE or stops merely because the reported lives reach zero.

A follow-up five-minute live seed-0 smoke run with the zero-demand guard completed without runtime errors, but scored only **4,875** at 3,062 ticks and reached one inferred scored epoch. The harness log showed target persistence and repeated short local returns while still short of that board's target set; the ending observation had 15 incomplete cubes. This does **not** confirm the forced-retreat hypothesis. I refined target handling so that after reaching a cube which still needs another landing, it rotates to a different unfinished target before returning, and replaced lifetime visit counts with a bounded recent-node penalty plus a small immediate-reversal penalty. A one-minute check then exposed a cross-effect: next-junction lookahead calls the route planner at the projected landing, so target rotation must not mutate the live target/cursor. The harness now snapshots and restores those two fields around the speculative Jev candidate search. A short smoke rerun will check that branch; these tests are debugging evidence, not held-out evidence.

The board ledger uses nonzero `needed_landings` from the public decoder when it is a valid 21-element nonnegative array, with score/palette evidence and the scored flash tally as cross-checks. In smoke testing, the adaptor briefly emitted an all-zero vector in the palette handoff; accepting it as final progress stalled the policy at NOOP. The ledger now treats that all-zero vector as ambiguous, uses palette evidence only if it identifies unfinished cubes, otherwise keeps a forward-progress fallback until the scored flash confirms a board transition. This keeps revisits and Sam reversions representable rather than reducing progress to a permanent visited bit. A board epoch still requires the broad palette flash/quiet sequence and a meaningful score tally; `completed_rounds=0` is not taken as proof that no board cleared. The zero-demand fix also received a one-minute seed-0 recheck (4,275 points, 0.11% stale) and the target-rotation/cursor-snapshot version completed another one-minute check at 4,225 points, 0.22% stale. Both stayed on their first board within the short window; neither establishes whether the targeted late losses are reduced. A five-minute run before target rotation scored 4,875 and recorded three life recoveries, so the target-rotation change remains exploratory rather than validated.

### 2. Change the corridor certificate at tips

For every candidate, the router evaluates immediate takeoff/landing exposure and the risk of outgoing edges from its landing at the next hop horizon. It counts ordinary certified exits separately from the incoming edge. If the destination has only one outgoing edge, that incoming edge is evaluated as the **mandatory retreat** instead of silently discarded: a dangerous forced return adds a larger soft cost to entering the tip. The router can choose a different unfinished cube first, but it does not make a tip illegal or wait forever; the route remains progress-seeking and the final two required cubes reduce the deferral cost.

Visible Coily is expanded into a small undirected reachable-node set over a short horizon; visible Purple/Red Balls expand only along the board's downward children. A recent rendered rolling-ball sighting also adds a small, decaying descendant-cone cost throughout the episode, not only after four inferred clears. These are possibility costs, not claims about a hidden object's current location. Old sightings expire, and a board or life transition clears the belief state.

The deterministic fallback ranks immediate risk, mandatory-return risk, unfinished demand, route distance, and the number of safe non-return exits. Even if all choices are risky, it chooses the least-cost legal one rather than turning a risk score into an unbounded NOOP loop.

### 3. Keep 0019's in-hop advice, but request and accept it selectively

During an active hop, code computes at most four candidate edges from the projected landing. The compact Jev state contains:

```json
{
  "current_hop": {"action": "DOWN", "projected_landing": 7},
  "position": {"projected_node": 7, "options": [
    {"direction": "RIGHT", "landing": "unfinished phase",
     "immediate_threat": "recent ball cone", "next_escape": "several",
     "forced_retreat": "not_a_tip", "code_rank": "first"}
  ]},
  "threats": [{"kind": "coily", "sighting": "visible",
                "relation": "near", "motion": "possible_pursuit"}],
  "progress": {"scored_board_clears": 3,
                "unfinished_bucket": "several",
                "priority": "survive and complete required phases"}
}
```

Code computes nodes, distances, legal edges, risk buckets, and corridor descriptions. It does not ask Jev to infer a raw coordinate mapping or invent an action. One batched request asks the same `Choice` twice with reversed option order; options are the supplied directions plus `keep_code_choice`. Jev is called no more often than every 18 ticks and only when the next junction has a meaningful threat/escape distinction (including recent threat evidence), not on every equivalent clear hop.

On arrival, code accepts advice only if both choices agree, both selected probabilities are at least **0.66**, the actual node is the projected landing, life/board/policy epochs still match, and the answer is at most **12 ticks** old. It must name a current candidate and pass a fresh deterministic safety check: either at least **0.45** less immediate risk, or nearly equal risk (within **0.15**) plus one more certified exit at no more than **0.8** extra route cost. It may not worsen a tip's forced-return certificate. Any timeout, malformed/stand-in answer, disagreement, stale node, low probability, or failed gate uses the freshly ranked deterministic edge.

### 4. Astra sets only durable posture

Astra runs asynchronously and at most twice. It sees a compact score/lives/clears snapshot, recent cube evidence, public threat labels, and the present policy. It may choose `coverage_posture` (`spine`, `upper`, `frontier`) and `leaf_caution` (`cautious`, `balanced`), initially or after a life/board transition. Whitelisted JSON only is accepted; arbitrary text is harmless. Astra cannot issue joystick actions, routes, node numbers, or hidden-hazard claims, and its slow response never blocks the motor.

## Freshness and fallback policy

Jev work is tagged with projected landing, creation tick, life epoch, palette epoch, and route-policy epoch. The code does not demand an identical hazard signature after the hop—doing so would throw away useful advice whenever the observed world changes—but it re-evaluates the selected option against the latest risks and exit certificate. It discards all pending advice on death, board transition, or policy change. While any call is pending, control continues on the deterministic handshake. The last posted action is never allowed to extend through a hop because the motor posts NOOP during animation.

## Confirmation and refutation

**Support:** relative to 0016, the early scored-clear pace stays similar, but routes into tips are delayed specifically when the measured mandatory-return risk is elevated; logs show a lower death rate after those decisions and more fifth/sixth clears. The rolling possibility cost is exercised before the fourth clear and changes only local alternatives. Jev's changed-route rate rises among its selected meaningful requests, with few expired/contradicted answers and no loss of the code safety gates.

**Refutation:** the same death timing follows verified tip-return routes despite a low/elevated-risk certificate, or the new forecast makes the agent defer tips until progress stalls without reducing losses. If Jev still rarely passes the safety/escape gate, its added role is not useful. If the opening score drops materially, the continuation penalty is too strong or the public-object reachability envelope is too broad. A score change without a corresponding change in risk-certified decisions would not confirm the hypothesis.

The goal is not to prove Coily or state 36 was the cause in every late death. It is to test one falsifiable structural gap in the parent controller: a corridor should include the safety of the only way back, while the model's latency is spent evaluating only a consequential next-junction choice.
