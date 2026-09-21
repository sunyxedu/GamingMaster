# Snake — strategy prior for Astra, Jev, and generated code

Research date: 2026-09-21. Target: the real-time, empty, walled rectangular board in `interfaces.md`, one food, growth one, immediate tail release on non-growing moves. There are no opponents, builds, weapons, rotations, or adversarial counters in this variant. The relevant counters are geometric traps, greedy mistakes, and missed deadlines.

## Evidence and the central strategy

[John Tapsell's Nokia Snake project](https://johnflux.com/2015/05/02/nokia-6110-part-3-algorithms/) uses a Hamiltonian cycle, then accelerates food collection with shortcuts that preserve cyclic tail/body/head order and growth space. He disables shortcuts at high occupancy. This is a strong practical starting point, not a claim that every shortcut implementation is safe.

[chynl/snake](https://github.com/chynl/snake) describes simulating a route to food, checking escape afterward, and otherwise taking a longer route toward the tail. Its reported graph-search success is 94% on its own 6×6 evaluation, not a universal win guarantee or a benchmark for this adaptor. The [Snake-on-graphs paper](https://arxiv.org/html/2506.21281v1) provides a formal basis for Hamiltonian strategies. Arbitrary obstacles and already-entangled states require more than simply drawing a cycle on the board.

The recommendations below are this study's synthesis and engineering design. Numeric thresholds and budgets are initial tuning choices, not experimentally optimized findings. Implement exact movement, search, counting, arithmetic, and freshness checks in code. Ask models to choose policies from supplied evidence, not to calculate whether a coordinate is occupied.

## 1. Establish a safe traversal before chasing food

**Owner: code. Astra chooses the policy family and reviews invariants. Jev is unnecessary for the baseline.**

A directed Hamiltonian cycle visits every board cell once and returns to its start. If the snake initially occupies a contiguous section in the matching order, repeatedly taking the cycle successor is collision-free until board completion under this ruleset. It also eventually encounters every possible food placement. This gives a reliable score-first baseline, albeit a slow one.

For an empty rectangle with both dimensions at least two and at least one even dimension, construct a cycle directly. Do not launch an exponential Hamiltonian search on an ordinary 10×10 rectangle. One construction when height is even: reserve the leftmost column as a return lane, traverse the top row left-to-right, traverse each later row alternating right-to-left and left-to-right across columns 1 through width−1, then enter the bottom of column 0 and travel up to close the loop. Transpose for an even width instead. This construction is a mathematical derivation for the selected geometry. Verify coverage, unique cells, adjacent edges, closure, and body order before using it.

Opening: inspect the three-cell starting body and heading, choose cycle orientation/transformation that agrees with it, and start moving. If no candidate cycle contains the body in order, use simulated safe repositioning or another candidate cycle; never snap the snake onto a cycle or assume arbitrary starts satisfy the invariant. The default symmetric central starts make this easy but the controller must still check.

An odd-by-odd rectangular grid has unequal checkerboard color classes, while a cycle alternates colors, so no cycle can cover every cell. This parity argument is a geometry limitation, not proof that every individual Snake run there is unwinnable. Treat odd grids and obstacles as separate strategy regimes.

Derived progress bound: on an N-cell cycle a persistent food is reached in at most N moves; starting from length L0, full completion takes at most N(N−L0) moves. The proposed 2N² evaluator cap therefore does not cut off a correctly initialized pure-cycle policy. This bound does not apply to a controller that loops on a smaller cycle or breaks its traversal invariant.

## 2. Add shortcuts only with an explicit safety condition

**Owner: code for cycle indexing and validation; Astra for risk/efficiency tradeoffs; Jev may select among already certified candidates.**

Assign each cell its cycle index. Maintain that body cells, read from tail to head, appear in forward cyclic order even when there are gaps from previous shortcuts. A shortcut must be one actual grid move, respect the heading rule, and land on an unoccupied cell strictly in the forward cycle interval between head and tail. Check the full post-move invariant by exact simulation during development; do not rely on a hand-waved distance-to-food metric. Equality with the old tail is a separate case and is legal only when it departs and the resulting order remains valid.

Preserving order ensures an ordinary cycle-successor fallback remains available. Growth must be included in the post-state check: a food move retains the tail, reducing slack. For a multi-step shortcut, simulate every intermediate state and food consumption, not just its endpoint. Stop the known-food route at the food; a newly sampled food invalidates predictions beyond it.

For progress, prefer forward shortcuts that do not jump past the current food's cycle position. Otherwise a safe policy can repeatedly skip food. At 50% occupancy, initially disable discretionary shortcuts and use the verified traversal; tune later only with held-out evidence. The source of this conservative starting threshold is [Tapsell's implementation discussion](https://johnflux.com/2015/05/02/nokia-6110-part-3-algorithms/); it is not a universal optimum.

Common failure: after a failed shortcut test, selecting an arbitrary locally legal neighbor breaks the cycle invariant. Keep the cycle successor as the actual fallback, and validate against the current body each tick. If the invariant is already broken, explicitly enter a recovery policy; do not claim guaranteed safety.

## 3. Faster food pursuit with escape checks

**Owner: code. Astra selects when to use it; Jev can compare summarized candidate routes.**

On an open board, BFS finds a shortest path through a static occupancy grid; A* can use Manhattan distance as a lower bound for four-direction movement. Neither alone solves Snake because the body moves. For each proposed food route, simulate the ordered body through the route, including the retained tail at eating time. Reject any intermediate collision or forced reversal. Inspect the resulting escape opportunities, not just whether the food is reachable now.

Use head-to-tail reachability and available-space calculations as filters. A static path to the tail after eating is encouraging, but **not a proof of eventual completion**: the tail moves, narrow passages can close, and a small closed loop may exclude the food. Likewise, a large flood-filled region may have an unusable bottleneck. Prefer a verified cycle/order certificate when available, then bounded exact search, then these heuristics. The simulate-food-then-tail idea comes from the [graph-search reference](https://github.com/chynl/snake); the conservative distinction between heuristic and certificate is this study's requirement.

For higher difficulty, search states containing the whole ordered body, direction, and current food, not just the head coordinate. A cell that is blocked now may be free later; a release-time estimate changes when food is eaten. A time-indexed head-only route can still be wrong if it ignores the new body deposited by its own route. Bound search by an actual local compute budget, provisionally 5 ms per 200 ms tick, and preserve a precomputed fallback if it expires. Benchmark before widening the search.

## 4. Survive traps without stalling forever

**Owner: code for metrics and candidate generation; Astra for choosing recovery rules; Jev for qualitative comparisons of supplied flags.**

If a safe food route is unavailable, use a tail-following detour that opens space, or follow the certified full-board cycle. Avoid permanently circling a small pocket. Maintain ticks since last food and repeat-state counters. If there is no food progress after N ticks in cycle mode, investigate a violated invariant or food-skipping policy; a valid pure-cycle traversal should already have found it. Other policies may legitimately need longer, so this diagnostic must not become an undocumented game starvation rule.

Flood fill, bottleneck detection, exit counts and simulated tail-release timing help rank alternatives. A “reachable space smaller than snake length” rule is only a warning: moving-tail access can change reachable space. A single empty neighbor is only one-tick safety. Evaluate future access and avoid food in a pocket whose entry the enlarged body will block.

When no certified move exists, choose the best bounded-survival candidate using exact simulation first and heuristics second. Expose `safety=heuristic` or `safety=unknown`; do not fabricate a guarantee. Record the trap and let Astra inspect the failure offline.

## 5. Real-time division of labor

**Code:** Execute every movement tick. Own body simulation, legal/reversal handling, pathfinding, cycle construction, occupancy, arithmetic, candidate scoring, replay bookkeeping, deadline checks and fallback selection. The game environment itself must not perform this assistance. The local controller is part of the generated harness and should be measured as such.

**Astra prompts:** Before start, ask Astra to choose a score-first or efficiency-focused policy and identify its assumptions. Between episodes, provide failed-state traces, aggregate held-out completion/latency results, and ask for one concrete policy revision. During play, accept only slowly changing policy parameters or strategies that code revalidates. A useful prompt is: “Choose among the implemented policies using these performance summaries. State when the chosen policy is valid, its fallback, and what evidence would justify switching. Do not return tick-level turns.” Astra cannot safely micromanage 200 ms actions.

**Jev questions:** Optional, asynchronous, bounded choice over code-generated candidates. Supply qualitative labels and computed quantities, not raw geometry that requires arithmetic or search. The baseline should remain fully functional without Jev. Model calls may only add value in ambiguous heuristic recovery/efficiency choices; do not spend one call per obvious safe move.

Suggested candidate record fields: `candidate_id`, `policy_id`, `episode_id`, `basis_tick`, `safety_certificate`, `post_food_escape_label`, `bottleneck_label`, `progress_label`, `fallback_available`, and code-computed efficiency/survival metrics. Examples of typed questions:

| Type | Question and allowed answer | Use |
|---|---|---|
| Choice | “Under the supplied score-first preference, which certified policy candidate is preferable: `cycle`, `shortcut`, or `keep_current`?” | Optional policy preference; candidates are already checked. |
| Choice | “Which recovery candidate best fits these supplied safety and progress labels: `A`, `B`, or `abstain`?” | Rank heuristic options without asking Jev to plan a route. |
| Yes/no | “Does this candidate's provided summary conflict with the instruction to avoid uncertified shortcuts?” | Instruction consistency, not a collision oracle. |
| Score | “Rate this summarized recovery option's desirability from 0 to 3 using the supplied rubric.” | Batch independent alternatives; local code decides using returned scores. |

Issue several independent questions in one call when useful. Require the exact enum or bounded integer; invalid/missing outputs become abstentions. Attach an episode ID and basis tick to every request and response. When a reply arrives, regenerate/revalidate the selected policy or candidate on the live board; discard stale route identities. A Jev reply spanning three ticks must never inject a turn computed for the old head position. A batch of independent typed answers is not a sequence of future moves.

## Failure patterns and counters

| Mistake / situation | Countermeasure | Owner |
|---|---|---|
| Greedy nearest-food pursuit seals a pocket | Simulate growth and assess the exit before committing. | Code; Astra sets conservatism. |
| Treating the old tail as always blocked or always free | Match exact growth-dependent transition order. | Code only. |
| Perimeter wall-hugging as a complete strategy | Use a traversal covering the interior too; perimeter loops miss interior food. | Code; Astra explains policy. |
| Assuming a Hamiltonian path is a closed cycle | Verify the final edge and current body order. | Code only. |
| Shortcut lands safely but destroys future traversal | Validate the post-move cyclic-order invariant. | Code only. |
| Shortest tail path repeats a tiny loop | Seek a productive detour/full traversal; track lack of progress. | Code; Jev may rank labeled alternatives. |
| Multiple quick turns enable accidental reversal | At most one direction decision per tick, relative to executed heading. | Environment and controller code. |
| Model response targets obsolete geometry | Episode/tick checks and revalidation; local fallback runs continuously. | Code only. |
| Rewarding lifetime causes endless circling | Food/completion primary; efficiency separate; external cap labeled truncation. | Evaluator code; Astra reviews objective. |
| Frozen debug environment hides latency failures | Evaluate real-time separately and include controller stalls. | Evaluator code. |

## Recommended harness-search order

First establish the pure-cycle baseline and deterministic replay, including its initialization assumptions. Add cycle-preserving shortcuts next, evaluating completion and ticks independently. Compare against simulated-food/escape search only after basic rule tests pass. Add Jev only if its asynchronous preferences improve a measured outcome over the same local controller alone; Astra should inspect failures and adapt policy between batches. Use identical scenario distributions and disjoint held-out seeds, and report performance by board size and timing profile so a fast small-board success does not hide large-board failures.
