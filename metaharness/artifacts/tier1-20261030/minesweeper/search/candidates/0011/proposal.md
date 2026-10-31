parent: 0008

## Hypothesis

Replace fixed-policy Monte Carlo continuation with **exact Bayesian decision search when the remaining posterior is small**. The parent values a candidate using three risky reveals under a fixed minimum-risk continuation policy. That can miss information whose optimal next action differs from the fixed policy, and finite samples can invent preferences between genuinely equivalent choices. A complete decision tree conditions on every possible visible clue and flood, then maximizes expected additional safe reveals until loss or completion. This is a different planning hypothesis, not a change to the parent's sample count, risk band, or rollout horizon.

## Evidence and limits of the diagnosis

The leader's three losses are all extremely late, with no episode error:

- `../0008/trace_seed602.jsonl`: action 229 is `{"type":"reveal","row":14,"col":0}`; terminal `safe_revealed` is 380 of 381. The preceding planner log gives both (14,0) and (15,0) `risk: 0.5`, `expected_gain: 0.5`, and `standard_error: 0.0`.
- `../0008/trace_seed608.jsonl`: action 246 reveals (0,9) and loses at 380/381. Both (0,9) and (1,9) again have exact risk .5 and estimated gain .5.
- `../0008/trace_seed611.jsonl`: action 17 reveals (8,1) and loses at 84/85. Its last planner log reports `chosen: [8,1]`, `baseline: [9,1]`, and equal .5 risk/gain for both. One decision earlier, all four candidates have risk .5 but sampled expected gains differ: .7299, .7132, .7689, and .752. The recorded estimates can therefore create a preference where exact analysis might show none.

Reconstructing **only the visible, nonterminal observations** confirms that each final pair has exactly two consistent completions, with one expected half-cell of progress whichever action is chosen. The preceding four-cell state in episode 611 has four consistent completions and exact optimal gain .75 for every action. These are genuine information limits, not broken arithmetic. An improved policy cannot identify the safe member from those observations. This experiment targets other small endgames where a full continuation tree can distinguish informative moves; it does not claim to repair an irreducible final coin flip. Any score movement caused solely by an exact tie is variance, not evidence for the hypothesis.

## Implementation

The parent's first corner reveal, safe deduction ordering, constraint inference, global posterior weighting, and early-board rollout planner are retained. When safe deductions stall with at most 24 unclassified hidden cells, the new branch:

1. Enumerates complete mine assignments satisfying every visible eight-neighbour clue, all previously proved mines, and the exact global mine count. Complete layouts have equal conditional prior weight. Limits are 4,096 layouts and 1,000,000 backtracking nodes; reaching either rejects the entire enumeration.
2. Precomputes each hypothetical reveal's actual number or zero-region feedback. Sampled or enumerated mines are used only by this simulator, never by the observation-conditioned policy or live fact state.
3. Groups surviving layouts by identical newly visible cell values. Each decision therefore receives only the information a real reveal would expose. A memoized Bellman recurrence sums safe cells opened in each branch and optimizes the continuation over every remaining candidate. Terminal mine branches retain earlier progress and add zero future progress, matching the score.
4. Reveals universally safe cells before gambling. This is lossless for the progress objective and reduces tree size. Universally mined cells are omitted from the action set, while still contributing to simulated clues. Memoization keys include the complete surviving-layout subset and hidden-cell mask.
5. Uses integer sums over the complete posterior, avoiding numerical/sample tie noise. If the existing risk/geometric baseline has exactly optimal value, preserve it; otherwise prefer the optimal action with lower exact mine probability and lower board degree. This is a general tie convention, not a coordinate/seed lookup.

Search is capped at 18 seconds, further reduced to one tenth of remaining episode wall time minus .5 seconds, and at 60,000 cached belief states. Cancellation is checked during enumeration, world construction, and recursion. An incomplete enumeration or incomplete root decision search contributes no live facts or preferred actions and falls back to the parent planner. The runtime also rebuilds residual equations after adding proven mines, avoiding stale residuals in the inherited probability call.

There are no model calls, model-output parsing, flags, background workers, direct backend calls, seed reads, archive reads, or cross-episode state in the harness. It uses only the Python standard library. It still performs one live reveal at a time and reparses the next observation.

## Validation

- Required preflight passed: score 1, 38 actions, zero model calls and no error. The warning that no System-2 answer was exercised is expected for a model-free harness.
- `verify_exact.py` compares complete posterior enumeration and every root Bellman value against an independent set/queue/fraction-free exhaustive oracle on 80 synthetic observed boards. It also compares three complete prior decision trees. All passed. Ten of the observed-board cases have equally minimum-risk actions with different exact optimal progress values, demonstrating that the objective can differ from immediate risk alone.
- Synthetic checks also cover zero floods, arbitrary previously proved mines, cancellation, rejecting an enumeration after its layout cap, and preserving either supplied baseline in a symmetric final pair. No game implementation or hidden archived layout was imported.
- One permitted dev evaluation was used: `dev/20261030-192223/`. Episode 2403 won in 34 actions / 7.87 seconds; 2404 won in 114 actions / 13.48 seconds. Both used zero model calls and had no errors. The digest and planning traces were inspected. **Neither episode invoked the new exact endgame branch**, so these wins establish regression behavior only, not superiority of the new planner. No second dev evaluation was necessary and no full training evaluation was run.
- Archived-state posterior probes used the public observations before the three losses, not terminal mine maps, and verified the exact symmetry described above. They were decision analyses, not reruns of the training evaluation.

## Confirmation or refutation

Support would be exact branches completing within the budget, selecting actions with strictly greater exact expected progress than the parent's fixed continuation policy, and higher progress on independent episodes that actually reach such endgames. Logs expose layout count, belief states, all exact root values, chosen/baseline cells, and elapsed search time.

The hypothesis is weakened if the branch rarely runs, most decisions are symmetric ties, search caps cause frequent fallback, or gains fail to generalize. The small-posterior cap leaves early uncertainty to the inherited Monte Carlo planner; its variable/component limits and three-gamble horizon remain limitations. The observed final 50/50s cannot themselves provide evidence of policy superiority.
