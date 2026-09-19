# Candidate 0002: Outcome-contingent experiment continuations

Parent: `0001`. One structural intervention: extend the parent's linear guarded batches into bounded, model-authored conditional trees.

## Hypothesis

The best candidate does **not** fail on the primary metric in the available archive. Its mean is already 2.0, both episodes complete, and neither errors. The remaining observed weakness is unnecessary reasoning after an experiment whose possible outcomes Astra has already anticipated. A linear plan can approach uncertainty and test it, but cannot attach different useful continuations to different observations. Consequently, recognizing one anticipated result costs another full System-2 call.

Allowing Astra to precompute a small number of exact outcome/continuation pairs should preserve completion while reducing calls. This is a control-flow change, not an attempt to tune exploration length or insert a puzzle solver. Success depends on whether Astra actually uses the new representation correctly; interpreter correctness alone cannot establish that.

## Archive evidence

Read both candidates' code, proposals, evaluations, and traces, including the completed baseline runs and both parent episodes. Line references below are physical JSONL lines in `../0001/` unless otherwise stated.

- `eval.json`: seed 0 completes two levels in 28 steps / 3 calls; seed 1 completes two in 22 steps / 4 calls. Seed 1 is the worse episode under the requested score-then-calls ranking, despite using fewer game actions. Both have `error: null`.
- Seed 1, `trace_seed1.jsonl:40`, explicitly anticipates alternatives: **“test downward entry to distinguish ordinary movement, obstruction, or transport toward the other orange beside yellow.”** Nevertheless its plan ends with `{"action":"ACTION4","before":{"agent":[[2,4]]},"uses":[],"probe":true}`. There is no representation for “if transported, do the already-understood next move.”
- Line 51 records transition 21: `agent` changes from `[[2,4]]` to `[[6,5]]`, `orange` changes from `[[3,4],[6,5]]` to `[[3,4]]`, while `yellow` remains `[[6,6]]`. The latter colour change matters to a correct branch guard: checking the agent alone would omit a selected state change.
- Line 52 says `"executed":5` and `"feedback":"Batch finished normally."` This was planned exhaustion, not a refuted prediction or interpreter error. Line 58 reports `"prediction_failures":0`.
- Line 53 spends another call, with recorded latency `19.66550612449646` seconds, to say **“Enter the adjacent yellow cell with ACTION1 to complete the final level.”** Line 54 supplies a one-action plan; line 55 reaches score 2.0. This is concrete evidence for an avoidable recognition/replanning boundary, although it does not prove the model will anticipate the exact observation under a new protocol.
- Seed 0, `trace_seed0.jsonl:67–68`, reports three calls, zero prediction failures, and full completion. Its two long checked routes demonstrate that ordinary linear batches already work and should remain available.
- The older baseline remains a useful control: `../0000-tier1_astra_naive/eval.json` records zero score, 15 steps, and 16 counted calls for each seed. The completed seed-0 trace's lines 44–54 repeatedly return `ACTION4`, spending separate calls on predictable movement. Candidate 0001 already fixed this general throughput problem; reverting to single-action reasoning would discard that gain.

Only two seeds are available, both labelled `teleport` in evaluation metadata. That metadata is used here to describe the evidence's limits, never read by the harness. There is no observed held-out result or demonstrated primary-score plateau across a broad seed distribution.

## What changed

An action may now end its plan segment with `cases`, each containing an exact post-action feature map `when` and a complete continuation `then`. The model can enumerate one useful outcome and leave all others to replanning, or provide several alternatives. For example, in an unrelated nonspatial state machine it could test a mechanism, continue a checked route if the resulting mode is ready and inventory has changed as expected, and request replanning if the mechanism stays idle. Names, values, meanings, and follow-up actions all come from the current model response.

Execution observes every action as before. Exactly one case must match. A case must specify every selected state feature that changed; unlisted state features implicitly assert no change. Missing selected paths, overlapping cases, or an unlisted result stop execution. Clock changes do not participate in that unchanged-state check. Boundaries always stop, and `ctx.done` always takes precedence. There is no wildcard, implicit common suffix, search routine, or executable model-generated Python.

The entire tree is validated before spending actions, including unselected branches. A branching action must be the last item in its containing list, cannot also assert a single `expect`, and has one to four alternatives. Every alternative tests at least one state feature. Limits are four nested branch layers, 64 total authored action nodes, 24 nodes per list, and 24 actions actually executed in a call. Empty `then` explicitly yields back to Astra. Continuations inherit the parent's preconditions, predictions, terminal checks, repeated-state checks, and four-consecutive-unchecked-probe cap. A matched branch can continue even after the final available model call.

The episode ledger retains exact branch alternatives and the selected index or fallback reason. `branch_selected` and `branch_fallback` events, plus aggregate hit/miss counters, make the intervention measurable. Retrieval now includes branch outcomes alongside prediction failures, recent steps, and the first/latest observation for each action. An unlisted alternative is recorded as an incomplete outcome set, not incorrectly asserted to falsify every action hypothesis. Ordinary failed `expect` checks still challenge associated hypotheses and cancel the suffix.

The rest of the parent design is retained: model-selected JSON-path features, optional value-location/count extraction, full current observations with row-oriented array formatting, recent scene snapshots, numbered measured deltas, untrusted hypothesis summaries, fresh model sessions, robust JSON/action parsing, and explicit remaining-call checks. State, action meanings, maps, and hypotheses stay local to one episode. Persistent memory contains only aggregate procedural statistics and a generic discovery method. No autonomous action selection is added: cheap exploration and continuations are explicitly authored by Astra.

Schema roles are restricted to the documented `state`, `boundary`, and `clock` values so a misspelled role cannot silently evade the new unchanged-state check. Imports remain standard-library only, and the entry point remains `build(ctx)` returning an object with `run_episode(game)`.

## Confirmation and refutation

The primary requirement is to maintain the parent's completion score. On these two archived seeds, a directional secondary-metric target is six total System-2 calls instead of seven, with a successful `branch_selected` continuation replacing seed 1's fourth call. This is a prediction, not a reported evaluation result. On broader and held-out evaluations, confirmation requires preserved or improved completion and fewer calls around experiments whose outcomes admit short, justified continuations.

Trace evidence should show the model authoring cases *before* seeing an experimental result, an exact match, and checked progress without an intervening model call. Count nonempty branch continuations separately from empty branches; a hit that immediately requests replanning does not save a call. Review completion and action use as well as call count, since speculation could increase costly game actions even while reducing calls.

Refutation includes lower completion, no model uptake of cases, frequent malformed-tree retries, or frequent unmatched/ambiguous outcomes that erase any savings. Increased prompt/response burden could also negate the latency benefit even if the ranking improves. The new full-tree validation can reject a useful prefix because a speculative branch is malformed; its frequency should be inspected. A selected branch followed by immediate prediction failure suggests continuations are insufficiently justified.

Feature selection remains the main generality risk: unselected observation changes are available as evidence but do not veto a continuation. Exact outcomes are also a poor fit for highly stochastic observations. Boundary changes still require a new call, and an unlisted result on the last call cannot be repaired. This proposal does not claim to solve any of those limitations. Held-out performance is essential: no specific grid key, colour, dimension, movement rule, goal condition, or action identity is compiled into this harness.

## Local validation

No model calls, target-game executions, or long experiments were run. Syntax and standard-library imports were checked. Deterministic fake contexts with nonspatial, differently named observations passed checks for unique outcome selection, continuation with the final model call, unknown outcomes, changed-but-unlisted state, overlapping alternatives, missing paths, phase changes, empty continuations, terminal stopping, invalid branch actions/features, forbidden shared suffixes, conflicting `expect`, clock-only branch rejection, ordinary prediction-failure stopping, the four-probe cap, zero remaining calls, parsing, scalar action arguments, and tree size/depth limits.

Offline replay of the parent's recorded model replies and observation deltas reproduced its action sequences: seed 0 used 28 steps / 3 recorded replies; seed 1 used 22 / 4. A second seed-1 replay manually attached its already-recorded final plan to an exact case on the preceding probe. The new interpreter then consumed only three supplied replies and executed the same 22 actions through termination, including the selected-state colour change. This is a hand-authored, hindsight-based interpreter test, **not** a new score or evidence that Astra will independently produce that branch. Replay fixtures and game-specific values were not embedded in the submitted harness.
