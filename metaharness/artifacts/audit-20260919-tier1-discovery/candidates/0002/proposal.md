# Candidate 0002: compile episode rules into a reusable checked policy

Parent: `0001-previous_astra_winner`. This is one structural experiment: replace repeated
concrete route generation with an episode-local program that Astra writes from public evidence.
The host remains a generic feature extractor, bounded data-language interpreter and checked executor.

## Diagnosis and evidence

The parent does **not** fail the primary metric in the archive: both episodes complete both
levels, with no reported errors. Its weaker episode is seed 0 by the secondary metric: four
System-2 calls versus three for seed 1 (`0001-previous_astra_winner/eval.json`). Both use 36
actions. The improvement opportunity is avoidable reasoning calls, not an invented score failure.

Two concrete control-flow boundaries discard useful computation:

1. In `0001-previous_astra_winner/trace_seed0.jsonl`, line 48, Astra explicitly predicts the
   outcome `{"when":{"agent":[[5,4]]},"then":[]}`. Transition 33 at line 75 selects precisely
   that outcome: `"selected":1,"matching_indices":[1]`. Line 76 reports
   `"continuation_length":0`; line 77 says
   `"Matched outcome explicitly requested replanning (empty then)."`
   The fourth call, line 78, supplies just three further actions, including another conditional
   test. The anticipated outcome was recognized correctly, but its empty continuation still
   required an expensive reasoning call. There is no demonstrated prediction failure here.
2. Line 47 in **both** parent traces cancels the plan at transition 20 on a level boundary.
   The next calls regenerate routes using controls already learned in the same episode. Seed 1,
   line 48, describes its rule as `"expected to persist across levels"` and emits another
   16-action concrete route. A new layout needs a new computed route, but may not need another
   foundation-model inference if the route computation itself is retained.

The baseline establishes the other extreme: `0000-tier1_astra_naive/eval.json` reports only
one level per seed after all 12 calls and 12 actions. Its trace endings, line 26 in each file,
confirm those counts. Keep the parent's observed-action checks and batching; do not regress to
one reasoning call per action. The parent's stored proposal only says it is a frozen prior winner,
so the diagnosis above comes from its actual code, evaluations and traces.

Only public descriptions, observations and transitions motivate the design. Evaluation-only
hidden-rule metadata is not used to supply rules or initialize the new harness.

## Hypothesis

After a short discovery phase, Astra can express the rules it has inferred and the procedure for
choosing the next action as a compact reusable program. Running that program against each fresh
observation should eliminate calls spent recognizing anticipated outcomes and generating another
route through a new scene. Exact action checks should preserve the parent's score while returning
genuinely novel or contradictory observations to Astra.

The targeted result is two calls on a straightforward episode: one for exploratory actions and
one to interpret their outcomes and compile a policy. This is a prediction to test, not an
evaluation result. Difficult mechanisms can still require further calls. Merely making the
existing prompt demand longer branches would leave the compulsory new-scene call intact.

## Changes from the parent

- **Astra supplies reusable computation.** A reply may contain `policy`, a source string defining
  `policy(obs, memory)`. It returns the parent's bounded plan format. The host invokes it again
  after a batch, a matched empty continuation, or a scene change. Prefer one checked action per
  invocation so decisions can use the latest observation. Explicit plans remain the discovery
  and repair fallback. The host contains no spatial search, action mapping, terrain interpretation,
  objective detector or built-in observation field name. Any such computation must be supplied
  by Astra from the current episode's public inputs.
- **Small interpreted language instead of unrestricted generated Python.** Ordinary functions,
  loops, containers and arithmetic suffice for many bounded search or rule procedures. No
  `exec`, `eval`, imports, arbitrary attribute access or access to ctx/game/files/network is
  provided. Each invocation has 60,000 AST evaluations, a recursion bound and collection limits.
  Expanded output and scratch data are checked before serialization. Unsupported syntax or
  runtime errors spend no game action and return the source/error to Astra for repair. These
  limits constrain computation; they do not prove a generated strategy correct.
- **Checked execution remains authoritative.** Actions are validated against `action_spec`.
  Guards run before each action; exact predictions and unchanged state features are checked
  afterward. Cases require a unique exact match and an unchanged frame for other state features.
  A mismatch, unlisted outcome, missing feature, invalid plan or runtime failure disables the
  compiled policy. Related hypotheses are marked challenged on observed prediction failures,
  and the transition is retained. Unchecked exploration is capped at four actions per model
  call across all policy invocations, rather than resetting with each generated batch.
- **Scene changes trigger local recomputation.** They discard the old suffix and clear scratch
  caches and repetition counters. The policy remains installed and receives the complete new
  observation. An expected scene transition is not counted as a refutation of an old-scene
  position prediction. If the policy cannot interpret the scene it can yield an empty plan
  with a reason; if it proposes a wrong action, the ordinary checks invalidate it.
- **Evidence stays explicit.** Retain first/latest examples of each action, recent transitions,
  counterexamples, branch outcomes and scene changes; show complete current observations with
  row-formatted scalar arrays, raw diffs, extracted features, hypotheses, strategy, the last two
  scene starts and the previous policy source when repair is needed. Scene snapshots serve as
  maps when the public observation includes one. Visit and state/action counts detect repetition
  while excluding model-labelled clock fields. Repeating the same state/action twice forces
  fresh reasoning; this is a generic heuristic, not a theorem about hidden state.
- **Contract corrections.** Inference uses only `ctx.think(..., session_id=None)`, as in the
  archived runtime. No inference exception is caught, so `ModelUnavailable` propagates. The
  hard fallback cap is 12 calls, further restricted by the available runtime budget. A working
  policy can continue after the final call. There are no spawned tasks or outstanding futures.
  Every applied action logs its originating model call and whether it came from the compiled
  policy or an explicit model plan; installation alone is not counted as a control action.

The manifest declares tier 1, `system1: null`, `max_system1_calls: 0`, and episode memory. This
candidate therefore uses synchronous stepping and no System-1 calls. It starts with no policy,
does not load previous `ctx.memory` entries, and stores only current-episode controller statistics
there. Learned action semantics and schema remain local to this run. It makes no tier-2 latency
or shared-memory claim.

## Confirmation, refutation and risks

Confirm the hypothesis if the evaluator preserves mean score **2.000** while reducing mean
System-2 calls below **3.5**, especially seed 0 below four calls. Traces should show actual
`control_action` events with `origin: compiled_policy`, continued compiled actions after
`scene_replan`, and recovery from anticipated branches without an intervening `model_call`.
Report action counts, source size, model latency, compilation errors, policy yields and
invalidations alongside the primary metrics; fewer calls may still mean longer individual calls.

Refute it if compilation/repair calls erase the savings, generated policies fail to cover new
scenes, scores decline, or Astra repeatedly falls back to explicit routes. A decline in the
evaluator's generic-game performance would also refute the generality claim. No held-out game
has been inspected or tested during this proposal.

The main risks are Astra's ability to generate valid code in the deliberately restricted language,
fuel exhaustion in expensive searches, insufficient state features allowing undetected surprises,
and local repetition detection interrupting legitimate revisits. A program may also preserve a
wrong hypothesis across scenes until its first contradictory step. Checks limit continued damage
but cannot undo that first action. The language and check machinery add complexity; this experiment
must earn that complexity through measured call savings, not just successful compilation.

## Local validation performed

No new model calls or game evaluations were run. Syntax parsing passed. A synthetic controller
fixture with arbitrary observation keys exercised compiled reuse over a scene change: four
checked actions used one stubbed model reply. Another fixture refuted the first prediction and
verified no stale second action and no call beyond 12. Further checks verified propagation of
a synthetic `ModelUnavailable`, bounded infinite loops, rejection of host-access syntax, helper
functions with a small arithmetic search, fenced JSON policy parsing and rejection of prose as
an action. Output/scratch checks additionally reject cyclic data and excessive expansion.

Replaying the two parent's **already archived** model replies against their recorded public
observation deltas reproduced all 36 actions per seed, with four and three stubbed calls
respectively. This checks backward compatibility of the explicit-plan executor, not the quality
of newly generated policies. Only the evaluator can establish the proposed improvement.
