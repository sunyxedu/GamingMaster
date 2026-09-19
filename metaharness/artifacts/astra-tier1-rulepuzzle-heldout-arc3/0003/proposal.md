# Candidate 0003: Checked, reusable feedback policies

Parent: `0002`.

## Hypothesis

The best candidate has **no observed primary-score failure**: both archived episodes complete two levels, with three System-2 calls each and no errors. Its remaining structural limitation is that it delegates a route through one already observed scene, not a procedure for planning from new observations. It can anticipate a handful of exact outcomes, but an unseen scene requires another reasoning call even when the relevant action rules still hold.

Test one new delegation unit: let Astra synthesize a reusable feedback policy after a few discovery experiments. The policy computes one checked action from the current observation, can run generic search or experimental control logic, and recomputes after each result. With explicit authorization it survives a scene change, discarding scene-local memory. The harness itself contains no planner for a particular game. The experiment asks whether model-authored computation can replace repeated model reasoning while preserving the parent's ability to stop on counterevidence.

A narrow supporting correction distinguishes a scene replacement from a refuted within-scene prediction. Both belong to the same intervention: policies need a well-defined scope for predictions and local state when moving between scenes.

## Concrete archive evidence

Read the code, proposals, evaluations and traces for candidates 0000, 0001 and 0002. The parent episodes tie under the actual score-then-calls ranking; seed 0 uses more actions and seed 1 has the longer latency and the only recorded prediction failure. Both are relevant, rather than inventing a failing-score episode.

Physical line references below are in `../0002/` unless otherwise specified:

- `eval.json`: seed 0 has score 2.0, 28 actions and three calls; seed 1 has score 2.0, 22 actions and three calls. Both errors are null. Only two seeds are available, with no held-out results.
- `trace_seed0.jsonl:2` requests four discovery probes. Line 13 then says **“Reach yellow along black cells through row 2 and column 7. Replan when the level changes.”** Its twelve-step route ends with an explicit prediction `"expect":{"level":2}`.
- Seed 0, line 38 observes that predicted boundary: `level` changes from 1 to 2, the controllable feature resets from `[[6,7]]` to `[[1,1]]`, and the observation delta replaces parts of the board. Line 39 cancels with `"boundary":["level"]` and an empty `"prediction_failure":{}`. This is not an unexpected result or an exhausted action budget: the new scene starts with 40 actions available.
- Seed 0, line 40 spends another call, with latency `30.453629970550537` seconds, to produce another twelve-step coordinate route. Its strategy explicitly says **“Movement rules are supported by the previous level; checked steps will detect any changed behavior.”** This supports the distinction between reusable action knowledge and scene-specific routing. It does not prove that the model can write a correct reusable planner.
- Seed 1, line 38 records transition 16 as both a boundary and a failed prediction: the plan expected `position:[[6,6]]` and `goal:[]`, but the next observation already belongs to the next scene. Line 40's request therefore labels hypothesis h3 **“challenged: prediction failed at transition 16”**. The response repairs the interpretation: **“entering yellow completes the level, explaining transition 16.”** This is concrete evidence of attribution noise introduced by comparing an old-scene prediction against a replacement scene. The trace does not establish how much of the call's latency that repair costs.
- Seed 1, line 40 takes `56.52229905128479` seconds and authors a conditional experiment tree. Line 52 records `"event":"branch_selected","transition":21,"case":0,"continuation_length":1`; line 56 reports three calls, one branch hit and no branch misses. Candidate 0002's prior intervention actually works here. Its case mechanism should remain available, rather than being discarded in the new experiment.
- The older baseline's `eval.json` reports zero score, 15 actions and 16 counted calls in both episodes. Candidate 0001 raised completion to 2.0 with seven total calls; candidate 0002 preserved completion with six. The new proposal keeps cheap multi-action execution rather than returning to one model call per action.

There is no evidence that an unseen scene is safely traversable by replaying the old route. The proposed improvement specifically requires reading its new observation and recomputing an action.

## What changed

### A model-authored feedback policy

A response may now contain `policy`, a source string defining exactly `policy(obs, features, memory, event)`, instead of a static plan. The four arguments are copied, ordinary data: the current complete observation, the current model-selected features, a local JSON dictionary, and the last measured transition. The function returns one action with the parent's preconditions and predicted postconditions, a bounded set of exact possible outcomes, or an explicit request for more reasoning.

The suggested workflow is still to gather informative initial probes. Once sufficient rules are supported, Astra can put their episode-specific interpretation in a short policy and compute decisions from current state. For example, it may implement search over an inferred transition model, or a finite-state strategy for a nonspatial task. No graph construction, movement, terrain, colour, goal, numeric token, action identity, board dimension or observation field is built into the submitted module. All such choices, if useful, have to appear in a response based on that episode's actual evidence.

The policy is invoked again after each action, so one response may cover up to 160 actions instead of one 24-action static batch. This longer horizon is enabled by repeated observation and prediction checks, not by increasing the length of an unchecked sequence. The four-consecutive-unchecked-probe limit remains. A policy must guard at least one selected state feature. A missing feature path, malformed returned action, runtime error, repeated selected state, failed prediction, or unmatched/ambiguous outcome returns control to Astra. The normal 15-call limit is checked before model calls, while a valid installed policy can continue after the last available call.

Policy cases use the same exact matching and unchanged-state frame as the parent. Their `then` lists are empty because the continuation is a fresh invocation on the new observation. Static plans retain their ordinary nested `then` continuations. No automatic action is chosen when a response cannot be parsed.

### Explicit scene transitions

Continuing a policy across a boundary requires both `policy_across_boundaries:true` at installation and `boundary_ok:true` on the particular action. Otherwise, a boundary still stops execution. When authorized, the executor clears policy memory and visit counts, then calls the same policy with the replacement observation. Any previously cached route disappears; supported rules in the model-written source remain episode-local.

An old-scene state/clock comparison that cannot hold after replacement is recorded as `scope_ended`, with its expected and actual values. It is not automatically attributed to a false action hypothesis. Explicit predictions about boundary features are still checked and can fail; a wrong boundary prediction prevents continuation even with authorization. Within-scene mismatches retain normal hypothesis challenges. Terminal status always stops. This is scope management, not a claim that every boundary means success.

### Restricted computation and repair

Generated source is parsed with the standard-library AST module and interpreted by an explicit evaluator. It is never passed to Python `exec` or `eval`. The language supports ordinary data, arithmetic, conditionals, bounded loops, comprehensions and a documented set of builtins/data methods. There are no imports, external calls, arbitrary attribute access, helper functions, recursion, host objects, filesystem access or model/game handles in the policy environment. Only the harness can call `ctx.act` and `ctx.think`.

Each invocation has a 120,000-operation fuel budget; containers have an 8,192-entry limit. Source, integer size, output depth, traversed output size and serialized output/local memory are also limited. Inputs are copied. Errors are caught before spending the invocation's action and reported with the policy source for repair. These are bounds on this interpreter, not a claim to support arbitrary Python or an operating-system sandbox.

The model may still choose static checked batches and contingent plans. The prompt describes both interfaces and encourages a compact policy only once enough semantics have been learned. This is a larger policy-synthesis burden than the parent's JSON-only response, and that burden is part of the experiment.

### Evidence and memory

The parent's evidence architecture remains: complete current observations with scalar matrices rendered row by row; model-selected JSON paths and optional exact-value location/count extraction; numbered transitions with raw deltas; first/latest action trials, recent transitions and counterexamples; recent scene-start snapshots; hypotheses kept separate from measurements; and fresh self-contained model sessions. Failed-policy source is also shown for repair.

Policies add scene-local mutable memory for a search, cache or experiment stage. It is reset on replacement and never crosses episodes. `ctx.memory` contains only procedural statistics and a generic discovery method, including policy action/error/boundary-resume counts. It never contains policy source, an action mapping, a feature schema, a map or inferred seed-specific rules.

Trace events `policy_installed`, `policy_error`, `policy_yield`, and `policy_boundary_resume`, plus policy-tagged transitions and final counters, make uptake and failures observable. Standard-library imports only; `build(ctx)` returns an object with `run_episode(game)`.

## Confirmation and refutation

Primary confirmation requires preserving or improving completed objectives, including on the held-out game. On the two archived seeds, the directional secondary target is **at most five total calls rather than six**, with at least one nonterminal scene boundary followed by useful checked policy actions without an intervening model call. Four total calls would be stronger evidence if both episodes admit one discovery call and one reusable policy. Neither number is a measured result.

The clearest confirming trace would contain: measured probes; policy installation based on those measurements; successful checked progress; `policy_boundary_resume`; different actions computed from the new observation; and continued progress. Count resumed actions and objective completion, not boundary-resume events alone. Code that simply returns `replan` at the next scene has not demonstrated the proposed benefit. Inspect game actions and latency alongside calls: search may save calls yet choose longer routes or increase synthesis latency.

Refutation includes lower completion, no model uptake, repeated unsupported syntax or resource exhaustion, expensive repair calls that erase savings, or policies relying on unjustified rules. Frequent `policy_yield` on replacement scenes would show that the program lacks reusable planning. Prediction failures soon after boundary resumption would suggest inappropriate transfer of episode rules or inadequate state features. Held-out failure despite target success would refute the generality claim.

Important remaining limits: checks only cover model-selected features; raw deltas do not automatically invalidate a policy. Incorrectly labelling a changing value as a boundary can mis-scope a prediction. Unchecked probes remain permitted within a short cap. The restricted language is not full Python. State-repeat detection can interrupt legitimate revisits. The model still has to discover mechanics, write correct code and request help for unknown contexts. Two successful archived seeds cannot establish that capability or broad transfer.

## Local validation

No new model calls, target-game executions or long experiments were run. Syntax and import checks passed. Deterministic fake contexts verified:

- A model-shaped graph-search policy solves a nonspatial two-scene task with different node names, connectivity, goals and required action sequences using one prerecorded reply. It continues after the final available model call; boundary memory reset and both required authorizations are checked separately.
- Checked failures revoke the policy and reach a static repair with counterevidence and challenged hypotheses in its prompt. Policy cases dispatch on new observations and reject unknown/ambiguous results. Unchecked probes stop at four, and repeated states stop a policy.
- Illegal actions, bad preconditions, missing feature paths, unsupported syntax/calls/attributes, endless loops, oversized sequence allocation and cyclic output/memory cannot cause an action from the rejected invocation. Supported loops, comprehensions, collection operations and copied-input isolation work.
- Fenced/prefixed policy JSON parses, action arguments survive, and zero remaining calls cause no model invocation.

Offline replay reconstructed observations from the parent's recorded initial observations and deltas, consuming its existing replies. It reproduced seed 0's 28 actions / three replies and seed 1's 22 actions / three replies through termination. The synthetic graph policy and replay fixtures are outside the submitted module. They are interpreter tests, not newly sampled policies, game scores or evidence that Astra will independently synthesize a successful controller. Evaluation remains to be run by the evaluator.
