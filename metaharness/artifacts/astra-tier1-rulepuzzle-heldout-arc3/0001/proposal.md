# Candidate 0001: Evidence-carrying action batches

Parent: `0000-tier1_astra_naive`.

## Hypothesis

The immediate failure is a mismatch between the granularity of reasoning and the two budgets. The baseline spends one scarce System-2 call per game action, including already understood movement. It runs out of calls before finishing an otherwise plausible route. A model-authored program of actions with checked intermediate predictions should turn each call into several useful steps while preventing an incorrect hypothesis from wasting the rest of a batch. Explicit transition evidence should let fresh reasoning sessions retain what was learned without carrying an ever-growing transcript.

This is one structural intervention: replace the single-action conversational loop with an evidence-carrying, guarded batch interpreter. Batching and its evidence/checking machinery are evaluated together; this candidate does not isolate their individual contributions.

## Concrete archive evidence

Both entries in the parent's `eval.json` have `score: 0.0`, `steps: 15`, and `system2_calls: 16`, with no errors. Each episode's final info reports `actions_used: 15`. Against the advertised 40-action first-level budget, that leaves 25 game actions unused. The baseline calls `ctx.think` before every action and catches `RuntimeError` by breaking. The context implementation increments its call counter before checking the budget, explaining the recorded sixteenth call without a sixteenth successful model response.

The JSONL files contain appended runs, so the following line references identify the relevant completed runs explicitly:

- `trace_seed0.jsonl:34–64` is the final completed seed-0 run. Lines 34, 36, 38, and 40 return `"ACTION1"`, `"ACTION2"`, `"ACTION3"`, and `"ACTION4\n\n"`. Subsequent observations show the controlled marker moving `(1,1) → (1,0) → (0,0) → (1,0) → (1,1)`, an informative four-action discovery cycle.
- In that same run, lines 44–54 repeatedly return `"ACTION4\n\n"` while the observed marker advances across row 2 from column 1 to column 7. These six predictable moves consume six separate reasoning calls. Lines 56–62 then continue downward. Line 64 records `"score": 0.0, "steps": 15, "system2_calls": 16`.
- `trace_seed1.jsonl:5–35` is the completed seed-1 run. Lines 11–17 show successive `ACTION4` responses moving downward; lines 19–29 show successive `ACTION1` responses moving right. The final request at line 33 still contains `"actions_left_this_level":26` and returns `"ACTION4\n\n"`. Line 35 again ends at 15 steps, zero score, and 16 counted calls.

These traces support a call-throughput diagnosis more directly than a failure to discover basic movement. They do not establish that either route would solve the hidden mechanics, or that unguarded batching would succeed. The held-out game remains untested.

## What changed

1. **Fresh, self-contained reasoning calls.** Each call receives the game description and action specification, the complete current observation, model-selected features, a strategy note, hypotheses, numbered transition evidence, recent scene-start observations, repeat-state counts, and executor feedback. Rectangular scalar arrays get one row per line; original keys and values are preserved. Sessions are not resumed.
2. **An observation schema supplied by Astra.** The interpreter supports ordinary JSON paths and optional exact-value location/count operations. Astra identifies state features, phase boundaries, and volatile counters from the actual observation and description. There are no built-in field names, colours, directions, objectives, dimensions, terrain rules, or game solvers. Nonspatial observations use the same path mechanism.
3. **Cheap execution between calls.** Astra can request up to 24 actions. Known steps carry exact feature predictions, optionally preconditions and hypothesis references. Initial discovery can use up to four consecutive explicitly authorized unchecked probes, recording every intermediate observation. The script selects no game-specific exploration actions on its own. A plain, unchecked action executes once before replanning.
4. **Cancel on counterevidence.** The executor checks every precondition and predicted feature. A mismatch, phase change, unexpected selected state-feature change, or repeated state cancels the remaining route. Prediction failures mark associated action hypotheses as challenged and retain the observed counterexample. This is a challenge to the claimed context/effect, not proof that the action is globally useless. Explicit experimental probes can change state without triggering the ordinary unexpected-feature stop.
5. **Measured episode memory.** The ledger stores before/after features and generic JSON deltas for every executed action. Retrieval favors recent steps, the first/latest trial of each action, and counterexamples, capped at 48 records. Large deltas explicitly flag overflow. Scene-start snapshots preserve earlier maps without a map-specific representation. The bounded model-written hypothesis list remains distinct from measurements.
6. **Robust parsing and budget accounting.** Fenced or prefixed JSON, plan arrays, and a conservative bare-action fallback are accepted. Names are matched exactly against `action_spec.names`, and scalar action arguments are preserved. Invalid plans never silently select the first action. The harness checks remaining calls before invoking Astra and uses only `ctx.act` to step the game.
7. **No seed-specific carryover.** `ctx.memory` retains aggregate episode/call/action/interruption counts and a generic discovery procedure. Schemas, action meanings, hypotheses, visited states, and maps are local to the episode. Rules must be rediscovered after each seed reset.

Only standard-library imports are used. The public entry point is `build(ctx)` returning an object with `run_episode(game)`. No game implementation, private game state, evaluation metadata, alternate solver, or additional model is accessed by the harness.

## Confirmation and refutation

The evaluator should first compare mean completed levels, then System-2 calls. Confirmation would be a positive mean score with multiple actions executed per call, at most 15 counted calls per episode, and successful continuation after scene changes. The strongest trace evidence would show short discovery batches followed by checked multi-step routes, with occasional mismatches cancelling the suffix and informing a successful repair. Held-out improvement is necessary to support the generality claim.

The hypothesis would be weakened if throughput rises but score stays zero: the bottleneck would then be rule inference or planning quality rather than call granularity. Frequent `plan_parse_error` events, invalid feature checks, one-step batches, or excessive prediction failures would show that the declarative protocol is too demanding. Frequent false interruptions would implicate schema choice or over-sensitive guards. Long batches that miss critical changes would show insufficient state features. Transfer failure would refute the proposed observation-schema abstraction even if this puzzle improves.

The interpreter only checks the features Astra chooses; raw deltas are evidence for the next call, not a complete learned simulator. Incorrect feature selection can miss important changes. Exact predictions may also cause unnecessary replanning in stochastic games. A last-call plan interrupted by a new scene or a failed check cannot be repaired without another call. These are deliberate limits to inspect in the evaluation, not claims of solved general planning.

## Local validation

No model calls, target-game episodes, or long experiments were run. A syntax check and deterministic fake-context checks passed for: arbitrary matrix/key names; checked multi-action execution; scene-boundary suffix cancellation; terminal stopping; a nonspatial observation with different keys; failed prediction evidence and hypothesis challenges; action arguments; the four-probe cap; avoiding an over-budget model call; fenced/prose JSON and bare-action parsing; rejection of out-of-vocabulary names; and absence of action semantics in persistent memory. These establish interpreter behavior, not game performance.
