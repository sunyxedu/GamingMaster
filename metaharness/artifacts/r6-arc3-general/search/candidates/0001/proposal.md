# Candidate 0001: faithful actions and measured, guarded planning

Parent: `0000-tier1_astra_naive`. One candidate; no game evaluation performed.

## Diagnosis from the archive

The parent scores 1, 0, 0 on seeds 0, 1, 2 and consumes all 40 Astra calls in every episode. Its code sends one action per call, renders observations through `obs_text`, parses replies against the initial action names, and supplies no explicit action-effect record. It also catches every `RuntimeError` around inference, which could hide model availability failures.

The strongest failure is an interface mismatch, not evidence that Astra cannot discover click-based rules. In `trace_seed1.jsonl`, model call 1 replies **`click 19 20`**, but the immediately following step is **`"action": "noop"`**. Call 11 replies **`click 43 44`**, again followed by `noop`. All **13** click replies in this zero-score episode become noops. The public action description in those same prompts explicitly says **`grammar: click X Y, each integer 0..63`**. Across all three parent traces, **24/24** click replies become noops. Astra's conversation contains its intended click, but the harness provides no explicit correction saying it actually posted noop. This can create false evidence about the game's controls.

The observation channel is also incomplete. All 40 seed-1 and all 40 seed-2 prompts end their observation inside the matrix with **`…(truncated)`**. Seed 2 has only 21 distinct complete prompt strings across 40 requests. Calls 1–7 all answer `up`; calls 18–22 all answer `down`. This does not prove those moves were wrong: some visible changes and repeated movements are legitimate. It does show that the entire reasoning budget buys only 40 requests, with no programmatic distinction between productive repetition, blocked movement, and a refuted hypothesis. Seed 0 first receives credit at request 38 (`action5`), leaving only two calls for subsequent levels.

## Hypothesis

Faithfully executing parameterized actions and presenting complete observations plus actual causal evidence will improve level completion. Guarded batches should then amortize reasoning across supported maneuvers without executing stale plans after a transition. The principal repair is the click interface; this candidate bundles supporting observation and control-loop changes, so a gain would not isolate their individual contributions.

This is an interface-and-evidence hypothesis, not a hypothesis about any particular puzzle mechanic. No game IDs, object roles, target masks, seed trajectories, or learned rules are embedded in the policy.

## Implementation

- **Actions:** Read `action_spec.names`, descriptions, and notes anew before every request. Recognize the public click example as a parameterized grammar, preserve arbitrary integer coordinates, and enforce bounds extracted from public documentation. Other action strings must match the current spec. Parse JSON plans, fenced JSON, and plain action lines without evaluating code. Malformed output gets a bounded legal exploration action, not a silently substituted noop. Trace parser fallbacks and every actual request.
- **Observation:** Discover rectangular integer matrices recursively by structure, with no dependence on field names. Compress only identical adjacent rows and columns, retain original inclusive coordinate bands and palette legend, and transmit every pixel. Preserve all other fields separately using JSON-pointer paths. Attach a current image when a supplied observation artifact exists. Components provide occupied-pixel click representatives; labels remain geometric, not semantic. Public animation artifact paths remain in metadata; this version does not read intermediate animation files.
- **Evidence:** Copy observation data before acting, measure pixel counts, changed bounds, recolors, pixel samples, and metadata changes. Store up to 256 transitions, 16 salient examples, state/action attempt counts, and action-effect statistics. Retrieve the latest batch, 12 recent events, salient counterexamples/progress, current-state attempts, and Astra's compact rule notebook. Fresh model sessions prevent stale full-observation histories from accumulating. The notebook asks for conditions, predictions, evidence IDs, counterexamples, and uncertainty.
- **Execution:** Astra chooses probe or execute mode. Probes stop at the first changed observation; execution can commit up to 32 actions, with a maximum repeat of 16 per entry. Check every result. Cancel the remaining batch on score/phase/action-set changes, a recovery state, explicit reset, a failed pixel/metadata/change prediction, two unchanged observations, or three within-batch visits to the same visible state. Read-only observation never pays an extra controller request.
- **Adaptation:** Astra identifies relevant metadata paths for phase, mode, telemetry, and recoverable terminal states from the actual observation and description. Recovery actions are legal-checked and limited to two automatic attempts per visible state before asking Astra again. Mechanics survive level transitions in the episode notebook; maps come from the new observation. No observation field names are hard-coded.
- **Cheap exploration:** Only absent/malformed/noop-only plans trigger the script. It chooses the least-tested current-state action, with occupied component pixels and coarse screen coverage as click candidates. There is no assumed inverse action, simulator, speculative graph traversal, or fixed opening. Useful navigation and rule composition remain Astra's responsibility.
- **Contract:** Standard-library imports only. Inference uses `ctx.think` exclusively; no Jev or background threads. Check remaining calls and wall budget, with a 40-call local ceiling. Model exceptions propagate unchanged, including `ModelUnavailable`. Only action `ValueError` is recovered; stop after three rejected requests. Replace this candidate's memory at each episode start even if the context is reused.

## Verification and evaluation criteria

Short synthetic checks passed using the prescribed Python 3.14 interpreter:

- Lossless grid reconstruction, thin features, and occupied click representatives for outlined shapes.
- All 24 archived click replies survive parsing and validation unchanged; bare, fractional, negative, and out-of-range clicks are rejected locally.
- Eleven malformed/plain/fenced stand-in reply forms complete without an unhandled parse exception.
- Batching and immediate interruption on progress, legal-action changes, renamed phase fields, and refuted change/pixel predictions.
- Mutable observation snapshots, invalid-action recovery, exhausted budgets, nonvisual renamed observations, and propagation of a `RuntimeError` subclass representing `ModelUnavailable`.

These are local stand-in checks, not the evaluator's official pre-flight or a game-score result. No model backend, live game, held-out result, or game implementation was accessed.

**Confirm:** mean levels completed exceeds 0.333 on the same development seeds; click requests in the trace match intended coordinates; legitimate multi-action plans increase actions per Astra call; explicit counterexamples lead to revised rules rather than repeated ineffective plans. Fewer calls breaks score ties.

**Refute or narrow:** clicks execute correctly and observations are complete, but completion does not improve. Then the bottleneck is more likely rule composition, goal inference, or exploration policy. If improvements occur only in click episodes, credit the parser repair before claiming that batching helped. If batches consume resources or die before a guard fires, require stronger predictions or shorter commitments. If guards repeatedly stop productive sequences because HUD pixels change or visuals alias hidden state, improve state modeling rather than merely raising repetition limits.

Remaining limitations: component summaries are capped at 64 per matrix (the full map is retained); visible fingerprints cannot prove underlying-state equality; a model can misidentify phase/recovery metadata; animation-only effects are not analyzed; no search over an executable learned mechanics model is implemented. The evaluator should determine whether these warrant the next structural change.
