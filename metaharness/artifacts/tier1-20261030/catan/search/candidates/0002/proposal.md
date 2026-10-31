parent: none

## Hypothesis

The baseline loses before strategy can be tested. In seed 101, the model response is exactly `build_road [19,21]`, while step 2 posts `build_road`. The returned observation says `Illegal, malformed, ambiguous, or stale action; choose a current legal_actions ID.` Revision remains 3. Every training seed finishes at one VP and zero normal turns. Its serialized prompt also ends with `…(truncated)` before the legal-action menu. The parser and observation presentation jointly destroy actionable information.

A different question from slot 1's fast action-ranking policy: **can Astra manage a compact portfolio of complete economic projects while a resource-aware executor handles the many intervening decisions?** A useful unit of reasoning is a settlement plus its required road route, or a city plus its exact maritime trades, rather than a single affordable action. This should conserve scarce wheat/ore, prevent gratuitous roads, and spend model latency on strategic alternatives.

## Design

Fresh implementation using only public observations. Code computes production by resource, pip-weighted sites, shortest unblocked expansion routes, build costs, exact deficit-filling maritime trades, and an expected-round affordability estimate. Projects include cities, connected settlements, development purchases and short routes that can take Longest Road. A bounded edge-trail search respects enemy building interruptions. Approximate estimates are labeled as such; actual awards and legality remain authoritative.

Astra chooses opening placements and a persistent economic mode/project preference, then revisits it after completion, loss of the target, a changed victory race, or seven own rounds without score growth. Prompts contain the entire relevant legal menu and derived features, rather than a blindly truncated raw map or growing chat history. At most twelve 300-second calls are attempted, with 380 seconds reserved before admitting each call. The measured 182-second maximum is below the timeout. Calls are fresh sessions with explicit compact memory.

The executor refreshes all project feasibility and the action menu after every action, submits only current exact IDs, and checks `ctx.progress()` plus building/road/trade postconditions. Rejections or mismatched effects trigger an evidence-bearing Astra replan; rejected actions are excluded until the revision changes. Invalid model JSON and model exceptions fall back to recomputed projects. Resource cards and robber/discard decisions use observed state, not hidden hands. The model can override with one current legal ID when recovery needs a tactical move.

## Tests and falsification

Support: reach completed games with zero rejected actions, substantially more than one VP, and a small number of strategic calls. Stronger support requires wins against the three search bots; report wins separately from mean terminal true VP. Refutation: the affordability relaxation repeatedly favors unproductive savings, cheap development purchases starve city builds, opening choices leave no expansion path, or project invalidations consume most of the clock. Those are specific places to improve the project abstraction, not reasons to memorize seed coordinates.

Preflight and development results will be appended after the complete files exist. No hidden state, game implementation source, or held-out results are used.

## Development evidence and revision

The first dev invocation (`dev/20261031-053752`) was interrupted after long service admission delays; it produced no completed episodes or digest. Its seed-302 first request took **827.5 seconds** from request to response, while the trace reports **68.1 seconds** of inference and a **759.3-second** gap before `t_start`. A 300-second backend timeout therefore did not bound the harness's total wait. The response itself was valid JSON, selected a live settlement ID, and both that settlement and the scripted road were accepted. This is interface/latency evidence, not a gameplay score.

The revision submits inference through `ctx.spawn` and bounds the foreground wait at **210 seconds**, above the archive's 182-second maximum observed inference latency. Only one advisor call can be outstanding. The executor resumes on the soft limit and polls the future after subsequent actions. A late answer can update strategy only; it can post an action only if its request revision still matches. Calls in flight at episode end are left to the evaluator's documented cancellation. No custom background loops are started. Initial-road scoring was also corrected to measure expansion improvements caused by that specific road, so the first settlement's existing network cannot mask a poor second road.

Preflight passed before and after this revision (`ok: true`, `ended_by: game_done`, 67 steps). Its stand-in episode scored 1.1, but this is **not** a development/training win and should not be treated as policy-strength evidence. The short preflight budget admitted no model calls, so separate observation-based tests exercised malformed JSON, live-ID fallback, exact city funding, empty-bank refusal, and 2:1 port conversion.

The completed development pair before the final conservative patch produced one win (seed 301, score 1.10, true VP 10, 103 steps) and one loss (seed 302, score 0.05, true VP 5, 109 steps), both valid and with zero illegal-action errors. Seed 302 is a concrete refutation of unconstrained award pursuit: the engine withheld Longest Road around occupied endpoints while the local graph estimate continued adding roads. The patch now commits to a funded settlement route and disables award-only projects after a contradicted postcondition; focused replay checks pass, and the final preflight remains valid.

Development win rate was **1/2 (50%)**, mean terminal true VP **7.5**, and mean composite score **0.575**. Neither episode received an advisor answer before termination (one submitted call each, zero S2-informed actions), so this validates executor viability and delayed-call survival, **not** Astra's strategic contribution. Both permitted dev invocations are used; the final commitment/award patch has focused recorded-observation checks and passing preflight, but has not been rerun on development games. Final selection remains the evaluator's training run.
