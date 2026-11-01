# 0003 — Ground the Jev-only action vocabulary

## Parent and isolated hypothesis

Parent: `0000-tier2_jev_only`. Keep its architecture: one synchronous Jev `Choice` over the current observation, with no Astra, no background inference, and no strategic state machine. **Change only the action interface:** replace bare action names with observation-grounded, fully parameterized commands and execute the selected command through the documented job/latch protocol.

Hypothesis: the parent's immediate bottleneck is not inadequate deliberation or latency. Its vocabulary cannot express a valid resource-gathering action. A typed classifier cannot supply arguments absent from its options. Giving the same controller executable alternatives should produce accepted jobs, real inventory changes, and opening milestones.

## Trace evidence

All three parent episodes score zero. The simplest diagnostic episode is seed 1: it stays alive and has nearby wood, so combat is not needed to explain failure.

* `0000-tier2_jev_only/trace_seed1.jsonl`, line 3: Jev chooses `gather` (probability **0.37**, confidence **0.35**). Its options contain only action names, not complete commands.
* Seed 1, line 458, model-request clock **55.628 s**, seq **305**: `last_result={"id":24,"action":"gather","status":"failed","error":"bad_arguments"}`. Final request still reports `gather / failed / bad_arguments` (id 410), an empty inventory, zero damage, and zero deaths. All **3,153** model-request observations have `task=null`. Applied tick actions include **1,287** bare `gather` and **438** `cancel`.
* Seed 0, line 183, clock **26.116 s**, seq **136**: `last_result={"id":8,"action":"gather","status":"failed","error":"bad_arguments"}`. That same latched failure remains in **2,012** subsequent request observations. **1,562/1,799** applied ticks are bare `gather`; every request has `task=null`; resource maxima stay empty. Initially exposed acacia/oak logs are present.
* Seed 2 applies `noop` on **1,523/1,799** ticks; it offers no evidence that mere valid syntax solves exploration. This is an explicit expected limitation, not a reason to introduce a second route-planning hypothesis.

Parent median Jev latencies are 0.237–0.282 s, maximum successful latency 2.533 s; it already has only about 4% stale ticks. A speculative async redesign or Astra planner would confound the syntax repair. No `lessons.md` exists in this archive.

## What changed

A stateless compiler instantiates finite command alternatives from public observations:

* `gather b:x,y,z 1` from actual exposed block candidates, with inventory/tool checks; commands name a specific resource, visibility and reach status.
* `craft ITEM NEW_COUNT [TABLE_ID]` from exact bootstrap recipe arithmetic and observed feasible-recipe hints. This exposes possibilities rather than automatically imposing a build order. Carried-item checks use **audited inventory only**, never cursor/output/client previews.
* Fully bound equip, eat, attack, shield, placement, furnace, bed, container and safe-stand navigation commands where public facts support them. Workstation placement requires an observed reference block and local collision evidence of a clear destination.
* While a bounded job owns the hands, only continue (`noop`) and interrupt (`cancel`) are offered. A dead player gets respawn/wait. Stale snapshots get stop/continue rather than newly invented targets.
* Selected commands are rebound on the latest observation. After posting, wait for a tick so the action reaches the game before another selection overwrites the mailbox. Identical completed one-shots get an intervening, applied `noop` before retry. These are executable-command bridge requirements, not a second strategic controller.

No seed coordinates are embedded, no unobserved terrain is queried, and no game implementation is inspected. Runtime model work remains entirely through `ctx.decide`.

## System-1 interface

**State:** the parent's public observation object, unchanged: self/survival state, authoritative inventory (labelled client previews remain distinguishable), candidates/entities, local collision, feasible recipes, task/results, feedback, progress and freshness. Retaining raw state deliberately avoids adding an independent observation-compression experiment. Dynamic option descriptions attach resource/action semantics, current stock, and code-computed compass/range/elevation labels. Stand options also say whether they get closer to or farther from the nearest observed wood, food animal or hostile. These are immediate target descriptions, not a route plan or exploration memory: Jev need not invent IDs, subtract coordinates or calculate recipes.

**Question:** one `Choice`, id `act`: the original game rules plus an instruction to select the best *complete executable command*. Criteria keys are exact whitespace action strings; descriptions say what the command would do and its relevant observed conditions. `noop` is an explicit wait/no-match alternative. Menu size is capped below the provider's 255-option limit. Jev still decides which eligible resource/recipe/movement/defense action to pursue.

**Answer-to-action:** membership-check `answer.value`, re-evaluate the same command's current preconditions, then `ctx.act(command)`. Unknown/garbage output becomes `noop`. A fresh Jev answer precedes every normal posted action. Logs record the chosen and posted commands, veto reason, menu size, snapshot seq, winning probability and confidence.

**Thresholds:** preserve the parent's argmax selection (no new minimum model-confidence threshold); probability/confidence are diagnostic only. Hard gates: snapshot age <=2 s for new jobs; response snapshot lag <=10; valid observed target IDs; actual inventory; safe stand navigation <=48 blocks; visible melee target <=3 blocks; no new job while busy. No new hazardous manual controls are offered.

**Timing/fallback:** synchronous calls retain the parent's 5 s timeout, above the observed successful tail. Existing bounded jobs continue during a delayed call; already completed one-shots remain latched. Timeout/transient failure posts `noop` and waits for the next tick. Budget/episode/backend-stop exceptions propagate normally. There are no outstanding worker loops or unbounded background work.

**Astra revision:** intentionally none in this refinement. Adding a planner now would prevent attributing improvement to action grounding. A later candidate should test Astra-authored goals/questions only if this repair yields accepted jobs but stalls on strategic choices (e.g. repeated nonproductive gathering, exploration or crafting).

## Confirmation and falsification

Confirm the immediate mechanism if `bad_arguments` disappears for emitted commands, `task` becomes non-null during real jobs, successful results and audited inventory deltas appear. Confirm performance relevance if the opening score becomes positive across training episodes without new errors.

Refute sufficiency if commands are valid yet accepted jobs fail, repeated choices make no resource progress, or exploration still idles. Distinguish navigation/visibility failures from malformed syntax; do not claim dragon-solving from an opening improvement. This deliberately has no persistent exploration memory, furnace scheduler or combat planner. Unrecognized container preview layouts do not authorize transfers.

## Local checks (no live game or backend calls)

* Imported the candidate using the evaluator's Python environment and checked all three archived spawn menus: complete command syntax, observed target membership, explicit `noop`, and provider menu-size bound.
* Replayed **226** sampled parent request observations across all three episodes through the compiler; no exceptions or malformed parameterized commands. Largest sampled menu: 12 options. All had `noop`.
* Synthetic assertions check audited-vs-client inventory separation, exact bootstrap recipes, nearby-table requirements, safe placement, furnace command arity, dead/busy/stale states, latest-state revalidation, and the one-shot rearm barrier.
* Six short 12-action stand-in loops passed: first-option answers, non-string garbage, missing answer, wrong answer-container type, intermittent timeout, and intermittent runtime failure.

These establish contract/bridge robustness only. Live action acceptance, navigation success, Jev choice quality and score gains remain **unevaluated**; the evaluator will run the real episodes.
