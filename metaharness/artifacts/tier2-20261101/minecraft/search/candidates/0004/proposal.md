# Candidate 0004 — compiled milestone options

Parent: `0002-tier2_idle`. Structural rewrite; no parent controller is copied.

## Hypothesis and evidence

The flat leaderboard is consistent with a missing action-argument bridge, not merely weak game strategy. In `0000-tier2_jev_only/harness.py`, the model chooses from `action_spec.names`, and the selected name is posted unchanged. Its seed-0 trace explicitly records tick 2 action `navigate` without a candidate argument, followed by tick 3 `cancel`. The public contract rejects bare parameterized actions. The same initial observation contains exposed `acacia_log` and `oak_log` candidates, so at least that opening is not blocked by a lack of observed wood. The idle parent does nothing and scores zero. The Astra-only trace begins with repeated stale noop ticks while its first model request is outstanding. No lessons.md was present at the archive root.

The experiment therefore moves argument construction, recipe dependency bookkeeping, job ownership, repetition latches, and safety interrupts into code. Jev chooses among concrete resource actions; Astra chooses only the macro resource priority. A valid wood-to-table chain should beat the flat baseline even if neither model produces useful advice. This is a partial opening/iron controller, not a claimed complete dragon solver.

## Control structure

Code derives inventory counts, candidate geometry and distances, prerequisite crafting, exposed resource choices, placement support candidates, furnace timing, and action strings. It attempts logs/table, a placed workstation, wood/stone picks, opportunistic food/bed collection, exposed iron, real smelting and a shield. It explores only listed stand candidates using bounded navigation, and hunts only observed visible passive animals. Gather actions carry observed candidate IDs. There is no seed lookup, privileged map access, console use, model-client bypass, or invented inventory.

A live bounded job receives noop until finished; emergency response first cancels it. Identical one-shot commands are separated by noop. Feedback failures temporarily suppress the last attempted action. Stale observations stop work. Drowning and visible close hostiles have deterministic survival interrupts. These guards are intentionally modest rather than a claim of complete hazard avoidance.

Astra runs in one background future, at most approximately once per 200 seconds and only with enough wall time left. It receives stage, inventory, player state, feasible options, recent actions and failure count. Its short validated JSON enum (`balanced`, `food`, `bed`, `iron`) reorders resource alternatives after prerequisites are met. Invalid output retains the previous priority. Its 160-second timeout allows for the archive's slow inference without blocking the tick loop. It does not generate code or revise arbitrary Jev prompts: this bounded revision hypothesis concerns changing which resource alternatives Jev prioritizes after local progress or failure evidence changes.

## Jev interface

Each request is a compact JSON object with `stage`, `macro_priority`, a code-derived health band, visible-hostile close/far labels, an option map of fully instantiated actions and their purposes, and the relevant navigation rule. Jev sees no inaccessible references or unbounded raw terrain dump.

Two independent questions are batched:

1. `action`, a Choice over at most six compiled options plus `unknown`, judges the best immediate resource opportunity.
2. `safe`, a Noul, judges whether the supplied hostile labels support ordinary resource work.

A selected action is accepted only when its winning-option probability is at least 0.60, safety probability at least 0.65, the answer is at most two clock-seconds and ten snapshots old, episode and macro-plan version agree, and the action still exists in the newly generated feasible menu. Otherwise code uses its prerequisite-first default. The model cannot invent a command or numeric parameter. Logs identify when Jev changed the default and when Astra changed the priority.

One Jev future may be outstanding. The tick loop posts noop while waiting for up to one second; if late, the default executes without waiting for the five-second inference timeout. Responses are consumed later and rejected if no longer applicable. Busy tasks continue without asking Jev to cancel/restart them. Backend unavailability and budget exhaustion propagate through the documented contract; transient inference failures fall back. All background tasks are single context-managed model calls, not persistent Python worker loops.

## Confirmation and falsification

Confirm with fully parameterized accepted actions, audited acquisition of logs/table, positive opening score, subsequent food/bed/iron milestones, and low stale-tick fraction. Compare Jev selections against logged defaults rather than interpreting call count as useful cooperation. Refute if correctly parameterized gathering and crafting still fail consistently, workstation placement loops dominate, inventory synchronization defeats prerequisite logic, or deterministic defaults achieve progress while model choices reliably reduce it. Those outcomes would motivate narrower feedback-specific repairs rather than additional free-form planning.

## Limits and validation

Written under the final stop directive. No live game experiments or model calls were run. A syntax-only check accompanies file creation; performance is unverified. Navigation, pickup, placement clearance, client-versus-audited inventory lag, and furnace loading remain live-mechanics uncertainties. The controller does not implement Nether portal construction, fortress search, stronghold triangulation, or the dragon fight; later iterations should extend verified skills only after breaking the zero-score opening failure.
