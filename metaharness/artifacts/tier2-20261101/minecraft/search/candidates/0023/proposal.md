# Candidate 0023 — do not gate resource search on finding the table

## Parent and targeted hypothesis

Parent: `candidates/0015`, the closed-loop compiler with feasible recipe counts, authoritative postconditions, explicit equip/place staging, and a small asynchronous Jev action menu.

The targeted hypothesis is that 0015’s cached crafting-table handoff is a **phase gate that prevents the next resource transaction**. The table-placement repair is recorded correctly, but when the placed table is not immediately present in the exposed candidate stream, the compiler returns before it reaches its stone-search branch. This is a controller deadlock, not evidence that no stone is available.

The parent’s seed-0 trace demonstrates the boundary: it records `table_place_observed` at about `10.0 s`, makes a wooden pick at `27.0 s`, and then repeatedly posts `goto -22.00 68.00 31.00` in phase `locate_table` (29 posts; the trace has 66 `locate_table` posts and 49 `stone_pick` posts overall). It never posts a stone gather/search transaction and finishes at score 3. The same pattern is visible in seed 2: after the opening, the trace records 12 invalidated table gotos while never reaching a stone or iron milestone. In contrast, the parent’s own recipe work is successful: all three episodes reach opening score 3, so this change does not revisit feasible craft counts.

## One change

The action compiler now distinguishes **resource acquisition** from **workstation return**. The cached `table_mark` wait/return branch is no longer unconditional. It is active only when the current inventory has enough real cobblestone for a pending table craft (stone pick, furnace, or later shield) or when another table-dependent craft is actually the next milestone. With a wooden pick and fewer than three cobblestone, the compiler is allowed to run its existing exposed-stone/scout policy immediately. Once cobblestone reaches the table-dependent threshold, the existing concrete table return and postcondition logic is reinstated before crafting the stone pick. The adjacent opening branch is also explicitly limited to `not table_mark`; otherwise bypassing the return branch would incorrectly recreate the already-placed-table phase from “no carried table, no visible table.”

No staircase algorithm, new seed knowledge, model prompt, recipe, or safety policy is added. The parent’s exact action compiler, failure cooldowns, Jev fallback, and real inventory checks remain unchanged. The isolated comparison is therefore whether removing the premature table gate lets 0015’s already-compiled stone transaction run and cross the first post-opening milestone.

## Expected evidence

Confirmation would be a trace with `wooden_pickaxe` followed by a stone gather/search action before repeated table gotos, then an observed cobblestone delta and (if the table can be reacquired) a feasible `craft stone_pickaxe ...` or iron/shield progress. The table-return count and `locate_table` dwell should fall sharply on seeds where exposed stone or a safe stone frontier exists.

Refutation would be either (a) the compiler still enters an unconditional table-return loop before cobblestone, indicating a different handoff bug, or (b) it gathers/escapes the gate but still cannot observe any stone on the tested terrain. In case (b), that would leave resource geometry as the next bottleneck; it would not be evidence against this gate hypothesis.

## Jev state, questions, and fallback

The state remains 0015’s compact, code-derived schema: `phase`, dimension, health and hunger bands, filtered authoritative personal inventory, visible hostile entities, the currently compiled exact command options, and recent normalized failures. Counts, recipe feasibility, coordinates, candidate IDs, and postconditions are computed by Python. Jev receives two independent questions in one request:

* `action`: a `Choice` over the exact freshly compiled command strings. Its value is accepted only if it is still in the menu, belongs to the same episode, is no more than ten observations old, and has probability at least `0.55`.
* `safe`: a `Noul` about whether no visible close hostile makes the listed work unsafe. A value below `0.35` rejects the answer.

There is one background Jev request at a time with a 2-second timeout. While it is pending, the loop posts `noop`; after 0.8 seconds it commits the first legal compiled option so the real-time game keeps progressing. Late, stale, malformed, unknown, or low-confidence answers are discarded and the first legal option (or `noop`) is used. Hard lava/fire/water/health/death overrides remain code-owned. Model-budget and model-unavailable exceptions retain the parent’s normal termination behavior; other parsing/posting failures are contained.
