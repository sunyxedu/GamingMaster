# 0013 — Hand-stage the opening workstation

## Parent and isolated hypothesis

Parent: `candidates/0008`. I keep its closed-loop frontier controller, code-owned
opening recipe chain, failure cooldowns, emergency overrides, and one background
Jev selector. This candidate tests one execution hypothesis only: **opening
workstation placement needs a hand-staged, unoccupied placement transaction**.
The item must be held, and the up-face target must not be the player’s own feet
cell.

The parent’s traces isolate this failure clearly:

* Seed 0 successfully gathered wood and crafted the table by about 9 seconds,
  then posted `place crafting_table b:-22,67,31 up` at 9.5 s. It received
  `action_invalidated ... reason=adapter_feedback` at 10.5 s and repeated the
  same placement at 36.0, 62.5, ... through 875.5 s. There were 34 such
  invalidations and the episode ended at opening score 3.0.
* Seed 2 had the same shape after it finally crafted a table: it posted
  `place crafting_table b:-113,66,-64 up` from 139.5 s onward, with repeated
  `adapter_feedback` invalidations (30 table-placement retries in the trace)
  through the deadline. It never reached stone or iron.
* This is not just a bad support candidate: the exact command and support stayed
  unchanged while inventory still contained the carried table. The parent never
  issued an `equip crafting_table` action. The earlier hand-staging experiment
  (`0010`) reported confirmed table placement in two seeds, but its episodes were
  invalidated by an unrelated `NameError: name 'threats' is not defined`; that
  plumbing failure means its positive placement signal was not a leaderboard
  result, but it is useful evidence for this narrowly scoped repair.
* A one-minute seed-0 smoke trace of the first equip-only version sharpened the
  precondition: it successfully posted `equip crafting_table hand`, then placed
  against `b:-24,67,30` while the player’s feet occupied the cell above that
  support, and again got `adapter_feedback`. The support filter now excludes
  that self-occupied target and chooses an adjacent exposed support.

The hypothesis is confirmed if 0013 converts the first table attempt into an
observed crafting-table candidate and proceeds to the existing wooden/stone
pick chain, without the long exact placement retry streak. It is refuted if the
equip-then-place pair is still rejected, or if a placed table is observed but
the controller immediately repeats an unrelated opening transaction. Seed 1’s
failure to acquire any wood is a separate frontier limitation and is not
silently claimed as fixed by this proposal.

The final two-minute seed-0 smoke test gave the expected local evidence:
`equip crafting_table hand`, one adjacent `place`, a `table_place_observed`
inventory delta, and a final observed `crafting_table` block, with zero stale
ticks and no errors. It still scored 3.0 because the unchanged downstream
policy later requested `craft acacia_planks 8` from only one log and fell back to
frontier navigation; that is deliberately outside this workstation test and a
clear next-iteration target.

## What changed

Only the workstation transaction is changed. When a carried table exists but
no table block is observed near the player:

1. The support compiler rejects a solid block directly under the player when
   its up-face target would be the player’s occupied feet cell.
2. If `self.held_item` is not `crafting_table`, the compiled menu contains
   `equip crafting_table hand` instead of `place`.
3. The normal active-job bookkeeping observes the held-item transition and
   releases the one-shot lease.
4. On the next fresh observation, the menu compiles
   `place crafting_table <fresh-support-id> up` only when the table is held.
5. If no safe adjacent support is exposed, the unchanged frontier navigation
   branch runs.

6. A successful place is recognized only from the real inventory delta. The
   controller latches the computed block coordinate, waits for the lagging
   candidate stream, and then returns to that coordinate instead of crafting a
   duplicate table. Crafting still requires an actually observed usable table;
   the latch is not a score or advancement claim.

There is no implicit placement success flag: a failed place that leaves the
inventory unchanged never arms the latch, and the existing feedback/cooldown
logic remains responsible for rejecting a failed support and avoiding an
immediate identical retry. No Astra calls are added; this keeps the test
focused on the parent’s workstation action bridge rather than strategic
replanning.

## Jev interface and real-time scheduling

The System-1 state is the parent’s compact, code-derived JSON object:

* `objective`, current code-owned `phase`, `dimension`, `health_band`, and
  `hunger_band`;
* audited personal `inventory`, filtered to relevant food, wood/planks, tools,
  cobblestone, iron, shield, furnace, table and sticks;
* visible hostile `threats`, represented by kind, `melee`/`near` range bucket,
  and entity ID;
* up to seven `options`, each an exact currently compiled command plus its
  code-derived purpose; and
* the last few normalized failed-command keys.

Coordinates, counts, recipe arithmetic, candidate legality, and table
postconditions are computed by Python. Jev is never asked to invent an item,
coordinate, or verb.

One request batches two independent typed questions:

* `action`: a `Choice` over exactly the supplied command strings. The answer is
  usable only if its value is still in the freshly compiled menu, has the same
  episode and is no more than ten observation sequence numbers old, and its
  returned option probability is at least **0.55**.
* `safe`: a `Noul` asking whether there is no visible close hostile threat. A
  safe probability below **0.35** rejects the Jev choice; code emergency rules
  remain authoritative regardless of this advisory answer.

At most one background request is outstanding, made with
`ctx.spawn(lambda: ctx.decide(..., timeout_s=2.0))`. While it is pending, the
controller posts `noop` to continue a bounded job. After **0.8 s**, it posts
the first code-compiled option as a deterministic lease fallback; a late
answer is harvested but cannot replace that action. Timeout, malformed answer,
unknown option, stale sequence, or low probability falls back to the first
legal option (or `noop`). `BudgetExceeded` and `ModelUnavailable` propagate as
normal episode termination conditions. Death, lava/water, low-health food,
visible hostiles, and stale observations bypass Jev through the parent’s safety
policy.

## Confirmation criteria

A useful positive trace should contain, in order, an `equip crafting_table hand`,
a held-item change, a single fresh `place` against a support, an inventory/table
postcondition, and a transition out of `place_table` into `wood_pick` or the
next resource phase. The key negative signal is absence of dozens of identical
`place crafting_table ...` posts and `adapter_feedback` invalidations. The
leaderboard test is successful if this preserves candidate 0008’s zero-error
status while raising the iron/stone/opening progression on the seeds where the
parent had already acquired wood.
