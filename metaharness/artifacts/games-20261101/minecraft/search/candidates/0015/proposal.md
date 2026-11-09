# 0015 — Compile only feasible recipe transactions

## Parent and isolated hypothesis

Parent: `candidates/0013`, which itself refined `0008` by hand-staging the crafting-table placement. I keep the parent’s architecture: Python compiles a bounded milestone menu and owns inventory/geometry/safety, while one background Jev request selects among exact currently compiled commands and a deterministic first-option lease keeps the real-time game moving.

The targeted hypothesis is that the table repair worked locally, but the next transaction is not precondition-complete: **fixed recipe output counts are causing the controller to ask the adaptor for impossible crafts, and the resulting invalidation sends it back into a navigation loop instead of producing a wooden or stone pick.**

The positive seed-0 trace from 0013 reaches the repaired workstation in order:

* `8.508 s equip crafting_table hand`
* `9.505 s place crafting_table b:-23,67,30 up`
* `10.503 s table_place_observed`

That is the parent’s intended success signal. Immediately afterwards, however, it gathers another log and repeatedly posts `craft acacia_planks 8`; those requests are rejected with `adapter_feedback` at `17.003`, `43.504`, `91.503`, `120.003`, `149.505`, and many later times. The inventory did not contain eight logs: the trace had just gathered one additional log after the initial table material. The same trace alternates those impossible crafts with `goto -22.00 68.00 31.00` and navigation, ending at score 3 without a wooden pick.

The seed-2 trace shows the same boundary from the other side: it successfully observes the table at `105.504 s`, then invalidates gathers at `108.002` and `108.503`, and spends the rest of the run trying to relocate/search rather than advancing to a verified wooden/stone pick. Thus 0013’s workstation experiment removed the parent’s 34/30 exact table-placement retry streak, but it exposed the next unverified transaction: recipe actions were compiled from desired counts, not from the authoritative inventory.

This is not a claim that frontier navigation or hidden-tree discovery is solved. Seed 1 still has the prior failure mode (`gather b:-12,64,-21 1` invalidated at `11.503 s` and again at `316.505 s`), and this candidate deliberately retains the failure cooldown/frontier fallback for that case. The isolated test is whether, once any seed has a log/table, feasible-count compilation lets it cross the first tool transition.

## What changed

The action compiler now treats every craft as a transaction with an input-derived upper bound:

1. `feasible_outputs(item, inventory)` computes the largest legal output count from the observed personal inventory. It handles the opening planks, sticks, crafting table, wooden pick, stone pick, stone sword, furnace, shield, and bucket recipes. It does not inspect hidden blocks, containers, cursor previews, or a seed.
2. `add_craft` clamps the requested `NEW_COUNT` to that bound and emits no craft command when the bound is zero. For example, one acacia log produces `craft acacia_planks 4`, not `craft acacia_planks 8`; a pick request is emitted only when at least three planks and two sticks are actually present.
3. The menu has a real inventory postcondition for each stage: it cannot request a wooden pick before a table, planks, and sticks; a stone pick before three cobblestone and two sticks; a furnace before eight cobblestone; or a shield before iron and planks. The same hand-staged equip-before-place transaction is used for both the crafting table and furnace.
4. Failed one-shot commands remain normalized and cooled by target. A craft with no inventory change is not treated as progress, and the current option set cannot replay it until its cooldown expires. A changed inventory, movement, dimension, or held item closes the active transaction; table placement additionally latches the observed inventory delta while the candidate stream catches up.
5. The parent’s safety overrides, bounded navigation, exposed/in-reach gathering rule, normal respawn, and Jev fallback are retained. No Astra call, privileged state, fixed seed fact, or command is added.

The expected causal trace is therefore `craft <wood>_planks 4` → observed inventory delta → feasible sticks/pick → observed wooden pick, rather than repeated `craft <wood>_planks 8` invalidations. If stone becomes exposed, the same compiler should emit `craft stone_pickaxe 1` only after the actual cobblestone and stick counts are present.

## Jev state, questions, and real-time policy

The state sent to Jev is a compact JSON object rebuilt from each observation:

* `phase`: a code-owned phase (`find_wood`, `make_table`, `place_table`, `wood_pick`, `weapon`, `stone_pick`, furnace/iron stages, or `resource_frontier`);
* dimension, health band, hunger band, and the filtered authoritative personal inventory;
* visible hostile entities as `{kind, range: melee|near, id}`;
* at most seven options, each containing an exact command string and a Python-derived purpose; and
* the last five normalized failed-command keys.

Counts, recipe arithmetic, candidate IDs, coordinates, hand requirements, and postconditions are code-owned. Jev is never asked to count inputs or invent a command.

One request batches two independent questions:

* `action`: `Choice` over the exact option strings. Its value is accepted only if it is in the freshly compiled menu, has the same episode, is at most ten observation sequence numbers old, and its returned probability is at least `0.55`.
* `safe`: `Noul` asking whether no visible close hostile makes the compiled work unsafe. A probability below `0.35` rejects a Jev-selected option. This is advisory; lava, death, stale observations, low-health food, and visible-threat overrides are decided in Python first.

There is at most one outstanding request, launched with `ctx.spawn` and `ctx.decide(..., timeout_s=2.0)`. While it is pending, the controller posts `noop`; after `0.8 s` it commits the first legal compiled option as a late-answer fallback. A completed answer after that commit is harvested but cannot replace the action. Timeout, malformed output, stale sequence, unknown command, or low probability falls back to the first option (or `noop`) without escaping `run_episode`; budget/model-unavailable exceptions propagate as normal episode termination. Every posted action is revalidated against the current menu.

## Confirmation / refutation

The hypothesis is confirmed by a zero-error trace with:

1. no `craft ... 8` request when only one log is present;
2. an observed feasible planks/sticks/wooden-pick sequence after the table postcondition;
3. a subsequent stone-gather/craft attempt or an iron/shield score component; and
4. fewer repeated craft/navigation invalidations than 0013’s seed-0 trace.

It is refuted if feasible-count actions are still rejected despite matching audited inputs, if the controller continues to alternate table `goto` and impossible craft after an observed table, or if the same 3/0/3 pattern persists on seeds where 0013 already placed a table. If seed 1 remains at zero because no exposed wood can be gathered, that is evidence against the separate frontier hypothesis, not against the recipe-count hypothesis. A positive wooden/stone transition with no later Nether progress would establish that this repair fixes the first progression bottleneck only, as intended.
