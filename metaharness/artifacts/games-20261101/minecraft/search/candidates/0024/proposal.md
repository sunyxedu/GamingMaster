# 0024 — Recover opening milestones and keep work local

## Parent, evidence, and hypothesis

Parent: `candidates/0013`; the best comparator is `candidates/0015`. This is a new, bounded opening policy rather than a copied version of either controller. It tests the hypothesis that the plateau is being reinforced by an overly sequential tool-first policy: the controllers secure logs/table, then spend the balance of the episode chasing stone and returning to a cached workbench, while **food and a bed—together worth seven of the ten opening points—are never treated as immediate scored objectives**. The test adds opportunistic, observation-grounded chest/food/bed transactions, and forbids a cached-workbench return loop.

The evidence is direct:

* In 0015 seed 0, the table and wooden pick transactions succeed early (`craft wooden_pickaxe 1` at about 27 s), but after that the action trace repeatedly posts `goto -22.00 68.00 31.00`; the action count shows 29 such posts, while the episode ends at 3.0 with no iron or later milestone. Seed 2 likewise posted `goto -121.00 63.00 -44.00` repeatedly after its wooden pick and finished at 3.0. This is a costly, stale-workstation policy, not a model-latency problem.
* The parent 0013's seed-1 trace made 44 Jev calls but only 0 score. It repeatedly navigated the same small set of stands and retried `gather b:-12,64,-21 1`; that exact gather was invalidated, while seed 1 obtained its first table only very late in the best candidate trace (0015: about 836 s). A rich policy that waits on its main dependency chain leaves no reliable score fallback.
* Across the leaderboard and Generation-7 lessons, candidates score at most 3 opening points. The episodes report no food or bed score component despite the score explicitly assigning three and four points to those flags. Generation-7's portable-table and staircase tests did not establish iron/shield progression; the evidence does not justify spending the final generation on another speculative underground route.

The hypothesis is deliberately narrower than “this beats the dragon”: a controller that opportunistically acquires a real food item and any real bed/sleep flag, while retaining the opening chain and not abandoning a local workbench, should improve mean milestone score and reduce time spent in table-return loops. The optional village/container branch only acts on visible, nearby containers; there is no structure or seed inference.

## What changed

1. **Opening score order:** the controller keeps the logs/table → wooden-pick dependency chain, then elevates visible food and bed opportunities ahead of speculative stone exploration. It can gather an observed bed, sleep in an observed bed when night permits, craft a bed only from three actually carried same-color wool and an observed table, acquire a visible edible block, harvest a visible animal for real drops, or inspect a nearby chest/barrel. It does not award progress until the adaptor's actual inventory/advancement facts change.
2. **Local-workbench rule:** recipes use a currently observed table. If a later stone/iron transaction needs a workstation but no table is locally observed, the harness uses an actual carried table or makes and places one when the inventory permits. It never keeps sending `goto` to a position cached from an earlier placement.
3. **One-action transaction gate:** an outstanding bounded job is continued; after a one-shot command, its inventory/hand/window/movement postcondition is checked before another command is selected. Explicit failures and action timeouts cool down the exact target. A successful inventory change is not confused with movement progress for mining.
4. **Risk control:** death uses normal respawn; lava/drowning and low-health eating are handled by code; visible nearby hostiles cancel an in-flight job and trigger shield, weapon attack, or retreat. The harness does not mine hidden interior blocks or assume a route to ore.
5. Stone and iron work are retained only as opportunistic extensions when exposed block candidates and real inputs are observed. They are not a required scoring claim of this experiment.

## Jev interface and timing

Astra is intentionally unused. The archived Astra latency fit is roughly 25 s plus output-token cost, and past sparse Astra calls did not change the opening-only result; the hypothesized failure is a locally executable priority/workbench policy, not missing long-form reasoning.

Each Jev request receives a fresh JSON object with: `objective`; code-owned `phase`; `dimension`; health/hunger bands; a compact audited personal-inventory subset (food, logs/planks, table, wool, picks, cobble, iron, shield); visible hostile kinds and `melee`/`near` buckets; and up to seven `{action,purpose}` options whose commands are completely compiled by Python. Coordinates, counts, item legality and transaction outcomes remain code-owned.

One batched request asks:

* `action`: a `Choice` over the exact current commands, choosing a safe useful action for the current phase;
* `clear`: a `Noul` asking whether the visible-threat list contains no nearby hostile making ordinary resource work unsafe.

A Choice is accepted only when its exact string remains in the newly compiled menu, episode matches, sequence age is at most ten observations, selected-option probability is at least **0.55**, and `clear` is at least **0.35**. Python safety overrides remain authoritative. At most one request runs in a background thread with a **2 s** call timeout. While pending, the game receives `noop`; after **0.7 s**, the first high-priority compiled option is committed as fallback, and a late answer cannot replace it. Timeout, malformed/garbage stand-in result, unknown command, stale result, or low confidence falls back to the first legal option. Budget exhaustion and model unavailability retain the contract's normal stop behavior. On return, the harness bounds cleanup of any one pending Jev future.

## Confirmation and refutation

Confirm the hypothesis if valid episodes show any real food and/or bed milestone after the initial table, yielding an opening score above 3 on at least one seed, while action traces no longer repeatedly return to the stale coordinates seen in 0015 seeds 0/2. Especially useful evidence is a window/transfer or visible-bed transaction followed by its authoritative postcondition, and reduced time between a wooden pick and the next useful action.

Refute it if visible nearby food/bed/container opportunities are present but are repeatedly rejected or never produce audited inventory/advancement changes; if the controller still spends long periods repeating one cached coordinate; or if the added side objectives delay the table/wooden pick enough to lower the mean score. No village, food drop, bed, ore, or dragon completion is assumed to exist in every seed.

## Implementation / safety note

The controller reads only the documented observation and the public candidate list; it never uses the seed, console, commands, hidden chunks, or container/cursor contents as personal inventory. Loot transfers require an observed open window and an item stack preview, and are bounded to the real stack count / 64. All menu commands use whitespace action grammar, and the evaluator's ordinary deadline/score behavior is unchanged.
