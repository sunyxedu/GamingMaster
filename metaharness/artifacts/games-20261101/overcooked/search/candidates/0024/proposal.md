# Candidate 0024 — consumption-priced counter inventory

## Parent, evidence, hypothesis

Assigned parent: **0013**, whose committed production and observation adapters
established the working pipeline. The implementation uses **0017**'s more robust
progress-bounded version of that pipeline as its control condition; its current
mean is 255.556. I retain its realtime loop, production branches, backup monitor,
parking, and recovery rather than repeat a failed traffic rewrite.

**Hypothesis:** counter stock is being treated as a mandatory reservation rather
than an optional ingredient source. A counter-first priority sends the chef on
long retrieval trips even when a dispenser would complete the pickup-to-pot
transaction sooner. The partner then finishes that pot, forcing the retrieved
onion back onto its original counter. This loses production time independently
of the extensively tested dish-parking hypotheses.

Concrete evidence: in `0017/trace_seed3.jsonl`, at tick **171** (line 252) the
controlled chef has just served at `[3,2]`; the partner is at `[2,1]`, the pot
contains one onion, and an onion remains on counter `[1,0]`. The controller
commits `take_counter [1,0]`. At tick **183** (line 266) it finally holds that
onion at `[1,1]`, but the partner has already filled the pot; it commits
`put_counter [1,0]`. The same retrieval/return pattern recurs at **215→225,
256→268, 298→310, and 339→351**. It ends at nine deliveries/180. The source
priority in both 0013 and 0017 strictly ranks counters ahead of dispensers,
before route length. The best controller therefore preserves a self-repeating
inventory detour. In parent 0013's weakest seed 0 the final result is also
nine deliveries with 87 collision ticks; its gate-release refinement did not
solve throughput. Lessons from generations 4–7 show that more gate leases,
parking locations, staging, or role locks did not add a cramped-room delivery.
This proposal instead tests the **source-to-consumption cost**.

## Change

When staged stock competes with dispensers, all compatible sources have equal
production urgency. Code enumerates pickup poses and estimates the complete
trip: current position and orientation → source, turn, pickup, source →
compatible pot, turn, fill. Present partner occupancy affects the first leg,
but is not a permanent lease; the second leg is a static geometric estimate.
An otherwise reachable contested first leg gets a two-tick delay cost.
The cheapest complete trip wins. The destination is an estimate for source
pricing, **not a reservation**: after pickup, the original opportunistic fill
selector rechecks which pots still accept the ingredient. Sources that cannot
reach either endpoint are omitted from the priced list. When there is no
matching counter stock, the parent's entire source policy remains unchanged.
The rest of the production motor is unchanged. This is not a new buffer,
counter preference, fixed role, or map-specific opening. No seed, layout name,
or hard-coded coordinate affects it.

## Confirm/refute

Confirm if `source_price_commit` shows cheap dispenser trips replacing distant
counter retrievals, the repeated take/return loop disappears, and earlier pot
starts yield additional deliveries without damaging the clean two-pot lanes.
Refute if the counter detours were necessary handoffs, if the full-trip estimate
fails to predict consumption, or if the change merely leaves useless stock
while throughput remains unchanged. Compare targeted cramped seeds as well as
the overall mean. Model calls, stale ticks, and every change of source are
logged. Live test results will be appended below.

## Tier-2 model bridge and fallback

Jev receives the 0017 strict-JSON schema: current tick, both players' poses and
held names, partner progress, authoritative pot phases/missing ingredients,
current job and epoch, recovery state, Astra hints, and up to ten legal medium
transactions. Each supply candidate now includes `source_to_pot_ticks`,
`destination`, and a code-derived `cost_band` of `fastest`, `near_equal`, or
`slower`. Jev is explicitly not asked to compare numbers: it recognizes the
named bands and any traffic reason to choose a near-equal alternative.

One asynchronous request batches:

* `transaction: Choice` over the current candidate IDs. It requires a selected
  probability of at least **0.55**, current epoch, a still-valid candidate,
  and at most one urgency band below the code choice. Ingredient choices must
  additionally be within **one tick** of the freshly recomputed cheapest full
  transaction; no answer may restore an expensive counter-first detour.
* `rescue: Noul` retains 0017's backup recognition. Probability ≥ **0.78** can
  only select an already-legal backup-dish candidate, never create a target.

One outstanding request; minimum interval 16 ticks, heartbeat 40 ticks, timeout
**2 seconds** (comfortably above recent ~0.2-second Jev medians and ~1-second
tails). Epoch mismatch, garbage, missing probability, unknown choices, timeout,
and failures leave the deterministic executor in control. Astra remains one
background reset-time review, with a 90-second timeout and enum-only tempo,
lane, and backup output. Its lane hint may only break equal source costs.
Astra should revise the broad strategy when repeated source choices do not
lead to consumption, not redefine geometry or arithmetic; this final-generation
candidate does not spend additional slow calls on that untested adaptation.

Every loop posts a legal primitive and calls `ctx.wait_for_tick`. Recipe
compatibility, explicit starts, facing, interactions, BFS motion, bounded job
expiry, and escape remain code-owned. The harness runs until game/context end;
it never waits for a model before acting. Only stdlib and `metaharness.*` are
imported.

## Validation

Both required files were written before live testing. Final-policy development
runs, one per training seed, produced:

| Seed | Archived 0017 | Candidate 0024 | Stale ticks | End |
|---|---:|---:|---:|---|
| 0 | 180 | 180 | 0 | game_done |
| 1 | 360 | 360 | 0 | game_done |
| 2 | 220 | 220 | 0 | game_done |
| 3 | 180 | **200** | 0 | game_done |
| 4 | 380 | 380 | 0 | game_done |
| 5 | 240 | 240 | 0 | game_done |
| 6 | 180 | 180 | 0 | game_done |
| 7 | 340 | 340 | 0 | game_done |
| 8 | 220 | 220 | 0 | game_done |
| **Mean** | **255.556** | **257.778** | **0 / 3,600** | **9/9 complete** |

These are **local smoke tests, not the official evaluation**. All use live
Jev; Astra was enabled only on seed 3. There were no episode errors or failed
model calls. Across the nine final runs there were 193 Jev calls, with pooled
median **0.177 s**, maximum **0.302 s**, and one successful Astra call lasting
**9.260 s**. Its policy (`steady`, `left`, `conservative`) was received at tick
56. The source pricer continued to prefer the right dispenser when its trip
was cheaper, so a late lane hint did not reinstate the detour. Official
all-Astra evaluation and held-out generalization remain untested.

The mechanism was exercised on its documented target. In
`tries/seed3-220555/trace_seed3.jsonl`, source pricing at ticks **52, 88, 124,
163** chooses the right dispenser at cost **6**, rather than counter stock at
**7** or the left dispenser at **8**. The five immediate same-counter
`take_counter → put_counter` loops in the archived seed-3 trace fall to **zero**;
`take_counter` commits fall **8 → 1**, and `put_counter` commits **9 → 3**.
Deliveries occur at ticks **52, 88, 124, 163, 200, 239, 277, 316, 353, 389**.
This supports the source-cost hypothesis on that episode, but does not solve
all cramped-room losses: seeds 0 and 6 remain at 180. It is also not a blanket
collision improvement: seed 3 has **40** collision ticks versus the parent's
**34**, despite its additional delivery.

### Refinement and negative result

The first version priced all ingredient trips and also preferred the predicted
pot after pickup. It scored **200 on seed 3**, but **340 on seed 1** and **200
on seed 2**, one delivery below their controls. I removed both the destination
preference and repricing when no counter stock exists, isolating the actual
inventory-choice hypothesis. The narrow final version restores **360/220** on
seeds 1/2, preserves the other five unaffected scores, and retains seed 3's
additional delivery. This does not separately identify which of those two
removed mechanisms caused the preliminary regression; it is a reason to keep
the final change narrow. Preliminary traces remain in `tries/` for inspection.

### Offline checks and reproducibility

`test_harness.py` passes **278** bridge checks over archived public observations,
**123** comparisons showing unchanged code choices/primitives when no counter
stock exists, and a **400-action** full-loop contract smoke test with garbage
stand-in model answers. It also checks current-counter usefulness, the known
expensive detour, near-equal hint limits, missing confidence, malformed Astra
JSON, arbitrary Jev values, and synthetic ingredient-label/control-index
transformations of public observations. These transformations are not live
held-out tests. No game implementation or held-out results were inspected.

`validation.json` records the final source fingerprint, exact trace paths,
per-seed outcomes, and measured latencies. Tests were run from the assigned
worktree using the documented CLI with `PYTHONPATH` pointing at its package
root; the first invocation without that package path failed before starting
an episode. No other candidate was changed.
