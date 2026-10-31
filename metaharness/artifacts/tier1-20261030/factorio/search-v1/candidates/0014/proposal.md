parent: 0007

## Hypothesis

The plateau is partly a manufacturing-allocation problem: independent intermediate stock targets consume scarce iron without reserving the rest of a complete science kit. This is distinct from changing furnace transfers or lab routing. The manual worker generally has spare time; missing *complementary inputs*, rather than nominal crafting speed, can idle science machines.

Concrete evidence is in `candidates/0007/trace_seed202.jsonl`. At turn 100 both green assemblers report `item_ingredient_shortage`, each holding six inserters and zero belts. The next action queues `handcraft iron-gear-wheel 12` before feeding them, then `handcraft iron-gear-wheel 24` after plate extraction. The post-planning ledger has zero iron plates and 103 copper cables. Two turns later the plan again spends on gears while downstream recipes compete for iron. At turn 127, green production/consumption is 210/208 and the planning iron buffer is zero. This supports testing coordinated material allocation, but does not prove the entire plateau is caused by worker scheduling.

## What changed versus the parent

The machine layout, construction order, research policy, remote transfers, and exception-only Astra repair come from 0007. The production worker is replaced with a joint bill-of-materials compiler once a green assembler exists. It expands a target vector of inserters, belts, and red-science gears against one shared inventory, including recipe output multiplicities and already-reserved pending products. It searches a small set of coverage horizons and chooses the largest complete target vector affordable from actual materials. A separate per-item `ensure()` call can no longer spend all iron on a large gear buffer while forgetting the plates needed by circuits, inserters, and belts.

The compiler executes only currently affordable jobs, final products before prerequisites. Pending outputs are planning evidence only and never become spendable until observed. Construction retains first access to stock. A small feasible single-item fallback allows progress when even a balanced minimum kit is unavailable. Red demand is reduced when its existing pack buffer is large. Coal, stock-to-machine transfers, simulation timing, and laboratory consumption remain normal public actions.

The episode still uses no model on the healthy path. Astra sees compact current economy, machine states, failures, and the new manufacturing bill only when construction stalls or an action fails. Its strict action parser, bounded asynchronous call, stale-advice rejection, and caught failures remain inherited. Traces now log raw stock separately from post-reservation stock, planned transfer remaining, and the chosen bill.

## Confirmation and refutation

Confirmation is fewer green `item_ingredient_shortage` turns, more balanced belt/inserter availability, and increased real green production/consumption without a larger red deficit. Compare the complete score and production/consumption ledgers, not just model-call count or nominal capacity. Earlier green commissioning is a possible benefit, not a requirement.

Refutation is lower throughput because the complete-kit affordability test is too conservative, synchronized targets perpetually favor prerequisites, construction still crowds out useful crafts, or reduced gear buffering starves red assemblers. A fully used transfer quota may remain the dominant limit even if worker allocation improves.

## Validation

Initial contract preflight passed: 37 actions, score 64.93, no model calls, no exception. That short run ends before the new green scheduler is active. Development seed 9001 completed at **4121.0** in 128 turns and 220 seconds, with no model calls, failed operations, or episode errors. The archived final 0007 dev on the same seed scored **4102.0**. First green production moves from turn 67 to 64. Across turn-80-through-127 machine snapshots, green `item_ingredient_shortage` counts fall from 26 to 20, while `working` counts rise from 63 to 66 and `full_output` counts rise from 7 to 10. At turn 127, green produced/consumed changes from 214/210 to 220/215; red changes from 404/372 to 397/368. This is modest support, with a clear red/green tradeoff. Batches reaching the 31-operation controller cap increase from 2 to 14, exposing an additional limit. These are development results, not training scores. No training evaluation has been run by the proposer.

Offline validation checks exact shared-iron affordability, belt output multiplicity, pending-output bookkeeping, malformed action rejection, and 1,000 randomized paid-crafting ledgers without spending future products or exceeding the worker time bound. All pass. The final preflight also passed (37 actions, score 64.93, no exception). Development seed **9002** then completed at **4078.0** in 128 turns, with no failed operations or episode error. Its one asynchronous advisor call began during a late construction stall and was cancelled normally at episode completion; no repair was applied. The two development scores average **4099.5**. Both permitted development evaluations are used. The final harness is exactly the revision tested in both development episodes.
