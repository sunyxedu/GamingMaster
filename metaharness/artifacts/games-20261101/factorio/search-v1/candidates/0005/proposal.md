parent: 0002

## Hypothesis

The leader solved execution throughput, but its late score is constrained by useful research demand rather than a lack of red packs. In its worst training episode, seed 202, the turn-80 `controller_state` lists `logistic-science-pack` among completed technologies while all three labs report `no_research_in_progress` with nine red packs each. Turn 127 has 232 red packs in stock, three stocked working labs, and still no green production; the final score is 2232.06. Green packs can unlock more valuable research consumption and spend that red surplus. This tests a different production objective, rather than refining the leader's three-red-assembler recipe.

## Change from parent

Keep the parent's seed-relative direct-feed smelters, observed-recipe bootstrap compiler, verified steam layout, tree clearing, and two early red assemblers with three labs. Replace the third red assembler with a green-science assembler, adding two iron smelter pairs and a second coal drill/chest first. This creates four iron pairs, one copper pair, two coal drill/chests and three science assemblers. Coordinates derive only from the current public map; no seed-specific coordinates or IDs are stored.

The manual worker makes cheap intermediates from observed recipes instead of building a separate automatic belt/inserter/circuit factory. Once the first assembler exists, it schedules multiple already-affordable independent craft jobs under a 29-second work budget, reserves inputs immediately, accounts for multiple output recipes, and never spends pending outputs. This saves transfers compared with shipping inputs and outputs through intermediate assemblers. Furnace, fuel, science and lab transfers share the existing 100-item ledger; fuel is topped up in larger batches when low to preserve operation slots.

The green machine stays empty until the exact recipe is unlocked. Research prioritizes automation, logistics, then logistic-science-pack, and afterward ranks available red/green technologies by weighted science consumption per lab second. Green-consuming research is selected only after observing green packs. Research and recipe availability are always read from live observations. Logs include production, lab consumption, exact recipes and action results to distinguish latent demand, insufficient materials and power faults.

Astra remains an asynchronous exception advisor using compact machine and economic observations, bounded complete action strings, stale-advice rejection and caught model/parse exceptions. Routine production needs no model calls.

## Confirmation and refutation

Confirmation requires actual engine green-pack production and lab consumption, a falling red surplus, more valuable red/green research, and a full-horizon score above 0002 without errors. Refutation includes green arriving too late, an overburdened transfer quota or manual worker, added smelting starving fuel, or a net loss from infrastructure investment. Preflight validates contract survival only; it cannot establish the late green branch because its short episode ends before that branch is built.

Initial preflight passed at 64.77 points in 36 turns with no model call. This was before the final worker and research refinements; exact-file validation and dev evidence follow below.

## Dev evidence and refinement

The first complete dev episode (`dev/20261030-012157`, seed 9001) finished all 128 turns in 214 seconds with score **2685.06**, one attempted System-2 call, and no episode error. Score components were red science **1780**, green science **886**, and bootstrap **19.06**. This supports the objective-mix hypothesis on a dev seed; it is not a training-score claim. The green assembler produced real packs, and mixed research consumed them.

Its trace exposed a startup cost: the worker spent construction turns away from red intermediates, and the endgame accumulated more than 600 iron plates while green supply limited laboratories. The refined scheduler therefore admits independent crafts once the first assembler exists, and uses spare worker time for at most two manual green packs when red stock exceeds six and fewer than eight green packs are waiting. These packs earn only legitimate laboratory-consumption credit. A redundant attempt to select already-completed Automation was removed; that was the sole per-operation error in the first dev trace.

## Final validation

The second and final allowed dev run (`dev/20261030-012754`, seed 9002) evaluated the final harness unchanged: **3741.06 points**, **128 turns**, **226.8 seconds**, **zero System-2 calls**, **zero failed operations**, and no episode error. Components were red science **1934**, green science **1788**, bootstrap **19.06**. First observed engine red production was turn 47 and green production turn 74. At turn 127 the ledger showed 120 engine-produced green packs and 189 green packs consumed in labs; the difference is legitimately paid manual production. These are pre-final-turn counters, while the score components are the final evaluator measurements.

The final-file preflight passed. Offline checks also pass for malformed model replies, full semicolon command preservation, terrain/bounds placement on four archived initial maps, transfer/work/operation limits and nonnegative stock across 128 captured economic states, and prevention of spending pending cable in circuit crafts. The final dev's largest submitted batch had 27 operations.

Remaining limitation: copper stock reached zero near the end of the final dev, while 162 iron plates and 89 red packs remained at turn 127. The branch demonstrably creates useful laboratory demand, but continued scaling now needs balanced copper supply and consumption timing. The two dev runs use different seeds and different revisions, so their score difference is not a controlled estimate of the scheduler refinement. Training selection and unseen-seed robustness remain for the evaluator. No full training evaluation was run.
