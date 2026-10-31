parent: 0005

## Hypothesis

The leader's green branch is constrained by a coupled copper/transport bottleneck. In its worst training episode (seed 202, score 3728.06), the turn-125 controller state has 143 iron plates, zero copper plates, and 84 spare red packs. Two labs report `missing_science_packs` despite carrying red packs. At turn 127 the green assembler reports `full_output`, and its input inventory is empty. This is not simply insufficient nominal production: the controller extracts every furnace output before servicing valuable science, and late action batches consume the 100-item transfer allowance before all machine feeds can be maintained.

A demand-driven transfer schedule should recover lost science throughput, while a second copper pair and second green assembler provide a balanced path to use existing iron/red surplus. Furnace inventories can buffer unneeded material without spending transfer quota. Even distribution of scarce green packs should activate more labs than banking several in one lab.

## Changes versus parent

Retain the observed-recipe compiler, paid manual intermediates, seed-relative smelters, verified steam layout, two red assemblers, and bounded asynchronous Astra exception repair. Add one copper drill/furnace pair, one green assembler, and one lab with connected poles after the parent's green branch. Prioritize science output extraction, emergency fuel, and evenly distributed lab packs over bulk material collection. Feed assemblers before topping up bounded raw-material buffers; throttle red feeding when stock already holds more than 24 red packs. Stop the manual green supplement after the second green assembler is observed, freeing worker time for intermediates. Log the same complete economic/machine state as the parent so the hypothesis is falsifiable.

## Confirmation and refutation

Confirm with fewer green-output blockage/ingredient starvation events, fewer green-starved stocked labs, sustained nonzero copper availability, increased engine-produced/consumed green science, and higher full-horizon score without episode errors. Refute if construction delay or the hard 100-item ceiling outweighs throughput gains, if bounded material collection starves the worker, or if extra science exceeds laboratory/power capacity. Preflight and at most two dev evaluations will be recorded below; no training evaluation will be run here.

## First dev result and diagnosed correction

Initial preflight passed (37 turns, score 64.84). The first dev run (`dev/20261030-014713`, seed 9001) completed 128 turns with zero System-2 calls and no failed operations, but scored only **2971.75**. Its green chain was starved by frequent manual coal jobs: the new fuel target deliberately left only four coal after a refill, which immediately triggered the inherited `stock < 8` emergency miner. That consumed the same worker needed for inserters/circuits. Turn 105, for example, queues `mine_resource ... 15` while green inputs are depleted. This refutes the initial small-fuel-buffer choice, not evidence that extra assemblers alone solve transfer limits.

The refined candidate restores at least 60 coal as the collection target, adds a third coal drill/chest to support the larger burner/steam load, retains larger construction material reserves, and moves first-green commissioning before the two later iron pairs. The second green assembler precedes the added copper pair. These are dependency/fuel corrections within the same demand-driven scheduler experiment. The parent's early science/red layout and all observed-recipe accounting remain intact. The second dev run will use the same dev seed so its before/after result is interpretable.

Offline checks passed over 512 archived economic states: nonnegative immediate stock, no spending pending outputs, at most 100 transferred items, at most 31 operations, and at most 1740 newly scheduled craft ticks. Malformed advisor reply parsing also passed. These checks are not live-engine evidence for the new late layout.

## Final dev evidence

The second and final allowed dev run (`dev/20261030-015437`, again seed 9001) evaluated the final harness: **4102.00 points**, **128 turns**, **214.8 seconds**, **zero System-2 calls**, **zero failed operations**, and no episode error. Its file hash matches the submitted harness. Components: red science **1918**, green science **2164**, bootstrap **20**. Green engine production first appears at turn **67**, versus **86** in the first revision; the complete expansion is observed at turn **87**, versus **96**. There are **zero** manual mining actions from turn 80 onward, versus **27** in the first revision. This same-seed comparison supports the fuel-buffer/dependency correction, though multiple linked corrections changed together.

At turn 127, the final ledger has **214** engine-produced green packs and **210** consumed green packs, compared with **110/110** in the initial revision. Final scoring includes one additional charged turn. The last planning state retains 27 copper plates and 35 red packs but zero immediate iron plates. The transfer ceiling remains binding: **57** turns exhaust all 100 items, and the largest batch has **31** operations. The experiment improves green throughput but has not removed the inventory transport bottleneck; automatic transport or more efficient use of transfers is a plausible next structural test. Do not mistake the displayed planning stock (already adjusted for queued operations) for the raw observation's pre-action stock.

This is a development score on one seed, not a training result or proof of superiority across the evaluation distribution. No full evaluation was run. Both allowed dev runs are used. Offline checks passed on 512 parent economic snapshots and all planned footprints on six archived initial maps (five distinct maps, because the dev runs reuse one seed). This checks bounds, water and mutual footprint overlap, not every engine placement rule. The second dev provides the live placement/production evidence for the added structures.

Final-file preflight **passed** at 64.77 points / 36 turns. It ended normally at the short preflight deadline, with no contract error; no System-2 answer was exercised because the exception advisor was not needed. Separate malformed-reply checks cover the parser, and the inherited advisor catches model and JSON exceptions. Final harness SHA-256: `48d8c189aa49d8afac84ef4bb3ce399f9086bf9129e4c606be12d36ad7bef33f`.
