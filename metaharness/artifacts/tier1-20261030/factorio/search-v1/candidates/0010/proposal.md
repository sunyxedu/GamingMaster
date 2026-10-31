parent: 0007

## Hypothesis

The leader is constrained by its remote logistics budget. In its weakest training
episode, seed 202 (4007 points), every turn from 90 through 128 submits exactly
100 transferred items. Across those 39 turns, science collection and lab loading
alone account for 585 items: 148 red extracted, 149 red inserted, and 144 green
in each direction. Turn 120 shows both green assemblers in
`item_ingredient_shortage`, all four labs in `missing_science_packs`, and zero
post-reservation iron/copper plates. The issue is not lack of nominal lab capacity.

Four local physical inserters should deliver science without spending the remote
quota, leaving more capacity for plates and intermediate preparation. They may
also eliminate the observation-boundary delay between production and consumption.

## Changes versus parent

Keep the parent's bootstrap, powered layout, smelting/coal capacity, research
selection, observed-recipe compiler, and exception-only Astra advisor. Add one
long-handed inserter between each existing assembler/lab pair. Coordinates rotate
with the observed validated pump site. Resolve routes from observed buildings,
power-network membership and recipe, never seed-specific IDs. Reserve science at
connected assembler outputs for the local lab and remotely distribute only surplus;
remote loading supplies the complementary science type. Missing or unpowered
routes retain the original remote-transfer behavior. All construction uses paid,
observed recipes and normal placement actions.

## Confirmation and refutation

Confirm with observed powered inserters delivering science, fewer manual science
transfers, higher plate availability and green production/consumption, and improved
full-horizon score without errors. Refute if placement or orientation prevents
delivery, local pack reservation starves other labs, construction delays first
green science, or freed transfer capacity fails to relieve the actual bottleneck.
Check the same raw economic telemetry as the parent and distinguish post-reservation
controller stock from pre-action stock. Pre-flight and at most two development
evaluations will be recorded here. No full training evaluation will be run.

## Dev correction

The first dev (`dev/20261030-024241`, seed 9001) scored 1274.32. All four inserters were observed as `waiting_for_space_in_destination`, and their reserved outputs were therefore withheld from remote delivery. The route geometry was wrong: centers were three tiles apart. The submitted revision moves each lab and inserter so assembler/lab centers are four tiles apart, places the inserter at the half-tile midpoint, and only treats a route as local when its observed inserter status is exactly `working`; otherwise the parent transfer scheduler handles all packs. Pre-flight after this correction passed (64.93 points, 37 actions). This corrected file is the one for evaluation.

The second dev of the four-tile revision reached 3885.0 with no errors, but still showed source/destination waiting on the integer midpoint. The final submitted revision uses the exact half-tile midpoint (6.5 in local coordinates) and fixes route lookup to the moved lab coordinate; pre-flight passes. This last geometry adjustment is motivated by the engine's half-tile port alignment and should be judged by the evaluator.
