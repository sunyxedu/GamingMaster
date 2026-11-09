parent: 0015

## Hypothesis

The research-aware lab router still applies at the wrong point in the turn. In the leader/0015 traces, the same late observations contain assembler `item_ingredient_shortage` or `no_ingredients`, lab `missing_science_packs`, and frequent 100-item transfer use. `service()` spends quota on lab packets before `produce()` has moved ingredients into assemblers, so a turn can feed labs while a scored red/green machine remains idle. The likely bottleneck is transfer ordering, not nominal machine capacity.

## What changed versus the parent

I retained 0015's seed-relative layout, recipe-derived paid crafting, three coal sources, fuel reserve, research-aware pack selection, and bounded asynchronous repair advisor. I split its transfer schedule into phases: service still drains assembler output, restores fuel, and collects bootstrap furnace material; `produce()` then sets recipes and feeds every observed assembler; only after those moves and furnace collection does `service_labs()` distribute the currently required science packs evenly across labs. Thus labs receive the residual shared quota after scored production inputs have been commissioned. No coordinates, recipes, IDs, grants, or model-generated actions changed.

The split is deliberately conservative: if there is no assembler, bootstrap collection is unchanged; if research is opaque, the existing all-pack fallback remains; and the existing 100-item and 31-operation caps are enforced by the same `move()`/`add()` methods.

## Evidence that would confirm or refute it

Confirm if assembler `item_ingredient_shortage`/`no_ingredients` turns decrease, first green production is earlier, and green produced and consumed science rise without more failed operations. A useful secondary signal is that lab loading occurs with less residual transfer pressure while assembler input inventories are nonzero. Refute if delaying lab delivery increases `missing_science_packs`, lowers lab consumption, or leaves transfer quota unused because fuel/furnace service already consumes it. Controller traces record raw statuses, inventories, production, consumption, and submitted batches for this comparison.
