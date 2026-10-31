parent: 0002

## Hypothesis

The leader’s rare-achievement gap is caused by station distance and repeated return trips after entering the iron/diamond phase. Candidate 0002 averages only 0.75 for collect_iron, 0.25 for collect_diamond, and 0.50/0.25 for iron pickaxe/sword. Its planner can gather a rare load but then must navigate back to the original table/furnace for each craft; this consumes the short surviving window and leaves the run with stone tools. The missing structural primitive is a temporary worksite near a remote resource.

## What changed vs. the parent

This candidate retains 0002’s verified observation bookkeeping, Dijkstra routing, asynchronous optional Astra adviser, placement target persistence, and one tick of feedback. It adds a portable micro-camp rule to crafting: when the nearest valid station is far away and enough wood/stone are carried, place a local table (and, for iron recipes, a local furnace) at the current expedition area before crafting. The table/furnace locations are learned from subsequent observations and recipe costs remain provisional until inventory deltas verify them. Rare resources are still gathered one at a time; the change is where they are converted into tools, reducing exposed return journeys.

## Confirmation / refutation

The hypothesis is supported if traces show iron collected, a nearby table/furnace placed, and iron tools crafted before the next long return, with higher collect_iron, make_iron_pickaxe, make_iron_sword and collect_diamond rates. It is refuted if carrying station materials costs more achievements than it saves, local placement fails on common terrain, or rare-resource rates remain unchanged while runs still die before crafting.
