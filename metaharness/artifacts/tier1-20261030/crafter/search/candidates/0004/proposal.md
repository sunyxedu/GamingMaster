parent: 0002

## Hypothesis

The leader’s main remaining failure is not recipe planning or model latency. It loses otherwise strong runs when a long objective remains active after dusk and the controller oscillates between two remembered walkable tiles. The evidence is explicit in the public traces: seed 1102 alternates `move_up`/`move_down` at positions `[10,50]` and `[10,51]` for dozens of ticks (steps 180–228), then remains exposed through the night; seed 1108 reaches 15 achievements but dies at step 227 with health 0 while still moving through hostile territory. The leader’s aggregate score is high (0.807), yet its weak seeds are dominated by late survival failures and the missing late-game achievements.

I test the structural hypothesis that a controller should switch from achievement pursuit to run preservation during deep night, while explicitly breaking stale route cycles. A short memory of recently seen hostiles also makes the route planner conservative when an enemy crosses the edge of the 9x9 view.

## What changed versus 0002

- Preserved 0002’s observation-driven Dijkstra planner, verified inventory-delta recipes, bounded placement retries, survival interrupts, and asynchronous Astra adviser. The adviser remains optional: model admission or malformed output cannot stop the game loop.
- Added a positional trace. An A/B/A/B cycle temporarily blocks the older edge for 55 ticks and clears the exploration target, forcing a fresh route or local choice instead of repeating two moves indefinitely.
- Added ten-tick threat memory. Current visible entities stay authoritative; recently seen hostile positions only add a danger cost to routing and expire quickly, so stale creatures cannot permanently freeze exploration.
- Added a night-return mode below daylight 0.23. It first follows a walkable tile adjacent to the nearest known table/furnace, then chooses a local open, low-danger step if no station route is known. It runs after urgent drink/food and combat handling and before strategic objectives, and automatically ends at dawn.
- At deep night, visible enemies farther than one tile trigger an escape step rather than pursuit. An adjacent finishing attack remains available. Sleep guards from 0002 remain unchanged.

## Evidence that would confirm or refute it

Confirmation would be fewer A/B loops, fewer deaths with 14–17 achievements, and higher minimum per-seed score while retaining the leader’s coal/iron/diamond progress. Traces should show a return toward a station or a short local route after daylight falls below 0.23, followed by resumed mining after dawn. Refutation would be a drop in early resource and rare-achievement rates because the night guard strands the player, or deaths despite the guard because the known map cannot connect to the station. A useful diagnostic is whether weak episodes end with health above zero and a larger fraction of `collect_coal`, `collect_iron`, `collect_diamond`, and skeleton achievements.

The pre-flight episode completed 432 actions with score 0.773 and no errors; it answered no Astra calls, so that smoke score is only a contract check. The real evaluation should compare survival and minimum seed score against 0002’s 0.682 worst seed and its documented night oscillations.
