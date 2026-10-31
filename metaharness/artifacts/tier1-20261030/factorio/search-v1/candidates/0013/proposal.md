parent: 0007

## Hypothesis

The leader's layout and production rates are adequate, but its fixed transfer order loses late science to the shared 100-item quota. Candidate 0012's furnace buffering confirms that bulk extraction is wasteful, yet its threshold policy overcorrects: seed 202 finishes with 34 iron plates but 25 copper plates in the planning stock while several red assemblers report `item_ingredient_shortage` and four labs report `missing_science_packs`; seed 101 finishes with 60 copper and only 1 iron. Across the 0012 traces, assembler output is repeatedly `full_output` and then ingredient-starved, while production ledgers continue rising. The useful unit of scheduling is therefore downstream deficit, not a fixed furnace buffer.

## What changed versus parent 0007

The proven 0007 machine plan, fuel reserve, recipe compiler, and exception handling are retained. Furnace extraction is replaced by a quota-aware deficit pass: first clear assembler outputs that would block production, then feed labs and machines, then allocate remaining transfers to the raw material whose observed furnace results and downstream deficits have the greatest weighted shortfall. Iron is explicitly prioritized when any red/green assembler lacks gears or plates; copper is prioritized only when circuit/cable inputs or copper furnaces are actually the limiting frontier. Extraction is capped to the demand of the next few turns and never drains both raw resources merely to maintain nominal stock targets. The policy also drains a nearly full result slot to prevent `full_output`, while preserving a small buffer when no consumer is waiting.

## Evidence that would confirm or refute it

Confirmation would be fewer late `item_ingredient_shortage`, `full_output`, and `missing_science_packs` statuses, a smaller gap between engine-produced and consumed green packs, and less frequent 100-item transfer exhaustion while retaining the leader's early red milestone. Refutation would be unchanged downstream starvation despite balanced raw buffers, or lower production because result slots remain full; the trace must then show whether output extraction or machine feeding is the binding limit. This is a single structural scheduling test, so scores should be interpreted with per-resource production, transfer use, and status telemetry.

## Development evidence

The initial deficit floor was too aggressive: on seed 9001 it scored 3667 and left late assemblers/labs starved. Raising the pressure policy's minimum raw buffers to 80 iron and 65 copper recovered the same seed to **4014.0** in 128 turns, with zero model calls, failed operations, or episode errors. The improvement supports the diagnosis that extraction demand must be bounded but substantial; the first revision's 35/30 floors underfed the chain. This second run is development evidence only and uses the same seed for comparison. The final harness has now used both permitted development evaluations.
