# Factorio strategy prior for Astra, Jev and harness code

Scope: base Factorio 2.0, no expansion mods, peaceful finite factory tasks, exact paused turns. The selected benchmark has remote placement/handling and a serial manual-work queue, so its opening differs from avatar speedruns. **Sources could not be retrieved because DNS failed; citations are reference targets from prior knowledge.** Numeric examples assume ordinary normal-quality machines, no modules, no bonuses and familiar base recipes. The pinned engine's prototypes override every hardcoded number. See [interfaces.md](interfaces.md) for the authoritative proposed action/economy/time/score contract.

## Division of labor

| Responsibility | Put in | Why |
|---|---|---|
| Recipe graph, exact ingredients/products, craft counts, footprints, tick costs, inventories, rates, quotas and score arithmetic | **Code** | Deterministic bookkeeping is cheaper and more reliable than repeated model reasoning. Account for multi-output recipes, reserves, partial batches, stock already committed to queues and machines, and technology locks. |
| Collision/port/orientation checks, nearest valid locations, pipe/belt connectivity, short transport routes, dependency ordering and stale-ID handling | **Code** | These are exact geometry or graph problems. Query engine placement validity before submitting work; a global layout still needs planning. |
| Choose milestones, allocate resources, arrange production blocks, select research, identify the dominant bottleneck, decide when to stop expanding | **Astra prompts** | These decisions require multi-step tradeoffs, horizon awareness and adaptation to seeded geography. Give Astra already computed costs/rates and concrete alternatives. |
| Classify precomputed machine summaries, choose a plausible failure class, rank a few repair candidates, flag a missing ingredient or unsafe frontier | **Jev questions**, optional | Typed recognition is useful when the state is large. Jev should neither calculate ratios nor invent action sequences. Code should directly recognize explicit engine statuses where sufficient. |
| Fixed-tick stepping, pause verification, action validity, reset, observation consistency, deduplication and reward accounting | **Code only** | These are evaluator rules, not agent choices. Neither model may change the clock, create materials or self-report score. |

For tier 1, Jev is **not required for reaction speed**. A paused factory gives Astra time to plan. Use Jev only if a batch of inexpensive judgments reduces observation/planning cost; prefer code for simple tests. [S1–S4]

## Objective-aware planning

The benchmark rewards real science production and laboratory consumption, with a small capped bootstrap reward for plates, gears and circuits. A large idle factory, stockpiles of unneeded buildings, and unlocked technologies that never help production are weak outcomes. The primary cold-start horizon is 128 turns × 30 simulated seconds = 64 minutes; it favors a reliable red/green science factory and selective expansion. A rocket is a long-term base-game objective, not a sensible mandatory opening goal for every short evaluation. [S1, S5–S7]

**Code:** expose remaining simulated time, current science rates, actual research consumption, ingredient shortages, projected infrastructure costs, and estimated payback in simulated seconds. Keep raw metrics alongside the scalar so low score can be traced to production, consumption or wasted time.

**Astra prompt:** “Choose the smallest complete chain that improves score within the remaining horizon. Name its bottleneck, full input/power/logistics requirements, one measurable success condition, and what will be postponed. Reserve the parts needed to keep fuel and power running.” Replan after a milestone, a failed action, or evidence that the expected rate is absent. Avoid replanning every individual belt tile.

**Jev question:** “Given these already computed candidate costs, earliest completion steps and expected score gains, which candidate is most plausible to finish before the horizon?” Supply choices, include “none”, and let code compute the numbers. Do not ask Jev to perform the payback calculation.

## Cold-start opening: dependency order, not fixed coordinates

### 1. Survey and reserve the bootstrap

Read iron, copper, coal and stone deposits, trees, coast and build corridors. Choose a compact initial production site with room to extend smelting and power. Establish stock and reserves before spending starter iron. Use the burner drill/furnace immediately on a useful resource chain; manually mine only what closes a bootstrap dependency. Harvest additional wood early for power poles. The starter wood alone is not a sustained power-grid supply. [S1–S3]

**Code:** rank valid ore/coast sites by distance, patch amount, collision-free area and transport cost; distinguish water tiles from a valid offshore-pump placement. Expand a bill of materials into exact handcraft steps, including cables/gears made first because the selected interface does not recursively handcraft intermediates. Reserve coal, stone and iron for the next miner/power chain. Track serial manual-worker completion times.

**Astra:** choose the initial geography and a short staged bill of materials. Do not copy coordinates from a previous seed. Compare temporarily moving the first drill with making another drill; movement returns an existing machine but produces nothing while it is elsewhere.

**Jev:** choose among already validated starter locations, or flag “no practical coal/water path” from explicit map summaries. A malformed map should be rejected by reset validation rather than repaired through free resources.

### 2. Stabilize coal and iron; add copper and stone

A burner miner must be fueled before its first output. Burner coal miners can be arranged to feed one another, creating a self-fueling starter supply; surplus still needs collection. A burner drill can directly feed a furnace if its output is correctly aimed, but the furnace needs its own fuel. Add iron smelting capacity early, one copper chain for early electrical/science parts, and stone for furnaces/boilers. Do not build every machine before establishing fuel flow. [S1–S3, S8]

**Code:** treat drill output direction, furnace input, burner fuel and blocked output as separate checks. Compare actual plate/fuel deltas after a charged interval with expected production. An idle furnace is not evidence that more furnaces are needed.

**Astra:** expand the slowest necessary material, usually iron in the early build. Keep enough fuel outside the main factory to restart a dead power chain. Spend limited manual transfers on bootstrapping; sustained production needs inserters and belts.

**Jev:** classify each miner/furnace as `fuel_missing`, `input_missing`, `output_blocked`, `working`, or `unknown`, based on supplied status and inventories.

### 3. Build a complete steam-power chain, then electric mining

The chain is water → offshore pump → boiler with fuel → correctly connected steam engine → poles → consumers. Water and steam connect to different boiler ports. Connect pole networks as well as covering consumers; a pole in range of a machine is useless if it belongs to an unpowered network. Electric drills and assemblers reduce manual work only after power and supply are reliable. [S9–S11]

**Code:** verify port alignment and fluid identity, power connectivity, fuel reserves and generation/demand. Use actual network data when possible; energy stored in an entity is not a sufficient power diagnostic. Compute engine counts from prototype flow/power values.

**Astra:** establish one functioning power block before spreading electric consumers. Use a robust coal supply; a burner inserter feeding a boiler can help recover from power loss, but it must itself have usable fuel/access to coal. Ordinary electric coal mining and electric boiler-feeding inserters can all stop together after a blackout.

**Jev:** choose `no_water`, `no_fuel`, `wrong_fluid_connection`, `disconnected_grid`, `generation_shortfall`, `healthy`, or `unknown` from explicit evidence. Do not label every slow machine a recipe shortage.

### 4. Establish and consume red science

Automation science requires a gear and copper plate in the familiar base recipe. Research Automation with early available packs to unlock assembling machine 1; **assemblers are not initially available merely because their recipe is visible**. A little handcrafting can bootstrap research, but the selected score excludes virtual handcraft production itself; genuine lab consumption still matters. Then automate gears and red packs and feed a powered lab with selected research. [S5, S12–S13]

**Code:** check technology availability and costs from the engine, queue handcraft subrecipes explicitly, and wait only the required charged turns. Confirm that a lab has both electricity and the packs required for its current research. Avoid making an assembler blueprint before its recipe is unlocked.

**Astra:** use just enough manual science to enable automation, then invest in automatic inputs. Research Logistics when belts/inserters/splitters or green-science progression require it; build a small reusable supply of belts, inserters, poles and assemblers rather than a huge early mall. Read the actual technology tree: recipe unlocks differ across versions.

**Jev:** `lab_ready`, `no_selected_research`, `wrong_pack_type`, `missing_required_pack`, `unpowered`, `research_complete`, or `unknown`.

### 5. Add circuits and green science without starving red

The familiar logistic science recipe consumes a transport belt and an inserter. These require plates, gears, circuits and copper cable. A usable green line therefore includes an entire intermediate chain and power/logistics; placing only a science assembler creates no throughput. Keep red production running because many technologies consume both types. Use direct cable insertion into circuits where practical. [S6, S14–S17]

**Code:** compute the marginal iron/copper and machine demand for the added line, reserve construction materials, and count belt recipe output correctly. A belt recipe can produce two belt items; “one craft” is not “one item.” Check the ingredient inventories of downstream inserter and science assemblers.

**Astra:** scale the bottleneck gradually. A compact shared gear supply can serve red packs, belts and inserters; provide enough capacity and avoid routing every user through a single underfed lane. Use short pipes/belts and direct insertion where they simplify a stable block.

**Jev:** choose the first missing dependency from a summarized chain, or choose which existing intermediate line is plausibly being starved by the green expansion.

### 6. Expand only after measuring; select the next milestone by payback

Steady red/green science often outperforms a half-built oil network at this horizon. Oil/chemical science requires additional research, power, steel, circuits, refineries, chemical plants, pipes, pumpjacks, and balanced fluids. Do not invest the final turns in unfinished higher-tier science. More advanced start families can make chemical/production/utility science a good objective. [S7, S18–S20]

**Code:** compare actual sustained production over several charged intervals; compute material/work deficits and time-to-first-pack. For mixed products, model all outputs and storage capacity rather than pretending one recipe has only its desired product.

**Astra:** choose between expanding an existing line, adding labs to consume its output, unlocking a useful upgrade, and starting the next science tier. Near the end, stop discretionary expansion and keep fueled machines/labs operating. Avoid “waiting” at an observation boundary: only an accepted turn advances the factory.

**Jev:** select `repair`, `scale_existing`, `add_consumption`, `advance_tier`, or `hold` from candidates whose cost/horizon evidence code has already computed.

## Quantitative facts worth encoding

These examples are useful sanity checks, **not substitutes for reading the selected version's prototypes**. They assume continuous inputs, available output space, full power and no productivity/speed effects. [S2–S6, S8–S17, S21–S23]

| Calculation / familiar base value | Application and owner |
|---|---|
| Recipe rate = `machine_crafting_speed / recipe_time_seconds`; item output rate multiplies by product count and applicable productivity | **Code:** use exact units and output counts. Astra chooses target rate, code sizes machines with ceiling arithmetic. |
| Resource rate = `mining_speed / resource_mining_time × product_amount`, adjusted for supported bonuses | **Code:** electric mining drill speed 0.5 on ordinary iron/copper/coal resource time 1 gives 0.5 ore/s. Burner drill speed 0.25 gives 0.25 ore/s under the same assumptions. Resource richness affects longevity, not this rate. |
| Iron/copper plate recipe time 3.2 s; stone furnace crafting speed 1 | **Code:** 0.3125 plate/s per furnace. One electric drill nominally supplies 1.6 stone furnaces; choose convenient integer blocks and measure. |
| Yellow belt capacity 15 items/s total, 7.5 per lane | **Code:** a fully balanced yellow ore belt needs 30 ordinary electric drills; a full yellow output belt takes 48 stone furnaces (24 per lane), ignoring logistics limits. A single full lane is half a belt. Avoid building this scale blindly in a tiny opening. |
| Steel uses 5 iron plates, familiar recipe time 16 s | **Code:** equal-speed iron and steel furnaces have a 1:1 machine-count ratio when iron feeds steel directly; steel *item* output is one-fifth of iron input. Do not confuse item ratio with furnace ratio. |
| Standard boiler 1.8 MW; steam engine 0.9 MW | **Code:** one fully supplied boiler supports two ordinary steam engines in the familiar base configuration. Verify fluid temperatures and rates from the pinned prototypes; do not transplant an offshore-pump ratio across versions unexamined. |
| Assembling machine 1 crafting speed 0.5; red recipe 5 s; green recipe 6 s | **Code:** one red assembler makes 6 packs/min, one green makes 5 packs/min. Five red and six green assemblers make 30/min of each with continuous supply. |
| Cable: 1 copper → 2 cable in 0.5 s; circuit: 1 iron + 3 cable → 1 circuit in 0.5 s | **Code:** equal-speed assemblers need **3 cable : 2 circuit** machines at full output. Direct insertion reduces belt volume and transport contention. Tiny green-science blocks may need much less than a full 3:2 module. |
| Lab capacity in research units/s = `effective_lab_speed / research_unit_time` | **Code:** derive pack demand using the current technology's packs-per-unit and research speed bonuses. Technologies have different unit times; no universal lab-to-science-assembler ratio exists. |

For a specific sanity-check target of **30 red + 30 green packs/min**, familiar recipes imply approximately **225 iron plates/min and 75 copper plates/min**, excluding fuel, construction, research-related infrastructure and other consumers. The derivation is 2 iron + 1 copper per red pack and 5.5 iron + 1.5 copper per green pack, with a two-output belt recipe. This is a reference load, not a mandatory opening layout.

At assembler-1 speed, a modest corresponding component block is 2 gear assemblers, 1 cable assembler, 1 circuit assembler, 1 belt assembler and 1 inserter assembler, plus the 5 red and 6 green assemblers. The input load is 3.75 iron plates/s and 1.25 copper plates/s, so ideal smelting requires 12 iron and 4 copper stone furnaces; ideal mining rounds up to 8 iron and 3 copper electric drills. **Power, inserters, belt lanes, lab demand and shared construction withdrawals can still bottleneck this design.** An individual assembler can often be underutilized; exact 100% ratios are less valuable than a complete reliable chain. Let code recompute the entire example from runtime recipes before presenting it as a plan. [S5–S6, S14–S17, S21–S23]

Use arithmetic in code, including fractional expectations and discrete completion counts near a horizon. Do not ask Jev “how many machines?” or rely on Astra remembering every ratio. Avoid claims of exact throughput from a one-turn sample when a slow recipe straddles the sample boundary.

## The universal stalled-factory diagnostic

Use this order for each important output chain; walk upstream only as far as the first real cause. [S3–S4, S8–S11, S18–S19, S22–S24]

1. **Is the machine valid and configured?** Correct entity, enabled compatible recipe, orientation and legal placement. An empty recipe is not an ingredient bottleneck.
2. **Does it have energy?** Burner fuel, connected electric grid with sufficient generation, or correct fluid at a generator. Low power slows several consumers at once.
3. **Does it have every input?** Inspect exact inventory roles/fluidboxes. Trace missing inputs to their source, not just the nearest belt. A stocked chest is irrelevant if no inserter can reach it.
4. **Can it release output?** Full inventory, backed-up lane, full destination or blocked coproduct fluid can stop an otherwise fully supplied machine. Adding upstream producers makes congestion worse.
5. **Can logistics carry the rate?** Wrong inserter direction, belt facing, lane imbalance, missing tile, splitter contention, insufficient transfer speed, misaligned pipe port or separate fluid networks.
6. **Is the available machine capacity sufficient?** Only after the above checks, add machines, upgrade speed/power, or reduce competing demand.
7. **Is there a sink?** A science line backs up if labs lack power, the current research needs another pack, or no valid research is selected. Downstream consumption is part of the production chain.

**Code:** build a dependency/status graph with exact evidence and derived likely causes. Prefer the actual engine status to visual guesses. Return “unknown” when a field is absent. Bound traversal and show the top few bottlenecks with action candidates.

**Astra:** select a repair that resolves the upstream cause and then predicts a measurable output delta. Do not redesign the entire factory in response to one stale reading.

**Jev:** classify the cause or choose among the candidate repairs. Request a typed enum and entity ID from an allowed set. Code validates that the choice still matches the current revision before converting it into an action.

## Layout, transport and expansion heuristics

- **Build modularly and leave extension space.** Keep smelting, components and science in compact blocks with room for another row and power poles. A main bus can organize a large factory, but an oversized bus consumes iron and space before producing useful science. **Astra:** architecture and expansion direction. **Code:** footprints, corridor reservations, collision/path checks. **Jev:** flag a proposed layout that blocks a known extension corridor. [S1, S22–S23]
- **Belts have two lanes and inserters have directional behavior.** An apparently full belt can carry only one useful lane; inserters usually drop on the far lane. Side-loading and merging affect distribution. **Code:** compute lane occupancy and directional connectivity. **Astra:** choose balanced lanes, splitters or separate lines. **Jev:** identify obvious lane starvation from summarized topology. [S22–S23]
- **Use direct insertion for bulky local intermediates.** Cable-to-circuit direct insertion avoids carrying three cable units per circuit over a general bus. Chests can buffer short interruptions, but large buffers hide shortages and consume resources needed for construction. **Astra:** choose buffer placement/caps. **Code:** track item flow and actual buffered stock. **Jev:** flag misleading “full buffer, no downstream use” cases. [S14–S17, S22–S23]
- **Separate fuel reliability from consumer expansion.** Scale coal extraction, transport and generation together; reserve restart fuel. **Code:** forecast fuel exhaustion from burner energy usage and available fuel value. **Astra:** prioritize a generation upgrade before connecting a large load. **Jev:** flag a supplied low-reserve/low-satisfaction state. [S8–S10]
- **Use an early small construction supply, not an unbounded mall.** Keep belts, inserters, poles, gears and assemblers available in finite amounts. Excess building stock earns little direct science reward. **Code:** inventory caps and committed-build costs. **Astra:** decide which supplies should be automated at this horizon. **Jev:** choose an overstocked category from code-computed targets. [S1, S12, S22]
- **Validate fluid arrangements by the pinned version.** Factorio 2.0 fluid behavior differs from older pipe-distance heuristics. Mixing fluids or connecting the wrong machine port still breaks production. Advanced oil can stop because one coproduct has nowhere to go. **Code:** fluid identity, connections, storage capacity and recipe balances. **Astra:** cracking and sink choices. **Jev:** classify which output is blocking a refinery. [S18–S19, S24]
- **Do not upgrade merely because a technology is available.** Better furnaces/assemblers or modules may improve throughput, footprint or efficiency, but require materials, power and payback time. **Code:** marginal rate/energy/cost comparison from prototypes. **Astra:** decide if it pays off before the final tick. **Jev:** rank supplied alternatives only. [S20–S21, S25]

## Common failures and their counters

| Failure | Correct response | Primary owner |
|---|---|---|
| Spending all iron on science before making power/mining equipment | Reserve a checked bootstrap bill of materials; include handcraft subrecipes and fuel. | Code reserves; Astra priorities |
| Many machines placed but no actual production | Inspect recipe, energy, every input, output and logistics before expanding. | Code diagnostic; Astra repair |
| Power loss stops electric coal mining, which stops boilers | Keep restart coal and a reliable fueling path; restore fuel/generation before adding consumers. | Code forecast; Astra design |
| A furnace/assembler remains idle with items “nearby” | Inspect the correct input inventory, inserter direction/reach, fuel and output buffer. | Code; optional Jev recognition |
| New green science consumes all gears/circuits and red stops | Budget combined demand; add shared intermediate capacity and prioritize balanced science. | Code demand graph; Astra allocation |
| Hundreds of packs accumulated but research is stalled | Select feasible research, power/feed labs, supply all required pack types; add labs if actual lab capacity is the bottleneck. | Code requirements; Astra selection |
| Long pipe/belt route crosses an occupied or invalid tile | Re-route with validated geometry and correct underground endpoints; recheck connectivity. | Code route search; Astra constraints |
| Using hand transfers for all transport | The selected 100-item quota will bind; automate sustained flow with inserters/belts and use manual handling for bootstrap/recovery. | Code quota; Astra logistics |
| Treating a place command as free construction | Charge the place item exactly once; no prototype creation without owned stock. | Trusted code |
| Assuming all craft actions yield one item | Use recipe execution counts and actual products, particularly cable and belts. | Code |
| Waiting in real time for a paused machine to finish | Submit a charged wait turn; observations alone consume no simulated time. | Code contract; Astra prompt |
| Replaying a timed-out step duplicates construction/production time | Retrieve the persisted step result using the same ID; do not resubmit as a new step. | Trusted code |
| Starting oil/blue science too late | Finish/scale useful red-green production or consume accumulated packs unless the new chain pays off within remaining ticks. | Code forecast; Astra decision |
| Grinding capped plates/gears for apparent progress | Inspect marginal reward; bootstrap categories stop scoring at their caps. Prioritize science and consumption. | Code reward calculation; Astra objective |
| Deleting/rebuilding or moving items to inflate production | Score genuine engine production and authorized research consumption; transfers and returns earn zero. | Trusted code |
| Copying a previous seed's blueprint without checking terrain | Translate/rotate and validate each footprint and connection; adapt to resource/coast topology. | Code geometry; Astra layout |

## Optional combat and later-game knowledge

There is no opposing player and no enemy base in the recommended primary scenario. Do not waste the primary run on turrets, ammunition, walls or military research. If a later family enables enemies, pollution can trigger attacks, time and nest destruction can affect evolution, and base expansion changes frontier risk. Factories need supplied turret coverage and ammunition; unsupported outward expansion can cost more than it earns. Walls alone do not kill enemies. Train/vehicle safety and pollution management are additional tasks only when those mechanics are actually present. [S26–S28]

**Code:** surface enemy sightings, damage, ammunition, attack paths, pollution reach where observed, and precomputed defensive coverage; never grant omniscient enemy information beyond the ruleset. **Astra:** decide military research, defensive chokepoints, pollution reduction and whether expansion is worth the risk. **Jev:** choose `immediate_threat`, `ammo_low`, `coverage_gap`, `safe`, or `unknown` from the current paused observation. Even combat can remain tier 1 when enemy simulation runs only during charged intervals; choose a smaller fixed interval for an explicitly separate combat ruleset if needed.

For advanced factories, rail signals, train schedules, oil coproduct balance, circuits, productivity and rocket requirements create distinct planning problems. Do not make the primary adaptor implement every late-game UI action before it can sustain a correct miner → furnace → assembler → lab chain. Add typed actions and regression checks as scenario requirements expand. Space Age, quality and elevated rails are excluded; their progression should not leak into base-game strategy. [S18–S20, S25, S29]

## Suggested model input and typed questions

An Astra turn should include: objective and remaining turns; recent actual/expected science and consumption rates; stock plus reserved inputs; power/fuel summary; current research; top blocked chains with reasons; last failures; a compact local map; and code-generated build/repair candidates with costs and feasibility. Ask for one prioritized batch, a prediction of what will improve, and a fallback if prerequisites fail. Store durable planning notes outside the game state, but treat the new engine observation as authoritative.

Useful independent Jev questions, when exact code cannot decide:

| Typed output | Question style | Required supplied evidence |
|---|---|---|
| Enum: `energy / input / output / logistics / configuration / none / unknown` | “Which cause best explains machine X's stall?” | Actual status, recipe, input/output buffers, power and connectivity summary. |
| Choice from candidate IDs plus `none` | “Which proposed repair most directly addresses the reported bottleneck?” | Candidate effects, legality and costs already computed by code. |
| Boolean plus `unknown` | “Does this proposed block obstruct the reserved expansion corridor?” | Explicit footprints/corridor map; use geometry code instead when available. |
| Enum: `repair / scale / consume / advance / hold` | “Which type of investment fits the supplied horizon/payback evidence?” | Code-computed remaining time, rates, costs and candidate completion times. |

Batch independent judgments, not sequential speculative actions. Do not ask Jev to plan a build order, calculate recipe expansion, infer hidden state, or answer from a screenshot when exact JSON already exposes the fact. The adaptor validates every suggested action against the current revision regardless of which model suggested it.

## Reference targets

These URLs were not successfully fetched in this session. The wiki is community-maintained mechanics/strategy knowledge; runtime prototypes remain authoritative for the pinned version.

- S1: https://wiki.factorio.com/Tutorial:Quick_start_guide
- S2: https://wiki.factorio.com/Mining ; https://wiki.factorio.com/Electric_mining_drill
- S3: https://wiki.factorio.com/Burner_mining_drill ; https://wiki.factorio.com/Stone_furnace
- S4: https://lua-api.factorio.com/latest/classes/LuaEntity.html ; https://lua-api.factorio.com/latest/classes/LuaFlowStatistics.html
- S5: https://wiki.factorio.com/Automation_science_pack
- S6: https://wiki.factorio.com/Logistic_science_pack
- S7: https://wiki.factorio.com/Science_pack ; https://wiki.factorio.com/Rocket_silo
- S8: https://wiki.factorio.com/Fuel ; https://wiki.factorio.com/Burner_inserter
- S9: https://wiki.factorio.com/Power_production ; https://wiki.factorio.com/Electric_system
- S10: https://wiki.factorio.com/Boiler ; https://wiki.factorio.com/Steam_engine
- S11: https://wiki.factorio.com/Offshore_pump ; https://wiki.factorio.com/Small_electric_pole
- S12: https://wiki.factorio.com/Automation_(research) ; https://wiki.factorio.com/Assembling_machine_1
- S13: https://wiki.factorio.com/Lab ; https://wiki.factorio.com/Research
- S14: https://wiki.factorio.com/Electronic_circuit
- S15: https://wiki.factorio.com/Copper_cable
- S16: https://wiki.factorio.com/Iron_gear_wheel ; https://wiki.factorio.com/Inserter
- S17: https://wiki.factorio.com/Transport_belt
- S18: https://wiki.factorio.com/Oil_processing ; https://wiki.factorio.com/Oil_refinery
- S19: https://wiki.factorio.com/Chemical_plant ; https://wiki.factorio.com/Cracking
- S20: https://wiki.factorio.com/Chemical_science_pack ; https://wiki.factorio.com/Production_science_pack ; https://wiki.factorio.com/Utility_science_pack
- S21: https://wiki.factorio.com/Steel_plate ; https://wiki.factorio.com/Steel_furnace
- S22: https://wiki.factorio.com/Belt_transport_system ; https://wiki.factorio.com/Tutorial:Main_bus
- S23: https://wiki.factorio.com/Inserters ; https://wiki.factorio.com/Splitter
- S24: https://wiki.factorio.com/Fluid_system
- S25: https://wiki.factorio.com/Module ; https://wiki.factorio.com/Productivity_module
- S26: https://wiki.factorio.com/Pollution ; https://wiki.factorio.com/Enemies
- S27: https://wiki.factorio.com/Gun_turret ; https://wiki.factorio.com/Wall
- S28: https://wiki.factorio.com/Defense
- S29: https://wiki.factorio.com/Railway ; https://wiki.factorio.com/Circuit_network
- Optional independent calculator for checking derived ratios: https://factoriolab.github.io/ ; https://github.com/factoriolab/factoriolab — do not depend on a live website at evaluation time.
