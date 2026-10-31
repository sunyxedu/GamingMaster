Design a harness for a **turn-based** game (tier 1: the world waits for the agent).

Game: `Factorio` — Factorio 2.0.72 base-only cold-coast-v1: manage a real factory remotely on a seeded 256x256 island/coast area. No avatar, enemies, pollution, expansion or GUI. Initial stock: 1 burner mining drill, 1 stone furnace, 8 iron plates, 1 wood in a 200-slot inventory. All other items require finite mining/crafting or real machine production. 128 turns, exactly 1800 engine ticks (30 seconds) per turn; thinking and observation cost no game time. A turn is one action string or up to 32 semicolon-separated operations, applied in order. Individual failed operations retain earlier successes and still charge the turn. Malformed strings change no world state and cost no turn. Wait also charges one turn. No arbitrary code, item grants or clock controls.
Actions: wait: advance one turn without operations; mine_resource ID COUNT: queue manual mining cycles on an ore tile; mine_tree ID: queue mining one real tree; handcraft RECIPE COUNT: reserve inputs and queue nonrecursive crafts; place ITEM X Y [DIR]: consume a building item; DIR is 0/4/8/12 (N/E/S/W); rotate ID: rotate an owned building clockwise; set_recipe ID RECIPE: set recipe on an empty, idle assembler; insert ID ROLE ITEM COUNT: move stock into a machine inventory; extract ID ROLE ITEM COUNT: move machine inventory to stock; mine_building ID: recover an empty building; fluids/burning fuel are lost; set_research TECHNOLOGY: select research with satisfied prerequisites; inspect OFFSET: read-only page of 100 buildings, no turn charged; recipe NAME: read-only exact recipe details, including locked recipes; technology NAME: read-only technology details.
Manual mining, tree mining and handcrafting share one serial worker: at most 32 queued jobs and 3600 remaining ticks, running concurrently with machines. Resource/tree mining uses prototype mining time divided by character mining speed; crafting uses recipe seconds at speed 1. Handcraft inputs must already exist; no recursive subcrafting, no fluid/probabilistic recipes. Craft counts mean recipe executions, including multiple outputs. Inputs reserved immediately; full stock blocks completion. Only categories allowed by the vanilla character prototype. Automated machines follow actual engine recipes, fuels, fluids, electricity and research unlocks. Vanilla craft-item technology triggers are also honored for completed, paid manual crafts, without adding production statistics; steam-power requires 50 engine-smelted iron plates, electronics 10 copper plates, then crafting a lab unlocks red science.
Transfer roles: fuel, source/result (furnace), input/output (assembler), lab, chest. Insert+extract share 100 actual items per turn. Empty buildings only can be mined; empty idle assemblers only can change recipe. Extract cargo first. Fluids, unfinished burning fuel are lost on mining. Normal-quality items only. Placement uses engine collision/terrain checks and consumes stock on success. Use half-tile coordinates for odd-size buildings; DIR 0/4/8/12 means north/east/south/west; drill direction points output, belts move in direction. Belts, pipes, inserters and power poles use ordinary engine connections. Fixed finite ores and wood vary by seed. No oil is supplied in this cold-start family, so chemical and later science are not reachable; focus on red/green science and research.
Observations: paused engine tick, charged_ticks, turn/budget; stock/slots; manual queue/outputs; exact transfer counts; last action results; bounds [-128,128); water_rows with inclusive x intervals (other tiles grass); patches and per-resource rows [id,name,x,y,remaining amount]; trees [id,x,y]; validated pump_site; up to 100 buildings with stable IDs, positions, boxes, status code, inventories, burner, fluid amounts and temperatures, stored joules and electric network ID (stored energy alone does not establish supply); inspect OFFSET selects another building page; entity_page_count is the actual page size (at most 100, reduced for large inventories). enabled_recipes lists exact ingredients/products/time; available_research lists prerequisites and science costs; research current/progress/completions; raw normal-quality engine production and consumption, laboratory science consumption and score. recipe NAME and technology NAME query locked details without time. IDs remain stable until mining/depletion; invalid IDs fail safely.
Score: bootstrap .01*min(iron plates produced,1000)+.01*min(copper plates,1000)+.02*min(gears,200)+.05*min(circuits,200), cap 34. Plus weights automation/logistic/chemical/production/utility/space=1/2/6/12/12/30 times (engine-produced science + 4*science actually consumed in labs). Initial stock, manual outputs, building recovery and transfers earn no production score. Manually crafted science earns consumption credit only when used by a real powered lab. Final score is measured after the last charged turn. This is a benchmark score, not an official vanilla score.

Actions:
- wait: wait: advance one turn without operations
- mine_resource: mine_resource ID COUNT: queue manual mining cycles on an ore tile
- mine_tree: mine_tree ID: queue mining one real tree
- handcraft: handcraft RECIPE COUNT: reserve inputs and queue nonrecursive crafts
- place: place ITEM X Y [DIR]: consume a building item; DIR is 0/4/8/12 (N/E/S/W)
- rotate: rotate ID: rotate an owned building clockwise
- set_recipe: set_recipe ID RECIPE: set recipe on an empty, idle assembler
- insert: insert ID ROLE ITEM COUNT: move stock into a machine inventory
- extract: extract ID ROLE ITEM COUNT: move machine inventory to stock
- mine_building: mine_building ID: recover an empty building; fluids/burning fuel are lost
- set_research: set_research TECHNOLOGY: select research with satisfied prerequisites
- inspect: inspect OFFSET: read-only page of 100 buildings, no turn charged
- recipe: recipe NAME: read-only exact recipe details, including locked recipes
- technology: technology NAME: read-only technology details
(real-time games: ticks apply 'wait' until the first action is posted; afterwards a tick without a fresh action repeats the last posted action)
One string, or up to 32 semicolon-separated operations per turn. Read-only queries must be alone.

Model available through `ctx`: `ctx.think(prompt, session_id=...)` — Astra (System-2). Budget: 40 calls
per episode; one call may return a plan of several actions. `ctx.act(action)` applies one action and
returns the next observation. Each episode has a wall-clock deadline of 1800 s, including model
latency (`ctx.budget.remaining()["wall_s"]`); when it passes, or the call budget is used up, the episode
ends with the score reached. Each episode starts cold: fresh `ctx.memory` and fresh module globals.

Evaluation: mean score over the training seeds (the game's own score, see the description), then fewer
System-2 calls, then fewer actions. A candidate can only be selected if none of its training episodes
ends with an error (see the contract: e.g. a TimeoutError from a model call that the harness does not
catch).

What the harness must design: how observations are shown to Astra (full history, a diff, an ASCII map
with a legend, only what changed), what to remember between calls (rules learned, states visited,
plans), how to parse Astra's reply into actions robustly, how many actions to commit per call, when
cheap scripted search or bookkeeping replaces a model call, and how to notice that a plan or belief was
wrong. If the description says that rules or action effects are not given, discovering them by
experiment is part of the task.

**Evaluation scope.** The harness is selected on the training seeds of this game and then reported on
unseen seeds of the same game; there is no held-out game. Rules, strategies and search that the game
description or the study establish may be implemented as code once they are verified against
observations — but nothing specific to the training seeds.

## Prior knowledge from the study phase
The study phase researched this game (rules, interface, community strategies). Use it: implement exact
rules, bookkeeping and search as code and put planning and adaptation into Astra prompts — but verify
every claim against observations, because sources can be wrong or outdated for this build of the game.

The full documents are readable in this archive under `prior/` (`prior/strategies.md`, `prior/brief.md`, `prior/interfaces.md`, `prior/study.json`, `prior/ADAPTOR.md`). Below: excerpts within a size budget; every cut is marked `…(truncated; full text in prior/<file>)`.

### study/study.json (summary)
- strategy_summary: Strong play builds complete fueled and powered production chains before expanding their nominal capacity. Bootstrap coal and iron, add copper and stone, establish water-boiler-steam-engine power, and research Automation before building assemblers. Automate red science and its laboratory consumption, then add the full belt/inserter/circuit chain for green science while maintaining balanced inputs. Encode recipe expansion, output multiplicities, footprints, resource reservations, throughput, quotas, tick accounting and bottleneck diagnostics in code using pinned engine prototypes. Astra should choose milestones, layout, research and investments whose completion and payback fit the remaining simulated horizon. Check configuration, energy, inputs, blocked outputs and logistics before adding more machines; keep restart fuel and consume available science near the end. Jev is optional for batched typed classifications of supplied evidence and repair candidates, never arithmetic or long-range planning.
- scoring: Proposed benchmark score, not an official vanilla or verified FLE score: let P(i) be genuinely engine-produced cumulative items after the reset baseline and L(i) be science packs genuinely consumed for research, verified against labs/research. B = 0.01*min(P(iron-plate),1000) + 0.01*min(P(copper-plate),1000) + 0.02*min(P(iron-gear-wheel),200) + 0.05*min(P(electronic-circuit),200), capped at 34. Final score = B + sum over automation/logistic/chemical/production/utility/space science of weights 1/2/6/12/12/30 times [P(i)+4*L(i)]. This provides an early bootstrap signal and rewards sustained science plus research consumption. Initial stock, transfers, returned buildings and virtual manual output never increase P; track manual output separately, and count its packs only if genuinely consumed in labs. Validate API counter direction/surface/quality and report raw production, consumption, research IDs/progress, transfers, manual output and exactly charged ticks alongside the scalar.
- interface.observation: Revision-tagged JSON at paused boundaries: episode/seed/tick/turn budget; stock and reserved manual jobs; bounded revealed terrain and resources; stable entity IDs, positions, orientations, statuses, recipes, inventories, fluids and power; recipe/prototype data with exact costs/times; current and available research; cumulative baseline-adjusted engine production/consumption, score components and prior action results. Page detailed queries and cache immutable prototypes. No screenshot or connected player is required.
- interface.actions: A new typed bridge exposes mine_resource, mine_tree, handcraft, place, rotate, set_recipe, insert, extract, mine_building, set_research and wait through local RCON. These are proposed adaptor names, not verified FLE methods. All construction and transfers use owned items; manual work consumes actual ingredients/deposits and simulated work time. Apply up to 32 operations sequentially at a paused boundary, preserve successful operations if later ones fail, and then run exactly 1800 ticks. Empty wait and accepted batches with failed operations still consume a turn. Malformed/stale envelopes change nothing. Read-only observe calls do not consume turns; no arbitrary Lua, item spawning, technology grants or clock changes are agent actions.

### study/strategies.md
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

…(truncated; full text in prior/strategies.md)

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

### study/brief.md
# Factorio study brief

Study date: 2026-10-29. Target: offline Linux x86_64 without a GUI; evaluator Python **3.12.3** at `/home/hemlix/GamingMaster-mh-tier1/runs/venvs/factorio/bin/python`.

**Evidence limitation:** HTTPS requests to the Factorio site, wiki and GitHub failed with “Could not resolve host,” and no browser/search tool or local game documentation was available. The URLs below are reference targets from prior knowledge, not sources successfully retrieved during this session. Current releases, licenses, FLE dependencies and version-sensitive server APIs remain to be checked against staged materials. Local checks confirmed the evaluator interpreter, x86_64, glibc 2.39 and basic archive tools; no Factorio executable was on PATH, and FLE/RCON packages were absent from the venv. No game-playing code or software installation was performed. See [the evidence ledger](research/evidence.md).

## Game, rules and objective

Factorio is a factory-building simulation. Mine iron, copper, coal and stone; smelt ores; manufacture components; build power and transport networks; produce science packs; and consume science in laboratories to unlock technology. The base game's long-term objective is a rocket launch, after which play can continue. A short factory benchmark should reward intermediate production and research rather than require a rocket. Space Age adds a substantially different progression and is excluded here. [B1–B3]

The dependency chain matters more than the number of placed buildings. A burner miner needs fuel; an electric miner needs power and a resource patch. A furnace needs ore, energy and room for output. An assembler needs an unlocked compatible recipe, all ingredients, power and output space. A laboratory needs power and every science-pack type required by the selected research. Recipes have finite times, and some produce multiple items per execution. Missing inputs and blocked outputs stop production. [B2, B4–B6]

Resources and inventories are finite. Research unlocks recipes but grants no ingredients. Coal is fuel and later a chemical input. Power poles must both cover consumers and connect to generation. Belts have two lanes, inserters have direction and reach, and fluids must connect to the correct ports. Pollution and enemy evolution/expansion create military pressure in normal freeplay; these are disabled in the primary task. [B6–B8]

## Selected environment and rule changes

Use the **official free-to-download Linux headless server**, one exact base-game **2.0.x** build, a new trusted **Lua control mod**, and local **RCON**. The adaptor phase must stage and test the engine; this study cannot truthfully specify a current release number. Freeze binary/mod/settings hashes and matching API documentation. Disable Space Age, quality and elevated-rail expansion mods. [B9–B11]

The official server is **proprietary, although it costs nothing to download**. This study assumes the user's explicit headless-server preference permits zero-cost proprietary software. If every component must have a libre/open-source license, Factorio cannot meet that requirement. The selected private offline setup needs no graphical client purchase, Factorio account, Steam login or model-provider API key. A locally generated RCON password is required; no secret value belongs in study files. Verify applicable engine terms when staging the archive. [B9–B10]

This is an explicitly modified **remote factory-manager benchmark**, not a claim to reproduce vanilla avatar movement or FLE's exact semantics:

- Start with 1 burner mining drill, 1 stone furnace, 8 iron plates and 1 wood. Use normal base-game unlocked recipes, normal-quality items and no technology bonuses.
- Reveal a bounded area, initially `x,y ∈ [-128,128)`, containing viable deposits, water and trees. Remote construction/handling is allowed within it; there is no walking avatar requirement.
- Construction consumes owned items. Manual mining depletes real deposits. Manual crafting consumes actual ingredients and modeled work time. A single serial manual-work queue runs concurrently with the real engine factory; at most 60 simulated seconds of pending manual work and 32 jobs may be queued.
- Manual stock-to-machine and machine-to-stock transfers share a limit of **100 items per turn**. This is a benchmark concession for bootstrapping. Belts, inserters and pipes carry sustained engine production without that quota.

…(truncated; full text in prior/brief.md)

## Reset variation

Use a deterministic seed to vary resource layout, richness, coast/tree placement and build corridors within feasible bounds. Reject starts lacking required resources, a valid pump site or usable space in the allowed area. For held-out evaluation, vary the actual geography rather than loading the same factory under a new seed label.

Keep cold starts separate from seeded repair scenarios and later-game starter factories; those families have different opportunities and must be reported separately or normalized. Reset the full save, technologies, world, stock, queues, counters and RNG, not just inventories. A later combat family should have separate settings and can still use paused turns.

## Reference targets

All links below were unavailable for retrieval in this session; verify version-sensitive claims against staged documentation.

- B1: https://www.factorio.com/
- B2: https://wiki.factorio.com/Tutorial:Quick_start_guide
- B3: https://wiki.factorio.com/Rocket_silo
- B4: https://wiki.factorio.com/Mining ; https://wiki.factorio.com/Stone_furnace
- B5: https://wiki.factorio.com/Assembling_machine_1 ; https://wiki.factorio.com/Lab
- B6: https://wiki.factorio.com/Belt_transport_system ; https://wiki.factorio.com/Inserters
- B7: https://wiki.factorio.com/Electric_system ; https://wiki.factorio.com/Fluid_system
- B8: https://wiki.factorio.com/Pollution ; https://wiki.factorio.com/Enemies
- B9: https://factorio.com/download-headless ; https://www.factorio.com/download
- B10: https://www.factorio.com/terms-of-service
- B11: https://lua-api.factorio.com/latest/classes/LuaGameScript.html ; https://lua-api.factorio.com/latest/
- B12: https://wiki.factorio.com/Time
- B13: https://wiki.factorio.com/Command_line_parameters ; https://github.com/JackHopkins/factorio-learning-environment
- B14: https://lua-api.factorio.com/latest/classes/LuaForce.html
- B15: https://lua-api.factorio.com/latest/classes/LuaFlowStatistics.html

### adaptor/ADAPTOR.md
# Factorio adaptor

This is a Tier 1 remote factory manager running the **real, official Factorio 2.0.72 Linux headless engine**. `game.py` implements `TurnGame`, and `bridge/control.lua` is a trusted control mod. No Factorio Learning Environment, graphical client, account, expansion, Docker, Python dependency or model API is needed. The engine is zero-cost **proprietary** software, not libre software; this follows the user's explicit preference for the free headless server. The adaptor code is MIT licensed. Upstream terms are at https://factorio.com/terms-of-service (staged in `runtime/terms-of-service.html`); third-party engine licenses ship in `runtime/factorio/data/licenses.txt`.

## Setup and offline use

Run `./setup.sh` once with network access, or stage the archive listed in `engine.sha256` under `runtime/` before running it offline. The script downloads an exact version, verifies its SHA-256 and extracts it locally. The archive and executable are already staged on this machine. Subsequent episodes never download anything. Python is `/home/hemlix/GamingMaster-mh-tier1/runs/venvs/factorio/bin/python`; no packages need installing (`requirements.txt` documents this).

All environment variables are optional and documented in `env.example`. Defaults use the staged executable and this directory's `runtime/`. An optional `FACTORIO_RCON_PASSWORD` is read from the environment; otherwise a fresh cryptographic local secret is generated in memory per episode. There are no hard-coded credentials. Each instance binds RCON and multiplayer exclusively to loopback, chooses ephemeral ports, disables public/LAN listing, online verification, updates, autosaves and empty-server automatic pause. No human client connects. Set no required environment variables to use the default installation.

Each reset creates a new save, force, surface, RNG, inventory, research state, counters and job queue in a unique `runtime/episode-*` directory, then starts one server process. Base and `mh-bridge` are explicitly enabled; space-age, quality and elevated-rails are explicitly disabled and checked by the mod. Close disconnects RCON and terminates/waits for the server, with a kill fallback; it is idempotent. Logs, saves and other generated artifacts remain in `runtime/` for diagnosis. The Lua source is copied into the runtime mod directory, never generated over pinned source files.

## Scenarios

Only **cold-coast-v1** is implemented. There are no pooled repair, later-start, or combat scenarios. All integer seeds are supported; explicit disjoint train/heldout/dev/validation seed lists are in `game.py`. Integers map to engine seeds by the first four bytes (big-endian) of SHA-256 of `mh-factorio-cold-v1:SEED`, with zero replaced by one. Hashing avoids nearby small seed aliases in the engine RNG.

This is a deterministic **custom scenario generator**, not Factorio's default freeplay terrain distribution. The engine's seeded RNG selects coast orientation, position and sinusoidal shoreline, a permutation of ore quadrants, patch-center distances (30–65 tiles), angles, elliptical radii (7–11 by 6–10 tiles), irregular patch edges, amounts (1,800–3,200 units per tile), and tree positions. All deposits, tiles, trees and machines are genuine engine entities. The revealed/buildable square is `[-128,128)` on both axes. Real finite iron, copper, coal and stone each exceed 200,000 units. A clear central area and open corridors make all patches accessible. There is coastal water, wood, and a pump placement checked by `LuaSurface.can_place_entity`. Generator invariants avoid missing resources by construction. On a failed resource/pump validation, reset retries up to three times, adding `0x9E3779B9` modulo `2**32-1` to the previous map seed each time. `map_attempt` and the actual map seed are observed. No retry has been needed for the declared seeds. Seed changes affect physical constraints, not just labels.

Initial stock is fixed: one burner mining drill, one stone furnace, eight iron plates and one wood, in a 200-slot engine inventory. There is no initial factory, research bonus, pollution or enemy generation. This bounded early-game family intentionally supplies no oil or uranium; red/green science and applicable research are its attainable science objectives. Later science weights are retained in the required score formula but have no attainable production chain in this family.

## Turns and actions

The engine starts paused at tick zero. Every accepted gameplay batch executes up to 32 typed operations, then runs **exactly 1,800 ticks**, stopping in the mod's tick handler. There are **128 turns**, totaling 230,400 charged ticks (64 simulated minutes). Simulation speed is 20; wall-clock thinking delay has no effect. No final uncharged interval is run. RCON remains usable while paused. Requests carry monotonically increasing turn IDs; resubmitting an acknowledged turn returns a duplicate acknowledgment without replaying operations. Lost connections or server failures invalidate the trial for evaluator retry.

Actions are strings; join operations with semicolons. All grammar and mechanics are supplied to the model by `describe()` and `action_spec`. Examples:

```
mine_resource 42 10; place stone-furnace 0 0
insert 1200 fuel coal 5; insert 1200 source iron-ore 20
extract 1200 result iron-plate 10; handcraft iron-gear-wheel 2
place burner-mining-drill 25.5 -30.5 4
set_research automation
wait
recipe lab
inspect 100
```

IDs must come from the actual observation. Directions are north/east/south/west = 0/4/8/12. Manual mining/handcrafting is one serial worker using actual prototype mining times and vanilla character mining speed, or recipe duration at crafting speed 1. Its queue is limited to 32 jobs and 3,600 remaining ticks. Crafts reserve real ingredients immediately; only enabled character-craftable recipes with deterministic item products are supported. Outputs blocked by stock capacity wait without loss. Ore mining reduces real deposits; tree mining uses the engine's normal products.

Insert/extract share a quota of **100 actual items per turn**. Placement requires stock and engine collision/terrain checks. Invalid per-operation attempts report errors and still consume that batch's turn. Malformed strings are rejected without advancing. Unknown/stale entities, invalid counts, and unsupported operations never become arbitrary Lua execution. Read-only `inspect OFFSET`, `recipe NAME` and `technology NAME` queries must be alone and charge no turn.

Two conservative differences from the proposed study interface are intentional: buildings must be empty before mining, and assemblers must be empty and idle before changing recipe. Cargo must first be extracted under the transfer quota. Mining loses remaining fluids and burning fuel, matching ordinary engine behavior. This prevents cargo/quota bypass and partially completed recipe duplication. Native craft-item research triggers are mirrored for **paid, completed manual crafts** with prerequisites satisfied (particularly crafting the first lab); no production counters are forged. Other production/research remains engine-driven.

## Observations and score

…(truncated; full text in prior/ADAPTOR.md)

```
B = .01*min(P(iron-plate),1000) + .01*min(P(copper-plate),1000)
  + .02*min(P(iron-gear-wheel),200) + .05*min(P(electronic-circuit),200)
score = B + Σ weight(i) * (P(i) + 4*L(i))
```

Weights for automation/logistic/chemical/production/utility/space science are 1/2/6/12/12/30. Bootstrap caps at 34. This is the requested benchmark score, not an official vanilla or FLE score. Initial stock, transfers, mining/rebuilding and virtual manual outputs do not increase `P`. `score_breakdown()` returns numeric components. `info()` reports validity, infrastructure failures, engine/family, runtime location, charged ticks and wall time. The model sees the raw economic/research ledgers in observations.

## Verification

`verify.py` exercises the actual staged engine, including privileged test fixtures that **are not part of the game action space**. It writes results to `runtime/verification.json` and `runtime/verification.log`. It checks a 60-second paused interval by querying the server twice, exact 1,800-tick steps, furnace production, manual/transfer exclusion, duplicate-step handling, powered assembler and lab production/consumption, malformed and illegal actions, physical differences and feasibility across all declared seeds, deterministic reset, random actions and crash invalidation.

Run the generic check from the repository root:

```
/home/hemlix/GamingMaster-mh-tier1/runs/venvs/factorio/bin/python -c "from metaharness.games.adaptor import validate_adaptor; print(validate_adaptor('runs/factorio/adaptor', expected_tier=1).to_markdown())"
```

Current test evidence is recorded in runtime files. Infrastructure failures are not silently converted into valid zero-score episodes.

### Recorded results on this machine

- Generic validator: **PASS**, including deterministic fresh-instance reset. An additional run with `max_actions=20` covers the entire declared action vocabulary; no warnings were reported.
- All **14** declared scenario seeds have distinct physical-layout SHA-256 fingerprints and validated resources, trees and pump sites (`runtime/verification.json`).
- A fueled stone furnace produces **9 plates in 1,800 ticks**, worth **0.09**. The powered assembler/lab fixture produced two red packs and consumed one in real research; combined with 20 furnace plates, its score was **6.2**.
- The public-action-only bootstrap (`verify_play.py`) reaches **180 iron plates, 30 copper plates, a lab, steam-power components and five manually crafted red packs in 50 turns**, scoring **2.1**. Native smelting triggers and the paid manual lab-craft trigger unlock the expected technologies. The manual packs add no production score.
- Inventory checks (`runtime/economy.log`) verify loaded-building rejection, empty-building recovery, the shared 100-item transfer quota, and exact preservation of a partially used science pack across extraction/reinsertion. Transfers copy engine stack metadata, not just item counts.
- The engine's virtual `science` production counter is included in the raw diagnostic ledger but never scored; only the named science-pack counters enter the formula.

Additional timing/horizon checks **passed** and are recorded in `runtime/clock-verification.json`: speed/inference-delay invariance, exact manual-worker time, and termination after 128 turns / 230,400 ticks with no post-horizon advance. `artifact-manifest.json` records the final source, engine archive and API-document hashes. All verification servers are closed after their checks.

### study/interfaces.md
# Factorio programmatic interfaces and adaptor specification

Target: offline Linux x86_64, no GUI; evaluator Python **3.12.3** at `/home/hemlix/GamingMaster-mh-tier1/runs/venvs/factorio/bin/python`. Study date: 2026-10-29. **Web retrieval failed at DNS. All source URLs are reference targets, not freshly verified pages.** See [the evidence ledger](research/evidence.md). In particular, no installed Factorio server or FLE checkout was available for an integration test.

## Recommendation and licensing boundary

Use the **official, zero-cost Linux Factorio headless server**, a small trusted **Lua control mod**, and **local Source RCON**, with a Python adaptor that exposes typed JSON actions. Pin one base-game **2.0.x** build and disable expansion mods. The mod is to be written in the adaptor phase; it does not exist yet. This is a modified, remote-manager factory benchmark whose production still runs in the actual Factorio engine. It deliberately has no GUI, human client, account login, cloud service, or language-model API dependency.

Factorio's server is proprietary even though its headless download is free. This recommendation interprets the user's preference for that server as allowing zero-cost proprietary software. If “free software” means exclusively libre/open-source licenses, **Factorio itself cannot satisfy the requirement**; FLE and an open-source RCON wrapper do not change the engine's license. Confirm the downloaded archive's applicable terms. A paid graphical client or Space Age purchase must not become an implicit prerequisite. [I1–I3]

## Practical interface inventory

Latency figures describe expected classes, not measurements on this machine. A local RCON exchange should usually be milliseconds to tens of milliseconds; large maps, Lua scans, and JSON serialization can make it much slower. Simulation wall time is separate from request latency.

| Interface | Maturity and exposure | Timing and constraints | Cost / license / suitability |
|---|---|---|---|

…(truncated; full text in prior/interfaces.md)

lasses/LuaFlowStatistics.html ; https://lua-api.factorio.com/latest/classes/LuaEntity.html ; https://lua-api.factorio.com/latest/classes/LuaInventory.html ; https://lua-api.factorio.com/latest/classes/LuaSurface.html — use matching-version APIs, not moving latest during evaluation.
- I5: https://wiki.factorio.com/Command_line_parameters ; https://wiki.factorio.com/Multiplayer ; https://wiki.factorio.com/Tutorial:Modding_tutorial — CLI, dedicated-server setup and mod packaging.
- I6: https://github.com/JackHopkins/factorio-learning-environment — FLE source, setup and actual tool contracts.
- I7: https://pypi.org/project/rcon/ ; https://github.com/conqp/rcon ; https://pypi.org/project/factorio-rcon-py/ ; https://developer.valvesoftware.com/wiki/Source_RCON_Protocol — optional clients and transport framing.
- I8: https://wiki.factorio.com/Console — console commands and scripting consequences.
- I9: https://github.com/factoriotools/factorio-docker — third-party container packaging.
- I10: https://factoriolab.github.io/ ; https://github.com/factoriolab/factoriolab — factory planning calculator, not an engine adaptor.


## Measured model latency (at archive creation, 2026-10-29 23:10)
- System-2 (Astra), one-word answer: 32 s
- System-2, an answer of ~967 output tokens: 79 s — latency grows with the length of the answer; size timeouts for the answers your prompts actually request
Later measurements per candidate are on the leaderboard (latency columns, a latency fit).