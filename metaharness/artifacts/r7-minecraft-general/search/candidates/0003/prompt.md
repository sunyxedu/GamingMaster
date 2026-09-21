# Meta-Harness iteration -> candidate `0003`

Working directory: `/Users/hemlix/Documents/GamingMaster/runs/r7-minecraft/search` (you have full read/write access here).

## Task
Design a harness for a **real-time** game (tier 2).

Game: `Minecraft Java 1.21.1 — natural survival` — Play ordinary Minecraft Java 1.21.1 Survival on Normal difficulty for 300 wall-clock seconds. The vanilla server and bot physics keep running while you think. Death ends the episode without respawn. You start healthy, empty-handed on an admitted natural dry surface, with nearby wood and shallow stone. Initial clear weather and time 1000 or 9000; night, weather, monsters, hunger, fall/fire/drowning damage, tool requirements, crafting and smelting follow vanilla rules. Unknown terrain is not supplied. Coordinates are x east, y up, z south.
Objective: maximize absolute score [0,100]. Resource credit uses episode maxima of simultaneously held personal inventory: min(unstripped Overworld logs,10)+0.5*min(cobblestone,20)+min(coal+charcoal,5)+min(raw_iron,5), up to30. First possession gives crafting_table5, wooden_pickaxe5, stone_pickaxe10, stone_axe5, furnace5, torch5, cooked_food5, iron_ingot5, shield5, up to50. Cooked food is cooked_beef/porkchop/mutton/chicken/rabbit/cod/salmon or baked_potato. Loot counts. Spent/dropped items retain earned maxima/milestone credit; repeated transfers do not add credit. Survive for 10*seconds/300 points. At an alive time-limit finish only, add 10*health/20; death subtracts20; clamp total. Full healthy idling yields20. Infrastructure failures invalidate a trial.
Observations include episode/sequence/freshness, elapsed and remaining seconds, self position/velocity/health/food/oxygen/equipment, authoritative personal inventory, predictive inventory_client, day/weather, nearby entities, local collision labels relative to your floored position, exposed block and stand candidates, last-observed furnace contents (not live after closing), feasible recipes, current task, action results, and score components/milestone times. Candidate IDs are strings b:x,y,z or s:x,y,z supplied by observations; resource candidates are limited to exposed loaded blocks within24 blocks, never a hidden ore scan. Stand points are locally checked; routes can fail. Entities may be occluded. Unknown chunks are omitted. No screenshot is needed. Inventory scoring comes from atomic read-only server queries, not client crafting previews. The policy cannot use console commands, teleport, creative inventory, raw packets or arbitrary code.
Use whitespace-separated action strings below. Craft NEW_COUNT means requested new outputs, rounded up to whole recipe executions (craft oak_planks 4 uses one log). Table recipes require a nearby placed table; optionally give its candidate ID. Furnace loading returns immediately; smelting takes actual time; retrieve later. Navigation/gather automatically choose a conservative route but never dig/bridge to travel. Gathering checks tool, reach and exposure, mines and tries to pick up a real drop. One job runs at a time. Deadlines are25s gather,15s navigation,10s others. noop lets a bounded job continue; cancel really clears controls. Wait and invalid actions still consume real time. Identical current actions are latched and do not replay after finishing: send noop then the same command to repeat deliberately. A different command while busy is rejected; cancel then wait for task=null. Actions receive feedback rather than raising for bad arguments. Optional dictionary envelope {episode,seq,action} rejects earlier episodes and observations more than10 snapshots old.
- noop: Continue an existing bounded job; otherwise stay still and release shield.
- cancel: Immediately stop the current job, navigation, digging, item use and window.
- navigate: navigate CANDIDATE_ID: travel to a listed stand or block interaction position.
- look_at: look_at X Y Z: look at a nearby point, avoiding known Enderman gaze.
- gather: gather BLOCK_ID [N=1]: mine and collect 1–4 exposed blocks of that type, choosing a held tool.
- equip: equip ITEM [DEST=hand]: equip inventory item to hand/off-hand/head/torso/legs/feet.
- craft: craft ITEM NEW_COUNT [TABLE_ID]: craft at least NEW_COUNT new outputs from inventory.
- place: place ITEM REFERENCE_BLOCK_ID FACE: place on up/down/east/west/north/south face.
- furnace_start: furnace_start FURNACE_ID INPUT COUNT FUEL COUNT: load an actual furnace, then close it.
- furnace_take: furnace_take FURNACE_ID: retrieve finished output, if any.
- eat: eat ITEM: consume held food when hungry.
- attack: attack ENTITY_ID: one visible in-range melee attack, respecting cooldown.
- shield: shield on|off: equip/use shield in the off-hand or release it.
- wait: wait [SECONDS=0.5]: wait for 0–5 seconds without freezing the world.
(no decision in time => 'noop')
Actions are whitespace-separated strings. Bare parameterized names are rejected as feedback, not Python exceptions. Only one bounded job owns movement/hands. Repeated identical current actions do not restart; send noop between deliberate repeats. noop preserves an existing job. cancel before replacing a busy job.

Actions:
- noop: Continue an existing bounded job; otherwise stay still and release shield.
- cancel: Immediately stop the current job, navigation, digging, item use and window.
- navigate: navigate CANDIDATE_ID: travel to a listed stand or block interaction position.
- look_at: look_at X Y Z: look at a nearby point, avoiding known Enderman gaze.
- gather: gather BLOCK_ID [N=1]: mine and collect 1–4 exposed blocks of that type, choosing a held tool.
- equip: equip ITEM [DEST=hand]: equip inventory item to hand/off-hand/head/torso/legs/feet.
- craft: craft ITEM NEW_COUNT [TABLE_ID]: craft at least NEW_COUNT new outputs from inventory.
- place: place ITEM REFERENCE_BLOCK_ID FACE: place on up/down/east/west/north/south face.
- furnace_start: furnace_start FURNACE_ID INPUT COUNT FUEL COUNT: load an actual furnace, then close it.
- furnace_take: furnace_take FURNACE_ID: retrieve finished output, if any.
- eat: eat ITEM: consume held food when hungry.
- attack: attack ENTITY_ID: one visible in-range melee attack, respecting cooldown.
- shield: shield on|off: equip/use shield in the off-hand or release it.
- wait: wait [SECONDS=0.5]: wait for 0–5 seconds without freezing the world.
(no decision in time => 'noop')
Actions are whitespace-separated strings. Bare parameterized names are rejected as feedback, not Python exceptions. Only one bounded job owns movement/hands. Repeated identical current actions do not restart; send noop between deliberate repeats. noop preserves an existing job. cancel before replacing a busy job.

Models available through `ctx`:
* `ctx.think(prompt, session_id=...)` — Astra (System-2, slow: seconds). Deep reasoning, can plan,
  can explain rules. Returns `.text` and `.session_id` (pass it back to keep conversation context).
  Budget: 40 calls per episode. Astra never sees the game unless you show it.
* `ctx.decide(state, questions)` — Jev (System-1, fast typed judgments). Budget: 4000 calls per
  episode. Read the complete Jev model context and local usage contract below before designing
  its state, questions and answer-to-action bridge. Actual latency must be measured.
* `ctx.spawn(fn)` — background thread, so Astra can think while Jev keeps acting.

Model economics (measured facts, not instructions): one Jev request answers *many independent typed
questions about one state in parallel* for the price and latency (~0.5 s) of one, and the Jev budget
is two orders of magnitude larger than Astra's; an Astra call takes tens of seconds to minutes and
the budget is a few dozen per episode. Whatever the design, the evaluator reports how many posted
actions were preceded by a fresh Jev / Astra answer, so the division of labour is visible.

The game ticks every 0.5s **on its own clock**; `ctx.act(a)` posts the current action (instant).
A tick with no fresh action re-applies the last one. Evaluation metric: mean score over seeds
(primary), then fraction of stale ticks (lower is better), then System-2 calls (fewer is better).
Clock mode: wall.

What the harness must design (this is the whole point): how Astra's slow output (a plan, a rule
summary, a policy sketch, a subgoal) is turned into *state + questions* for Jev when needed, how
Jev's typed answer is turned into a game action, when to go back to Astra, and what to do when a
decision is late.

**Generality requirement.** The best harness is re-evaluated on a *held-out game* with the same
interface (`game.describe()`, `game.action_spec`, dict observations, `ctx`) but different rules and
different observation keys. Do not hard-code this game's rules or observation fields; discover them
from `game.describe()` / the observation, or ask the System-2 model. Game-specific solvers score 0
on the held-out game.

## Prior knowledge from the study phase
The study phase researched this game (rules, interface, community strategies). Use it: implement exact
rules/bookkeeping/search as code, put planning and adaptation into Astra prompts, and fast recognition
into Jev questions — but verify every claim against observations, because sources can be wrong or
outdated for this build of the game.

### study/brief.md
Minecraft: Java Edition — study brief

Researched 2026-09-21. Target the **vanilla Java Edition 1.21.1 dedicated server**, one Survival player, Normal difficulty, local offline authentication, and **300 seconds of active play**. This is a proposed early-game benchmark, not Minecraft's built-in scoring system. No game-playing code was written or server launched during this study.

**Objective and rules.** Minecraft is an open-ended sandbox rather than a game with one compulsory victory condition. Survival play involves gathering blocks and items, crafting equipment, managing hunger and health, and avoiding environmental and creature damage. This benchmark rewards becoming productively equipped within five minutes; reaching the End or defeating the dragon is outside its scope. [Mojang's beginner overview](https://www.minecraft.net/en-us/minecraft-tips-for-beginners), [Creative versus Survival](https://www.minecraft.net/en-us/article/creative-vs-survival-mode).

- Begin in the Overworld with an empty inventory, full health and food, no experience, and no operator privileges. The adaptor owns a separate administrative console for setup and shutdown.
- The normal progression is logs → planks/sticks/crafting table → wooden pickaxe → cobblestone → stone tools/furnace → fuel, cooked food, and optionally iron/shield. Tool suitability matters: a stone pickaxe can harvest iron; merely breaking a block does not guarantee its useful drop. [Mojang's pickaxe guide](https://www.minecraft.net/en-us/article/taking-inventory-pickaxe).
- Inventory crafting provides a 2×2 grid; a placed crafting table enables 3×3 recipes. Crafting consumes ingredients; the output count may exceed one. Furnaces consume input and fuel over time. [First-day guide](https://www.minecraft.net/en-us/article/how-survive-your-first-day), [furnace guide](https://www.minecraft.net/en-us/article/block-week-furnace).
- Health and food are each represented on a 0–20 scale; health points are half-hearts. Food supports natural healing. Death normally allows respawn, but **this benchmark terminates on the first death**. Disable Mineflayer's automatic respawn. [Mineflayer API](https://github.com/PrismarineJS/mineflayer/blob/4.39.0/docs/api.md), [death/spawning guide](https://www.minecraft.net/en-us/article/spawning-and-dying).
- Darkness can produce hostile mobs; caves are dangerous even during daylight. Light and a small enclosed shelter are useful. Beds can skip nighttime, but are optional in a short episode. [First-night guide](https://www.minecraft.net/en-us/article/how-survive-your-first-night-minecraft).

**Pace and tier.** This is **tier 2**. The dedicated server normally targets **20 ticks/second**, or 50 ms per tick. The day/night cycle is 24,000 ticks, about 20 minutes; sunset begins around tick 12,000. The world continues while Python waits for a model. A five-minute dawn-only episode would usually miss surface nighttime threats, so include dusk starts. [Mojang's tick command specification](https://www.minecraft.net/en-us/article/minecraft-java-edition-1-20-3), [Mineflayer time implementation](https://github.com/PrismarineJS/mineflayer/blob/4.39.0/lib/plugins/time.js).

Use **decision_latency_s = 0.5** as a proposed tactical response budget, not a guaranteed safe delay. Falling, drowning, lava, or an already-primed creeper can require immediate code-level action. Creepers have an ordinary fuse of roughly 1.5 seconds, making a multi-second reasoning call unsuitable for emergency response. Safe gathering or sheltering can tolerate several seconds, but the episode clock still runs. [Creeper mechanics](https://minecraft.fandom.com/wiki/Creeper).

Recommended scheduling: local physics and safety checks every tick; JSON observations at 5 Hz plus urgent events; Jev choices every 0.5–1 second when needed; Astra asynchronous plans at milestones, failures, or roughly 15–30 second intervals. Astra's actual latency may exceed that interval, so permit only one outstanding planning call and keep executing its bounded prior plan. These are design targets, not measured performance. Vanilla `tick freeze` does **not** turn this into a proper turn-based environment: players and their mounts are exempt. Do not pause or slow the world during scored play. [Tick behavior](https://www.minecraft.net/en-us/article/minecraft-java-edition-1-20-3).

**Controls and agent actions.** A human normally moves with WASD, jumps with Space, sneaks with Shift, sprints with Ctrl, looks with the mouse, attacks/mines with the left mouse button, uses/places with the right, opens inventory with E, and selects hotbar slots with 1–9. Key bindings are configurable. The recommended agent interface instead exposes bounded navigation, look, gather, equip, craft, place, furnace input/output, eat, attack, shield, cancel, and wait actions. These map to Mineflayer and pathfinder operations; they must obey range, visibility, tool, collision, inventory, and time constraints. [Mojang's controls overview](https://www.minecraft.net/en-us/article/minecraft-controls), [Mineflayer](https://github.com/PrismarineJS/mineflayer), [pathfinder](https://github.com/PrismarineJS/mineflayer-pathfinder).

**Exact proposed score.** Report an absolute score in [0, 100], its components, and optional step reward as the change in absolute score. Let T = 300 seconds and a be wall-clock seconds survived since the ready observation, capped at T. Setup/loading time is excluded; model reasoning time is included. Let H be final health clamped to [0,20], A = 1 if alive when the time limit is reached and otherwise 0, and D = 1 if the episode ended in death and otherwise 0.

Resource high-water marks are the **largest simultaneously held, server-confirmed inventory count** for each category during the episode. Aggregate slots before taking a maximum. Count personal storage and equipped items once; exclude crafting result previews, the cursor, and container/furnace slots until transferred into personal inventory. Never sum all positive inventory deltas: tossing and reclaiming items would otherwise farm points.

| Component | Definition | Maximum |
|---|---|---:|
| Logs | 1 × min(log high-water mark, 10) | 10 |
| Cobblestone | 0.5 × min(cobblestone high-water mark, 20) | 10 |
| Fuel | 1 × min(combined coal + charcoal high-water mark, 5) | 5 |
| Raw iron | 1 × min(raw_iron high-water mark, 5) | 5 |
| Equipment milestones | First verified possession of each item in the next table | 50 |
| Survival time | 10 × a/T | 10 |
| Healthy finish | 10 × A × H/20 | 10 |
| Death penalty | −20 × D | −20 |

Final score = clamp(resources + milestones + survival time + healthy finish − death penalty, 0, 100). Retain precision internally and display one decimal. Logs means the eight unstripped Overworld log items in 1.21.1: oak, spruce, birch, jungle, acacia, dark_oak, mangrove, cherry. No crafted wood blocks, stripped logs, stems, or planks count in that category. Cobblestone means `cobblestone` only.

| First possession milestone | Points |
|---|---:|
| crafting_table | 5 |
| wooden_pickaxe | 5 |
| stone_pickaxe | 10 |
| stone_axe | 5 |
| furnace | 5 |
| torch | 5 |
| Any cooked food in the explicit list below | 5 |
| iron_ingot | 5 |
| shield | 5 |

Cooked-food items: `cooked_beef`, `cooked_porkchop`, `cooked_mutton`, `cooked_chicken`, `cooked_rabbit`, `cooked_cod`, `cooked_salmon`, `baked_potato`. This is an equipment/progress score: legitimate looting also counts, not only crafting. Milestones latch once and survive later placement, consumption, breakage, or death. Resources already credited also survive consumption. The high-water design deliberately rewards keeping a modest useful reserve; it does not claim to measure lifetime mining. No extra points for repeatedly crafting the same tool, breaking player-placed blocks, kills, distance, or XP. An idle healthy survivor earns 20; substantial early-game progress can earn much more. Full marks are an upper bound, not an expected outcome on every 
…(truncated)

### study/interfaces.md
Minecraft: Java Edition — adaptor interface specification

Researched 2026-09-21. This document specifies a future adaptor; it contains no implemented game-playing logic. All proposed files, downloads, caches, worlds, and logs belong under `/Users/hemlix/Documents/GamingMaster/runs/r7-minecraft/study/`. Treat timings and resource budgets below as engineering estimates until measured.

**Choose Mineflayer through a persistent Node bridge, connected to an adaptor-owned vanilla 1.21.1 server on loopback.** Python 3.14.5 owns the lifecycle and sends JSON Lines over subprocess pipes. Node owns the connected player and uninterrupted physics. The server uses `online-mode=false`; the bot uses `auth: 'offline'`. No Microsoft/Mojang account, launcher, access token, API key, renderer, or paid hosting is technically required for this connection. The mature Mineflayer API supplies world, inventory, movement, digging, crafting, and entity abstractions that a raw protocol client would have to rebuild. [Mineflayer project](https://github.com/PrismarineJS/mineflayer), [offline protocol login](https://github.com/PrismarineJS/node-minecraft-protocol).

**Practical interface alternatives.** Latency refers to expected local responsiveness, not measured benchmarks. All choices still need a legal Minecraft installation/server where applicable. Bot use on unrelated public servers depends on their rules; this study only proposes a private loopback server.

| Interface | Maturity and exposure | Latency / costs / risks | Assessment here |
|---|---|---|---|
| Mineflayer + pathfinder | Established Node ecosystem; protocol-backed blocks, entities, inventory, health, recipes and physical actions. No rendering required. | Local events usually arrive within ticks; actions take their actual game duration. Free open-source libraries; CPU/RAM for server. Version compatibility and pathfinding failures still need tests. | **Selected.** Best fit for installed Node and no account. |
| `minecraft-protocol` directly | Mature parser/serializer with offline authentication and keepalives; packet-level access. | Low transport overhead; substantial work for physics, state reconstruction, inventory transactions and navigation. Free, same local-server constraints. | Useful diagnostics beneath Mineflayer, poor primary interface. [Project](https://github.com/PrismarineJS/node-minecraft-protocol). |
| Python pyCraft / Quarry | Existing protocol libraries, not complete survival agents. | Asynchronous packet access; uncertain modern-version and Python 3.14 dependency coverage. Physics/planning burden; free. | Do not assume `pip install` creates a playable agent. [pyCraft](https://github.com/ammaraskar/pyCraft), [Quarry](https://github.com/barneygale/quarry). |
| Rust Azalea / C++ Botcraft | Active specialist bot/client libraries; world and movement facilities, Botcraft optional rendering. | Low-level native efficiency; Rust/C++ toolchains, compilation and version selection add setup. Open source. | Plausible headless alternatives, more integration work than existing Node stack. [Azalea](https://github.com/azalea-rs/azalea), [Botcraft](https://github.com/adepierre/Botcraft). |
| Server stdin, RCON, commands, datapacks | Vanilla administrative controls can configure time, difficulty, seed-related setup, entities, scoreboards and inspect data. Datapacks can implement objectives/events. | Console commands execute on server ticks; text parsing or RCON framing required. No RCON cost, but authentication/network exposure if enabled. | **Use stdin for lifecycle and setup only.** Teleport/give/setblock are privileged world edits, not survival player actions. RCON needs a secret; avoid it with owned pipes. [Server properties](https://docs.papermc.io/paper/reference/server-properties/). |
| New official Java server management API | Newer releases provide JSON-RPC over WebSocket for server management. | Structured admin requests, but not a rich embodied player API; requires newer server and authentication/TLS configuration. | Not available on pinned 1.21.1; no reason to upgrade for this task. [Mojang announcement](https://www.minecraft.net/de-de/article/minecraft-snapshot-25w35a). |
| Fabric / Forge / NeoForge mod, including Carpet/Scarpet and GameTest-style fixtures | Mature mod platforms; custom tick hooks, exact world/player state, fake-player or custom action implementations. Carpet provides server scripting. | Can run every tick; free tools but Java build/mapping/mod compatibility work. Easy to accidentally grant omniscience or bypass game mechanics. | Useful future reset/instrumentation layer, unnecessary initially. [Fabric](https://docs.fabricmc.net/develop/), [Carpet](https://github.com/gnembon/fabric-carpet). |
| Paper / Spigot / Bukkit plugin | Mature server event and world/inventory APIs; good authoritative scoring hooks. | Tick events and plugin RPC are fast. Free software, extra Java plugin build; server implementation can change vanilla behavior. | Strong fallback if trusted scoring or complex fixtures outgrow vanilla console. Need a separate connected player/client for ordinary movement. [Paper development](https://docs.papermc.io/paper/dev/). |
| Baritone client mod | Established automated pathfinding, mining and building with Java API. | Tick-level control inside a client; full client/mod setup and Java integration, graphics/client ownership issues. | More automation but poorly matched to no-account headless setup. [Baritone](https://github.com/cabaletta/baritone). |
| Minescript client mod | Python scripts can query/control a running modded client. | Tick/IPC overhead; convenient Python access but requires full client, version-matched mod and Python integration validation. | Python branding does not prove evaluator 3.14.5 support or remove client/account requirements. [Minescript](https://minescript.net/). |
| Project Malmo / MalmoEnv | Influential Microsoft research platform: XML missions, observations, reward channels and command actions; MalmoEnv has a Python-to-Java environment. Legacy/archive status. | Experiment-style stepping is possible; old Minecraft/Java builds and native bindings can be expensive to revive. No modern cp314 wheel evidenced. | Do not install legacy Malmo into the evaluator. A separate older runtime would require another bridge. [Malmo](https://github.com/microsoft/malmo), [MalmoEnv](https://github.com/microsoft/malmo/tree/master/MalmoEnv), [published wheels](https://pypi.org/project/malmo/). |
| MineRL | Established research tasks/datasets and Gym-style interaction; rich task ecosystem. | Legacy JDK 8/Gradle/client rendering stack and older dependencies; startup can dwarf a short episode. Python 3.14.5 compatibility not established. | Good research precedent, not a practical dependency choice here. [Project](https://github.com/minerllabs/minerl), [installation](https://minerl.readthedocs.io/en/latest/tutorials/index.html). |
| MineDojo / MineStudio | Research environments with visual observations, actions and task tooling; MineDojo includes voxel/inventory information. | Java simulator/rendering dependencies; MineStudio recommends Linux/Conda, JDK 8 and rendering support. “Python ≥3.9/3.10” does not certify 3.14.5. Free software, potentially significant compute/setup. | A subprocess in another Python could work, but would duplicate functionality already in Mineflayer. [MineDojo](https://github.com/MineDojo/MineDojo), [MineStudio](https://github.com/CraftJarvis/MineStudio). |
| Voyager | Research agent architecture and Mineflayer bridge precedent, not a minimal environment. | Includes an LLM stack, mods and setup assumptions; model calls add cost. | Borrow skill composition and feedback ideas; do not install its whole stack or require its account/API-key workflow. [Voyager](https://github.com/MineDojo/Voyager). |
| Screen capture + keyboard/mouse | General fallback on the real graphical client; pixels/UI rather than structured world state. MSS captures; PyAutoGUI injects 
…(truncated)

### study/strategies.md
Minecraft: Java Edition — strategy prior for Astra, Jev and the harness

Researched 2026-09-21 for vanilla **1.21.1**, Normal Survival, empty inventory, five-minute episodes and the score specified in brief.md/interfaces.md. These recommendations optimize that task, not a dragon speedrun, PvP match or long-term building world. The timings and thresholds below are initial controller choices, not universal Minecraft rules.

**Core strategy.** Turn a short wood opening into stone equipment quickly, then secure cheap furnace/fuel/food milestones while avoiding damage. Explore only as far as necessary to find the next reachable prerequisite. Obtain exposed iron when it is cheap; stop pursuing it when terrain, darkness or time makes survival and already-known rewards more valuable. A small safe work area and reliable item pickup beat impressive but unfinished plans. The community beginner guide specifically favors a wooden pickaxe first and an early upgrade after three cobblestone; this avoids spending time making a whole wooden tool set. [Community beginner guide](https://minecraft.wiki/w/Tutorial%3ABeginner%27s_guide), [Mojang first-day guide](https://www.minecraft.net/en-us/article/how-survive-your-first-day).

**Opening and build order.** Treat this as a dependency graph with branches, not a rigid timed script. Code computes feasible recipes and distances; Astra chooses the next short sequence and a fallback. Jev selects among currently valid nearby targets or recognizes when to interrupt.

| Stage / rough elapsed budget | Strong action and rationale | Ownership |
|---|---|---|
| 0–15 s: orient | Inspect nearby exposed wood, accessible stone, footing, hazards and time. Select a short route to a harvestable trunk; remember a safe retreat position. Avoid wandering to an ideal scenic base. | Code computes candidates/paths; **Astra** selects a local work area; **Jev** judges which legal trunk approach is least obstructed or threatening. |
| 10–60 s: wood bootstrap | Gather about three logs initially if stone is close; convert enough to make a table, sticks and one wooden pickaxe. Place the table on accessible dry ground. Keep its coordinates. Add wood if the nearest stone requires travel. | **Code** handles quantities, recipe variants, placements and confirmation; **Astra** decides whether to work here or move toward stone. |
| 40–100 s: stone upgrade | Mine three ordinary stone blocks with the wooden pick; pick up their cobblestone; craft/equip a stone pick. Use it for subsequent stone. Avoid granite/diorite/andesite when the target is cobblestone. | **Code** checks block/drop/tool eligibility; **Astra** orders the upgrade; **Jev** may choose an exposed safe face among code-approved candidates. |
| 70–160 s: productive tools | Craft stone axe and furnace. Axe accelerates additional wood gathering and can handle nearby passive animals. Stone sword is optional if combat is likely; it has no direct milestone reward here. | **Astra** chooses axe/furnace order based on resources and threats; **code** executes the material graph. |
| 100–240 s: fuel and food | Prefer reachable surface coal; otherwise reserve logs and smelt charcoal with planks. Make torches. Start cooking local food or smelting exposed raw iron while working nearby. | **Astra** chooses coal versus charcoal and workstation placement; **code** schedules furnace jobs; **Jev** judges interruption/threat and nearby prey choices. |
| 160–280 s: optional iron | A stone pick can harvest iron. Smelt an ingot; a shield uses little iron and benefits survival. Gather up to the resource caps only if doing so is safe and cheap. Deep mining is usually a poor five-minute bet. | **Astra** weighs expected progress and remaining time; **code** estimates costs/cooldowns; **Jev** recognizes a dangerous approach. |
| Last 20–40 s | Collect finished furnace output, eat if needed, and finish cheap already-feasible milestones. Stay in a verified safe place; do not start a long cave trip. | **Code** enforces the deadline and pending work; **Astra** picks final tasks; **Jev** monitors safety. |

The bootstrap arithmetic is simple but belongs in code: three logs produce 12 planks. Four planks make the table, three make the wooden pick head, and two plank-to-stick operations consume four more planks to produce eight sticks. That leaves one plank and enough sticks for wooden pick, stone pick and stone axe. The three stone tools/workstations listed here consume 3 + 3 + 8 = **14 cobblestone** total. Gather more wood for fuel, shield, torches and reserves. Do not delay the stone pick merely to maximize the cobblestone inventory high-water score.

**Recipe and resource facts to implement as exact rules.** Use the pinned version's actual registry/recipes as authority, rather than an LLM's remembered recipe. Different wood species have separate plank recipes even where mixed planks can be ingredients. The following is a compact planning reference. [Minecraft data registry](https://github.com/PrismarineJS/minecraft-data), [Mineflayer crafting semantics](https://github.com/PrismarineJS/mineflayer/blob/4.39.0/lib/plugins/craft.js).

| Result | Inputs / important condition | Assignment |
|---|---|---|
| 4 matching planks | 1 ordinary Overworld log | Code |
| 4 sticks | 2 planks | Code |
| 1 crafting table | 4 planks, inventory grid sufficient | Code |
| 1 wooden pickaxe | 3 planks + 2 sticks, placed table | Code |
| 1 stone pickaxe or stone axe | 3 suitable stone-tool ingredients + 2 sticks, placed table; cobblestone is the intended local source | Code |
| 1 stone sword | 2 suitable stone-tool ingredients + 1 stick, placed table | Code; Astra decides whether worthwhile |
| 1 furnace | 8 valid furnace stone ingredients; cobblestone is the baseline source | Code |
| 4 torches | 1 coal OR charcoal + 1 stick | Code |
| 1 charcoal | Smelt 1 eligible log/wood item with separate fuel; planks are fuel, not charcoal input | Code; Astra chooses this fallback |
| 1 iron ingot | Smelt 1 raw iron with fuel | Code |
| 1 shield | 6 planks + 1 iron ingot, table | Code; Astra prioritizes survival value |
| 1 bed | 3 matching-color wool + 3 planks, table | Code; optional Astra opportunity |

Mojang describes the table/tool opening, furnaces and shield ingredients in its [first-day guide](https://www.minecraft.net/en-us/article/how-survive-your-first-day), [furnace guide](https://www.minecraft.net/en-us/article/block-week-furnace), and [shield guide](https://www.minecraft.net/de-de/article/taking-inventory--shield). The [pickaxe progression](https://www.minecraft.net/en-us/article/taking-inventory-pickaxe) explains why stone tools unlock iron. In this version, ordinary iron mining yields **raw_iron**, which must be smelted; do not search the inventory only for `iron_ore`. Avoid importing later-version copper-tool progression into 1.21.1.

One normal furnace operation takes approximately **200 ticks / 10 seconds** at 20 TPS. Coal/charcoal fuel supports eight operations; two planks support three. Code should know fuel arithmetic, remaining cook time and required inventory space, while Astra decides what to process first. Keep a reserve of sticks/planks before consuming everything as fuel. One early charcoal can supply torches or further smelting: prioritize based on available coal and the approaching night. [Smelting reference](https://minecraft.fandom.com/wiki/Smelting), [community smelting planning](https://minecraft.wiki/w/Tutorial%3ASmelting), [Mojang charcoal guide](https://www.minecraft.net/de-de/article/charcoal).

**High-value heuristics and adaptation.** These are proposed policies grounded in the mechanics above. Numeric thresholds should be tuned on training seeds and frozen for held-out evaluation.

| Situation | Heuristic / contingency | Code | Astra prompts | Jev questions |
|---|---|---|---|---|
| Multiple resource candidates | Prefer verified short safe paths, appropriate tool, exposed face and reliable pickup. Euclidean-nearest can be 
…(truncated)

### adaptor/ADAPTOR.md
# Minecraft Java Edition adaptor

This is a **Tier 2, real vanilla Minecraft 1.21.1** adaptor. Python owns a private dedicated server and a persistent Node/Mineflayer player on `127.0.0.1`. There is no simulator substituted for Minecraft, no graphical client, and no Microsoft account or credentials. The bot uses the server's explicitly configured offline authentication. It is not intended for public servers.

## Install and run

```sh
cd /Users/hemlix/Documents/GamingMaster/runs/r7-minecraft/adaptor
./setup.sh
/Users/hemlix/Documents/GamingMaster/.venv/bin/python -m pip install -r requirements.txt
/Users/hemlix/Documents/GamingMaster/.venv/bin/python -c "from metaharness.games.adaptor import validate_adaptor; print(validate_adaptor('/Users/hemlix/Documents/GamingMaster/runs/r7-minecraft/adaptor').to_markdown())"
```

The runtime has already been installed and tested with the specified Python **3.14.5**, Node **26.0.0**, and local Temurin **21.0.12.1+1**. Python uses only its standard library plus the evaluator's `metaharness`. Node dependencies are `mineflayer@4.39.0`, `mineflayer-pathfinder@2.4.5`, and `vec3@0.1.10`; `bridge/package-lock.json` fixes their complete dependency tree. Setup uses `npm ci` with a local cache, not a global installation.

`setup.py` downloads Mojang's version manifest, resolves **1.21.1**, and checks the server JAR against its official SHA-1. The tested JAR has SHA-1 `59353fb40c36d304f2035d51e7d6e6baa98dc05c` and SHA-256 `e3bc55693e93cda0188f2e60aea28113fc647c5e85a15fa3d1b347349231b4bb`. Java is downloaded from Adoptium for the current architecture and checked against its published SHA-256. Effective paths and versions are recorded in `runtime/artifacts.json`; download metadata is retained. Runtime files stay beneath this adaptor by default. The server setup writes `eula=true` as part of the requested installation; Mojang's [EULA](https://www.minecraft.net/en-us/eula) applies. Downloaded server/JDK binaries and generated worlds are excluded from source control.

No environment variables are required. `env.example` lists every supported override. `MC_HOST` must be `127.0.0.1`, and `MC_VERSION` must be `1.21.1`. Duration is fixed at **300 seconds**; `MC_EPISODE_SECONDS` is intentionally not supported. `make_game()` also accepts `run_dir`, `port`, and `username` keyword arguments. Unknown options raise an error. The returned object has `close()` and context-manager support; callers should always close it.

```python
from game import make_game
import time
with make_game() as game:
    observation = game.reset(0)
    while not game.done:
        game.tick('noop')
        observation = game.observe()
        time.sleep(game.tick_s)  # A production runner schedules its own clock.
    result = game.info()
```

`tick_s=0.5`; `ticks` counts calls that applied a current action, including rejected policy commands and noops. `tick()` does not sleep for a game turn. The Java server and Mineflayer physics run continuously at their ordinary target 20 Hz, including during model inference. A separate Python monotonic-clock monitor ends the episode even if the runner stops calling `tick()`. Setup and shutdown are excluded from scored time. Typical measured warm reset is about 5 seconds; initial generation/admission is longer and can take minutes for rejected seeds.

## Scenarios and pristine resets

Benchmark ID: `vanilla-1.21.1-admission-v3-score-v1`. `DETERMINISTIC=False`.

For integer harness seed `s`, the world seed is the first eight bytes of SHA-256 of UTF-8 `minecraft:1.21.1:world:{s}`, interpreted as a signed big-endian 64-bit integer. It is stored/transmitted as a **decimal string**, never a JavaScript Number. It does not depend on policy outcomes or Python's randomized hash.

Time stratum uses the low bit of byte zero of SHA-256 of `minecraft:1.21.1:time:{s//2}`, XOR `s % 2`: zero selects **1000**, one selects **9000**. Every complete even/odd pair has one of each stratum. Prequalify complete pairs and retain only pairs with both starts admitted to keep evaluation strata exactly balanced. Do not balance results after seeing policy scores.

Admission is preparation-only, before any scored play. Its fixed predicate is implemented in `bridge/terrain.js`:

* Begin at the actual vanilla world spawn with `spawnRadius=0`, avoiding randomized player spawn offsets. Examine 13 candidate centers in fixed order: origin; four cardinal offsets of 48; four diagonal offsets of 48; four cardinal offsets of 96. For each center choose the nearest dry surface within eight blocks, breaking ties by x then z.
* Use full natural ground blocks with clear feet/head space, excluding liquids, fire, cactus, powder snow, magma, campfires, and berry bushes and adjacent hazards. Non-cubic natural ground such as snow layers is conservatively excluded.
* Flood reachable surface cells using cardinal steps, at most one block up and two down, with step head clearance and a **48-block Euclidean radius** around the proposed start. Lava/water crossings and larger drops are excluded.
* Require at least **12 unstripped Overworld log blocks** reachable from those surface cells within ordinary interaction reach, and **24 ordinary stone blocks** within that radius beneath at most two cover blocks. Shallow stone must have solid support below, excluding cave drops and adjacent liquid hazards. Admission may inspect privileged terrain, but those scans and witness locations are never given to the player.

Each attempt, witness set, selected spawn, world seed and rejection reason is saved in a baseline manifest. A rejected harness seed raises `InadmissibleSeed`; it is **never silently replaced with a different world**. Infrastructure/readiness failures likewise do not become low-score trials. Prepare a declared evaluation set ahead of time:

```sh
/Users/hemlix/Documents/GamingMaster/.venv/bin/python prepare_seeds.py 100000 100001 100002 100003
```

This command only prepares worlds and prints admission results; the above held-out example has not been run during development. Development exercised seeds **0, 1, 2, 3**; exclude those entire world seeds from held-out evaluation. Repeat matched trials because mob AI, item trajectories/drops, entity scheduling and network timing are not bitwise deterministic. Both policies must use the same admitted seed list, configuration, dependency lock and 300-second limit.

Preparation uses a temporary spectator and setup-only server commands. Before caching, the server is stopped cleanly, the temporary player's `playerdata`, statistics and advancements are removed, Normal difficulty and ordinary regeneration/spawning/daylight rules are restored, and the chosen world spawn is recorded. A baseline contains an unplayed natural world and manifest, **not a played player**. Preparation data is discarded by launching a new bridge for actual play.

Every `reset` stops the previous owned processes, copies that stopped baseline into a new UUID episode directory (APFS copy-on-write when available), and creates a fresh unprivileged Survival player. It verifies full health/food, empty server inventory, recorded position, loaded chunks and absence of a nearby observed hostile. Initial clear weather and the selected time are set at the final setup barrier. No administrative gameplay mutation is exposed to the policy or issued after that barrier. Read-only inventory audits continue during play.

Baseline creation has a per-seed filesystem lock. Failed/incomplete cache creation is not reused. Stopped played worlds are retained for the last three episodes; older world and extracted server-cache directories are pruned, while scenario manifests, score results, inventory journals and logs remain. Baselines are retained separately. `server_owner.py` stops and reaps Java on controller EOF or termination; bridge EOF disconnects the player. Cleanup affects only owned process handles. `close()` is idempotent. Closing an unfinished episode freezes its score and marks it i


## Jev model context and usage for Meta-Harness

Snapshot: 2026-09-19. This is design context for the Astra proposer and for
runtime Astra when it designs Jev questions. The Python interface below is
GamingMaster's wrapper, which differs from the provider SDK. The model profile
and episode budgets in the experiment manifest are authoritative.

### What Jev supplies

Jev is TypeSafe's System One model for fast, focused judgments. It produces
typed answers and probability distributions, not prose, code, plans or reasoning
traces. It accepts text and structured JSON, not images, audio or video. A visual
game therefore needs a separate perception step before Jev can judge its state.
Calibration describes aggregate predictions, not correctness of each answer.
See [System One](https://docs.typesafe.ai/concepts/system-one).

In this architecture, Astra can interpret unfamiliar rules, choose subgoals and
design the decision interface. Jev can recognize situations, assess whether a
subgoal's preconditions still hold, or choose among well-described alternatives.
Code should handle exact arithmetic, legal-action checks, coordinate transforms,
bookkeeping and execution. Do not ask Jev to invent an unknown numerical action
mapping or perform an entire planning algorithm in one classification.

### Context that each request must carry

Jev does not inherit the proposer's prompt, Astra's conversation, `ctx.memory`,
previous requests, game description or observations automatically. This wrapper
sends only the supplied `state` and `questions` with the configured model.
Construct the necessary context explicitly on every call.

Prefer a named JSON object containing the current observation summary, the
relevant rule/field meanings, Astra's current subgoal and constraints, and any
short recent evidence needed for this decision. Include candidate descriptions
when asking Jev to compare candidates. Supply actual values, not variable names
or a reference to an inaccessible file. Keep irrelevant history out; preserve
evidence that distinguishes the options. English instructions are preferred for
this model. See [State](https://docs.typesafe.ai/concepts/state).

Batch independent judgments about that state into one call. They share the
input but do not see each other's answers. If a question really requires an
earlier answer, put that answer in the next call's state. Otherwise ask both
speculatively and combine them in code. Put the complete judgment in
`instructions`: question IDs are invisible to Jev. Refer to input fields
explicitly, e.g. `plan.preconditions`. See
[Questions](https://docs.typesafe.ai/primitives).

### Exact local call and answer contract

Import `Choice`, `Score`, `Noul` from `metaharness.models.jev`. Call
`ctx.decide(state, questions, timeout_s=5.0)`; it returns
`dict[str, System1Answer]` directly, keyed by your question IDs.

| Question constructor | Meaning of `answer.value` | Other local fields |
|---|---|---|
| `Choice(instructions, criteria={option: description, ...})` | Selected option key, a string | `.probabilities`, `.confidence` |
| `Score(instructions, criteria=[low_description, ..., high_description])` | Fractional level index, from 0 to number of levels minus 1 | `.probabilities` with string index keys, `.confidence` |
| `Noul(instructions, criteria={"true": description, "false": description})` | Probability of yes, between 0 and 1 | `.probabilities` contains `"true"`/`"false"`; `.confidence` is normally `None` |

All answers also have `.type` and `.top(k)`. There is no `.answers` container,
`.choice`, `.score`, `.noul`, generated explanation, or Score `.legend` on the
local return objects. Keep the Score level descriptions in your own code.
`Noul` criteria are optional. JSON-serializable question dictionaries with
`type`, `instructions` and appropriate `criteria` also work. Credentials and
the HTTP client are managed outside the harness; do not instantiate `JevClient`
or bypass `ctx.decide`. See the [HTTP API](https://docs.typesafe.ai/api) for the
provider's different wire format.

For Choice, define distinguishable options and include an explicit unknown or
no-match option when needed; there can be at most 255 options. Structured
descriptions can spell out inclusion, exclusion and examples when labels are
ambiguous. See [Choice](https://docs.typesafe.ai/primitives/choice).
For Score, use 2–10 concrete ordered levels for one dimension; the returned
number is an expected level, not necessarily an integer or a probability.
See [Score](https://docs.typesafe.ai/primitives/score).

Confidence summarizes the distribution and is not interchangeable with the
winning option's probability. Inspect both when useful. A Noul near 0.5 is
uncertainty about a yes/no proposition, not medium severity. Handle absent
confidence explicitly; never turn `None` into certainty. Choose thresholds
using training evidence and action consequences, with a fallback or Astra
escalation for ambiguity. See [Confidence](https://docs.typesafe.ai/confidence).

### Small usage example, not a game adapter

This helper illustrates one batched request about an Astra plan and an
observation summary. The proposer still designs the summaries, questions,
thresholds, action mapping and scheduling for its candidate.

```python
from metaharness.models.jev import Choice, Noul, Score


def assess_plan(ctx, plan, observation_summary):
    state = {"plan": plan, "observation": observation_summary}
    answers = ctx.decide(state, {
        "condition": Choice(
            "Do the facts in `observation` support `plan.preconditions`?",
            {
                "supported": "All required preconditions are supported by the observations.",
                "contradicted": "An observed fact contradicts a required precondition.",
                "unknown": "The observations cannot establish all required preconditions.",
            },
        ),
        "progress": Score(
            "How much of `plan.goal` is visibly achieved in `observation`?",
            ["No visible progress.", "Some progress, but the goal is incomplete.",
             "The stated goal is visibly complete."],
        ),
        "finished": Noul(
            "Does `observation` explicitly establish completion of `plan.goal`?"
        ),
    }, timeout_s=5.0)
    condition = answers["condition"]
    return {
        "condition": condition.value,
        "condition_probability": condition.probabilities.get(condition.value, 0.0),
        "condition_confidence": condition.confidence,
        "progress_level": answers["progress"].value,
        "completion_probability": answers["finished"].value,
    }
```

### Timing and the generated bridge

One admitted `ctx.decide` request consumes one System-1 call, even with several
questions. Requests are synchronous unless submitted via `ctx.spawn`. Batching
reduces round trips but does not promise constant latency. Measure actual
latency from traces: Jev can take longer than a game tick. Keep an executable
fallback while waiting, limit outstanding requests, and associate each response
with the observation and plan version that produced it. Recheck relevant
preconditions and legality on the latest observation before acting. A cached
semantic interpretation may outlive an individual action decision; define its
invalidation rule rather than treating every answer as permanently reusable.

The context caps each request's timeout by the remaining wall budget. Handle
transient failures within that budget, propagate `ModelUnavailable`, and join
all spawned futures. The standard inference calls already log request,
response, latency and token usage. Add harness events explaining when a Jev
answer changes a chosen action, invalidates a plan, or triggers Astra; call
counts alone do not establish useful cooperation.

In `proposal.md`, specify the Jev state schema, each question's role, how its
answer affects control, freshness/confidence handling, and the hypothesis about
when Astra should revise the interface. These choices belong to the generated
harness; this guide supplies no environment-specific adapter.

When runtime Astra is asked to create or revise Jev questions, include this
guide (or its relevant interface sections) in that `ctx.think` prompt too.
The proposer context is not automatically forwarded to runtime Astra. A harness
can obtain the exact same bundled text with
`from metaharness.models.jev import usage_guide` and append `usage_guide()` to
the interface-design prompt, together with public observations and action
specifications. Do not send the full guide to Jev on every tick; send the
decision-specific state and questions it describes.


## Harness contract (must be satisfied exactly)
```
The harness contract — the *only* thing a proposed harness must satisfy.

A harness is a Python module that defines::

    def build(ctx: HarnessContext) -> HarnessProgram

and ``HarnessProgram`` has one method::

    def run_episode(self, game) -> None

The harness owns the whole control loop for one episode. Everything it may
use comes through ``ctx``:

* ``ctx.system2``   — Astra (slow, reasoning). ``ctx.think(prompt, session_id=...)``
                      wraps it with tracing + budget accounting.
* ``ctx.system1``   — Jev (fast, typed decisions). ``ctx.decide(state, questions)``
                      wraps it with tracing + latency accounting. Tier 1 tasks
                      pass ``system1=None``.
* ``ctx.spawn(fn)`` — run ``fn`` on a background thread (e.g. an Astra call
                      while Jev keeps the game alive). Returns a Future.
* ``ctx.clock``     — the game clock. In *virtual* mode, model latency is
                      charged to the clock deterministically (tests); in
                      *wall* mode it is real time.
* ``ctx.memory``    — a dict reset per episode by default. Explicit shared
                      memory evaluations allow learning between seeds; the
                      evaluation manifest records which protocol was used.
* ``ctx.log(...)``  — free-form events into the trace.
* ``ctx.budget``    — remaining System-2 calls / tokens; exceeding it ends the
                      episode (score kept), so harnesses must be thrifty.

Real-time games (tier 2) expose ``game.tick_s``. The runner keeps ticking the
game on its own clock: in wall mode a thread ticks every ``tick_s``; in
virtual mode each model call's latency advances the game by
``latency / tick_s`` ticks (using the *last action the harness posted*).
The harness posts actions with ``ctx.act(action)``; posting is instant
(virtual mode charges a 1 ms overhead so a loop that decides without any
model call still lets time pass). ``ctx.wait_for_tick()`` blocks until the
next tick boundary — use it instead of spinning.
Turn games (tier 1) are stepped with ``ctx.act(action)`` synchronously and
return the next observation.

Exact signatures (use these; anything else is a contract error caught in pre-flight)::

    ctx.think(prompt: str, *, session_id: str | None = None, system: str | None = None,
              images: list[str] | None = None, timeout_s: float = 300.0) -> System2Result   # .text .session_id
    ctx.decide(state: Any, questions: dict[str, Choice | Score | Noul], *, timeout_s: float = 5.0)
              -> dict[str, System1Answer]                                                   # .value .probabilities .confidence
    ctx.act(action) -> observation | None      ctx.observe() -> observation      ctx.done -> bool
    ctx.wait_for_tick() -> None                ctx.spawn(fn) -> concurrent.futures.Future
    ctx.log(**fields) -> None                  ctx.memory: dict     ctx.clock.now() -> float (s)
    ctx.budget.remaining() -> {"system2_calls": int, "system1_calls": int, "wall_s": float}
    game.describe() -> str    game.action_spec.names / .descriptions / .noop / .notes    game.observe()
    game.tick_s (realtime only)    game.ticks    game.score    game.done
```


## Archive
# Leaderboard

| # | candidate | mean score | episodes | stale ticks | S1 calls | S2 calls | Jev→act | Astra→act | status |
|---|---|---|---|---|---|---|---|---|---|
| 0000 | 0000-tier2_jev_only | 15.266 | 3 | 38% | 967 | 0 | 100% | 0% | 1 episode errors |
| 0001 | 0001-tier2_astra_only | 6.348 | 3 | 93% | 0 | 111 | 0% | 100% | ok |
| 0002 | 0002 | 20.000 | 3 | 0% | 0 | 24 | 0% | 0% | ok |
| 0003 | 0003 | – | – | – | – | – | – | – | not evaluated |

`Jev→act` / `Astra→act`: fraction of posted actions preceded by a fresh System-1 / System-2 answer since the previous action (reported, not used for selection).

Best so far: `0002` (mean score 20.000)

Prior candidates (read their code, proposals, eval.json and the JSONL traces — traces contain every
model call with request/response/latency and every game tick/step):
- candidates/0000-tier2_jev_only/ (harness.py, proposal.md, eval.json, trace_seed*.jsonl)
- candidates/0001-tier2_astra_only/ (harness.py, proposal.md, eval.json, trace_seed*.jsonl)
- candidates/0002/ (harness.py, proposal.md, eval.json, trace_seed*.jsonl)

## What to do
1. Read the best candidate(s) and the traces of their worst episodes. Diagnose the failure mode
   concretely (quote evidence from traces).
2. Write `candidates/0003/harness.py` — a complete module with `build(ctx)`.
   It must import only the standard library and `metaharness.*`.
3. Write `candidates/0003/proposal.md`: hypothesis, what changed vs. the parent
   candidate, what evidence would confirm/refute it.
4. Append 3-6 bullet lessons to `notes.md` at the archive root (a running lab notebook: what
   generalised, what was a dead end, which contract details bit you). Read it first.
5. If you cannot use tools, reply with the full harness in ONE ```python fenced block and the
   proposal text after it.
Before the real evaluation the candidate is *pre-flighted*: one short episode with stand-in models
that return arbitrary text / first options on a virtual clock. Any exception there (a contract
misuse, an unhandled parse failure of model output) is sent back to you once for repair; a harness
must survive garbage model output without crashing.
Do not modify any other candidate. Do not run long experiments yourself; the evaluator will.
Read only this archive and the interface documentation above. Do not inspect game implementation
source, other experiment archives, credentials, or held-out games/results. Infer semantics from the
public game description and observations supplied through ctx. Do not call model backends directly:
all inference must go through ctx.think/ctx.decide so the evaluator can account for it.
ModelUnavailable must propagate. Join all ctx.spawn futures before returning; no background work
may escape the episode. The default evaluation resets both memory and module globals per episode;
only an explicit shared-memory protocol allows cross-episode caching. The manifest is authoritative.