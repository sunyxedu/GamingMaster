Design a harness for a **real-time** game (tier 2).

Game: `Minecraft Java 1.21.1 — Survival to Free the End` — Play Minecraft Java 1.21.1 Survival, Normal difficulty, in a fresh naturally generated world. Defeat the Ender Dragon: success is the server advancement minecraft:end/kill_dragon (Free the End).
Time limit 3600 wall-clock seconds, excluding setup. The actual server and bot physics run continuously at target 20Hz even when you do not call tick or observe. Input cadence is 0.5 seconds. Death drops inventory (keepInventory=false); use respawn to continue normally. No cheats, bonus resources, teleport, modified drops, peaceful mode or privileged map information. Initial day 1000 and clear weather are set before joining; ordinary time, weather, mobs, hunger and damage then run freely. Spawn is an ordinary vanilla world spawn with spawnRadius=0. Natural seeds are not filtered for easy wood/food; seed is evaluator-only. Coordinates: +x east, +y up, +z south.
Dragon score [0,100]: opening 10 (having held both logs and crafting table 3, food 3, any bed or sleep advancement 4); iron 5 and shield 5; Nether entry 8, fortress advancement 6, blaze rod 6; pearl 4, eye 6, stronghold advancement 5, seen reachable active End portal 5; End entry 10; dragon advancement 30. Flags are permanent, no points from idling or duplicate transfers. Successful dragon completion is always score 100; any unearned preparation weights become a separately reported completion bonus. Primary comparison is success, then milestone progress; raw deaths/damage/resources remain reported. Specialized resource scenarios score proportional current possession; advancement scenarios score 100 on completion only. Non-infrastructure death is not an invalid trial.
Observations: episode/seq/age; self position/velocity/yaw/pitch/ground, survival mode/dimension, health/food/saturation/oxygen/effects, water/lava/dead, equipment/XP/bed knowledge; authoritative audited personal inventory and labelled client inventory/slot previews; open container/cursor; world day/time/biome/weather and last server-time age; nearby entities (some occluded, visible flag); exposed block candidates and collision labels; limited block changes/events; server-completed advancements; feasible recipe hints; current bounded task, action results, score and raw progress. Candidates b:x,y,z and s:x,y,z come from loaded nearby terrain. Exposure is adjacent free space, NOT pixel-equivalent vision; visible is a raycast from your eyes. Unknown chunks, seed, underground solid interiors, structure coordinates and console are not exposed. Inventory score excludes cursor, recipe output preview and containers. Snapshot data can lag; inspect observation_age_s, task and feedback.
Actions are whitespace-separated strings, never code or commands. One bounded asynchronous job owns hands/navigation. noop continues a job, but when idle releases controls/item use; cancel interrupts. Busy commands are rejected: cancel then wait for task=null. Identical one-shot actions are latched, so send noop between deliberate repeats; controls renew each tick, attacks repeat with cooldown. controls KEYS is a comma-list of forward,back,left,right,jump,sprint,sneak or none, relative to current yaw; its 0.8s lease clears if ticker stalls. use_item holds 0–5s then releases (bow); equip first, look_at first. shield on stays active until another action/noop/cancel. Gathering chooses tools/path then mines 1–4 blocks, tries real pickup, and times out after25s. Navigate/goto have15s deadline; others10s. Navigation never mines or bridges, avoids known hazards and drops >2, and cancels on damage; manual controls can deliberately move into water/portals and are not a survival autopilot. dig is direct in-reach mining with held tool, no auto-pickup; no mining your support. place takes ITEM REFERENCE_ID FACE with up/down/east/west/north/south, crafting wants NEW_COUNT outputs (rounded to whole recipe executions). Furnaces consume real input/fuel and take real time. Container windows remain open for transfer/move_slot; cancel/close_window closes them. Trade and enchanting consume ordinary payment/XP. Rejected or malformed input reports feedback, not a Python exception. Optional {episode,seq,action} envelope rejects stale (>10 snapshots) observations.
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
- wait: wait [SECONDS=0.5]: wait 0–5 seconds without pausing the world.
- controls: controls forward,jump|none: hold listed movement keys; renew each tick; expires after 0.8s.
- goto: goto X Y Z: path to loaded safe feet position within 48 blocks; no automatic mining/bridging.
- dig: dig X Y Z: mine one visible in-reach block with the held tool, without pickup automation.
- activate_block: activate_block BLOCK_ID [FACE=up]: right-click block with held item (doors, bucket, flint, eyes).
- activate_entity: activate_entity ENTITY_ID: ordinary right-click a visible entity within 4 blocks.
- use_item: use_item [SECONDS=0.2] [hand|off-hand]: hold then release use (food, bow, eye, bucket).
- hotbar: hotbar SLOT: select a hotbar slot 0–8.
- open_container: open_container BLOCK_ID: open a nearby chest/barrel/container and retain its window.
- close_window: Close the current container window.
- transfer: transfer take|put ITEM COUNT: move 1–64 real items between open container and inventory.
- move_slot: move_slot SOURCE DEST: move a real stack between window slot numbers.
- toss: toss ITEM COUNT: drop 1–64 carried items (including gold for piglin bartering).
- sleep: sleep BED_ID: use an Overworld bed when vanilla permits; set ordinary respawn point.
- wake: Leave the bed.
- respawn: Respawn normally after death; inventory remains dropped, no restoration.
- inspect_trades: inspect_trades VILLAGER_ID: open nearby villager; report its offered trades.
- trade: trade VILLAGER_ID INDEX [TIMES=1]: perform a real offered trade with carried payment.
- enchant: enchant TABLE_ID ITEM CHOICE: enchant carried item with lapis/XP, choice 0–2.
- chat: chat TEXT: normal player chat only; slash commands/control characters are forbidden.
(real-time games: ticks apply 'noop' until the first action is posted; afterwards a tick without a fresh action repeats the last posted action)
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
- wait: wait [SECONDS=0.5]: wait 0–5 seconds without pausing the world.
- controls: controls forward,jump|none: hold listed movement keys; renew each tick; expires after 0.8s.
- goto: goto X Y Z: path to loaded safe feet position within 48 blocks; no automatic mining/bridging.
- dig: dig X Y Z: mine one visible in-reach block with the held tool, without pickup automation.
- activate_block: activate_block BLOCK_ID [FACE=up]: right-click block with held item (doors, bucket, flint, eyes).
- activate_entity: activate_entity ENTITY_ID: ordinary right-click a visible entity within 4 blocks.
- use_item: use_item [SECONDS=0.2] [hand|off-hand]: hold then release use (food, bow, eye, bucket).
- hotbar: hotbar SLOT: select a hotbar slot 0–8.
- open_container: open_container BLOCK_ID: open a nearby chest/barrel/container and retain its window.
- close_window: Close the current container window.
- transfer: transfer take|put ITEM COUNT: move 1–64 real items between open container and inventory.
- move_slot: move_slot SOURCE DEST: move a real stack between window slot numbers.
- toss: toss ITEM COUNT: drop 1–64 carried items (including gold for piglin bartering).
- sleep: sleep BED_ID: use an Overworld bed when vanilla permits; set ordinary respawn point.
- wake: Leave the bed.
- respawn: Respawn normally after death; inventory remains dropped, no restoration.
- inspect_trades: inspect_trades VILLAGER_ID: open nearby villager; report its offered trades.
- trade: trade VILLAGER_ID INDEX [TIMES=1]: perform a real offered trade with carried payment.
- enchant: enchant TABLE_ID ITEM CHOICE: enchant carried item with lapis/XP, choice 0–2.
- chat: chat TEXT: normal player chat only; slash commands/control characters are forbidden.
(real-time games: ticks apply 'noop' until the first action is posted; afterwards a tick without a fresh action repeats the last posted action)
Actions are whitespace-separated strings. Bare parameterized names are rejected as feedback, not Python exceptions. Only one bounded job owns movement/hands. Repeated identical current actions do not restart; send noop between deliberate repeats. noop preserves an existing job. cancel before replacing a busy job.

Models available through `ctx`:
* `ctx.think(prompt, session_id=...)` — Astra (System-2, slow). Deep reasoning, can plan, can explain
  rules. Returns `.text` and `.session_id` (pass it back to keep conversation context).
  Budget: 40 calls per episode. Astra never sees the game unless you show it.
* `ctx.decide(state, questions)` — Jev (System-1, fast typed judgments). Budget: 4000 calls per
  episode. Read the complete Jev model context and local usage contract below before designing
  its state, questions and answer-to-action bridge. Actual latency must be measured.
* `ctx.spawn(fn)` — background thread, so Astra can think while Jev keeps acting.

Model economics (facts, not instructions): one Jev request answers *many independent typed questions
about one state in parallel* for the price and latency of one, and the Jev budget is two orders of
magnitude larger than Astra's; an Astra call takes far longer and its latency grows with the length
of its answer. Design timeouts and pipelining from the measured latencies (below, if calibrated, and on
the leaderboard). Whatever the design, the evaluator reports how many posted actions were preceded by a
fresh Jev / Astra answer, so the division of labour is visible.

The game ticks every 0.5s **on its own clock**; `ctx.act(a)` posts the current action (instant).
A tick with no fresh action re-applies the last one. Each episode has a wall-clock deadline of
900 s (`ctx.budget.remaining()["wall_s"]`); the game's own end, the deadline or a used-up call
budget end the episode with the score reached. Each episode starts cold: fresh `ctx.memory` and fresh module globals. Clock mode: wall.

Evaluation: mean score over the training seeds (primary), then fraction of stale ticks (lower is
better), then System-2 calls (fewer is better); real-time ties favour longer survival. A candidate can
only be selected if none of its training episodes ends with an error (see the contract). The idle
baseline (never acts) shows what doing nothing scores.

What the harness must design (this is the whole point): how Astra's slow output (a plan, a rule
summary, a policy sketch, a subgoal) is turned into *state + questions* for Jev when needed, how
Jev's typed answer is turned into a game action, when to go back to Astra, and what to do when a
decision is late.

**Evaluation scope.** The harness is selected on the training seeds of this game and then reported on
unseen seeds of the same game; there is no held-out game. Rules, strategies and search that the game
description or the study establish may be implemented as code once they are verified against
observations — but nothing specific to the training seeds.

## Prior knowledge from the study phase
The study phase researched this game (rules, interface, community strategies). Use it: implement exact
rules, bookkeeping and search as code, put planning and adaptation into Astra prompts, and fast
recognition into Jev questions — but verify every claim against observations, because sources can be
wrong or outdated for this build of the game.

The full documents are readable in this archive under `prior/` (`prior/strategies.md`, `prior/brief.md`, `prior/interfaces.md`, `prior/study.json`, `prior/ADAPTOR.md`). Below: excerpts within a size budget; every cut is marked `…(truncated; full text in prior/<file>)`.

### study/study.json (summary)
- strategy_summary: Strong play is milestone-driven and risk-adjusted: secure logs, food, a bed/shelter, shield, iron, blocks, and a return route before taking the Nether. Use a fortress-first route by default, with a bastion/barter speedrun branch only when the state supports its risk; track portal and dimension coordinates and compute Eye of Ender triangulation in code. Prepare an explicit inventory and escape checklist before the End, destroy healing crystals, avoid the void/endermen, and attack the dragon only during safe windows. Code owns recipes, counts, geometry, path safety, cooldowns, inventory, and scoring; Astra chooses macro routes and adapts; Jev recognizes immediate threats and selects a typed tactical response. The prior and detailed placement rules are in strategies.md, with factual references to https://minecraft.wiki/w/Survival, https://minecraft.wiki/w/Tutorial:Speedrun, https://minecraft.wiki/w/Piglin, https://minecraft.wiki/w/Eye_of_Ender, and https://minecraft.wiki/w/Tutorial:Defeating_the_ender_dragon.
- scoring: For the default dragon scenario, report raw terminal facts and use the Java Free the End advancement or verified dragon-death event as primary success. Rank unsuccessful episodes by deterministic milestone progress: safe opening/bed, iron and shield, Nether portal, fortress/blaze rods, pearls/eyes, stronghold, active End portal, End entry, crystals/dragon damage, and dragon kill; also report elapsed ticks, deaths, inventory loss, damage, and resource state. A transparent optional weighted score is 10% opening, 10% iron/shield, 20% Nether/fortress, 20% eyes/stronghold/portal, 10% End entry, and 30% dragon kill, with raw flags and deaths retained. Do not award commands, creative mode, teleport, item generation, keepInventory assistance, or operator intervention. Sources: https://minecraft.wiki/w/Survival, https://minecraft.wiki/w/Advancement, https://minecraft.wiki/w/Tutorial:Advancement_guide/The_End_tab
- interface.observation: The adaptor should emit JSON containing episode and snapshot IDs, server tick estimate, dimension, position/velocity/yaw/pitch/on-ground, health/max health, food/saturation/oxygen/effects/fire, respawn knowledge, selected slot/held item/armor/offhand, full inventory and open window, nearby entities with type/position/health/hostility/visibility, bounded nearby block and fluid/light observations plus block diffs, time/weather, chat, path/action status, damage/death/spawn events, and code-computed milestone/resource facts. Mineflayer provides bot.entity, health, food, foodSaturation, oxygenLevel, inventory, heldItem, quickBarSlot, entities, players, blockAt, findBlock(s), blockAtCursor, nearestEntity, time, and event streams; see https://github.com/PrismarineJS/mineflayer/blob/master/docs/api.md.
- interface.actions: Expose validated semantic JSON actions with IDs and deadlines: hold/release movement controls, look, safe navigate to a goal, dig, place, activate block/entity, use item, attack, equip, select hotbar slot, craft, smelt/trade/window operations, inventory moves, and ordinary chat. The arbiter must clear controls on timeout/death/kick, reject stale or lethal actions, verify reach/line-of-sight/held item/inventory, cancel paths on hazards or block/entity changes, and record action results. Do not expose RCON, operator commands, teleport, give, gamemode, or gamerule mutation to the game-playing model; see https://github.com/PrismarineJS/mineflayer/blob/master/docs/api.md and https://github.com/PrismarineJS/mineflayer-pathfinder.

### study/strategies.md
# Minecraft strategy prior

## What “strong play” means here

Minecraft has no single dominant strategy until the scenario names an objective. For the default study objective—finish a fresh Survival world by defeating the Ender Dragon—strong play is a **risk-adjusted progression**: secure food, a bed, tools, and a safe return path; acquire Nether resources; locate and activate the End portal; then win the End fight. Speedrunners deliberately accept much more variance to minimize time, while a robust autonomous player should prefer a safe route unless the seed gives an unusually favorable shortcut. The [Survival overview](https://minecraft.wiki/w/Survival) explicitly describes the open-ended resource/build/explore/survive loop and treats the dragon as a way to “beat the game” without ending play; the [speedrun tutorial](https://minecraft.wiki/w/Tutorial:Speedrun) is useful for the aggressive branch, not as a rule that every run should imitate.

The division below is deliberate:

* **Code** owns exact facts, bookkeeping, geometry, timers, action preconditions, invariant safety, and scoring. It must not ask a language model to count items, calculate a distance, or remember an attack cooldown.
* **Astra** owns slow macro planning, route/equipment tradeoffs, resource prioritization, interpretation of unusual structures, and replanning after a meaningful state change. Astra should produce a bounded plan with stop/abort conditions, not a stream of unvalidated keystrokes.
* **Jev** owns fast recognition and a typed immediate choice from a small legal action set. Jev can recognize a creeper, lava, a blocked route, an opening to attack, or a safe retreat. It should not do arithmetic or invent a long plan.

## Default dragon route: robust build order

The following is a seed-agnostic baseline. The exact minutes are deliberately omitted: biome, structure, terrain, and network/TPS make a fixed clock brittle.

| Phase | Strong-player behavior | Put it in |
|---|---|---|
| **Spawn scan and opening** | Read the spawn biome and nearby resources before committing to a direction. Collect enough logs for a crafting table, tools, sticks, and emergency blocks; make stone tools quickly; acquire a renewable/reliable food source. If a village, exposed iron, exposed lava, or ruined portal is visible, treat it as an option rather than abandoning all safety checks. | **Code:** bounded scan, item counts, and map markers. **Astra:** choose between ordinary opening, village opening, or speedrun opening. **Jev:** recognize immediate cliffs, lava, hostile mobs, and whether movement can safely continue. |
| **First shelter and respawn** | Make a bed if wool is available, use it successfully, and establish a lit, enclosed staging point with a chest, furnace, crafting table, and marked coordinates. A bed is not just a night skip: it is a recovery location. Keep a small emergency kit rather than taking every resource on the first expedition. See [Bed](https://minecraft.wiki/w/Bed) and [spawning and dying](https://www.minecraft.net/en-us/article/spawning-and-dying). | **Code:** verify the bed interaction and save its coordinates; inventory ledger and base waypoint. **Astra:** choose where the base/portal should go. **Jev:** answer whether a nearby mob/terrain state makes the shelter interaction unsafe. |
| **Iron and first defense** | Prioritize a shield, bucket, pickaxe, iron weapon/armor as available, furnace fuel, torches, and food. A shield plus cover is more reliable than trying to out-click skeletons. Do not wait for perfect diamond gear before progressing, but do not enter the Nether with no escape blocks, food, or return plan. | **Code:** recipes, required item counts, slot/equipment bookkeeping, and preflight checks. **Astra:** decide whether to mine, cave, trade, or exploit an exposed resource. **Jev:** choose “fight / block / retreat / eat / flee” when a threat is already present. |
| **Mining and resource safety** | Prefer inspected caves, a staircase, or a controlled branch/strip route. Light the route, mark junctions, avoid long blind drops, and never assume a solid-looking floor is safe. “Never dig straight down” is a community maxim because it can open into lava or a mob-filled cave; the [Minecraft tips](https://minecraft.wiki/w/Tutorial:Tips_and_tricks) and [things not to do](https://minecraft.wiki/w/Tutorial:Things_not_to_do) pages document safer alternatives. | **Code:** block/fluid/light sampling, fall/lava vetoes, route map and return waypoint. **Astra:** decide how much mining is justified by the current gear. **Jev:** recognize a visible lava flow, drop, hostile ambush, or blocked escape and select the emergency action. |
| **Food and time discipline** | Keep food in a known hotbar slot and eat before a long trip or dangerous fight, not after hunger has already prevented sprinting. Return or sleep before a risky night expedition if the route is not lit. Use a compact base rather than repeatedly losing time to unmarked travel. Hunger and health are continuous state, not background flavor. | **Code:** food/health thresholds, hotbar selection, day/night and elapsed-tick bookkeeping. **Astra:** schedule trips and decide when a detour is worth it. **Jev:** choose “eat now / continue / retreat” from a precomputed safe-state label. |
| **Portal preparation** | Build a Nether portal only after recording the Overworld location and carrying blocks, food, a pick, shield, and a way to relight/repair the portal if the scenario permits. The Nether coordinate scale makes it an efficient travel route, but portal exits can be dangerous or displaced. Horizontal coordinates scale approximately 1:8; Y does not. See [Nether portal](https://minecraft.wiki/w/Nether_portal). | **Code:** dimension-aware coordinates, portal waypoint pair, return-path validation, and no-water-in-Nether action veto. **Astra:** select a fortress/bastion search direction. **Jev:** recognize an unsafe portal spawn, lava, fire, ghast shot, or piglin cluster and select cover/retreat. |
| **Nether resources** | Wear at least one gold armor piece before ordinary piglin contact. Use walls/blocks and controlled sightlines. Locate a Nether fortress for blaze rods; optionally use a bastion and piglin bartering for pearls if the seed and skill justify it. Gold armor reduces ordinary piglin hostility but does not make it safe to attack piglins, mine gold in front of them, or open nearby containers. See [Piglin](https://minecraft.wiki/w/Piglin), [Nether fortress](https://minecraft.wiki/w/Nether_Fortress), and [Nether survival](https://minecraft.fandom.com/wiki/Tutorials/Nether_survival). | **Code:** dimension/coordinate ledger, gold-armor and escape-block preconditions, blaze/pearl/gold counts, and bounded search. **Astra:** choose safe fortress-first versus high-variance bastion/barter branch. **Jev:** identify piglin/blaze/ghast/lava danger and select block, shield, retreat, or attack. |

…(truncated; full text in prior/strategies.md)

Useful Astra questions include:

1. “Which of the safe standard, village, fortress-first, or bastion/barter routes best reaches the next milestone from this state, and what makes you abort it?”
2. “Is this expedition worth continuing given the computed food/blocks/health/escape budget, or should the bot return?”
3. “Choose a safe End preparation checklist and attack phase plan from the available inventory; do not assume missing gear.”
4. “A block/entity event invalidated the route. Replan from the last safe waypoint and state the next Jev tactical policy.”

### Jev typed questions

Jev should receive a small JSON state and answer one of finite enums or booleans. Batch independent questions in one call when possible:

* `threat_class`: `none | creeper | ranged | blaze | piglin | brute | enderman | dragon | lava | fire | fall | void | drowning | multiple`;
* `immediate_action`: `continue | stop | shield | retreat | place_cover | attack_target | switch_target | eat | equip | jump | look_away | wait_for_cooldown | abandon_path`;
* `safe_to_continue`: yes/no, based on code-labelled hazard and visible escape state;
* `target_priority`: one entity ID from a provided list or `none`;
* `route_response`: `follow | replan | return_to_waypoint | shelter`;
* `combat_phase`: `approach | defend | ranged | melee_window | disengage`;
* `interaction_safe`: yes/no for opening a container, mining a gold block, entering a portal room, or collecting a drop;
* `visible_structure`: enum such as `village | fortress | bastion | stronghold | portal_room | none | uncertain`;
* `which_known_hotbar_slot`: select from slots already supplied, never infer an item count.

The adaptor should precompute distances, attack readiness, health bands, item counts, line-of-sight, and legal action choices. Jev recognizes and selects; code executes and can override a lethal action. Do not ask Jev to find a stronghold, calculate an intersection, plan a Nether expedition, or count crafting materials.

## Sources

* [Survival](https://minecraft.wiki/w/Survival)
* [Tutorial: Speedrun](https://minecraft.wiki/w/Tutorial:Speedrun)
* [Tick](https://minecraft.wiki/w/Tick)
* [Bed](https://minecraft.wiki/w/Bed), [Death](https://minecraft.wiki/w/Death), and [Item entity](https://minecraft.wiki/w/Item_%28entity%29)
* [Tutorial: Tips and tricks](https://minecraft.wiki/w/Tutorial:Tips_and_tricks) and [Tutorial: Things not to do](https://minecraft.wiki/w/Tutorial:Things_not_to_do)
* [Nether portal](https://minecraft.wiki/w/Nether_portal), [Nether fortress](https://minecraft.wiki/w/Nether_Fortress), [Piglin](https://minecraft.wiki/w/Piglin), and [Bastion Remnant](https://minecraft.wiki/w/Bastion_Remnant)
* [Eye of Ender](https://minecraft.wiki/w/Eye_of_Ender), [Stronghold](https://minecraft.wiki/w/Stronghold), [The End](https://minecraft.wiki/w/The_End), and [Ender Dragon](https://minecraft.wiki/w/Ender_Dragon)
* [Combat](https://minecraft.wiki/w/Combat), [Melee attack](https://minecraft.wiki/w/Melee_attack), [Shield](https://minecraft.wiki/w/Shield), [Creeper](https://minecraft.wiki/w/Creeper), [Skeleton](https://minecraft.wiki/w/Skeleton), [Enderman](https://minecraft.wiki/w/Enderman), [Blaze](https://minecraft.wiki/w/Blaze), and [Ghast](https://minecraft.wiki/w/Ghast)
* [Defeating the Ender Dragon](https://minecraft.wiki/w/Tutorial:Defeating_the_ender_dragon) and [Advancement guide: The End](https://minecraft.wiki/w/Tutorial:Advancement_guide/The_End_tab)
* [Mineflayer API](https://github.com/PrismarineJS/mineflayer/blob/master/docs/api.md), [Mineflayer Pathfinder](https://github.com/PrismarineJS/mineflayer-pathfinder), and [Mineflayer PVP](https://github.com/PrismarineJS/mineflayer-pvp)

### study/brief.md
# Minecraft study brief

## Scope and assumptions

This study treats the game as **Minecraft: Java Edition, Survival mode**, on a version-pinned private vanilla server. The user has explicitly classified it as tier 2; that is correct. Minecraft is not a turn-based game: the server simulates the world continuously, hostile entities can move and attack, hunger and fire/lava damage continue, and the player can fall while the agent is thinking. The adaptor should pin one exact server version (for example, the version of the downloaded `server.jar`) rather than assuming that “Minecraft” is a stable protocol or ruleset.

Minecraft has no universal win screen or built-in numeric score in Survival. A useful default evaluation contract is a fresh random-seed world, normal difficulty, no commands/creative assistance, and **reach and defeat the Ender Dragon**. The dragon is the conventional “beat the game” endpoint but is optional: defeating it does not end the world. A scenario may instead request exploration, a build, a farm, a particular advancement, or a resource; those are different objectives and should be named in the reset configuration. The official Java server download is at [Minecraft Java server download](https://www.minecraft.net/en-us/download/server); the community-maintained [Survival overview](https://minecraft.wiki/w/Survival) documents the open-ended survival loop and the dragon endpoint.

## Rules and state that matter

### World and player

* The Overworld, Nether, and End are persistent, procedurally generated dimensions. The seed, version, difficulty, gamerules, structures, and spawn point materially change a run.
* Survival players gather and craft resources, place and break blocks, explore, fight or avoid mobs, eat, use an inventory/hotbar, and can die. A normal death returns the player to world spawn, a bed spawn, or a Nether respawn anchor where applicable; dropped inventory and experience are at the death location. See [Survival](https://minecraft.wiki/w/Survival) and Mojang's [spawning and dying article](https://www.minecraft.net/en-us/article/spawning-and-dying).
* A bed sets an Overworld respawn point when successfully used. A robust run makes a bed and safe shelter early, then records the location; it does not assume a respawn point remains valid if the bed is destroyed or obstructed. See [Bed](https://minecraft.wiki/w/Bed).
* The default `keepInventory=false` makes death costly: dropped item entities normally despawn after 6,000 game ticks (five minutes) while in an entity-ticking loaded chunk. A reset may deliberately use `keepInventory=true` for a training scenario, but the default dragon scenario should expose the normal risk. See [Item entity](https://minecraft.wiki/w/Item_%28entity%29) and [Death](https://minecraft.wiki/w/Death).
* Hunger is both a resource and a movement constraint. Sprinting/jumping consume it; low hunger prevents sprinting and reduces recovery. The adaptor must expose health, food, saturation, oxygen, effects, and held/armor items rather than letting a planner infer them from position alone.

### Time and hazards

* The Java game normally runs at **20 ticks per second**, one tick every 0.05 seconds. A full day/night cycle is 24,000 ticks, or 20 minutes. Redstone, crop growth, mob AI, fire, falling, projectiles, damage, and many interaction timers are tick-driven. If server MSPT rises above 50 ms, the effective rate falls. See [Tick](https://minecraft.wiki/w/Tick).
* Night, caves, the Nether, and the End introduce hostile entities. In current Java versions, Overworld hostile spawning requires block light level 0; lighting and spawn-proofing are therefore useful base and route protections. See [Light](https://minecraft.wiki/w/Light) and [Spawn-proofing](https://minecraft.wiki/w/Tutorial:Spawn-proofing).
* The most dangerous continuous hazards for an automated client are lava/fire, void or long falls, drowning, explosions, projectile damage, being surrounded, and a stale path after a block/entity update. A path that is geometrically short is not necessarily survivable.
* The Nether is not merely a travel shortcut: it is needed for blaze rods and normally for the resources used to reach the End. Horizontal travel in the Nether corresponds approximately to 8x the Overworld distance; the Y coordinate does not scale. Record both dimension and coordinates for every waypoint. See [Nether portal](https://minecraft.wiki/w/Nether_portal) and the [speedrun tutorial](https://minecraft.wiki/w/Tutorial:Speedrun).

### Progression to the default objective

1. **Opening:** obtain logs, crafting table, wooden/stone tools, reliable food, shelter/bed, torches, and a stone or iron weapon. Do not spend the first night wandering without food or a respawn plan.
2. **Iron and safety:** make a shield, bucket, iron tools/armor as available, furnace, and a chest; mark the base and portal coordinates. Use a staircase or other inspected mining route rather than blindly mining into a fall or lava.

…(truncated; full text in prior/brief.md)

The server tick facts come from [Tick](https://minecraft.wiki/w/Tick); the chosen 0.5 s is an engineering budget, not a game rule. A slow Astra call may safely last tens of seconds while the bot is in a lit base with movement stopped, but it is not safe to hand Astra direct control during a fight or near a ledge.

## Reset and scenario semantics

`reset(seed)` should create or load an isolated world with the pinned version and seed, stop/restart the server cleanly, remove old player data, and return the bot to a known spawn. Prefer a separate world directory per seed rather than trying to undo a persistent world. Set difficulty, gamerules, structures, and `keepInventory` explicitly and report them in the initial JSON state. Keep commands/creative/operator powers in the server-side reset harness only; the gameplay client must join as an ordinary survival player.

Held-out seeds should vary spawn biome, nearby food/wood, village/portal availability, terrain, ocean/mountain/cave access, structure distances, Nether layout, and stronghold location. Include at least one easy calibration seed, then evaluate on unseen random seeds; do not train only on a seed with a village or exposed lava. For specialized scenarios, vary the requested build site, target coordinates, difficulty, weather/day, and nearby opponents while holding the objective constant.

## Sources

* [Minecraft Java server download](https://www.minecraft.net/en-us/download/server)
* [Minecraft Usage Guidelines](https://www.minecraft.net/en-us/usage-guidelines) and [Minecraft EULA](https://www.minecraft.net/en-us/eula)
* [Minecraft Survival](https://minecraft.wiki/w/Survival), [Tick](https://minecraft.wiki/w/Tick), [Death](https://minecraft.wiki/w/Death), and [Bed](https://minecraft.wiki/w/Bed)
* [Eye of Ender](https://minecraft.wiki/w/Eye_of_Ender), [Stronghold](https://minecraft.wiki/w/Stronghold), [Nether portal](https://minecraft.wiki/w/Nether_portal), and [Tutorial: Speedrun](https://minecraft.wiki/w/Tutorial:Speedrun)
* [Mineflayer README](https://github.com/PrismarineJS/mineflayer), [Mineflayer API](https://github.com/PrismarineJS/mineflayer/blob/master/docs/api.md), and [Mineflayer Pathfinder](https://github.com/PrismarineJS/mineflayer-pathfinder)

### adaptor/ADAPTOR.md
# Minecraft Java Survival adaptor

## What it runs

This adaptor is a **Tier 2 real-time** interface to a real, local vanilla Minecraft Java Edition **1.21.1** server. Python owns the Java server process and a Node.js Mineflayer client. The client connects over loopback with offline authentication; no Microsoft account, GUI, GPU, or public server is required. The gameplay action schema never exposes console, RCON, teleport, operator, creative, `/give`, `/gamemode`, or gamerule mutation.

The pinned bridge uses `mineflayer@4.39.0`, `mineflayer-pathfinder@2.4.5`, and `vec3@0.1.10`, locked in `bridge/package-lock.json`. Java 21 and Node 24 are supported by the checked setup. Runtime downloads, worlds, logs, npm cache, and temporary files stay under `runtime/`; source files are never rewritten during play.

## Setup

```sh
cd /home/hemlix/GamingMaster-mh-tier2/runs/minecraft/adaptor
./setup.sh
/home/hemlix/GamingMaster-mh-tier2/runs/minecraft/.venv/bin/python -c \
  "from metaharness.games.adaptor import validate_adaptor; print(validate_adaptor('/home/hemlix/GamingMaster-mh-tier2/runs/minecraft/adaptor', expected_tier=2).to_markdown())"
```

`setup.sh` uses the evaluator venv next to this adaptor by default. `setup.py` downloads Mojang metadata, resolves exactly 1.21.1, verifies the official SHA-1 and SHA-256, checks Java 21 and Node >=22, installs the locked Node dependencies, and writes `runtime/artifacts.json`. The server setup writes `eula=true` as requested; Minecraft's EULA applies.

No credentials are required. `env.example` lists optional overrides. The supported variables are loopback-only `MC_HOST`, `MC_PORT`, `MC_USERNAME`, `MC_VERSION=1.21.1`, `MC_AUTH=offline`, `MC_SERVER_JAR` beneath `runtime/`, `JAVA_BIN`, `NODE_BIN`, and `MC_RUN_DIR` beneath `runtime/`. `make_game()` also accepts `run_dir`, `port`, `username`, `max_seconds`, `objective`, `target`, and `target_count`.

## Reset and scenarios

Each `reset(seed)` stops any previous owned instance and creates a new isolated server directory and world. The server is configured before the client joins with:

- vanilla 1.21.1, Java 21, Survival, Normal difficulty, non-hardcore;
- `online-mode=false`, `enforce-secure-profile=false`, `allow-nether=true`, generated structures enabled;
- `keepInventory=false`, ordinary mob spawning, daylight/weather cycles, natural regeneration, fire, mob griefing, drops, insomnia, patrols, traders, wardens, and random tick speed 3;
- `spawnRadius=0`, clear weather, and time 1000 before the player joins.

The player then joins an ordinary newly generated world at the server-selected vanilla spawn with an empty inventory. No spectator preparation, teleport-only reset, terrain admission filter, generated starting resources, or easy-seed selection is used. Seed `s` maps deterministically to the signed 64-bit value represented by the first eight bytes of SHA-256(`minecraft:1.21.1:world:{s}`); the decimal string is written to `level-seed` and recorded in `scenario.json` under `runtime/episodes/<episode>/`. Different seeds therefore vary biome, wood/food proximity, villages, exposed lava and ruined portals, terrain/caves, structures, Nether layout, and stronghold location through the real vanilla generator. The evaluator may keep seeds private; the observation does not reveal the seed.

The default objective is the Java advancement `minecraft:end/kill_dragon` (**Free the End**). Optional specialized objectives are supported through `objective='advancement', target='minecraft:...'` or `objective='resource', target='item_name', target_count=N`. Resource and advancement objectives are useful for calibration; the default benchmark is dragon completion. The default episode duration is 3,600 seconds and can be bounded for a local smoke test with `max_seconds`, without changing the default benchmark contract.

## Observations

The Node bridge normalizes Mineflayer state into compact JSON snapshots, pushed at approximately 5 Hz and kept below 200 kB in validation. The Python adaptor adds episode, elapsed/remaining time, observation freshness, authoritative inventory, score/progress, validity, and action feedback.

A snapshot contains:

- player position/velocity/yaw/pitch/ground state, health, food, saturation, oxygen, effects, water/lava flags, death state, dimension, Survival mode, equipment, selected hotbar slot, XP, and known bed position;
- authoritative personal inventory from periodic read-only server `data get entity <player> Inventory` audits, plus explicitly labelled client inventory slots and open-window/cursor previews;
- world time/day, biome when available, weather, server-time age, physics tick count, and bounded collision labels;
- nearby entities with stable IDs, position/velocity/distance, visibility, hostile and conditional-threat labels;
- exposed nearby blocks and safe stand candidates, raycast visibility, block properties, workstations, feasible recipe hints, and recent local block changes;
- server-sent advancement completion, bounded events/messages, current task/deadline, action result, and combat telemetry such as observed End Crystal deaths and Ender Dragon boss-bar health.

Unknown chunks, hidden underground ore, seed, structure coordinates, console responses, and privileged preparation data are not exposed. Candidates are client-visible loaded terrain, not a map oracle. The player sees structured state rather than screenshots; there is no renderer in the normal run.

## Actions

`action_spec` exposes every action name and `describe()` documents the compact whitespace grammar. All actions are ordinary player operations:

- `noop`, `cancel`, `wait [seconds]`, `controls keys|none`, `goto x y z`, `navigate candidate`;
- `look_at x y z`, `hotbar slot`, `equip item [destination]`, `gather candidate [count]`, `dig x y z`, `place item reference face`;
- `craft item count [table]`, `furnace_start`, `furnace_take`, `eat item`, `attack entity_id`, `shield on|off`;
- `activate_block candidate [face]`, `activate_entity entity_id`, `use_item [seconds] [hand]`, `open_container candidate`, `close_window`;
- `transfer take|put item count`, `move_slot source destination`, `toss item count`, `sleep bed`, `wake`, `respawn`;
- `inspect_trades villager_id`, `trade villager_id index [times]`, `enchant table item choice`, and ordinary `chat text`.

Jobs are bounded and asynchronous: navigation 15 seconds, gathering 25 seconds, other jobs 10 seconds. Navigation does not mine, bridge, tower, parkour, or sprint; it avoids known hazards and drops greater than two blocks and cancels on damage. Manual controls expire after 0.8 seconds unless renewed by the runner. Malformed, stale, busy, or illegal actions return structured feedback rather than raising from `tick()`.

`tick_s=0.5`. `tick()` only posts the current action to the Node mailbox and returns immediately; the Java server and Mineflayer physics continue while the model reasons. `ticks` counts posted calls. The adaptor has an independent monitor for time limits, stale server/bridge/observation streams, non-Survival mode, and completion.

## Score and evidence

`score` is the transparent dragon progress score, with raw facts retained in `observe()['progress']`, `info()['progress']`, `score_breakdown()`, `scenario.json`, `result.json`, and the inventory audit journal:

- opening 10%: logs plus crafting table (3), food (3), bed/sleep (4);
- iron/shield 10%: iron (5), shield (5);
- Nether/fortress 20%: Nether entry (8), fortress advancement (6), blaze rod (6);
- eyes/stronghold/portal 20%: pearl (4), eye (6), stronghold advancement (5), active portal evidence (5);
- End entry 10%;
- dragon kill 30%, based on `minecraft:end/kill_dragon`.

The score records milestone times, advancement IDs, current resources, per-episode resource maxima, elapsed server ticks, deaths, estimated dropped inventory, damage taken, crystal deaths, and minimum observed dragon health. Completion fills the score to 100 while preserving the raw milestone facts. Idle survival earns no artificial completion credit. Infrastructure failures set `info()['valid_trial']=False` and an invalid reason; normal death is valid, drops inventory, records a death, and can be followed by ordinary `respawn`.

## Verification and limits

The required validator passes against a real local 1.21.1 server. The live test also passes real log mining, crafting planks and a crafting table, progression scoring, asynchronous realtime actions, malformed/random action rejection, JSON-size checks, and process cleanup. Final validation evidence is written under `runtime/validation-final.txt`; live mechanics evidence is under `runtime/live-test-final.log` and `runtime/live-test.json`.

This is a bounded client/action adaptor, not an autonomous Minecraft strategy agent. Mineflayer pathfinding can fail around occlusion, water, narrow caves, hostile combat, portals, or changing blocks. There is no guarantee of dragon completion. The observation stream is network/client state and can lag; stale infrastructure invalidates the trial. The server and offline identity are local-only and must not be exposed to an untrusted network.

### study/interfaces.md
# Minecraft programmatic interfaces

## Recommendation in one sentence

Use a **standalone Node.js Mineflayer client** connected to a version-pinned private vanilla Java server, with `mineflayer-pathfinder` for navigation and an optional `mineflayer-pvp` plugin for a bounded combat primitive. Let the Python harness talk to a small Node control process over JSON Lines (or localhost TCP), and use RCON/server process control only for reset and measurement outside the scored gameplay loop. This gives structured state and low-latency input without a display, GPU, Microsoft account, or client-side mod installation.

There is no official Mojang bot API or official benchmark environment for Java Survival. The official [Java server download](https://www.minecraft.net/en-us/download/server) provides the simulation, while the client connects through the Java protocol. The practical choices below are community-maintained bridges or administrative interfaces; test the exact Minecraft version before committing to a run.

## Comparison of practical interfaces

| Interface | Observation and control | Maturity and latency | Cost / legal and operational risk | Decision |
|---|---|---|---|---|
| **Mineflayer standalone bot** | Structured client state: entity/player maps, health/food/oxygen, inventory and windows, held item, position/time/weather, blocks, chat, block/entity updates, and server events. Controls include movement keys, look, dig, place, activate, equip/craft/window operations, attack, item use, and chat. | High maturity and a large PrismarineJS ecosystem. Event-driven packet handling is normally one local server tick plus network/processing delay; local latency should be low tens of milliseconds, but mining/crafting still takes game time. | Free, MIT-licensed open source; Node.js is already installed. Public servers may ban bots or treat automation as cheating; use only the user's private authorized server. Offline mode has weak identity, so bind locally/firewall it. | **Chosen.** Best structured state/control-to-effort ratio for a headless agent. See [Mineflayer](https://github.com/PrismarineJS/mineflayer) and its [API reference](https://github.com/PrismarineJS/mineflayer/blob/master/docs/api.md). |

…(truncated; full text in prior/interfaces.md)

## References

* [Mineflayer repository](https://github.com/PrismarineJS/mineflayer)
* [Mineflayer API](https://github.com/PrismarineJS/mineflayer/blob/master/docs/api.md)
* [Mineflayer Pathfinder](https://github.com/PrismarineJS/mineflayer-pathfinder)
* [Mineflayer PVP](https://github.com/PrismarineJS/mineflayer-pvp)
* [Minecraft protocol](https://github.com/PrismarineJS/node-minecraft-protocol)
* [Prismarine Viewer](https://github.com/PrismarineJS/prismarine-viewer)
* [Baritone](https://github.com/cabaletta/baritone)
* [Paper development docs](https://docs.papermc.io/paper/dev/) and [Fabric docs](https://docs.fabricmc.net/)
* [Minecraft Java server download](https://www.minecraft.net/en-us/download/server)
* [Minecraft EULA](https://www.minecraft.net/en-us/eula) and [Usage Guidelines](https://www.minecraft.net/en-us/usage-guidelines)
* [Server setup](https://minecraft.wiki/w/Tutorial:Setting_up_a_Java_Edition_server), [RCON](https://minecraft.wiki/w/RCON), and [Server List Ping](https://minecraft.wiki/w/Java_Edition_protocol/Server_List_Ping)


## Measured model latency (at archive creation, 2026-10-30 11:29)
- System-2 (Astra), one-word answer: 43 s
- System-2, an answer of ~1112 output tokens: 155 s — latency grows with the length of the answer; size timeouts for the answers your prompts actually request
- System-1 (Jev), one request with the game's initial observation and a Choice over its actions: median 213 ms (= 0.4 ticks), p90 236 ms, max 246 ms over 12 warm requests; first request on a new connection 299 ms — a timeout below the tail turns slow answers into failed calls
Later measurements per candidate are on the leaderboard (latency columns, a latency fit).

## Jev model context and usage for Meta-Harness

Snapshot: 2026-10-29 (docs.typesafe.ai: API, models, primitives, confidence, patterns, cookbooks, request
rules and the jev-1.13 known-failure page; plus the design of deployed open-source Jev agents). This is design context for the Astra proposer and for runtime Astra
when it designs Jev questions. The Python interface below is GamingMaster's wrapper, which differs
from the provider SDK. The model profile and episode budgets in the experiment manifest are
authoritative.

### What Jev supplies

Jev is TypeSafe's System One model for fast, focused judgments. It produces
typed answers and probability distributions, not prose, code, plans or reasoning
traces. It reads text and structured JSON. (The provider has an experimental image
input; this wrapper does not send images, so a visual game needs a separate perception
step — code, or Astra via `ctx.think(images=[...])` — before Jev can judge its state.)
Calibration describes aggregate predictions, not correctness of each answer.
See [System One](https://docs.typesafe.ai/concepts/system-one).

Model ids: `jev-latest` and `jev-preview` both resolve to jev-1.14.0 at this snapshot (released
2026-10-14; jev-1.13.0 remains available). An alias can move between calls; every answer records
the version that produced it, and the leaderboard warns when an archive mixes versions. Limits:
64k tokens per request, 32k per question; English is the primary training language.

In this architecture, Astra can interpret unfamiliar rules, choose subgoals and
design the decision interface. Jev can recognize situations, assess whether a
subgoal's preconditions still hold, or choose among well-described alternatives.
Code should handle exact arithmetic, legal-action checks, coordinate transforms,
bookkeeping and execution. Do not ask Jev to invent an unknown numerical action
mapping or perform an entire planning algorithm in one classification.

### Known failure modes (provider's list for jev-1.13) and what to do instead

* **Literal reading.** It answers the question written, not the one meant: state the
  exact condition in `instructions` and put boundary cases in the criteria.
* **Math, counting, comparisons of numbers and dates.** Unreliable: compute them in
  code and send the result or a named bucket ("adjacent", "two cells away"), never ask
  Jev to count, add or compare magnitudes.
* **Numeric representations.** Semantic words beat raw numbers: colour names over RGB/hex
  or palette indices, "wall to the left" over coordinates. Translate in code first.
* **Indirection.** Several hops (a property of a property, double negatives) cost
  accuracy: name the relevant state field directly, e.g. `threats.nearest`.
* **Large states with irrelevant detail.** Unrelated content distracts and costs accuracy
  ("context rot"): filter in code and send only the fields this decision needs.
* **Contradictory instructions and criteria.** Keep them aligned; a Noul whose `true`
  means "no" performs worse.
* **Choice option order.** It leans toward the option listed first: do not always list a
  default/no-op first; when a decision matters, ask the same question twice with the
  options reordered (one request, two question ids) and act only if both agree.
* **Generation.** It is not a generator: turn extraction into a Choice over candidates
  that code (or Astra) produced.

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
this model. A request must fit 64k tokens, and the state plus any single question
32k. See [State](https://docs.typesafe.ai/concepts/state).

Batch independent judgments about that state into one call. They share the
input but do not see each other's answers. If a question really requires an
earlier answer, put that answer in the next call's state. Otherwise ask both
speculatively and combine them in code. Put the complete judgment in
`instructions`: question IDs are invisible to Jev. Refer to input fields
explicitly, e.g. `plan.preconditions`. See
[Questions](https://docs.typesafe.ai/primitives).

### Patterns that deployed Jev systems share (provider cookbooks, open-source agents)

* **Code computes facts, Jev judges, code decides.** Successful controllers send a small JSON
  state of *derived facts* (distances as named buckets, what is adjacent, time to contact,
  the last action and its measured outcome, the measured age of the observation) rather than
  raw grids or numbers, then ask atomic questions and combine the probabilities in code.
* **One request, several atomic questions.** A per-tick request typically carries 3–8
  independent questions (e.g. a Choice over the legal actions, a Score of immediate danger, a
  Noul per precondition or per candidate), including speculative ones whose answers code may
  ignore; a request with many questions costs about the same latency as one with one.
* **Structured criteria.** Option descriptions as objects (`what`, `not_for`, `examples`) and
  instructions that name the state fields (`` `threats.nearest` ``) measurably beat prose.
* **Freshness and fallbacks.** Every answer is tagged with the observation / plan version it
  judged; a late answer is dropped, and an executable fallback (the last safe action, a
  conservative rule) runs while waiting. Deadlines per question are short and explicit.
* **Confidence gates per consequence.** Irreversible or lethal actions need a higher top
  probability than cheap ones; a top probability under ~0.6 is treated as "undecided" (route to
  the fallback or to System-2), not as a weak yes. Ask high-stakes Choices twice with the options
  reordered and act only when both agree.
* **Large option sets** (more than 255, or many near-identical options) are handled in two
  stages: Score or Noul each candidate independently, then one Choice over the shortlist.
* **System-2 authors, System-1 runs.** The slow model designs the state schema, the questions,
  the thresholds and the fallback policy — and re-authors them when the evidence says the
  questions are wrong — rather than being asked per tick. Harnesses that let System-2 revise
  the System-1 interface on observed failures improved most across iterations.

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
For Score, use 2–10 concrete ordered levels for one dimension (the API rejects more
than 10; the constructor raises); the returned number is an expected level, not
necessarily an integer or a probability — use it for thresholds, not to interpolate
magnitudes. See [Score](https://docs.typesafe.ai/primitives/score).

Confidence summarizes the distribution and is not interchangeable with the
winning option's probability. Inspect both when useful. A Noul near 0.5 is
uncertainty about a yes/no proposition, not medium severity. Handle absent
confidence explicitly; never turn `None` into certainty. Choose thresholds
using training evidence and action consequences, with a fallback or Astra
escalation for ambiguity. See [Confidence](https://docs.typesafe.ai/confidence).

What `ctx.decide` raises: `TimeoutError` when the answer (including retries of
rate-limited or transient server errors) does not arrive within `timeout_s`;
`RuntimeError` for any other failed call (e.g. a malformed question rejected by
the API); `BudgetExceeded` / `EpisodeOver` when the System-1 budget is used up or
the episode is over; `ModelUnavailable` when the key is rejected or the account is
persistently throttled (the run stops; catching it does not hide it).

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
latency from traces and the task's calibration: Jev can take longer than a game
tick, and the first request on a new connection (each thread keeps its own) pays
the connection setup. Keep an executable fallback while waiting, limit outstanding
requests, and associate each response with the observation and plan version that
produced it. Recheck relevant preconditions and legality on the latest observation
before acting. A cached semantic interpretation may outlive an individual action
decision; define its invalidation rule rather than treating every answer as
permanently reusable.

The context caps each request's timeout by the remaining wall budget. Handle
transient failures within that budget, let `ModelUnavailable` propagate, and stop
background work when `ctx.done`. The standard inference calls already log request,
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
