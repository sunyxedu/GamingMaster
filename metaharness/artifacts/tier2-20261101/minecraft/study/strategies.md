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
| **Blaze rods and pearls** | Kill enough blazes for the intended eye budget, using fortress corridors, cover, a shield, and a retreat route. Obtain ender pearls by controlled enderman encounters, piglin bartering, or another scenario-legal source. Do not turn an incomplete resource count into an irreversible portal trip; reserve food and blocks for the return. Blaze rods and pearls combine into Eyes of Ender; see [Eye of Ender](https://minecraft.wiki/w/Eye_of_Ender) and the [speedrun tutorial](https://minecraft.wiki/w/Tutorial:Speedrun). | **Code:** item count and craft arithmetic, recipe validation, route/escape bookkeeping. **Astra:** decide whether to farm more rods/pearls or return. **Jev:** recognize when a blaze is exposed, when a pearl/rod drop is safe to collect, and whether to disengage. |
| **Stronghold search** | Return to the Overworld, throw an Eye of Ender at separated positions, record both the player location and the eye direction/landing behavior, and triangulate the likely stronghold with code. Walk to the estimate, then search systematically rather than repeatedly throwing eyes at every block. Eyes can be consumed/break, so preserve a reserve. The [Stronghold](https://minecraft.wiki/w/Stronghold) and [Eye of Ender](https://minecraft.wiki/w/Eye_of_Ender) pages cover the mechanics. | **Code:** vector/line intersection, coordinate arithmetic, eye inventory, portal-room search map, and no-stale-observation checks. **Astra:** choose throw locations, interpret odd terrain/stronghold layouts, and plan the portal-room approach. **Jev:** recognize a silverfish/portal-room/void or immediate mob threat and choose the tactical response. |
| **Portal-room preparation** | Secure a small staging area, light/close side corridors, refill food, equip the selected weapon/armor, carry blocks and a water bucket where useful, and put irreplaceable surplus in a safe chest. Activate the portal only when the inventory and recovery plan pass the preflight. | **Code:** exact inventory/armor/food/portal-frame checklist and save-point ledger. **Astra:** choose safe staging and decide whether one more preparation trip is worth it. **Jev:** decide whether the visible portal room is safe to cross or whether to place cover/retreat. |
| **End entry and crystals** | Enter with a clear objective: avoid the void, avoid staring at endermen, destroy healing crystals, and attack the dragon during safe windows. A bow is the robust general tool for exposed crystals and a flying dragon; pillar cover and disciplined movement matter more than speed. The [Ender Dragon tutorial](https://minecraft.wiki/w/Tutorial:Defeating_the_ender_dragon) documents the crystal/dragon fight. | **Code:** End dimension detection, void/edge veto, crystal/entity tracking, health/food/arrow ledger, and attack cooldown. **Astra:** choose a crystal-first route and switch between ranged/melee phases. **Jev:** identify exposed crystal, dragon landing/opening, enderman threat, or immediate retreat. |
| **Dragon kill and exit** | Do not rush a low-health dragon into a void death. Confirm the dragon-death event/`Free the End` advancement and record the terminal result. If the scenario continues after the kill, collect the egg/gateway only under a new plan; the dragon kill is the default terminal success, not the end of the persistent world. See [Advancement guide: The End](https://minecraft.wiki/w/Tutorial:Advancement_guide/The_End_tab). | **Code:** advancement/event confirmation, score, death/episode status, and artifact log. **Astra:** decide whether the requested objective ends at the kill. **Jev:** only handles the immediate final-fight response; it does not declare success from a visual guess. |

## Alternative openings and when to select them

### Safe standard opening

This is the default for held-out random seeds: logs → stone tools → food/bed/shelter → shield/iron/bucket → controlled Nether. It has lower variance and gives the agent a known recovery point. Use it when no village, ruined portal, or exposed lava route is immediately available, when health/food are low, or when terrain is dangerous. The plan should stop scouting once the minimum opening kit exists; wandering for perfect resources is a common way to lose the first night.

### Village opening

A nearby village can supply beds, food, crops, hay, chests, and trading opportunities. It is strong when visible and safe, but the agent must not assume every village is safe: ravines, mobs, raids/illagers in a scenario, and night exposure still matter. Code records the village and accessible containers; Astra chooses whether to harvest beds/food or establish a base; Jev handles villagers and nearby hostile threats. Later, villager trading can convert renewable materials into emeralds and gear; see [Trading](https://minecraft.wiki/w/Trading). Do not make villager transport a required early step unless the scenario specifically rewards it.

### High-variance speedrun opening

If the spawn reveals an exposed lava pool, water, village, ruined portal, or unusually favorable structure, a skilled player may go directly toward a Nether entry, bastion/barter route, fortress, and stronghold. This minimizes time but is brittle: one lava, fall, piglin, or missing food event can erase the run. Astra may select it only when a precomputed gate says the route has enough blocks/food and a return or recovery plan; code enforces the gate; Jev handles the immediate hazards. Treat the [speedrun tutorial](https://minecraft.wiki/w/Tutorial:Speedrun) as a source for this branch, not a promise that it dominates safe play on every seed.

## Nether route choices and counters

### Fortress-first, low-variance route

Search from the portal using a bounded, marked exploration pattern; keep the portal coordinate and a direction/return marker. At a fortress, use corridors, pillars, and blocks to break blaze line-of-sight. Collect the required rods, then leave before health/food/blocks are exhausted. Never let Pathfinder choose an uninspected drop through a fortress or basalt/Nether terrain. This route is slower than a perfect bastion barter but easier to audit.

### Bastion/barter route

A gold-rich bastion can provide gold for piglin bartering and valuable resources, and bartering can supply ender pearls. It is also one of the most lethal environments: piglins, piglin brutes, vertical drops, lava, and chest-opening anger rules create clustered threats. Wear gold for ordinary piglins, but isolate targets, block sightlines, do not mine gold in a piglin's view, and do not open a container while relying on neutrality. The code should have a “bastion not safe” abort condition based on health, block count, escape connectivity, and visible threats; Astra chooses whether the expected time saving justifies it; Jev makes the moment-to-moment cover/retreat choice. See [Piglin](https://minecraft.wiki/w/Piglin), [Bastion Remnant](https://minecraft.wiki/w/Bastion_Remnant), and [speedrun](https://minecraft.wiki/w/Tutorial:Speedrun).

### Nether-specific hazards

Water buckets are not a general Nether escape tool; blocks, ladders/scaffolding where safe, fire resistance if acquired, and a known portal route are more reliable. Lava flows and ghast fireballs can destroy an unprotected path. The action arbiter should stop movement on a newly observed fluid/edge, rather than trusting a path calculated before the block update. Jev receives a precomputed `hazard` enum and chooses among cover, retreat, stop, or continue; it is not asked to estimate lava physics.

## Combat heuristics

### General melee

Java combat rewards waiting for a charged attack rather than spamming weak hits; see [Combat](https://minecraft.wiki/w/Combat) and [Melee attack](https://minecraft.wiki/w/Melee_attack). A robust bot keeps distance when possible, uses a shield/cover against ranged attacks, attacks a reachable target, and retreats before health or escape space becomes critical. The code maintains attack cooldown, reach, line of sight, weapon slot, target validity, and damage/knockback observations. Astra sets a fight policy—avoid, isolate, or engage—while Jev answers a typed question such as `immediate_action = {attack, shield, retreat, eat, place_cover, switch_target}`.

Do not make the PVP plugin the only combat policy. `bot.pvp.attack` is useful after a target and arena are approved; it must be cancellable when health drops, a second hostile enters, fire/lava appears, the target disappears, or the path becomes unsafe. Use a deterministic emergency escape that does not need a model call.

### Threat-specific responses

| Threat | Strong response | Code / Astra / Jev placement |
|---|---|---|
| Creeper | Do not let it close while mining or inventory-managing; retreat to open space/cover, use knockback or a shield/solid block, and cancel long actions when its fuse/position makes the area unsafe. | **Code:** distance, line-of-sight, fuse/damage veto, cancel. **Jev:** recognize creeper-close and choose retreat/cover/attack. **Astra:** avoid route patterns that repeatedly enter enclosed creeper spaces. See [Creeper](https://minecraft.wiki/w/Creeper). |
| Skeleton/stray | Shield or use cover while closing distance; avoid standing in an open straight line; do not trade arrows at low health. | **Code:** projectile/cover and health flags. **Jev:** shield/cover/close/retreat. **Astra:** select a route with cover. See [Skeleton](https://minecraft.wiki/w/Skeleton) and [Shield](https://minecraft.wiki/w/Shield). |
| Zombie/husk/spider | Isolate one target, maintain spacing, and do not become surrounded in a cave or doorway. Husk hunger effects and cave geometry make a retreat better than a greedy extra hit. | **Code:** target count, reach, exit path. **Jev:** target/retreat/cover choice. **Astra:** decide whether to clear or bypass. |
| Enderman | Avoid crosshair contact and use a two-block shelter or water where appropriate; do not start an Enderman fight while also managing another threat. | **Code:** gaze/target and shelter checks. **Jev:** recognize enderman aggro and select shelter/retreat. **Astra:** plan pearl farming in controlled terrain. See [Enderman](https://minecraft.wiki/w/Enderman). |
| Blaze | Use fortress cover, shield/blocks, and a clear retreat; collect drops only after nearby fire/hostiles are accounted for. | **Code:** fire/health/escape checks and rod count. **Jev:** immediate cover/attack/retreat. **Astra:** decide when enough rods have been collected. See [Blaze](https://minecraft.wiki/w/Blaze). |
| Ghast | Use solid blocks and move out of a predictable fireball line; do not stand on a narrow bridge trying to return a shot. | **Code:** projectile/edge detector and movement veto. **Jev:** cover/retreat. **Astra:** choose whether to cross or detour. See [Ghast](https://minecraft.wiki/w/Ghast). |
| Piglin/piglin brute | Gold armor helps with ordinary piglins only. Separate groups, block sightlines, avoid chest/gold triggers, and retreat from a brute or clustered aggro rather than trusting armor. | **Code:** armor, block/container action gates, cluster count and escape route. **Jev:** identify neutrality/aggro/cover/retreat. **Astra:** select fortress-first or bastion route. See [Piglin](https://minecraft.wiki/w/Piglin). |
| Lava/fire/fall/void | Stop, look, sample blocks, place safe blocks, and replan. A shortest path is invalid if its risk mask contains lava, a fatal drop, or a void edge. | **Code:** hard veto and emergency stop; never delegate this invariant. **Jev:** classify the local response. **Astra:** replan the macro route. See [Tutorial: Things not to do](https://minecraft.wiki/w/Tutorial:Things_not_to_do). |
| Ender Dragon | Destroy crystals, stay away from the void, use ranged attacks while flying, and use melee only in a safe landing/opening. Avoid beds unless the scenario explicitly tests the expert explosion method. | **Code:** crystal/dragon state, edge/health/arrow checks, event confirmation. **Jev:** opening/retreat/target recognition. **Astra:** phase plan and equipment tradeoff. See [Defeating the Ender Dragon](https://minecraft.wiki/w/Tutorial:Defeating_the_ender_dragon). |

## Navigation, mining, and bookkeeping heuristics

1. **Safety mask before shortest path.** Build a local occupancy/risk map that marks solid/passable blocks, fluids, fire, exposed edges, maximum drop, hostile entities, and stale/unloaded space. Pathfinder's `canDig`, `canPlace`, drop and entity settings are inputs, not proof. Code rejects paths through forbidden hazards and Astra chooses whether to spend time making a bridge/staircase.
2. **Use landmarks and coordinates.** Save spawn, bed, base, portal pair, fortress, bastion, stronghold, portal room, and death location. Store dimension with every coordinate. Convert Nether/Overworld horizontal coordinates in code; do not ask Jev or Astra to do the arithmetic. The 1:8 relationship is documented at [Nether portal](https://minecraft.wiki/w/Nether_portal).
3. **Mark explored routes.** Torches/blocks and a cached waypoint graph make a return route explicit. A bot that reaches a resource but cannot return has not made progress. Code tracks last safe waypoint and a route risk budget; Astra decides when to turn back.
4. **Treat chunks and events as asynchronous.** A block update, entity spawn, portal transfer, death, or server lag invalidates a plan. Cancel/replan on `blockUpdate`, `entitySpawn`, `health`, dimension change, or stale timestamp. This is code, not a prompt instruction.
5. **Inventory is a finite state machine.** Keep food, blocks, weapon, pick, torches, and shield in stable slots; count exact rods/pearls/eyes, arrows, gold armor, and spare tools. Before opening a container, trading, crafting, or entering the End, snapshot inventory and verify the expected delta. Code owns this ledger; Astra gets a summarized deficit and can choose a route to fix it.
6. **Stop before the red line.** Return when food, blocks, health, durability, or escape space falls below a precomputed threshold. The death penalty includes dropped items that can despawn after five minutes in a loaded ticking chunk; see [Item entity](https://minecraft.wiki/w/Item_%28entity%29). Astra may override a conservative threshold only with an explicit reason and safe recovery plan.

## Stronghold and End planning details

### Eye throws and triangulation

The eye search is a geometry/bookkeeping task, not a language task. At each throw, code stores the bot's X/Z and the normalized horizontal direction of the eye; after at least two sufficiently separated observations, it estimates an intersection/uncertainty region. Astra chooses where to take the second observation if the first line is noisy or terrain blocks a direct route. Code decides whether the eye budget permits another throw and records a search spiral around the estimate. Jev only recognizes a visible eye/stronghold/portal-room hazard or chooses a local action.

When the stronghold is reached, search rooms systematically, mark dead ends, keep the portal room safe, and do not let a generic “go to nearest portal-frame block” path walk through lava or silverfish ambushes. A portal-frame/eye count and an activated portal are separate milestones.

### End preparation checklist

The code must report a pass/fail checklist before activation:

* acceptable armor and weapon, shield/blocks, food, pick, and a recovery plan;
* enough ranged ammunition or a declared melee-only plan;
* hotbar slots known and no open container/window;
* health/food not in a danger band and no active fire/poison/fall;
* portal-room coordinates and return/death evidence logged;
* objective says whether the episode ends at dragon kill or continues.

Astra chooses how much preparation is worth the time. Jev answers only immediate choices once inside. This separation prevents a slow Astra call from leaving the bot motionless on the End island.

### Dragon phases

1. **Arrival:** stop at a safe area, orient, and avoid accidental enderman aggro. Code checks edge/void risk.
2. **Crystal phase:** use ranged attacks where possible; pillar crystals may need a careful climb/cover approach. Do not stand below a dragon knockback path or on a narrow edge.
3. **Flying phase:** track dragon position and attack windows; keep moving in a safe arena rather than chasing every moment.
4. **Perch/landing phase:** use a charged melee attack only when the dragon head/body is reachable and the escape region remains safe.
5. **Low health:** preserve a valid escape; no greedy final hit. Confirm `Free the End`/dragon death from event or advancement state.

Beds exploding in the Nether/End can produce a fast kill but are an expert, high-risk technique. It belongs in a separate strategy prompt with an explicit block-cover calculation, not in the default prior. See [Bed](https://minecraft.wiki/w/Bed) and the [dragon tutorial](https://minecraft.wiki/w/Tutorial:Defeating_the_ender_dragon).

## Common mistakes and known counters

| Mistake | Why it loses | Prevention/owner |
|---|---|---|
| Wandering for resources after the opening kit is complete | Night, hunger, and mobs accumulate while no milestone is secured. | **Astra:** stop conditions and milestone-driven route. **Code:** time/food/bed flags. |
| Digging straight down or sprinting across uninspected terrain | Lava, caves, falls, and mobs turn a small shortcut into death. | **Code:** fluid/drop/edge veto; **Jev:** emergency stop; source [things not to do](https://minecraft.wiki/w/Tutorial:Things_not_to_do). |
| Entering the Nether without a gold piece, food, blocks, or a portal ledger | Piglin aggro, fire, and a lost return route are common run-ending states. | **Code:** portal preflight; **Astra:** route choice; **Jev:** local cover/retreat. |
| Assuming gold armor makes piglins harmless | Chest opening, gold mining, attacking, and brutes can still trigger danger. | **Code:** interaction gates; **Jev:** aggro recognition; [Piglin](https://minecraft.wiki/w/Piglin). |
| Letting a pathfinder walk through a newly changed block/entity situation | A path was safe when computed but not when executed. | **Code:** event cancellation, freshness and hazard mask; **Jev:** choose the immediate alternate. |
| Fighting in a doorway or cave while surrounded | Knockback and line-of-sight prevent escape; one extra hit becomes a death. | **Astra:** choose isolation/retreat; **Jev:** count threats; **Code:** minimum escape-space invariant. |
| Spam-attacking in Java | Uncharged attacks do less damage and lose critical/sprint timing. | **Code:** attack cooldown; **Jev:** attack now vs wait/retreat; [Combat](https://minecraft.wiki/w/Combat). |
| Staring at an Enderman | It creates a second hostile threat at the worst time. | **Code:** gaze/target check; **Jev:** shelter/retreat; [Enderman](https://minecraft.wiki/w/Enderman). |
| Spending every Eye of Ender or entering the End with no reserve | Eye breaks/search error or a bad portal preparation ends the run. | **Code:** eye budget and End checklist; **Astra:** preparation tradeoff. |
| Treating dragon health as success | A final hit can still be followed by void death or a false visual assumption. | **Code:** advancement/event confirmation and terminal score. |
| Dropping or storing the only food/blocks/tool in an unsafe container | A window/action interruption leaves the bot unable to escape. | **Code:** inventory transaction verification and emergency hotbar. |
| Using RCON/commands during the scored episode | Teleport/give/gamerule changes make the score meaningless. | **Code/harness:** hard separation between admin channel and gameplay action schema. |
| Calling Astra on every tick | The world continues while reasoning; combat and falls punish waiting. | **Architecture:** Jev at 2–10 Hz for compact emergency choices; Astra at safe checkpoints. |

## Prompt/interface design for the two models

### Astra macro-planning prompt

Give Astra a structured summary, not a screenshot-only transcript:

* objective and terminal predicate;
* current phase/milestones, dimension/coordinates, time/day, health/food/effects;
* complete inventory/armor and deficits against the next milestone;
* base/portal/waypoint graph and local hazard summary;
* visible structures/entities and recent event log;
* server tick/latency freshness and any action currently in progress;
* candidate routes with estimated risk computed by code.

Require a structured response containing: next phase, one primary goal, ordered subgoals, resource budget, route/waypoints, expected stop condition, abort triggers, and what Jev should be asked during execution. Ask Astra to choose between safe and speedrun branches explicitly and explain the tradeoff. Reject any plan that uses an unavailable item, admin command, unbounded exploration, or an action outside the adaptor schema.

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
