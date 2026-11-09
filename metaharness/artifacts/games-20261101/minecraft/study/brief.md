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
3. **Nether:** build and light a portal. Bring food, blocks, a shield, a pick, and at least one gold armor piece to reduce ordinary piglin hostility. Locate a Nether fortress for blaze rods. Piglins can still become hostile if the player attacks them, opens containers nearby, or mines gold-related blocks; treat a gold helmet as mitigation, not immunity. See [Piglin](https://minecraft.wiki/w/Piglin), [Nether fortress](https://minecraft.wiki/w/Nether_Fortress), and [Tutorial: Nether survival](https://minecraft.fandom.com/wiki/Tutorials/Nether_survival).
4. **Eyes and stronghold:** turn blaze rods into blaze powder and combine with ender pearls to craft Eyes of Ender. Barter gold with piglins or fight endermen for pearls. Throw eyes in the Overworld, follow their direction, and use two separated observations for a ray/triangulation estimate rather than wasting eyes at every step. The [Eye of Ender](https://minecraft.wiki/w/Eye_of_Ender) and [Stronghold](https://minecraft.wiki/w/Stronghold) pages describe the search and portal mechanics.
5. **End preparation:** before entering, refill food, bring blocks, a water bucket, a strong melee weapon, bow/arrows if possible, armor, spare tools, and a safe recovery/bed plan outside the portal room. Preserve an explicit inventory ledger: do not put every irreplaceable item into one unprotected trip.
6. **Dragon:** enter the End, avoid the void and endermen, destroy the healing End Crystals (usually with a bow or carefully reached pillar), and damage the dragon during safe attack windows. A bed-explosion strategy is a high-risk expert option, not a default policy. Killing the dragon grants the Java advancement **Free the End**; it does not delete or stop the world. See [Defeating the Ender Dragon](https://minecraft.wiki/w/Tutorial:Defeating_the_ender_dragon), [The End](https://minecraft.wiki/w/The_End), and [Advancement guide: The End](https://minecraft.wiki/w/Tutorial:Advancement_guide/The_End_tab).

## Objective and score/progress

The server does not supply a single fair “score,” so the adaptor should report raw facts plus a deterministic evaluation score. Raw facts should include:

* terminal status: dragon killed / objective completed, death count, episode end reason, and elapsed game ticks;
* dimension, position, health/food at checkpoints, current respawn point, and inventory/armor;
* advancement IDs and completion timestamps, especially the `story`, `nether`, and `end` trees;
* milestone flags: first shelter/bed, iron/shield, portal, fortress/blaze rods, sufficient pearls/eyes, stronghold, active End portal, End entry, dragon kill;
* resource and safety counters: diamonds/netherite if relevant to the scenario, food/blocks, damage taken, mob kills, deaths, and whether the bot lost its inventory.

For the default dragon scenario, use a primary binary success for `Free the End`, then rank unsuccessful runs by the furthest completed milestone. A transparent secondary score can be the weighted sum of milestone flags (early survival 10%, iron/shield 10%, Nether/fortress 20%, eyes/stronghold/portal 20%, End entry 10%, dragon kill 30%), multiplied by a death/safety factor only if the evaluator wants efficiency. Store the unweighted flags and death count so a weighted score cannot hide reckless play. Do not award points for commands, creative mode, teleporting, `keepInventory`, generated items, or operator intervention.

For build/exploration scenarios, replace the dragon terminal flag with a scenario-specific predicate (for example, an exact block/structure check or an advancement) and keep survival/death and resource ledgers. Advancements are a better general progress signal than raw play time; [Advancement](https://minecraft.wiki/w/Advancement) and [Scoreboard](https://minecraft.wiki/w/Scoreboard) document the server-side records available for evaluation.

## Controls and practical actions

A human uses keyboard/mouse: WASD movement, jump, sneak, sprint, hotbar selection, left-click attack/mine, right-click use/place, inventory/container clicks, and chat/commands. The bot adaptor should expose semantic actions and retain low-level escape hatches:

| Action family | Examples | Safety/verification |
|---|---|---|
| Motion and view | hold/release forward/back/left/right/jump/sprint/sneak; look at a point or yaw/pitch; mount/dismount | stop all controls on timeout, death, kick, or hazard; verify position and on-ground state |
| Navigation | navigate to a block/near point/coordinate; follow entity; stop/replan | pathfinder success is not proof of safety; reject lava/void/unsafe drops and replan on block/entity updates |
| World interaction | select hotbar slot, equip/unequip, dig a block, place a block, activate a block/entity, use an item | check target/face, held item, reach, and inventory; wait for completion or failure event |
| Inventory/crafting | craft recipe, move/toss item, open/close chest/furnace/villager/enchantment UI, smelt/trade | serialize the window and inventory before/after; never leave a container open during danger |
| Combat | attack a selected entity, use bow/projectile/shield/food, retreat, place cover | Jev chooses an immediate response; code enforces range, cooldown, health and escape rules |
| Communication | chat, whisper, receive/parse chat and server messages | do not use chat commands as a gameplay shortcut; reserve privileged RCON commands for reset/telemetry |

The Mineflayer API exposes these operations as structured JavaScript methods/events; exact names are in [`mineflayer`'s API reference](https://github.com/PrismarineJS/mineflayer/blob/master/docs/api.md). The adaptor should translate every completed/failed action into a small JSON event with action ID, start/end tick, result, and error reason.

## Pace and tier rationale

The simulation never waits for Astra. At 20 TPS, a falling player, a creeper fuse, an arrow, lava, fire, or a melee mob can change the state within a fraction of a second. Ordinary harvesting and route planning allow seconds or minutes, but combat and hazard avoidance require a fast policy loop. Set the operational decision budget to about **0.5 seconds** for an emergency choice: Jev should receive compact structured snapshots and issue a short action/hold decision at roughly 2–10 Hz, while an action arbiter can continue a validated movement for several ticks. Astra should plan macro goals at safe checkpoints and when Jev reports a hazard, not block the low-level loop. This is tier 2 even though many individual actions (crafting, inventory management, path calculation) are asynchronous and can be awaited.

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
