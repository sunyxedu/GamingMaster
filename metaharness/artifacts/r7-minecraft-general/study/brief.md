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

Cooked-food items: `cooked_beef`, `cooked_porkchop`, `cooked_mutton`, `cooked_chicken`, `cooked_rabbit`, `cooked_cod`, `cooked_salmon`, `baked_potato`. This is an equipment/progress score: legitimate looting also counts, not only crafting. Milestones latch once and survive later placement, consumption, breakage, or death. Resources already credited also survive consumption. The high-water design deliberately rewards keeping a modest useful reserve; it does not claim to measure lifetime mining. No extra points for repeatedly crafting the same tool, breaking player-placed blocks, kills, distance, or XP. An idle healthy survivor earns 20; substantial early-game progress can earn much more. Full marks are an upper bound, not an expected outcome on every natural seed.

Death is `terminated=true`; reaching T alive is `truncated=true`. An infrastructure crash, login failure, or absent required chunks is an invalid trial, not a zero-score gameplay loss. Do not award healthy-finish credit on premature disconnects. Keep the score in trusted adaptor bookkeeping rather than asking either model to estimate it.

**Scenarios.** Use fresh seeded natural worlds, Normal difficulty, and empty inventory. Derive stable independent world/start-time choices from the evaluator seed. Use equal daytime (tick 1,000) and late-afternoon (tick 9,000) strata: dusk arrives approximately 150 seconds into the latter. Prequalify viable surface starts with nearby wood and shallow reachable stone using a fixed geometry/resource predicate, never a policy's achieved score. Restore pristine per-seed snapshots or create a completely new world for each reset; teleporting an old character or changing `level-seed` alone is insufficient. Vary wood species, hills, water obstacles, ore visibility, and passive animals. See interfaces.md for the reset contract and strategies.md for the model division of labor.

**Evidence limits.** Runtime versions were inspected locally. Web documentation and tagged source were checked, but shell network access failed DNS resolution, so package installation and a server/bot handshake remain adaptor-phase checks. Some Minecraft Wiki pages were available only as search excerpts; the linked community pages supplement primary Mojang documentation. Scoring, budgets, safeguards, and scenario selection above are benchmark proposals.
