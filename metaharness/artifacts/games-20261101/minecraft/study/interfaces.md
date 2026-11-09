# Minecraft programmatic interfaces

## Recommendation in one sentence

Use a **standalone Node.js Mineflayer client** connected to a version-pinned private vanilla Java server, with `mineflayer-pathfinder` for navigation and an optional `mineflayer-pvp` plugin for a bounded combat primitive. Let the Python harness talk to a small Node control process over JSON Lines (or localhost TCP), and use RCON/server process control only for reset and measurement outside the scored gameplay loop. This gives structured state and low-latency input without a display, GPU, Microsoft account, or client-side mod installation.

There is no official Mojang bot API or official benchmark environment for Java Survival. The official [Java server download](https://www.minecraft.net/en-us/download/server) provides the simulation, while the client connects through the Java protocol. The practical choices below are community-maintained bridges or administrative interfaces; test the exact Minecraft version before committing to a run.

## Comparison of practical interfaces

| Interface | Observation and control | Maturity and latency | Cost / legal and operational risk | Decision |
|---|---|---|---|---|
| **Mineflayer standalone bot** | Structured client state: entity/player maps, health/food/oxygen, inventory and windows, held item, position/time/weather, blocks, chat, block/entity updates, and server events. Controls include movement keys, look, dig, place, activate, equip/craft/window operations, attack, item use, and chat. | High maturity and a large PrismarineJS ecosystem. Event-driven packet handling is normally one local server tick plus network/processing delay; local latency should be low tens of milliseconds, but mining/crafting still takes game time. | Free, MIT-licensed open source; Node.js is already installed. Public servers may ban bots or treat automation as cheating; use only the user's private authorized server. Offline mode has weak identity, so bind locally/firewall it. | **Chosen.** Best structured state/control-to-effort ratio for a headless agent. See [Mineflayer](https://github.com/PrismarineJS/mineflayer) and its [API reference](https://github.com/PrismarineJS/mineflayer/blob/master/docs/api.md). |
| **Mineflayer Pathfinder** | Adds A* navigation and goals such as `GoalBlock`, `GoalNear`, `GoalXZ`, follow, look-at/place-block, configurable digging/placing, drop limits, sprinting, parkour, avoidance, and path status events. | Mature companion plugin. `goto`/`setGoal` are asynchronous; planning can take from a few milliseconds to seconds depending on map/goal. Its documented defaults include a 5,000 ms think budget and 40 ms per-tick budget, which should be bounded for this game. | Free/MIT ecosystem code, but it can walk a bot into lava, fire, void, mobs, or newly changed blocks unless the adaptor adds hazard checks. | **Use with Mineflayer**, but treat it as a route generator, not an autonomous safety policy. See [mineflayer-pathfinder](https://github.com/PrismarineJS/mineflayer-pathfinder). |
| **Mineflayer PVP** | High-level `bot.pvp.attack(entity)` and `bot.pvp.stop()` plus combat state/events; handles pursuit/attack logic using Pathfinder. | Useful but narrower and less deterministic than a custom action arbiter. Combat still needs a fast threat policy, shield/food/retreat logic, target validation, and cooldown handling. | Free/MIT. Bot combat is likely disallowed on public servers; harmless on the authorized private evaluation server. | Optional primitive only; Jev and safety code must retain authority. See [mineflayer-pvp](https://github.com/PrismarineJS/mineflayer-pvp). |
| **Raw `minecraft-protocol` client** | Packet-level JavaScript objects: login/status, entity/world packets, chat, inventory and movement packets, and arbitrary packet writes. Can make a custom client/server. | Mature, version-aware protocol library and very low overhead, but every version-sensitive world, physics, inventory, and interaction behavior becomes adaptor code. | Free open source. A malformed packet or protocol mismatch can crash/disconnect the bot; packet-level automation has the same public-server rules risk. | Keep as a fallback or instrumentation layer under Mineflayer, not the primary API. See [node-minecraft-protocol](https://github.com/PrismarineJS/node-minecraft-protocol). |
| **Vanilla server commands, RCON, and world files** | RCON executes administrative commands and returns text; commands can query/set gamerules, players, scores, data, advancements, time, and world state. Server logs/world files provide reset and post-run evidence. It does **not** provide a normal player view or safe player input. | Very mature and low command round-trip latency on loopback, but responses are text, can be fragmented, and are not a continuous observation stream. | RCON is unauthenticated-by-network-design, unencrypted, and dangerous on an exposed network. Never expose its port; use loopback/firewall and a strong secret. Commands used during play can invalidate scoring. See [RCON](https://minecraft.wiki/w/RCON). | Use only for pre/post episode reset, seed/configuration, health checks, and scoring evidence. Keep it out of the agent action vocabulary. |
| **Paper plugin or Fabric server-side mod** | A custom server component can observe authoritative ticks, entities, blocks, inventories, advancements, damage, and events, and expose a local JSON/WebSocket/HTTP telemetry or reset endpoint. It can also accidentally grant powers or change simulation. | Lowest observation latency (same server tick, normally ≤50 ms at 20 TPS) and strongest scoring truth, but requires maintaining version-specific Java code and replacing vanilla with Paper/Fabric. [Paper development docs](https://docs.papermc.io/paper/dev/) and [Fabric docs](https://docs.fabricmc.net/) are the relevant frameworks. | Free frameworks; custom code is extra implementation and a modded server changes the reproducibility claim. Use a server-side-only component only for telemetry/reset if strict vanilla gameplay is required. No public-server anti-cheat issue on the private server, but deployment/version risk is high. | Valuable second phase for authoritative scoring, not needed for the first working bot. |
| **Prismarine Viewer / headless render** | Web or headless rendering of the Mineflayer bot's nearby world; screenshots/video/browser view can aid debugging. It does not improve control. | Mature companion visualization, but frame/render latency and limited field of view are worse than structured state. Headless mode can avoid a physical display/GPU, yet must be smoke-tested on this Linux image. | Free/MIT. Renderer dependencies and textures add install complexity; screenshots can be misleading or stale. See [prismarine-viewer](https://github.com/PrismarineJS/prismarine-viewer). | Optional debugging fallback, not Jev's primary observation. |
| **Baritone client/mod** | Java client-side pathfinding and mining commands/API (`#goto`, `#mine`, etc.) with rich path planning. It runs inside Forge/Fabric/NeoForge client environments rather than as a simple standalone protocol bot. | Mature and fast pathfinder across many versions, but a headless client, loader, exact mappings, and client lifecycle are more work. It is a second autonomous planner competing with Astra unless carefully wrapped. See [Baritone](https://github.com/cabaletta/baritone). | Free LGPL-3.0; client mods/automated movement are often prohibited on public servers and can trigger anti-cheat. Needs a client and potentially software rendering, contrary to the simple no-display setup. | Not chosen; Mineflayer gives the needed headless state directly. |
| **Python pyCraft or other native Python protocol clients** | A Python client can connect and handle a subset of packets, with custom actions and observations. | `pyCraft` is well documented but its stated support is approximately Minecraft 1.8–1.18.1 and only a subset of packets. Modern forks may cover newer versions, but maintenance and feature coverage are less certain than Mineflayer. See [pyCraft](https://github.com/ammaraskar/pyCraft). | Avoids a Node bridge but would require implementing modern login, chunks, physics, inventory, and interactions. The evaluator's Python 3.12 is not evidence that an old client supports it. | Not chosen for a current Java server. Use Python as the harness, not the Minecraft protocol client. |
| **Screen capture + keyboard/mouse injection** | A full Java client under Xvfb/software rendering, screenshots via X11/ffmpeg/OpenCV, input via xdotool/uinput/pyautogui. It observes pixels and controls the same pixels a human does. | Lowest robustness and highest latency; GUI/render stalls, focus, camera calibration, menus, and occlusion create failures. No GPU/display makes setup especially fragile. | Requires a client distribution/account path, more RAM, and potentially Microsoft authentication. Pixel automation can violate server rules and is difficult to audit. | Last-resort fallback for a GUI-only scenario, not appropriate here. |
| **Server List Ping / status protocol** | MOTD, version, player count/max players and status JSON; no gameplay observation or control. See [Server List Ping](https://minecraft.wiki/w/Java_Edition_protocol/Server_List_Ping). | Trivial and fast, but only a liveness/version preflight. | Free and safe on a private server. | Use as a preflight health check, never as the game interface. |

## Chosen architecture

### Processes

1. **Server process:** a version-pinned Mojang Java server running `--nogui` on loopback. Use Java 21 and a Java-21-compatible release (1.21.x is a practical starting family); do not use a floating `latest` version. The exact version is put in `MC_VERSION`, in the server configuration, and in the Node lockfile.
2. **Node bot process:** Mineflayer creates one offline-mode bot. It loads Pathfinder and, only where needed, PVP. It owns the network connection, action execution, event subscriptions, world cache, reconnect/death handling, and compact JSON snapshots.
3. **Python harness/model layer:** the harness spawns the Node process and sends typed actions over JSON Lines. It gives Jev compact state and receives one short tactical decision. Astra receives summaries/checkpoints and returns macro plans; Astra never blocks an emergency loop. The Python harness may use `mctools`/`rconx` in the supplied venv for server reset, but those commands are not exposed to the game-playing model.
4. **Optional debugging process:** Prismarine Viewer on a local-only port, or a headless frame dump when structured state does not explain a failure.

A subprocess JSONL bridge is preferable to trying to import Node modules into Python: the Node process remains the authoritative Minecraft client, while the evaluator can stay in `/home/hemlix/GamingMaster-mh-tier2/runs/minecraft/.venv/bin/python`. Use one message ID per action and never mix log text with stdout; send diagnostics to stderr.

### Observation contract to implement

Mineflayer exposes raw objects, but the harness should normalize them into deterministic JSON. At minimum, every snapshot should include:

* `episode_id`, monotonically increasing `snapshot_id`, server/client tick estimate, wall time, dimension, world seed only if the scenario allows it, position/velocity/yaw/pitch/on-ground, and current path/action status;
* health/max health, food, saturation, oxygen, active effects, fire/lava/fall/void indicators, death/kick state, and current respawn/bed knowledge;
* selected hotbar slot, held item, armor/offhand, complete inventory as `{slot,name,count,metadata/nbt-if-needed}`, open window/container and cursor item;
* nearby entities within a bounded radius as `{id,type,name,position,velocity,health,hostile,visible,distance}` plus online players; use a stable ordering so Jev sees compact repeatable input;
* nearby block observations around the bot and targeted block: block name/state, collision/replaceability, light where available, fluid/lava/fire, and a short diff stream for changed blocks. Maintain a larger server-side world cache for Pathfinder, but do not send an entire chunk to Jev every call;
* time-of-day/day, weather, difficulty/gamerules known to the bot, chat/server messages, inventory changes, damage/death, spawn, block/entity updates, advancements, and completed action results;
* milestone/resource facts calculated by code: logs/food/iron/shield/bucket, portal, gold armor, blaze rods, pearls/eyes, stronghold/portal, End entry, crystals/dragon status, and a risk flag.

Relevant Mineflayer names include `bot.entity`, `bot.health`, `bot.food`, `bot.foodSaturation`, `bot.oxygenLevel`, `bot.inventory`, `bot.heldItem`, `bot.quickBarSlot`, `bot.entities`, `bot.players`, `bot.blockAt`, `bot.findBlock(s)`, `bot.blockAtCursor`, `bot.nearestEntity`, `bot.time`, and `bot.isRaining`. Subscribe to `spawn`, `health`, `move`, `blockUpdate`, `entitySpawn`, `entityGone`, `chat`, `windowOpen`, `diggingCompleted`, `death`, `kicked`, and `error`; verify names against the pinned API version. See the [Mineflayer API reference](https://github.com/PrismarineJS/mineflayer/blob/master/docs/api.md).

Use event-driven updates for hazards and a compact periodic snapshot at roughly 5–10 Hz. A snapshot should be immutable for one Jev question batch. Do not make Jev infer arithmetic such as distances, cooldowns, item counts, or coordinate transforms: code computes those fields first.

### Action contract to implement

Give the model only validated semantic actions, each with an ID, a maximum duration, and a cancellation rule:

* `controls`: hold/release `forward`, `back`, `left`, `right`, `jump`, `sprint`, `sneak`; clear all controls on timeout, death, kick, or hazard;
* `look`: point at a world position/entity or set yaw/pitch; rate-limit impossible rotations and verify line of sight;
* `navigate`: `GoalNear`/`GoalBlock`/safe coordinate with a route profile; stop/replan on lava, void, damage, entity threat, block update, dimension change, or stale observation;
* `dig`, `place`, `activate_block`, `activate_entity`, `use_item`, `attack`, `equip`, and `select_hotbar`; check reach, target state, held item, and inventory before issuing them;
* `craft`, `smelt`, `open_container`, `trade`, `enchant`, `move_slot`, `toss`, and `close_window`; wait for the relevant window/inventory event and compare before/after state;
* `chat` only for ordinary communication. Do not include `/op`, `/gamemode`, `/tp`, `/give`, `/data modify`, or other administrative commands in the gameplay action schema.

The arbiter should reject a command if the bot is dead, disconnected, in a nonmatching dimension, has a stale snapshot, or lacks the required item. It should be able to interrupt any long navigation with an emergency Jev decision, and it should record start/end tick, packet/action result, and failure reason.

## Installation and setup for the next adaptor phase

### Software

* Java 21 (already present) and one exact Mojang Java server JAR compatible with Java 21; use the [official server download](https://www.minecraft.net/en-us/download/server).
* Node.js 24 and npm (already present).
* npm packages: `mineflayer`, `mineflayer-pathfinder`, `mineflayer-pvp`, `minecraft-protocol`; add `prismarine-viewer` only for debugging. Save an npm lockfile and pin versions after the exact server-version smoke test. Mineflayer documents Node.js >=18, so Node 24 satisfies its stated minimum, but the lockfile test remains necessary.
* Optional Python-side reset/status package in the exact evaluator venv: `mctools` (RCON/query/ping) or `rconx` (RCON). Verify installation with `/home/hemlix/GamingMaster-mh-tier2/runs/minecraft/.venv/bin/python`, not the system interpreter. A local server process controller can avoid this dependency.

### Values the user/harness must provide as environment variables

Do not commit a password, token, account credential, or private address to the study files. The adaptor should fail closed when required values are missing.

| Variable | Required | Description |
|---|---:|---|
| `MC_HOST` | yes | DNS name or IP of the private Java server; for the recommended same-machine setup, the user supplies the loopback address. |
| `MC_PORT` | yes | Java gameplay TCP port selected by the server. |
| `MC_USERNAME` | yes | Unique bot username for the private offline-mode server; this is not a Microsoft account credential. |
| `MC_VERSION` | yes | Exact Java protocol/game version matching the server JAR, not `latest`. |
| `MC_AUTH` | yes | Must be the offline-auth setting for this private offline-mode server; do not put Microsoft tokens in the adaptor. |
| `MC_RCON_HOST` | no | Loopback/private RCON host if the reset harness uses RCON rather than the server process stdin. |
| `MC_RCON_PORT` | no | RCON TCP port selected in `server.properties`. |
| `MC_RCON_PASSWORD` | no | User-provided RCON password; never log or store it. Required only when RCON reset/telemetry is enabled. |
| `MC_WORLD_DIR` | no | Isolated world directory used by the reset controller; required if the harness deletes/recreates worlds itself. |
| `MC_SERVER_JAR` | no | Local path to the pinned server JAR if the reset controller starts the server. |

`MC_HOST`, `MC_PORT`, and RCON ports must be reachable only from the local harness or a protected private network. Offline authentication lets anyone claim an arbitrary username, so `online-mode=false` is not a security boundary; use loopback/firewall and do not expose the server or RCON port to the Internet. The [server setup guide](https://minecraft.wiki/w/Tutorial:Setting_up_a_Java_Edition_server) and [RCON documentation](https://minecraft.wiki/w/RCON) describe the relevant settings.

### Setup sequence

1. Create a new server directory outside the bot source, put in the exact server JAR, and run it once with `--nogui`; accept the EULA in the server directory, not by writing a credential into this study.
2. Configure a private Survival server: `online-mode=false`, `gamemode=survival`, an explicit `difficulty`, `hardcore=false`, `allow-nether=true`, `generate-structures=true`, and a blank/local `server-ip`. Choose `level-name` per episode. Decide and record `keepInventory`, `doMobSpawning`, `doDaylightCycle`, and other gamerules rather than inheriting defaults.
3. Bind gameplay and any RCON listener to loopback or a firewall-protected interface. If RCON is enabled, use a strong user-supplied password, `broadcast-rcon-to-ops=false`, and a non-public port. Use an offline username allowlist only as a convenience; it is not authentication.
4. In a clean Node adapter directory, install and lock the four core npm packages. Confirm the selected Mineflayer version supports `MC_VERSION`. Load Pathfinder after Mineflayer and PVP after Pathfinder if combat is enabled. Do not use a GUI or renderer in the normal run.
5. Start the server, perform a status ping, connect the bot with `host`, `port`, `username`, exact `version`, and offline auth, and wait for `spawn`. Test spawn/state JSON, movement and stop, a safe block interaction, inventory serialization, death/reconnect, kick/error handling, and a clean disconnect.
6. Implement reset outside scored play: stop the bot; stop the server cleanly; create a fresh world directory or select a unique world name; set the requested seed/configuration before generation; start the same pinned server; wait for status; connect the bot; verify dimension, spawn, difficulty, gamerules, and empty inventory. Return the initial snapshot and the seed only if the scenario contract permits it.
7. Make the score collector read advancement/milestone evidence after the episode, death and action logs, and server/world evidence. Stop the bot before using RCON commands that could affect gameplay. Never let the model request reset, teleport, item creation, operator status, or gamerule changes.
8. Run calibration episodes on a known seed and then held-out seeds with no Astra call near hazards. Measure actual local RTT, snapshot freshness, server MSPT/TPS, action completion times, disconnect/reconnect behavior, and Mineflayer version compatibility before tuning Jev's cadence.

## Why the alternatives lose

A raw protocol client has lower abstraction overhead but forces the adaptor to rebuild physics, chunks, inventory transactions, and reliable interaction. A server plugin would provide the best telemetry but changes the server and requires a custom Java component before there is a working player. Baritone and a full GUI client introduce a client/mod/display lifecycle that the stated headless Linux setup is specifically trying to avoid. Screen automation is only appropriate when no network/client interface exists. Mineflayer is the smallest mature interface that provides both a JSON-friendly world model and direct actions, while its protocol and viewer can remain available as escape hatches.

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
