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
