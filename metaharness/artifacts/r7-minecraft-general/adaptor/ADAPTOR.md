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

Baseline creation has a per-seed filesystem lock. Failed/incomplete cache creation is not reused. Stopped played worlds are retained for the last three episodes; older world and extracted server-cache directories are pruned, while scenario manifests, score results, inventory journals and logs remain. Baselines are retained separately. `server_owner.py` stops and reaps Java on controller EOF or termination; bridge EOF disconnects the player. Cleanup affects only owned process handles. `close()` is idempotent. Closing an unfinished episode freezes its score and marks it invalid/closed.

## Observations and actions

Observations are JSON dictionaries, typically tens of kilobytes, with no binary data or screenshots:

* Episode/sequence, observation freshness, elapsed/remaining wall time, terminal status and trial validity.
* Player coordinates, velocity, orientation, ground state, health, hunger, saturation, known oxygen, equipment, dimension and game mode.
* **Authoritative personal `inventory`** plus explicitly labelled predictive `inventory_client`; never use the latter as scoring evidence.
* Day/weather, last server-time packet age, client physics count, and nearby collision labels. Collision entries are `[dx,dy,dz,label]` relative to the player's floored position, for dx/dz −2…2 and dy −1…2; labels are solid/passable/hazard/unknown and do not reveal buried material identities.
* Bounded nearby entity summaries and exposed resource/workstation candidates within 24 blocks; local stand candidates up to six blocks away. Exposure means a loaded adjacent air/passable cell, not necessarily visibility from the current eye. Thus this is structured client perception, not pixel-equivalent vision. Hidden solid ore interiors and admission scans are not disclosed. Unknown chunks are omitted and not traversed.
* Feasible recipe names, last-observed furnace contents/progress (explicitly not live after closing), active job/deadline, latest results and score components/milestone times.

Snapshots are pushed about 5 Hz, with resource candidates refreshed about once per second and after actions. Data can become stale. Candidate/action preconditions are rechecked against the live client cache. Pathfinding can fail or a drop can be unreachable; these produce feedback, not invented success.

`describe()` gives the policy the complete rules, score formula, observation meanings and all action syntax. `action_spec` includes these 14 names and their one-line meanings:

| String grammar | Meaning |
|---|---|
| `noop` | Continue a bounded job, otherwise stay still and release shield. |
| `cancel` | Cancel the current job and immediately clear movement, digging, item use and open windows. |
| `navigate CANDIDATE_ID` | Walk to a listed safe stand or block interaction position. |
| `look_at X Y Z` | Look at a point within 32 blocks; reject known Enderman gaze. |
| `gather BLOCK_ID [N=1]` | Mine and collect 1–4 exposed blocks of that type, selecting an available suitable tool. |
| `equip ITEM [DEST=hand]` | Equip to hand, off-hand, head, torso, legs or feet. |
| `craft ITEM NEW_COUNT [TABLE_ID]` | Request new output items, rounded to whole recipe executions; optional observed table. |
| `place ITEM REFERENCE_BLOCK_ID FACE` | Place on up/down/east/west/north/south reference face. |
| `furnace_start ID INPUT COUNT FUEL COUNT` | Load actual carried ingredients/fuel, then close the furnace; smelting continues in real time. |
| `furnace_take ID` | Retrieve actual finished furnace output. |
| `eat ITEM` | Consume carried food when hungry. |
| `attack ENTITY_ID` | One visible, in-range attack with a conservative weapon cooldown. |
| `shield on\|off` | Equip/use a shield off-hand or release it. |
| `wait [SECONDS=0.5]` | Wait 0–5 seconds while the world continues. |

Candidate IDs (`b:x,y,z` or `s:x,y,z`) must come from observations. A bare parameterized action is rejected through structured feedback, allowing the generic validator to probe every action name safely. Unsupported input cannot become JavaScript, raw packets, console commands, teleportation or creative inventory operations.

Only one job owns movement/hands/windows. Deadlines are 25 seconds for gathering, 15 for navigation, and 10 for other jobs. A different command while busy is rejected; use `cancel` and wait until `task` is null. Identical current strings are latched, so a repeated mailbox value cannot repeatedly craft after finishing; send `noop` before a deliberate repeat. `noop` lets a previously accepted bounded job finish. Optional `{episode, seq, action}` envelopes reject earlier episodes, future sequences and observations more than ten snapshots old.

Navigation disables autonomous digging, bridge placement, towers, parkour and sprinting; drops are limited to two blocks, with liquid/hazard avoidance. Block interaction uses an eye-position ray and reach check. Gathering requires a suitable tool, refuses mining the player's support or exposing adjacent hazards/falling blocks, and checks actual pickup. Cancellation stops game controls, interrupts interruptible waits/pathfinding/digging, prevents subsequent cancelled inventory clicks, and serializes pending completion. If a library operation cannot settle within its cancellation grace, the trial is invalidated rather than allowing an old job to resume into a newer one. A 50 ms local heartbeat cancels actions on immediate liquid/fire exposure and attempts to surface in water; it is not a complete combat or rescue policy.

## Authoritative score

`score` is an absolute float, clamped to [0,100]. `info()['progress']` and observations report all components, resource maxima and first-possession times.

Resource points are `min(logs_max,10) + 0.5*min(cobblestone_max,20) + min(coal_plus_charcoal_max,5) + min(raw_iron_max,5)`, at most 30. Each maximum is of the **simultaneous held total**, not lifetime collection. Logs are oak, spruce, birch, jungle, acacia, dark_oak, mangrove and cherry unstripped logs only. Coal and charcoal share one simultaneous total.

Possession milestones: crafting table 5; wooden pickaxe 5; stone pickaxe 10; stone axe 5; furnace 5; torch 5; cooked food 5; iron ingot 5; shield 5 (total 50). Cooked food is exactly cooked beef, porkchop, mutton, chicken, rabbit, cod, salmon, or baked potato. Legitimate loot counts. Spent or dropped items retain their earned maxima/milestones. Dropping/recollecting, equipping and window transfers do not give duplicate credit.

Add `10 * seconds_survived / 300`. Only an alive time-limit finish adds `10 * final_health / 20`. Terminal death subtracts 20. Full healthy idling earns exactly **20**, leaving 80 for progress. Score and final health freeze at the terminal boundary. Infrastructure errors set `valid_trial=false`; their numeric contract score must not be included in evaluation averages.

The trusted Python process obtains atomic, read-only `data get entity PLAYER Inventory` responses about 10 Hz, plus packet-triggered and serialized inventory-action barriers. Crafting is divided into recipe executions with server verification between them. This avoids trusting optimistic client windows. Only NBT personal slots 0–35, armor 100–103 and off-hand −106 count. Cursor items, crafting inputs/previews, furnace/container slots and items nested in container components are excluded. Inventory snapshots and monotonic times are journalled per episode. As with any network observation, sub-tick changes not exposed before disconnect cannot be reconstructed; stale audit/bridge/server-time streams invalidate the trial.

## Verification and limits

See `VALIDATION.md` and the retained `runtime/*-test.*` artifacts for measured results. Test scripts use the actual installed server. `test_live.py` uses natural survival actions only. `test_fixture.py` builds a disposable mechanics fixture **before** its test clock and is explicitly not a natural benchmark score. `test_idle.py` runs the full unshortened 300-second clock; `test_lifecycle.py` deliberately terminates owned processes to test invalid-trial handling. `test_scoring.py` checks hand-calculated inventory traces.

This adaptor supplies bounded actions, not a Minecraft strategy agent. Conservative movement can reject feasible routes, especially around overhangs, water, narrow passages or unreachable drops. There is no visual renderer, free-form chat, chest interface, arbitrary command endpoint, deep exploration planner or guaranteed survival reflex. Network/client snapshots may lag. Natural seeds and ordinary mob/drop randomness require repeated matched evaluation, and arbitrary cold/rejected seeds can exceed the generic validator's overall time budget. Prequalify evaluation baselines first.

Upstream references: [Mineflayer API](https://github.com/PrismarineJS/mineflayer/blob/4.39.0/docs/api.md), [pathfinder](https://github.com/PrismarineJS/mineflayer-pathfinder/tree/2.4.5), [Mojang 1.21.1 release](https://www.minecraft.net/en-us/article/minecraft-java-edition-1-21-1).
