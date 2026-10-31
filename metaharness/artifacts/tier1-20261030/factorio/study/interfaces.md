# Factorio programmatic interfaces and adaptor specification

Target: offline Linux x86_64, no GUI; evaluator Python **3.12.3** at `/home/hemlix/GamingMaster-mh-tier1/runs/venvs/factorio/bin/python`. Study date: 2026-10-29. **Web retrieval failed at DNS. All source URLs are reference targets, not freshly verified pages.** See [the evidence ledger](research/evidence.md). In particular, no installed Factorio server or FLE checkout was available for an integration test.

## Recommendation and licensing boundary

Use the **official, zero-cost Linux Factorio headless server**, a small trusted **Lua control mod**, and **local Source RCON**, with a Python adaptor that exposes typed JSON actions. Pin one base-game **2.0.x** build and disable expansion mods. The mod is to be written in the adaptor phase; it does not exist yet. This is a modified, remote-manager factory benchmark whose production still runs in the actual Factorio engine. It deliberately has no GUI, human client, account login, cloud service, or language-model API dependency.

Factorio's server is proprietary even though its headless download is free. This recommendation interprets the user's preference for that server as allowing zero-cost proprietary software. If “free software” means exclusively libre/open-source licenses, **Factorio itself cannot satisfy the requirement**; FLE and an open-source RCON wrapper do not change the engine's license. Confirm the downloaded archive's applicable terms. A paid graphical client or Space Age purchase must not become an implicit prerequisite. [I1–I3]

## Practical interface inventory

Latency figures describe expected classes, not measurements on this machine. A local RCON exchange should usually be milliseconds to tens of milliseconds; large maps, Lua scans, and JSON serialization can make it much slower. Simulation wall time is separate from request latency.

| Interface | Maturity and exposure | Timing and constraints | Cost / license / suitability |
|---|---|---|---|
| **Headless + RCON + trusted Lua mod** — selected | Official server, console, runtime scripting and mod APIs are established; the new adaptor needs implementation. Reads actual entities, terrain, inventories, recipes, force research, fluids and production counters. Can schedule exact game operations. | No rendering. Explicit engine pause and tick advancement must pass the tests below. Small local requests; simulation throughput depends on factory size. RCON alone is a console transport, not a game-state JSON API. | Zero purchase cost for the headless engine; proprietary terms apply. User's own offline server has no third-party bot service or anti-cheat requirement. A freely licensed custom mod can supply the bridge. [I1–I5] |
| **Factorio Learning Environment (FLE)** | Research environment built around a real Factorio server and programmatic factory tools. Published examples include inspection, resource finding, placement, crafting, recipes, inventory movement, movement, and evaluation/production facilities. Source exists, but current API/version support was not retrieved. | Convenient higher-level Python tools and structured results; action time costs, sleeps, `eval` calls, game speed, reward, and pause semantics depend on revision. Docker/process startup can dominate reset. Do not assume Gymnasium-style `reset/step` or deterministic fixed ticks. | Framework and dependencies must have their licenses inspected. Factorio remains proprietary. Offline execution needs all engine, image, Python, and source artifacts staged; evaluation/model integrations are optional and should be excluded. Best fallback if a known working checkout is supplied. [I6] |
| **RCON Python libraries** (`rcon`, `factorio-rcon-py`) | Reusable clients for authentication, request IDs and command responses; not factory environments. Need the same Lua bridge for state and mechanics. | Low overhead; check multipart response handling, persistent connections, timeouts and Python 3.12 compatibility. | No paid service; verify pinned release license/dependencies. Optional convenience, not required by the selected standard-library client design. [I7] |
| **Lua scenario instead of a mod** | Official scenario `control.lua` can own map creation, events, scoring and a remote interface. Similar capabilities to a mod; useful for completely custom maps and rules. | Requires correct scenario packaging and lifecycle. RCON still supplies external control. A scenario or mod is preferable to a long sequence of ad hoc console mutations. | Local script can be freely licensed; engine terms unchanged. We choose a control mod so normal `--create` / `--start-server` saves remain usable. [I4–I5] |
| **Console `/c` or `/sc`, `remote.call`, server stdin, logs** | Official privileged Lua/console access; useful for installation diagnostics and trusted administration. `remote` is an in-engine interface between scripts, not a TCP/HTTP server. `script-output` files can export snapshots. | Stdin lacks a clean request-response protocol; logs/files need framing and polling. Console Lua may disable achievements and can trivially grant items. | Suitable only behind a trusted adaptor. Do not expose arbitrary Lua or shell commands as game actions. Local benchmark score is independent of Steam achievements. [I4–I5, I8] |
| **Docker server packaging** (e.g. factoriotools) | Common server packaging with mounted saves/config/mods. Can host either FLE or a custom bridge. It is not another observation API. | Needs a working daemon and preloaded exact images for offline use; permission/network/mount errors add setup risk. | Wrapper license differs from engine license. No Docker dependency in the chosen design. Docker CLI presence does not prove daemon access. [I9] |
| **Graphical client + screenshots + input injection** | The human UI exposes the full normal game. Could use X11/Wayland automation, screenshots and OCR, but no native typed factory observations. | Frames and input delivery depend on a running graphical stack; headless server renders no screenshots. Long mouse sequences are fragile. | Full graphical client normally requires ownership; a virtual display does not supply a license. Contradicts the no-GUI, zero-purchase-cost preference. [I1–I3] |
| **Free demo / tutorial automation** | Limited free content rather than the full headless factory environment. No verified general offline Python environment for it. | Different scenarios and progression; GUI/runtime capabilities must be separately investigated. | Not an equivalent replacement for this evaluation. [I2] |
| **Save-file parsing, memory hooks, network bots, emulation, external factorio planners** | Saves are good reset artifacts but parsing internals is not a supported live control API. An ordinary multiplayer bot must implement a version-specific protocol. Calculators such as FactorioLab solve rates, not the game. | High maintenance for binary hooks/protocols; planners are useful offline support tools but cannot observe authoritative state or execute actions. | No need for these approaches when official scripting is available. Do not mistake a recipe calculator or simulator for the actual engine. [I4, I10] |

## Software and offline artifacts

Required:

1. Official **Linux x86_64 headless archive**, one exact base-game 2.0.x version; extract within this workspace. Record filename, version output, SHA-256 and terms. Do not use a moving `latest` URL in the final reproducibility manifest. The exact release is intentionally unresolved because downloads were inaccessible.
2. Host runtime libraries required by that binary. Host x86_64 and glibc 2.39 are confirmed; run the archive's executable with `--version` / `--help` and check its shared-library dependencies after staging. No GUI library stack should be necessary for the headless distribution.
3. Evaluator Python **3.12.3**, already present. A Source RCON client can use Python's `socket`, `struct`, `json`, `subprocess`, `pathlib`, `hashlib`, and `time`; **no pip dependency is mandatory**. The game simulation and Lua runtime come from the server.
4. A new `study-control` Lua mod, matching the selected Factorio major version, with a registered console command and tick controller described below. Use normal engine recipes and entities; no prototype modifications are necessary.
5. Version-matched runtime API docs / JSON, server-settings example, map-generation example, map-settings example, plus offline copies of any strategy references wanted at runtime. Use docs shipped with the chosen release if available, or stage the matching version from the official API site. [I1, I4–I5]

Optional: a pinned RCON wheel and its entire dependency closure, with license and hashes, tested on CPython 3.12.3. If chosen later, install with `/home/hemlix/GamingMaster-mh-tier1/runs/venvs/factorio/bin/python -m pip install --no-index --find-links <staged-wheelhouse> <pinned-package>`. Never install into system Python. The current task's sandbox permits writes only in this study directory and `/tmp`; modifying the sibling evaluator venv would require a later execution context that permits it. No installation was needed or attempted here.

All artifacts must arrive from an already available local bundle or a separately network-enabled provisioning phase. A runtime `pip install`, `git clone`, Docker pull, mod-portal login or download is incompatible with the offline requirement. This machine currently has neither a verified engine archive nor the required docs.

## Configuration and credentials

These environment names are **our proposed adaptor contract**, not names that Factorio or FLE automatically reads.

| Variable | Required? | Meaning |
|---|---|---|
| `FACTORIO_BINARY` | Yes | Absolute path to staged `factorio/bin/x64/factorio`. |
| `FACTORIO_WRITE_DIR` | Yes | Absolute writable directory for saves, config, mods, logs and script output; use a child of this study directory. One directory per parallel server. |
| `FACTORIO_RCON_HOST` | No | Local server address; default `127.0.0.1`. |
| `FACTORIO_RCON_PORT` | No | TCP port; default `27015`, unique per concurrent worker. |
| `FACTORIO_RCON_PASSWORD` | Yes | A locally generated RCON secret. Pass at process launch; never put its value in study files, observations, traces or committed configs. No external account credential is needed. |
| `FACTORIO_GAME_PORT` | No | UDP game port; default `34197`, unique per concurrent worker. No graphical player connects. |
| `FACTORIO_ARTIFACT_MANIFEST` | No | Path to locally staged binary/mod/docs hashes and scenario ruleset metadata. |

No Factorio username, multiplayer authentication token, Steam credentials, OpenAI key or other model-provider key is required by the selected game adaptor. Disable public/LAN listing and online user verification for this private local server. Keep RCON reachable only locally; determine supported bind flags from that binary's `--help` instead of assuming a flag from another version. RCON exposes an administrative console and is not an encrypted remote service. [I1–I5]

## Installation and server lifecycle

This is the concrete work for the adaptor phase; no following commands or game scripts have been executed in this study.

1. Stage and verify the binary; run `--help`. Store an explicit `config.ini` with `read-data` pointing to the archive's `data` directory and `write-data` pointing to `FACTORIO_WRITE_DIR`. Use absolute paths to avoid home-directory defaults.
2. In `<write-dir>/mods/study-control_0.1.0/`, create `info.json` and `control.lua`. Mod metadata declares a 2.0-compatible `factorio_version`, a dependency on `base`, and no expansion dependency. Enable `base` and `study-control` in `mod-list.json`; disable `space-age`, `quality`, `elevated-rails` if present. Pin all actual enabled mod versions. Implement control logic described below using 2.0's `storage`, not old `global` examples.
3. Start from the selected binary's example settings. Set server visibility public=false and lan=false, `require_user_verification=false`, and **`auto_pause=false`**. The mod owns pausing; the “pause when nobody is connected” setting would otherwise prevent intended stepping. Omit account/token fields. Use no password value in the study's examples.
4. Produce a per-seed map-generation JSON with finite iron, copper, coal, stone, access to water and trees. Disable enemy-base generation; disable enemy expansion/evolution and pollution in map settings for the primary family. Peaceful mode alone is insufficient to remove enemies or stop simulation. Pre-generate the evaluation area before the initial snapshot and validate starts as described below.
5. Create a save using the enabled mod and settings; then run it as a dedicated server. Representative CLI shapes (check with the staged binary):

   ```text
   <factorio> --config <config.ini> --mod-directory <mods-dir> --create <seed-save.zip> --map-gen-settings <map-gen.json> --map-settings <map-settings.json>
   <factorio> --config <config.ini> --mod-directory <mods-dir> --start-server <seed-save.zip> --server-settings <server-settings.json> --port <game-port> --rcon-port <rcon-port> --rcon-password <secret-from-environment>
   ```

   The adaptor should launch an argument vector without shell interpolation and redact the password from logs. Verify the accepted options rather than assuming every release uses identical flags. Do not create the initial save without the mod and then accidentally load it with a different mod set.
6. Connect RCON and complete a `hello` exchange returning engine version, bridge schema version, ruleset ID, seed, mod manifest, tick, phase and pause state. Initialize/baseline all score counters **after** map setup. Save the paused initial state for reproducible resets, or re-create it deterministically.
7. Run all acceptance checks below before evaluation. Reset by stopping the previous process and starting a pristine per-seed save, clearing Python caches and session IDs. Restoring only inventory is not a reset: machines, fluids, terrain, queues, technologies, counters and RNG also matter.

## Trusted mod and RCON protocol

Register a command named `study` through the official `commands` API. Accept console requests in the form `/study <one-line-JSON>`; parse with the pinned version's JSON helper and return JSON through `rcon.print`. In Factorio 2.0 the relevant helpers are expected under `helpers`; verify signatures in the staged API. The names below are **new bridge methods**, not existing FLE methods.

Use a JSON envelope with `protocol_version`, `episode_id`, `request_id`, `method`, and `params`. Step requests also carry `expected_revision` and `step_id`. Responses include IDs, phase, tick, revision, success/errors and result. Reject unknown fields where ambiguity would hide an error, nonfinite numbers, invalid counts, excessive payloads, inaccessible coordinates and stale episode/revision IDs.

- `hello`: version/ruleset/seed handshake. Read-only.
- `observe`: compact summary, paged entity/terrain/prototype requests, production and research details. Read-only; while running, return status rather than a mixed-time state.
- `submit_step`: at a paused boundary, apply a validated bounded action batch, acknowledge it, and schedule the fixed simulation interval.
- `step_status`: query whether the acknowledged step has reached its target tick; return the final revision only after it pauses.
- `reset` is owned by the Python process manager, not an unrestricted agent action. Administrative bootstrap and save commands likewise stay outside the action space.

Use standard Source RCON framing (little-endian length, request ID and packet type, NUL-terminated body plus terminator), authentication and bounded response reads, or a tested library. Factorio console output is text; Python parses only the tagged JSON envelope for the matching request. Page large results into conservatively small responses (e.g. roughly 2 KiB ASCII JSON bodies), verify response IDs/completeness, and test multipart handling. Never assume one TCP read contains an entire reply. Restrict snapshots by area/fields so dumping every entity does not dominate inference time. [I5, I7]

Keep durable episode state in mod storage: schema/ruleset, force and surface identities, permitted area, stock inventory, manual queue, step ID, target tick, revision, transfer quota, production baseline, and a bounded request/result deduplication table. Register commands/events/interfaces in the mod lifecycle correctly; do not reinitialize saved state on every load. There is no connected human player: `game.player` may be nil. Use the explicit player force, surface and a script-owned inventory; no avatar or GUI is required.

## Tier-1 clock contract: release gate

Recommended fixed budget: **128 turns, 1,800 simulation ticks per turn**, at 60 ticks per simulated second. Thus the episode contains 64 simulated minutes. Up to 32 typed operations can be submitted in one turn. These are benchmark choices, not vanilla defaults. Every accepted turn, even an empty wait or a turn with individually failed actions, uses the same tick budget. Inspection does not consume a turn.

1. End initialization with `game.tick_paused=true`, an empty pending-step record and phase `paused`. Ensure no built-in empty-server auto-pause interferes.
2. Validate the step envelope before changing anything. During this paused command handler, perform bounded instantaneous placement/configuration/transfers and queue manual jobs. Set `target_tick = current_tick + 1800`; persist the step ID, change phase to `running`, and unpause.
3. The mod's `on_tick` handler advances virtual manual jobs by actual elapsed ticks and watches `target_tick`. At the target, finish due work, pause using the engine flag, finalize the authoritative observations/counters, increment the revision and mark the step complete. Test event-order effects at the boundary; scoring must be sampled consistently.
4. The submitting RCON handler returns an acknowledgment; Python polls `step_status`. Do not hold an RCON callback open waiting for `on_tick`, and do not expect an `on_tick` callback to respond to an old RCON request.
5. While Astra reasons, simulation tick, production, research and enemies must remain unchanged. A proposed 3,600-second decision timeout is a harness resource limit, **not a gameplay deadline**. A verified paused simulation has no natural finite decision deadline.

`LuaGameScript.tick_paused`, `tick`, and the relevant event behavior must be checked on the selected **dedicated multiplayer server**; do not assume a single-player example proves server compatibility. `ticks_to_run` may be a useful alternative if documented and tested for that build. Never approximate this contract with Python `sleep`, setting a small nonzero `game.speed`, or a periodic wall-clock pause: those give different production to slow and fast reasoning. `game.speed` may accelerate an active interval after invariance tests, but it does not define the amount of simulated time. [I4]

If paused RCON/resume or exact target stopping fails, the selected tier-1 bridge is not ready. Fix the bridge, or investigate a known FLE build with proven stepping; do not silently evaluate a continuously running factory as tier 1.

## Actions and physical/economic rules

The evaluated agent is a **remote factory manager** over an initially revealed finite area, initially `x,y ∈ [-128,128)`. Actions do not require a walking avatar. This convenience must be disclosed in the benchmark manifest. No arbitrary item spawning, research unlocking, terrain/resource editing, force bonuses, infinite energy, console execution or unrestricted teleporting belongs in the action set.

Initial shared stock: 1 burner mining drill, 1 stone furnace, 8 iron plates and 1 wood. Use one normal base-game force, normal-quality recipes/items, ordinary mining/crafting speeds and no bonus technologies. Store stock in a real Lua inventory with documented capacity (recommend 200 slots); expose actual free slots and use stack sizes from prototypes. Stock capacity/overflow may not silently delete or duplicate items.

| Typed action | Arguments and semantics |
|---|---|
| `mine_resource` | Resource entity ID and positive cycle count; queue bounded manual work for iron ore, copper ore, coal or stone. Only accessible real deposits; each completed cycle removes actual resource amount and credits only the base recipe's product. Time comes from the resource's mining time and the baseline character mining speed in prototypes. Reject fluids, infinite resources and unsupported products. |
| `mine_tree` | Tree entity ID; queue its normal mining time, then mine the actual tree into stock. Use the real entity's mining products; do not award wood once per retry. |
| `handcraft` | Enabled hand-craftable recipe ID and recipe-execution count. Inputs must already be in stock; **no recursive automatic subcrafting**. Reserve inputs, queue serial work using recipe energy and the baseline character crafting speed, and credit completed outputs exactly once. No fluid recipes. For example, one copper-cable craft yields multiple items; counts refer to recipe executions. |
| `place` | Item name, position, direction and optional explicit entity name where an item has multiple place results. Validate force permissions, allowed area, collision, terrain/water and stock. Consume the actual place item only if placement succeeds. Use engine placement checks and creation; script creation alone does not charge stock. Verify ordinary neighbor connections, especially pole wires, belts and pipes; explicitly reproduce normal permitted connections if script creation does not do so automatically. |
| `rotate` | Entity ID and legal direction/rotation. Update the actual engine entity; report the resulting orientation. Respect limits on nonrotatable entities. |
| `set_recipe` | Entity ID and enabled compatible recipe. Validate crafting category, ingredient/recipe restrictions and normal quality. Return displaced ingredients to bounded stock, charging the manual-transfer quota, or reject if stock space/quota is insufficient. Do not silently discard ingredients during routine configuration. |
| `insert` / `extract` | Entity ID, explicit inventory role, item and positive count. Transfer existing physical items to/from stock using engine inventory operations. Valid roles include fuel, furnace source/result, assembler input/output and lab input as appropriate. Return exact moved counts. No direct fluid insertion, remote belt-item injection or arbitrary inventory index chosen by the model. |
| `mine_building` | Owned entity ID; return normal mined building items and contained items to stock if capacity permits. Charge contained cargo against the manual-transfer quota; reject if quota is insufficient, so mining a chest cannot bypass the transfer limit. Follow engine mining behavior, including loss of fluids/partial processes where applicable, and report losses. No new-production reward for returned buildings. Restrict ownership and reach area. |
| `set_research` | Enabled technology ID with prerequisites satisfied. Choose one current technology; retain ordinary engine progress behavior when switching, report progress. Do not grant completion or bypass prerequisites. |
| `wait` | Empty batch / no operation; factory still advances the fixed interval. |

Placement, rotation, recipe selection, research selection and inventory handling are instantaneous at a boundary; their frequency is limited by the 32-operation cap. Manual mining/tree mining/handcrafting use **one serial worker queue** concurrently with the automated factory. Bound pending manual work to at most 60 simulated seconds, including unfinished work, and at most 32 queued jobs; reject requests beyond these limits. Unfinished work persists across turns. This prevents unlimited instant manual production. Reserve handcraft inputs at enqueue, finish outputs at the modeled completion tick, and expose blocked jobs if output stock is full. Cancellation/refunds need a precisely specified contract if added later; omit cancellation from the first version.

Manual stock↔machine movement has a shared **100-item limit per turn**, counting both insert and extract quantities, cargo recovered when mining a building, and ingredients returned by recipe changes. Only the actual building item returned by mining is exempt, along with building items consumed by placement and genuine manual mining/crafting outputs; these have their own rules. Apply the quota to actual moved items, not requested quantities. This is a deliberately generous remote-handling concession for bootstrapping, not vanilla logistics. Expose transfers per episode as a diagnostic. Belts, inserters, chests, pipes and machines do unlimited ordinary engine handling while time runs.

Agent-visible action failures are precise: insufficient stock, full destination, unknown/stale entity, out of area, locked/incompatible recipe, blocked placement, queue/quota limit, or wrong phase. Prevalidate the envelope; within an accepted batch execute operations in order with **per-operation success**, not pretend all-or-nothing rollback. Successful earlier operations remain committed. Record results and advance the turn. Syntactically invalid or stale envelopes mutate nothing and do not advance time; bound retries to avoid unbounded evaluation work. Return stable bridge entity IDs even for entities without an engine `unit_number`.

Virtual manual crafting is a benchmark convenience. It must consume real ingredients and time, but script-inserted outputs need not appear in engine production statistics. Keep a separate audited manual-output ledger and **exclude it from scored production**. Do not forge statistics via `on_flow` to make manual bookkeeping look like engine production. Automated mining, smelting, assembling and research are executed by the real engine.

## Observation contract

Every final observation is JSON tied to one paused tick and revision. Provide a small summary plus read-only detail queries:

- Episode: version/seed/ruleset, step and remaining steps, tick, phase, action limits, last action outcomes, remaining quota and manual queue with reserved ingredients/completion estimates.
- Stock: counts by item and quality, stack/capacity information. IDs and quantities must be exact integers; distinguish stock from items still inside machines or travelling on belts.
- World: revealed tiles, water/coast, trees/rocks, resource type/amount/position; buildable bounds, occupied footprints, and stable IDs. Pre-generate/reveal a bounded area rather than silently expose an unlimited map.
- Entities: prototype, position, orientation, bounding box, health, force, actual status, recipe, progress, inventory roles/counts, burner fuel, fluidbox contents, and relevant connections. Query detailed belt-lane contents or power-network data only when needed.
- Power: connected networks/consumers, generation and demand/satisfaction where exposed. An entity's stored `energy` alone is not proof of full network supply. Keep the API's units explicit.
- Recipes/prototypes: ingredients, product quantities/probabilities, recipe seconds, categories, unlocks, machine crafting speed, mining speed, power/fuel, footprint, fluid ports and item stack sizes. In 2.0 these are accessed through the prototype API/global mappings rather than blindly reusing 1.1 `game.*_prototypes` examples. Cache immutable data outside repeated observations.
- Research: currently selected technology, prerequisites, remaining pack requirements, progress, completed technology IDs and available recipes. Include partial research to avoid losing credit for useful work near the horizon.
- Production: baseline-adjusted cumulative **produced** and **consumed** quantities, plus tick-normalized rolling deltas, scored components and excluded manual output. Rates computed over zero elapsed ticks are undefined, not infinite.

Use the official entity/surface/force/inventory/prototype APIs to build these views. State exact API field names in the implementation only after reading the pinned version. Notably, 2.0 force production statistics may require per-surface getter methods; do not copy old `.item_production_statistics` code unexamined. Query only the evaluation force and surface and normal-quality items. [I4]

## Score and exploit-resistant accounting

No universal vanilla scalar “factory score” exists for this task. Define the following **benchmark score**, and expose its components. Let `P(i)` be the cumulative nonnegative count of item `i` actually produced by the evaluation factory since the reset baseline, using engine flow statistics. Let `L(i)` be science packs actually consumed for research, checked against lab consumption and research changes. Counter baselines are taken after initialization, not after the first action.

For science types automation, logistic, chemical, production, utility and space, assign weights `1, 2, 6, 12, 12, 30` respectively:

```text
B = .01*min(P(iron-plate),1000)
  + .01*min(P(copper-plate),1000)
  + .02*min(P(iron-gear-wheel),200)
  + .05*min(P(electronic-circuit),200)

score = B + Σscience_i weight(i) * (P(i) + 4*L(i))
```

The bootstrap cap is **34**. Production of science is useful progress; actually consuming it in research earns additional credit. These weights are policy choices, not exact ore costs. For the primary 64-minute cold-start family, red/green science is the realistic focus; space science is optional for later-start families. Report final score, per-step delta, raw production/consumption, research IDs/progress, manual-output ledger, transfers, ticks and elapsed wall time. Score at the fixed horizon, with no uncharged final production interval.

Read cumulative counters (`LuaFlowStatistics` input/output counts using the pinned API), not GUI average graphs, net inventory, or wall-time-dependent rates. Test whether “input” means production and “output” means consumption in the relevant statistics object before wiring names. Science consumption must not include packs deleted, discarded or removed by scripts; verify research consumption behavior on that build. In this base-only ruleset science packs have no alternative allowed manufacturing sink, simplifying attribution. If this assumption changes, explicitly instrument authorized laboratory consumption.

Moving items between inventories, repeatedly mining/replacing a building, reset stock, console insertions and virtual manual output must never increase `P`. Leave built-in production counters untouched. Validate these invariants with fixtures; do not merely rely on a field name. A non-monotonic cumulative counter or a broken baseline is an error, not something to hide with unconditional clamping. Read the engine's actual completed recipe outputs, including multi-output recipes and any future supported productivity, without double-counting ingredients.

FLE's production score, if adopted instead, must be inspected in the pinned source for prices, baselines, inventory dependence, time behavior and handcraft accounting. Do not label the formula above “the FLE score.” [I4, I6]

## Seeded resets and meaningful held-out scenarios

Keep the ruleset fixed across a comparison. Derive the engine map seed and bounded map-setting choices from the evaluation seed. Use three separately reported families: (A) cold start from the stock above; (B) a small powered but unbalanced red/green factory with a reproducibly selected bottleneck; (C) a later starter with specified recipes/research unlocked, oil access and a finite inventory. Family B/C state creation is privileged reset work and excluded from score baselines. Do not pool their raw scores with family A without normalization.

For family A, vary patch shape, distance and relative direction, resource richness within a documented range, water/coast layout, trees and available build corridors. Reject maps missing accessible iron/copper/coal/stone, harvestable trees, a viable offshore-pump site and useful construction space within the allowed area. Pre-generation, validation and deterministic retry-seed derivation must be reproducible. Keep total accessible resources comfortably above the planned horizon's needs; do not let unlucky generation make an episode impossible. Validation guarantees feasibility, not an identical optimal layout.

Held-out seeds must change these physical constraints, not just a seed label over the same save. Snapshot initial entities, technologies, queues, stock, RNG state, counters and ruleset; do not carry learned-in-episode stock/research between resets. A future combat family needs defenses and enemy settings in its own manifest and score normalization. It can remain tier 1 with the same pause contract.

## Acceptance checklist before connecting Astra

1. **Offline boot:** with networking unavailable, create/load a local save and authenticate RCON, with no graphical client or account login.
2. **Paused observation:** two queries separated by at least 60 wall seconds show identical tick, inventories, production and research. Both queries work while paused.
3. **Exact step:** 1-tick, 60-tick and 1,800-tick diagnostic intervals stop at the exact target, including an otherwise empty server. Only 1,800 is exposed in scored evaluation.
4. **Timing invariance:** identical action traces with different inference delays and tested simulation-speed settings yield the same authoritative factory state/score at corresponding ticks. Normalize session-specific IDs when comparing.
5. **Economy:** blocked placement consumes nothing; successful placement consumes the right item and creates expected normal connections; recipes enforce unlocks and ingredients; full inventories and failed transfers duplicate nothing; manual jobs respect actual work time and stock reservations. Mining a loaded chest or changing recipes cannot bypass the transfer quota.
6. **Production/research:** a known fueled furnace produces the expected plates over enough ticks; an assembler records the actual output count; a powered lab consumes valid science and advances a chosen technology. Establish baselines after scripted setup; don't count those inputs as earned progress.
7. **No free reward:** transfer/extract/insert, mine/rebuild and handcrafted bootstrap operations cannot create scored production. Duplicate network delivery of the same step neither repeats actions nor advances another interval.
8. **Recovery/reset:** disconnect after step acknowledgment, reconnect, and discover that step's result without re-execution; reloading the initial save reproduces the seed and clears the previous episode; a crash is reported as infrastructure failure rather than a clean zero-score game.
9. **Observation consistency:** entity deletion invalidates its ID; stock/inventory totals agree with the engine; recipe counts and units are correct; large observations are paged without truncation or mixed revisions.
10. **Budget enforcement:** turn count, allowed area, operation count, manual queue time and transfer quota are enforced by the trusted mod, not only by prompts. Final score uses exactly the charged simulation ticks.

## FLE fallback: what to inspect before choosing it

Use the repository at [I6], record a commit and license, and read the actual README, Python metadata and lockfiles. Locate its server/bootstrap assets, Factorio version, container tags, save files, RCON setup, tool implementations, serialization/reset paths and score calculation. Confirm whether it needs a 1.1-era engine or supports the intended 2.0 build; do not mix a current server with an old Lua toolset. Make a small capability matrix mapping its actual methods to observation, crafting, mining, placement, insertion/extraction, research, stepping, reset and score.

Check distribution/import names rather than assume `pip install fle` installs this project. If its declared Python version excludes 3.12, that checkout is incompatible with the evaluator unless a tested compatibility change is made; silently switching Python is not allowed. Stage wheels for this platform/interpreter, exact container images if needed, Factorio binaries, and example saves. Avoid optional agents/LLM runners that demand API keys. Use only the local environment/tool layers, and run the same acceptance tests above.

FLE may offer stronger native avatar/movement semantics and reduce implementation effort. Its version-sensitive dependency footprint and unverified time/reward semantics are why the smaller direct bridge is the study's default. No FLE method signatures or version pins are presented as verified facts here.

## References

All links were **unretrievable in this session**; validate against the staged version.

- I1: https://factorio.com/download-headless — official headless download.
- I2: https://www.factorio.com/download — downloads, demo and platforms.
- I3: https://www.factorio.com/terms-of-service — applicable proprietary license/terms; inspect the archive too.
- I4: https://lua-api.factorio.com/latest/ ; https://lua-api.factorio.com/latest/classes/LuaGameScript.html ; https://lua-api.factorio.com/latest/classes/LuaForce.html ; https://lua-api.factorio.com/latest/classes/LuaFlowStatistics.html ; https://lua-api.factorio.com/latest/classes/LuaEntity.html ; https://lua-api.factorio.com/latest/classes/LuaInventory.html ; https://lua-api.factorio.com/latest/classes/LuaSurface.html — use matching-version APIs, not moving latest during evaluation.
- I5: https://wiki.factorio.com/Command_line_parameters ; https://wiki.factorio.com/Multiplayer ; https://wiki.factorio.com/Tutorial:Modding_tutorial — CLI, dedicated-server setup and mod packaging.
- I6: https://github.com/JackHopkins/factorio-learning-environment — FLE source, setup and actual tool contracts.
- I7: https://pypi.org/project/rcon/ ; https://github.com/conqp/rcon ; https://pypi.org/project/factorio-rcon-py/ ; https://developer.valvesoftware.com/wiki/Source_RCON_Protocol — optional clients and transport framing.
- I8: https://wiki.factorio.com/Console — console commands and scripting consequences.
- I9: https://github.com/factoriotools/factorio-docker — third-party container packaging.
- I10: https://factoriolab.github.io/ ; https://github.com/factoriolab/factoriolab — factory planning calculator, not an engine adaptor.
