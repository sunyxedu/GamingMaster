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
- Accept at most **32 typed operations per turn**, then advance exactly **1,800 simulation ticks**. An empty wait turn advances the same amount. Inspection does not advance time. Individual failed operations within an accepted batch do not undo successful earlier operations and still consume the turn.
- End after **128 turns**: 230,400 ticks, or **64 simulated minutes** at 60 ticks/second. No uncharged final production interval is allowed.
- Real engine machines perform automated mining, smelting, assembly and research. No agent action can spawn free stock, edit deposits, unlock research, change game speed/tick budgets or execute arbitrary console Lua.

These constants are benchmark recommendations, not official Factorio rules. The complete implementable setup, API contract, error handling, scoring and acceptance gates are in [interfaces.md](interfaces.md). [strategies.md](strategies.md) explains how to play this particular ruleset and which decisions belong in code, Astra or Jev.

## Pace and tier

**Keep tier 1, conditional on verified pause-and-step control.** Ordinary Factorio is real-time and normally advances 60 simulation ticks per second. Even peaceful mode continues consuming fuel, producing items and researching while a model thinks. A normal unpaused headless server would therefore be tier 2. [B12]

The selected adaptor must stop the simulation at every decision boundary, accept a batch, and run a fixed tick interval before stopping again. The stop must occur at an engine tick target, not after a Python wall-clock sleep. RCON must remain usable to inspect and resume the paused dedicated server. `game.tick_paused` and an `on_tick` target are the proposed mechanism, **not a tested capability on this machine**; verify multiplayer behavior on the chosen build. If it fails, the tier-1 adaptor is not ready. [B11]

At a verified paused boundary, a decision can take seconds or minutes without affecting the game. There is no natural finite gameplay deadline. `study.json` uses **3600.0 seconds** as an explicit proposed harness timeout to satisfy the numeric field; it does not mean the game becomes unsafe after one hour. Jev is optional for state classification, not needed for real-time reactions. Both models' wall-clock delays must be irrelevant to earned production.

## Controls and observations

Human play uses movement, mining, handcrafting, placing/rotating buildings, choosing recipes, moving inventory and selecting research through a GUI. The headless server has no rendered game view. The proposed typed actions are `mine_resource`, `mine_tree`, `handcraft`, `place`, `rotate`, `set_recipe`, `insert`, `extract`, `mine_building`, `set_research` and `wait`; read-only `observe` calls provide detail without advancing a turn. These names describe a new bridge, not official Python or FLE methods. [B11, B13]

Observe JSON containing tick/revision/remaining turns; stock and queued manual work; revealed terrain/resources; entity positions/directions/status; inventories, recipes, fluids and power; available technologies and research progress; actual production/consumption counters; and score components. Cache versioned recipes/prototypes and query detailed local topology only when needed. Use explicit force/surface references because an empty dedicated server may have no `game.player`. [B11, B14]

## Progress and score

There is no universal vanilla freeplay scalar suitable for this evaluation. Define a benchmark score from **actual engine production and research**, with capped early bootstrap credit. This is not asserted to be the existing FLE score.

Let `P(i)` be cumulative genuinely produced items since the post-setup baseline, from engine flow statistics. Let `L(i)` be science packs genuinely consumed for research, verified against laboratory consumption and research state. For automation, logistic, chemical, production, utility and space science, use weights `1, 2, 6, 12, 12, 30` respectively.

```text
B = .01*min(P(iron-plate),1000)
  + .01*min(P(copper-plate),1000)
  + .02*min(P(iron-gear-wheel),200)
  + .05*min(P(electronic-circuit),200)

score = B + Σscience_i weight(i) * [P(i) + 4*L(i)]
```

The bootstrap term caps at **34**. Science production earns progress, and useful consumption earns additional credit. Weights are evaluation choices, not exact recipe prices. Report the scalar plus raw produced/consumed quantities, researched IDs, partial research, manual output, transfers and simulated ticks so behavior is auditable. [B14–B15]

Initial supplies, transfers, returned mined buildings and script insertions never count as newly produced items. Virtual manual-craft outputs are tracked separately and excluded from `P`; those packs can still earn credit when genuinely consumed in laboratories. Do not falsify engine statistics to credit scripted stock insertion. Score counters must use the correct force/surface and verified production/consumption directions. Negative or reset counters should be diagnosed, not silently clamped. [B14–B15]

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
