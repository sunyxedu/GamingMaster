parent: 0000-tier1_astra_naive

# Hypothesis

The leader is primarily limited by how much useful game time it can execute per slow model call, not by a difficult survival decision. Its worst traces end before meaningful play: seeds 1106 and 1108 record zero steps and one pending System-2 call; seed 1104 records seven movement steps, no `do`, zero achievements, and full health at the 1800-second deadline. Even seed 1101, tied for best, executes only 29 steps. Its successful transitions establish `tree + do -> wood`, a table costing two wood, a wood pickaxe costing one wood, and stone mining with that pickaxe. These tiny deterministic transitions should not each need a 40–80 second reasoning call.

A controller that compiles achievement objectives into observation-checked skills should lift both the mean and the weak seeds, while using far fewer Astra calls. Batching is closed-loop: it reads and checks the next observation after every action, rather than blindly replaying an action string.

# Change

Replaces the one-action-per-call loop with persistent terrain memory, fresh dynamic entity observations, weighted path search through walkable and mineable terrain, explicit facing interactions, and a dependency compiler for tool progression. Survival interrupts every skill. Resource collection, placement, crafting, farming, drinking, sleep and adjacent combat are handled by code. Failed placement/crafting states are tracked; recipe hypotheses are verified from inventory/achievement deltas rather than silently assumed correct.

Astra receives a compact local ASCII map, nearest known resource coordinates, inventory, missing achievements, verified/hypothesized recipes, and recent transition events. At most six strategic reviews occur, separated by at least 220 game steps. Astra can select an achievement objective, exploration direction, bounded diagnostic actions or a recipe correction. Invalid JSON and model errors fall back to the controller; reviews have a 150-second timeout and reserve time for useful actions. No initial model response is required to make progress.

All memory is episode-local; maps and routes come exclusively from observations. The harness imports only standard-library modules. It never reads simulator internals or uses seed-specific rules.

# Confirm / refute

Confirm if the early wood/table/tool chain reliably completes before the first review, the minimum score rises, hundreds of checked actions execute per episode, and System-2 calls drop well below the leader. Inspect option_unlock and strategic_review events plus inventory deltas to distinguish productive skills from action throughput alone.

Refute if the agent merely accelerates starvation, loops on invalid crafting/placement, or repeatedly crosses unsafe terrain. In particular, plant maturity fields, adjacency requirements and unverified iron recipes need observation-based validation. Such failures would motivate targeted option repairs rather than increasing Astra call frequency.

# Validation

Initial implementation written before testing. Pre-flight and dev results will be recorded below after execution.

## First dev evaluation and refinement

`dev/20261030-091848`: seed 3301 reached 15/22 in 233 steps with one Astra call; seed 3302 reached 10/22 in 204 steps with zero calls. Mean 0.56818; both ended through the game, without errors. The digest and full transition traces showed that the structural throughput improvement works, but survival is now the limiting factor. Seed 3302 repeatedly commuted between its original table and distant resources; it still had only a wooden sword when multiple zombies killed it. Seed 3301 found food too late, unlocking eat_cow on its death tick.

Refinements retain the same hypothesis: gather materials for several recipes per trip (eight initial wood; seven stone for the stone tools, furnace and placement), reset exploration destinations when the needed resource changes, retain recent cow sightings, start food search sooner, and recognize the observed `plant_growing` entity label. Optional wake-up sleep waits until a stone sword is ready. First pre-flight: 18/22, 299 steps; revised pre-flight: 16/22, 321 steps. Both passed with no exceptions; these are contract smoke tests, not selection scores. Six additional snapshot fixtures exercised malformed model output and caught TimeoutError/RuntimeError successfully.

## Second dev evaluation and final repair

`dev/20261030-092828`: seed 3301 reached 13/22 in 202 steps with zero Astra calls; seed 3302 reached 16/22 in 235 steps with one call. Mean 0.65909, minimum 0.59091; neither episode errored. These dev seeds differ from the leaderboard's training seeds, so the two means are not a paired comparison. Both dev evaluations support the throughput mechanism; neither demonstrates long-term survival or reliable iron/diamond completion.

Offline replay of the second seed-3301 trace identified a precise placement failure: steps 101–137 alternated `move_right` and `move_left` while trying to place a furnace, despite having four stone. Position-only navigation returned to the table, while the placement helper immediately moved away to change facing. The final harness searches `(position, facing)` states for a legal placement. A regression fixture reconstructed the observed map at step 100 and reached a legal furnace placement in two actions (`move_down`, `place_furnace`) instead of oscillating. This change is general to all placement actions and uses only observed terrain.

The final repaired file passed pre-flight: `ok: true`, no exception, 15/22 achievements, 312 actions. Its full dev score is **not measured**: the two permitted dev evaluations were already used. The preceding dev score must not be attributed to the final placement repair. Plant maturation, crowd avoidance, and health recovery remain useful next hypotheses; batch provisioning alone did not prevent deaths to night mobs.

Exact parent trace evidence: seed 1108's `episode_end` contains `"score": 0.0`, `"steps": 0`, `"system2_calls": 1`, and `"ended_by": "deadline"`. Seed 1104's seven step actions are `move_up, move_up, move_up, move_up, move_left, move_left, move_up`, with no interaction. In contrast, verified first-dev crafting deltas include `{"wood":-1,"stone":-1,"stone_pickaxe":1}` and furnace `{"stone":-4}`. Iron recipes remain hypotheses pending an observed successful craft.
