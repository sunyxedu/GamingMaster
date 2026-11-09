
### Candidate 0001 — closed-loop options (iteration 1)

- The naive leader's worst seeds 1106/1108 executed zero actions; seed 1104 executed seven moves and no interaction. Model latency and queueing dominate the opening, so useful scripted work before the first model call is a major improvement opportunity.
- Reading every observation does not require calling a model every tick. Weighted path search, facing checks, resource skills and inventory-delta verification produced 10–15 achievements on the first two dev episodes with only one Astra call total (mean 0.568, no errors).
- Single-item procurement is a dead end: repeated table–forest–quarry round trips delay the stone sword until night. Batch resources for multiple recipes, but preserve safety interrupts and inspect the next observation after every action.
- Adaptor observations use `plant_growing` (not just `plant`). Object vocabularies must be learned from trace observations; otherwise farms silently disappear from memory. Reset an exploration destination when the required resource changes, especially when food interrupts mining.
- A valid pre-flight can exercise no System-2 responses. Separate fixtures for malformed text/JSON, wrong-shaped fields, TimeoutError and RuntimeError all passed for 0001; six fixtures explicitly exercised the review fallback.
- Candidate 0001's second dev mean was 0.659 (13/22 and 16/22, no errors, one Astra call total). Replay exposed a 37-turn furnace-facing oscillation. The final placement search includes `(position, facing)` and reaches a legal placement from that recorded state in two actions; final pre-flight passed, but the dev budget was exhausted before this last repair, so 0.659 is not a measured score of the final file.

### Candidate 0002 — asynchronous adviser with verified execution (iteration 1)

- Model-call latency omits a consequential wait: dev 3301/3302 took 731.53/380.32 seconds from request to next action while reporting only 22.48/20.60 seconds of inference, despite a 150-second timeout. The final controller never waits for advice; only completed futures can alter its objective.
- Sleeping is a commitment with a severe combat cost. Both second-dev episodes lost seven health to a single sleeping hit (ticks 146 and 187), and the next attempted movement did not immediately move. Start sleep in strong daylight, check its estimated duration against dusk, and exclude every visible hostile.
- Fixed exploration durations were a dead end: the first dev saw stone at tick 109 but followed exploration until tick 160. New resource discovery must invalidate an exploration objective; an inventory gain should complete a collection objective.
- Exact inventory deltas support verified recipe bookkeeping, while failed signatures prevent repeated impossible crafts. Keep model-proposed costs provisional and protect observed costs from later overrides. Portable tables can save long single-item return journeys.
- The adaptor emits plant_growing; normalize semantic object variants and distinguish crop memory from transient creature memory. Recipe/placement failures and farming status must be grounded in the observation vocabulary.
- Final robustness checks exercised malformed JSON/text, wrong-shaped fields, TimeoutError, RuntimeError, stale advice, and an adviser that never finishes; 226 replay ticks completed without waiting. Pre-flight alone answered no S2 calls, so it did not cover these cases. The second dev mean was 0.454545 with no errors, before the final asynchronous and safe-sleep repairs; do not report it as a final-revision score. Final pre-flight passed at 19/22 in 552 steps; that is a smoke test, not a training estimate.

### Candidate 0003 — persistent camp and expeditions (iteration 1)

- A camp is a testable structural alternative to roaming objectives, but the farming premise remains unproven: neither dev seed unlocked eat_plant. Replacing zombie-destroyed crops before stone weapons was a dead end; bound farming effort and prioritize the stone sword.
- Batch abundant resources, not necessarily rare ones. Common wood/stone loads improved the second dev mean from 0.455 to 0.705; demanding two iron before crafting the first iron pickaxe cost seed 3302 roughly 230 additional ticks. Craft the first rare upgrade promptly.
- A placement target must persist while the player changes position/facing. Recomputing an adjacent target each tick made place_stone walk west until terrain accidentally allowed placement. A four-orientation fixture catches the problem without a model call.
- Replanning every tick can oscillate when threats cross the observation boundary. Dev 3302 steps 509–520 alternated left/right as the same zombie appeared/disappeared. Brief threat memory used only for route costs removes that reversal in observation replay; visible empty cells must invalidate the memory.
- Camp return safety is still unresolved. The stronger dev episode gathered iron but died on the return without crafting iron tools. A future experiment could compare a second temporary crafting site against a long dangerous return; current results do not establish that one permanent camp is optimal.
- Both dev runs were error-free; the second scored 14/22 and 17/22 with one/two Astra calls. Final placement, iron-load and threat-memory fixes passed pre-flight and focused response/replay checks but have no third dev evaluation. Pre-flight exercised no model response; explicit malformed-output/timeout/runtime-error tests were necessary. Do not label the second-dev 0.705 mean as a final-revision score.

- Leader 0002’s weak episodes show a late failure mode distinct from recipe errors: repeated A/B movement at dusk can consume an entire night and expose the player to mobs.
- A short hostile-memory window is useful for route costs, but it must expire quickly; current local observations remain authoritative so stale enemies do not become permanent walls.
- Deep-night behavior should be a mode switch with an explicit return target or local safe step. Continuing the strategic objective after daylight falls below roughly 0.23 is costly even with a sword.
- Positional cycle detection is cheap bookkeeping and catches planner failure without another model call; temporarily blocking one edge is safer than globally rewriting the map.
- Pre-flight stand-in models may answer zero Astra calls, so contract checks do not validate adviser parsing or asynchronous completion paths; keep parsing isolated and all exceptions contained.


### Candidate 0005 — portable rare-resource micro-camp experiment

- Candidate 0002’s leader score is strong on common achievements but weak on rare conversion: collect_iron 0.75, collect_diamond 0.25, iron pickaxe 0.50 and iron sword 0.25.
- Candidate 0004 is already testing dusk retreat and cycle breaking; a separate direction is to shorten the iron/diamond station loop with a temporary local table/furnace.
- Station placement must be confirmed by the next semantic map and inventory deltas; guessed recipe costs remain provisional.

### Candidate 0009 — vital-reserve governor

- Parent traces show a distinct late failure: food, drink, and energy reach zero while 15–21 achievements are already unlocked; movement then oscillates until health reaches zero.
- A hostile-triggered panic mode is insufficient when the enemy is outside the 9x9 view or when starvation damage is the main threat. Survival needs a reserve policy independent of visible threats.
- Refill thresholds must account for route length. The governor starts at food/drink ≤7 and refills to 9, before the parent's ≤5 emergency threshold.
- When no water or food is remembered, exploration is the only useful deterministic fallback; allowing Astra to continue crafting at zero vitals wastes the remaining ticks.

### Candidate 0007 — hard food reserve

- Candidate 0006's retreat logic can avoid an immediate mob hit yet still die after a long route: seed 1101 reached food 0 around step 277 and remained in a boundary oscillation until health reached zero.
- A low-food mode must suspend progression itself; merely ranking food ahead of the objective still lets exploration/crafting consume the remaining reserve.
- Release a starvation lock only after a genuine reserve (food 7 here), because one eat followed by immediate mining recreates the same failure.
- Cows are mobile and should be treated as visible targets; when none is visible, food search must expose new map cells rather than trusting a stale coordinate.

### Candidate 0008 — rare-resource checkpoint experiment

- Leader 0002’s concrete late failure is conversion latency: seed 1102 collected iron at step 536 but ended at step 577 without either iron tool; seed 1108 stopped at 15/22 before coal/iron.
- A rare-resource objective needs a bounded expedition and a nearby legal station; an open-ended frontier search can spend the whole reserve before crafting.
- Temporary stations should be verified from map/inventory deltas, and a resource gain should immediately invalidate the old exploration objective.
- Keep model advice subordinate to a code-owned checkpoint budget; stale or malformed advice must not extend a dangerous leg.

### Candidate 0011 — hostile escape lock

- Late leader failures can be diagnosed from positions as well as vitals: seed 1102 repeatedly alternated between two adjacent cells while zombies approached, even though it had already unlocked 16 achievements.
- A moving-hostile controller needs hysteresis.  Recomputing the nearest threat every tick can reverse a valid retreat and turn a safe edge into a death loop.
- A remembered safe anchor is useful only when learned from a quiet, healthy observation; it must remain episode-local and be scored as a route preference rather than treated as a fixed world coordinate.
- Preflight stand-in models may answer zero Astra calls, so the new threat and cycle paths must be deterministic and garbage-response safe; candidate 0011 passed preflight with 432 actions and no errors.

### Candidate 0010 — rare conversion lock and cycle escape

- The strongest concrete failure is post-discovery: seed 1102 collected iron at step 536, then repeated an A/B route until death at step 577 with no iron tool.
- A code-owned conversion lock should outrank stale model objectives immediately after an iron/diamond inventory increase, while remaining bounded so a bad station hypothesis cannot consume the episode.
- Temporary edge blocking is a cheap way to break positional cycles, but must expire quickly and only suppress one edge; permanent map bans would erase valid routes.
- Inventory deltas are more reliable than achievement flags for arming conversion because a pickup can be observed even when counters are delayed or variant-shaped.

### Candidate 0012 — crop revisit and vocabulary probe

- Across the archived 0002 traces, `place_plant` succeeds but the only emitted crop object is `plant_growing`; `eat_plant` remains zero in every candidate. A parser that waits for a nonexistent ripe token silently disables an achievement.
- Remembering plant coordinates and first-seen ticks makes a hidden maturation process testable without simulator access. Probe old crops with a cooldown and let inventory/achievement deltas, rather than object names, decide whether the hypothesis was right.
- Farming must be scheduled before the long iron/diamond leg, but bounded: one extra planting and 25-tick probe spacing keep a bad crop hypothesis from trapping progression.
- Preflight stand-in observations may not contain crops or answer Astra, so crop-specific behavior needs explicit replay fixtures in addition to the contract preflight.

### Candidate 0014 — dusk shelter mode

- A late death can be a scheduling failure even when recipes are correct: seed 1108 reached 15/22 before daylight fell to 0.046 and hostiles surrounded the player.
- A station-adjacent tile is an observation-grounded shelter anchor; capture it only while healthy, fed, bright and quiet so a dangerous mining position is not remembered as home.
- Dusk mode must be bounded and forget disconnected anchors. A permanent return lock can turn a mistaken station hypothesis into a worse oscillation than the original planner.
- Preflight again answered zero Astra calls; contract safety therefore depends on deterministic dusk paths and garbage-safe parsing, while advisor behavior needs separate replay or dev evidence.

### Candidate 0013 — conversion plus dusk lock

- Candidate 0002's traces separate two late failures: iron can be held for 40+ ticks without conversion, and a 15-achievement run can die in a hostile cluster at daylight 0.046.
- Inventory deltas are a stronger trigger than achievement flags for rare-resource conversion; a bounded lock should outrank stale Astra objectives while leaving the map and recipe checks in code.
- Dusk shelter must be bounded and observation-grounded: learn a station-adjacent anchor only while healthy, fed, bright and quiet, then forget it when disconnected or after 150 ticks.
- A/B position cycles are detectable from the last six positions; temporarily blocking one cell is safer than permanently banning a route.
- The combined experiment risks competing locks: conversion is attempted before dusk, while night mode takes precedence after daylight falls below 0.30.

### Candidate 0015 — vital reserve lock

- Late deaths can follow a successful dependency chain even when the immediate cause is consumable depletion; an urgency ranking is weaker than a mode that suspends progression.
- Reserve locks need both a refill target and a hard expiry. Without an expiry, an unseen water source or mobile cow can turn safety code into an episode-long stall.
- Inventory thresholds are more stable than achievement counters for starvation prevention: food/drink can fall for many ticks without any new achievement event.
- A reserve fallback should reveal nearby map cells while avoiding craft/sleep commitments; this keeps recovery possible without reopening the original long expedition.

### Candidate 0017 — bounded route watchdog

- Candidate 0015 added reserve and rare-conversion locks, yet its traces still show long no-achievement movement tails after an iron pickup; urgency controls do not invalidate stale route commitments.
- Exact A/B cycle detection is too narrow: a planner can loop over three or four remembered cells when a hostile or mobile resource changes the local topology.
- A short, expiring blacklist plus clearing the current objective is safer than permanently deleting map cells; it forces a fresh frontier choice while preserving cross-seed adaptation.
- Seven unchanged actions are a useful cheap signal for failed interaction/placement or a departed mobile target; resetting bounded commitments avoids wasting a model call.
- Preflight must continue to tolerate stand-in observations with no model responses; watchdog logic is deterministic and should be replay-tested with malformed model text separately.

### Candidate 0018 — bounded post-iron diamond expedition

- Leader traces show a large delay between iron-pickaxe craft and diamond collection (seed 1101: steps 281 to 512); some seeds craft the pickaxe but never enter a focused diamond search.
- Inventory transitions are a safer trigger than achievement counters: arm the expedition only when `iron_pickaxe` actually increases, then clear it on diamond gain or a hard tick/action bound.
- Reuse the existing observation-driven pathfinder inside the lock so the experiment changes scheduling rather than hard-coding routes or seeds.
- Safety interrupts must remain ahead of the expedition; a rare-resource lock that ignores combat, hunger, thirst, or sleep would confound conversion with avoidable deaths.

### Candidate 0016 — bounded dusk return and cycle escape

- The leader’s worst traces end after substantial progress: seed 1108 had 15/22 achievements but died at daylight 0.047; this supports a late survival hypothesis rather than an opening dependency failure.
- A shelter trigger must be conditional on both dangerous light and evidence of vulnerability (low health or a hostile crowd). Unconditional dusk mode in later candidates interrupted healthy progression and scored poorly.
- Learn shelter only from an observation-grounded station-adjacent tile while healthy, fed, hydrated, bright and quiet; discard it immediately if the current explored graph cannot reach it.
- A short A/B cycle lock is safer than a permanent route ban: suppress one cell for 12 ticks, reset exploration, and let the normal planner recover.
- Preflight still answers no Astra calls, so asynchronous advice parsing remains separately guarded and the deterministic safety paths must tolerate arbitrary stand-in observations.
