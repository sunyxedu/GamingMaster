Design a harness for a **real-time** game (tier 2).

Game: `Doom (ViZDoom ASYNC_PLAYER)` — Classic Doom single-player on one IWAD map per episode. The engine runs headlessly at 35 tics per second in ViZDoom ASYNC_PLAYER mode; monsters, projectiles, doors, weapon cooldowns, and damage continue while the controller is thinking. Reach and activate the map exit to succeed. Death and the episode timeout end the episode. The seed selects a deterministic map/skill scenario and seeds ViZDoom's own RNG: declared train seeds use early maps at skills 2-3, held-out seeds use a mixed set including unseen maps at skill 4, and VIZDOOM_SKILL=5 is a separate stress run. Doom 1 uses E1M1-E4M9 and Doom II uses MAP01-MAP32; Doom and Freedoom are never mixed.

- noop: Hold no buttons; safe initial and missed-decision action.
- forward: Run forward while keeping the current view.
- backward: Run backward; useful for retreating from a threat.
- turn_left: Turn left in place.
- turn_right: Turn right in place.
- strafe_left: Run sideways to the left without turning.
- strafe_right: Run sideways to the right without turning.
- forward_left: Run forward and strafe left (a moving turn/circle).
- forward_right: Run forward and strafe right (a moving turn/circle).
- backward_left: Run backward and strafe left.
- backward_right: Run backward and strafe right.
- fire: Fire the selected weapon without moving.
- fire_forward: Fire while running forward.
- fire_strafe_left: Fire while strafing left.
- fire_strafe_right: Fire while strafing right.
- use: Pulse USE once for a door, switch, or exit.
- next_weapon: Pulse selection of the next available weapon.
- prev_weapon: Pulse selection of the previous available weapon.
- select_weapon_1: Pulse selection of weapon slot 1 (if owned).
- select_weapon_2: Pulse selection of weapon slot 2 (if owned).
- select_weapon_3: Pulse selection of weapon slot 3 (if owned).
- select_weapon_4: Pulse selection of weapon slot 4 (if owned).
- select_weapon_5: Pulse selection of weapon slot 5 (if owned).
- select_weapon_6: Pulse selection of weapon slot 6 (if owned).
- select_weapon_7: Pulse selection of weapon slot 7 (if owned).
(real-time games: ticks apply 'noop' until the first action is posted; afterwards a tick without a fresh action repeats the last posted action)
Doom runs at 35 tics/s. Movement and fire actions are held until a new action is posted; USE and weapon-selection actions are one-tic pulses. The optional compact grammar is 'move <forward> <turn>', where each integer is -1, 0, or 1 and positive turn means right, or 'select <1-7>'. Unknown or malformed actions are ignored and reported in the next observation.

Observations are compact JSON with IWAD family, map, skill, seed, Doom tic and elapsed time, health/armor/death, weapon/ammo ownership, kill/item/secret counts, player position/angle/velocity, visible labels with names/categories and pixel boxes, the latest and total ViZDoom reward, and terminal status. Labels are only currently rendered visible objects; whole-map objects, sectors, and omniscient automap data are disabled. Map totals are not exposed by the native API, so exact kill/item/secret percentages are null; bounded count progress is secondary to exit success.

Actions:
- noop: Hold no buttons; safe initial and missed-decision action.
- forward: Run forward while keeping the current view.
- backward: Run backward; useful for retreating from a threat.
- turn_left: Turn left in place.
- turn_right: Turn right in place.
- strafe_left: Run sideways to the left without turning.
- strafe_right: Run sideways to the right without turning.
- forward_left: Run forward and strafe left (a moving turn/circle).
- forward_right: Run forward and strafe right (a moving turn/circle).
- backward_left: Run backward and strafe left.
- backward_right: Run backward and strafe right.
- fire: Fire the selected weapon without moving.
- fire_forward: Fire while running forward.
- fire_strafe_left: Fire while strafing left.
- fire_strafe_right: Fire while strafing right.
- use: Pulse USE once for a door, switch, or exit.
- next_weapon: Pulse selection of the next available weapon.
- prev_weapon: Pulse selection of the previous available weapon.
- select_weapon_1: Pulse selection of weapon slot 1 (if owned).
- select_weapon_2: Pulse selection of weapon slot 2 (if owned).
- select_weapon_3: Pulse selection of weapon slot 3 (if owned).
- select_weapon_4: Pulse selection of weapon slot 4 (if owned).
- select_weapon_5: Pulse selection of weapon slot 5 (if owned).
- select_weapon_6: Pulse selection of weapon slot 6 (if owned).
- select_weapon_7: Pulse selection of weapon slot 7 (if owned).
(real-time games: ticks apply 'noop' until the first action is posted; afterwards a tick without a fresh action repeats the last posted action)
Doom runs at 35 tics/s. Movement and fire actions are held until a new action is posted; USE and weapon-selection actions are one-tic pulses. The optional compact grammar is 'move <forward> <turn>', where each integer is -1, 0, or 1 and positive turn means right, or 'select <1-7>'. Unknown or malformed actions are ignored and reported in the next observation.

Models available through `ctx`:
* `ctx.think(prompt, session_id=...)` — Astra (System-2, slow). Deep reasoning, can plan, can explain
  rules. Returns `.text` and `.session_id` (pass it back to keep conversation context).
  Budget: 40 calls per episode. Astra never sees the game unless you show it.
* `ctx.decide(state, questions)` — Jev (System-1, fast typed judgments). Budget: 4000 calls per
  episode. Read the complete Jev model context and local usage contract below before designing
  its state, questions and answer-to-action bridge. Actual latency must be measured.
* `ctx.spawn(fn)` — background thread, so Astra can think while Jev keeps acting.

Model economics (facts, not instructions): one Jev request answers *many independent typed questions
about one state in parallel* for the price and latency of one, and the Jev budget is two orders of
magnitude larger than Astra's; an Astra call takes far longer and its latency grows with the length
of its answer. Design timeouts and pipelining from the measured latencies (below, if calibrated, and on
the leaderboard). Whatever the design, the evaluator reports how many posted actions were preceded by a
fresh Jev / Astra answer, so the division of labour is visible.

The game ticks every 0.02857142857142857s **on its own clock**; `ctx.act(a)` posts the current action (instant).
A tick with no fresh action re-applies the last one. Each episode has a wall-clock deadline of
900 s (`ctx.budget.remaining()["wall_s"]`); the game's own end, the deadline or a used-up call
budget end the episode with the score reached. Each episode starts cold: fresh `ctx.memory` and fresh module globals. Clock mode: wall.

Evaluation: mean score over the training seeds (primary), then fraction of stale ticks (lower is
better), then System-2 calls (fewer is better); real-time ties favour longer survival. A candidate can
only be selected if none of its training episodes ends with an error (see the contract). The idle
baseline (never acts) shows what doing nothing scores.

What the harness must design (this is the whole point): how Astra's slow output (a plan, a rule
summary, a policy sketch, a subgoal) is turned into *state + questions* for Jev when needed, how
Jev's typed answer is turned into a game action, when to go back to Astra, and what to do when a
decision is late.

**Evaluation scope.** The harness is selected on the training seeds of this game and then reported on
unseen seeds of the same game; there is no held-out game. Rules, strategies and search that the game
description or the study establish may be implemented as code once they are verified against
observations — but nothing specific to the training seeds.

## Prior knowledge from the study phase
The study phase researched this game (rules, interface, community strategies). Use it: implement exact
rules, bookkeeping and search as code, put planning and adaptation into Astra prompts, and fast
recognition into Jev questions — but verify every claim against observations, because sources can be
wrong or outdated for this build of the game.

The full documents are readable in this archive under `prior/` (`prior/strategies.md`, `prior/brief.md`, `prior/interfaces.md`, `prior/study.json`, `prior/ADAPTOR.md`). Below: excerpts within a size budget; every cut is marked `…(truncated; full text in prior/<file>)`.

### study/study.json (summary)
- strategy_summary: Strong play keeps moving, uses strafe/circle movement and cover, prioritizes immediate hitscan and body-blocking threats, and uses monster infighting to conserve ammunition. Shotgun/chaingun handle routine targets, heavy weapons are reserved for crowds and major threats, and rockets are avoided at close range. The agent should route to required keys and the exit, clear only blockers by default, and treat 100% kills/items/secrets as a declared secondary objective. Code should enforce timing, legal weapon/ammo choices, state bookkeeping, and emergency retreat rules; Jev should make fast typed local choices while Astra maintains map routes and resource plans.
- scoring: Make map-exit success the dominant score: report completed-by-exit, death or timeout, map, skill, seed, elapsed tics/seconds, and ViZDoom total reward. Add secondary normalized progress from KILLCOUNT, ITEMCOUNT, and SECRETCOUNT (and percentages when map totals are available), plus final health/armor and remaining ammo; use time as a tie-breaker. Do not reward optional kills or secrets enough to outweigh failing to exit.
- interface.observation: A compact JSON projection of ViZDoom GameState: tic and state number; map/skill/seed; named health, armor, death, weapon, ammo, kill, item, secret, damage, position, angle, and velocity variables; visible state.labels entries with object_id, object_name, object_category, label value, world position/velocity, and pixel bounding box; optionally an in-memory grayscale screen and explored-only automap. The raw labels_buffer is an 8-bit array and screen output is never rendered to a display. Keep all-object and all-sector buffers disabled for fair play unless explicitly allowed.
- interface.actions: Translate a typed Jev/Astra intention into the configured ViZDoom button vector. Hold movement, turning, SPEED/STRAFE, and ATTACK as needed; pulse USE and weapon-selection buttons for about one tic; use set_action in ASYNC_PLAYER and refresh every 4-7 tics. For deterministic tests, PLAYER plus make_action(action, 1-4) is acceptable but pauses the world. Read button order from get_available_buttons() rather than assuming indexes.

### study/strategies.md
# Classic Doom strategy prior

## Strategic objective

A strong general player optimizes **survive -> reach the exit -> finish efficiently**, not “stand still and clear every actor.” Full kills/items/secrets are valuable tie-breakers or a separate 100% objective, but a fast safe exit is the default. Map knowledge changes the answer: the best route is often a key/weapon/exit route that skips optional fights, while a 100% run deliberately detours for monsters and secrets. The [DoomWiki Doom overview](https://doomwiki.org/wiki/Doom), [intermission reference](https://doomwiki.org/wiki/Intermission), and [ViZDoom Doom config](https://github.com/Farama-Foundation/ViZDoom/blob/main/scenarios/doom.cfg) support this distinction: the level ends at the exit, while kills/items/secrets and time are separately measurable.

There is no universal RTS-style opening/build order. The repeatable opening is: acquire the first useful weapon/ammo, run while scanning the room and exit direction, identify locked-door/key dependencies, then choose a route that preserves health and ammunition. Astra should learn map-specific routes from repeated trials rather than pretending every map has the same opening.

## Core combat and movement heuristics

### Keep moving and preserve a retreat lane

- Hold run while moving unless a precise interaction requires a tap. Strafe and turn together; do not walk straight at a projectile or stand in a doorway.
- Circle-strafe around exposed monsters while keeping the crosshair on the threat. Zig-zag against rockets and revenant missiles; use corners, pillars, doors, and room edges to break line of sight.
- Enter a room with an exit lane already in mind. Avoid getting trapped in a corner, between a monster and a wall, or in a narrow path where a lost soul can body-block.
- Treat hitscanners as immediate threats: a Former Human/Shotgun Guy can damage the player without a visible projectile, so break sight or kill the closest dangerous hitscanner before leisurely fighting distant melee monsters.

The community guide [meow1917's Doom tips](https://meow1917.neocities.org/doom/) emphasizes continuous movement, circle-strafing, retreat geometry, and not approaching dangerous monsters in a straight line. These are good prompt priors, but they must yield to local geometry in the label state.

### Use monster infighting

A projectile fired by one monster can hit another monster and cause the victim to attack the shooter. The player can briefly expose or reposition to make a mixed group cross-fire, then circle the outside of the fight while conserving ammunition. Do not stand in the crossfire, and do not assume it will solve a hitscanner, an isolated monster, or a monster immune to the relevant attack. See [DoomWiki: monster infighting](https://doomwiki.org/wiki/Monster_infighting).

Code should detect a nearby multi-monster situation and the presence of projectiles/cover; Jev can answer “fight now / bait infighting / retreat” from that JSON. Astra should choose which group to bait and whether the time/health risk is worth it.

### Target priority

Use a threat score rather than a fixed monster-name list. A reasonable initial order is:

1. A close hitscanner with a line of fire, especially a Shotgun Guy.
2. Arch-vile or Pain Elemental when present and exposed: the former can inflict severe damage/resurrect monsters; the latter can multiply lost-soul pressure.
3. A close fast melee/body-block threat (pinkie, spectre, lost soul) that removes the retreat path.
4. Homing/projectile threats such as Revenants and heavy monsters that control the arena.
5. Routine distant imps/zombies and monsters already occupied in infighting.

Override this ranking when a Cyberdemon, Spider Mastermind, or similarly lethal line of fire is immediately active: first reach cover or establish a dodge orbit, then shoot. The agent should prioritize **the threat that can damage it before the next decision**, not the monster with the highest nominal health. The [Doom monsters reference](https://strategywiki.org/wiki/Doom/Monsters) and [classic tips discussion](https://www.doomworld.com/forum/topic/141484-tips-for-classic-doom-general/) are useful community references; verify names/health behavior against the selected IWAD/mod.

### Weapon and ammo policy

- **Pistol:** fallback only; conserve it for weak isolated targets when better ammo is scarce.
- **Fists/chainsaw:** chainsaw weak melee enemies in a safe open geometry; never commit to melee when a retreat lane is blocked or a high-damage monster can hit back. Berserk enables stronger melee but does not remove positioning risk.
- **Shotgun:** reliable workhorse for imps, zombies, pinkies, and mid-range targets. The Super Shotgun is the Doom II workhorse at close/medium range, but its long reload and shell cost matter.
- **Chaingun:** good for hitscanners, pain-stun pressure, and sustained fire at targets where shotgun spread is wasteful. Stop firing if it draws a dangerous enemy while the player is exposed.
- **Rocket launcher:** strong against clustered or large targets in open space. Never fire into a nearby wall, floor, or a target that can rush into the splash radius; be especially careful around lost souls, Pain Elementals, and pinkies.
- **Plasma rifle:** high-rate emergency weapon for crowded rooms and tough enemies when available; reserve enough cells for the next known arena if the map route requires it.
- **BFG9000:** save for major crowds/boss arenas. Use cover and a deliberate release rather than wasting a shot while turning or firing into a wall.

The [Doom weapon reference](https://doomwiki.org/wiki/Weapon), [weapon-use discussion](https://www.doomworld.com/forum/topic/104607-how-to-use-the-weapons-properly-in-classic-doom/), and [classic tactics guide](https://meow1917.neocities.org/doom/) support the workhorse/area-denial/splash-risk distinctions. Exact damage, ammo, and weapon availability vary by IWAD, skill, and mod, so the adaptor should use `SELECTED_WEAPON_AMMO`, `WEAPON0`--`WEAPON9`, visible labels, and Astra's map memory rather than hard-coded “always use X.”

### Health, armor, and pickups

- Do not spend a medikit or armor pickup at full/near-full capacity if it can safely be left for a known return route; do not die while saving it, either.
- Track health and armor separately. A small amount of armor can make a future hitscan exchange survivable; a low-health player should avoid optional fights even if ammo is plentiful.
- Pick up weapons/ammo that change the available action set, but do not route through an arena solely for a minor item when the exit is known.
- Leave a dangerous room by the shortest safe route after the required key/switch is acquired. Search for secrets only when health, ammo, and time make the detour positive.

`HEALTH` and `ARMOR` can exceed 100 in ViZDoom, so code must not clamp them to a 0--100 display scale. The official [GameVariable definitions](https://vizdoom.farama.org/api/python/enums/) explicitly call this out.

## Navigation and level routing

1. **Orient:** record spawn position/angle, visible doors, keys, weapon pickups, monsters, and possible exit architecture. If a normal automap is enabled, use explored geometry; do not assume unvisited map data.
2. **Acquire capability:** route to shotgun/chaingun/ammo or a defensible room before taking a large fight when the map permits it.
3. **Solve locks:** identify red/yellow/blue key requirements and switches. Use `USE` at doors/switches, and confirm a state/message/position change rather than repeatedly holding it.
4. **Clear only what blocks the route:** kill or bypass the monsters that control the corridor, key, or exit. Use infighting for optional groups.
5. **Revisit deliberately:** remember which doors, lifts, teleports, and one-way traps changed. A route planner should not oscillate between two doors after a failed `USE`.
6. **Exit:** verify the exit cue/teleporter/position change and stop issuing combat actions once the map is terminal.

Astra should maintain a per-map route graph: node = observed room/door/key/switch/teleporter, edge = movement or interaction, annotations = threat, ammo/health cost, and whether the edge is one-way. Code should maintain exact current position, angle, key inventory inferred from messages/route memory, and interaction cooldown. Jev should classify visible “door / switch / key / exit / threat” and select a typed local intention; it should not invent a full route.

## Difficulty and scenario policy

Use S3/Normal for the baseline, then evaluate S1/S2 and S4/S5 separately. Difficulty affects monster behavior and available resources; do not mix skill in one score. Nightmare or fast-monster variants demand shorter action holds and more conservative exposure. Use the [ViZDoom skill/map wrapper definitions](https://github.com/Farama-Foundation/ViZDoom/blob/main/gymnasium_wrapper/base_gymnasium_env.py) for the 1--5 scale and map IDs, and report IWAD/map/skill/seed with every result.

A held-out scenario should vary **map**, **skill**, and **Doom RNG seed**, not only the starting seed on the same easy map. A useful split is:

- warm-up: E1M1/E1M2 or MAP01/MAP02 at S2--S3;
- development: mixed early and mid maps, S3;
- held-out: maps never used for route memorization, plus S4 and unseen seeds;
- stress: projectile-heavy arenas, tight corridors, and optionally S5, with a separate score because exact map knowledge and reaction cadence dominate.

Do not use random map changes inside a single episode. Reset starts one known map; a campaign harness may chain the next map only after recording the previous map's result.

## Common mistakes and counters

| Mistake | Counter-policy |
|---|---|
| Standing still to ask Astra or Jev | Keep a code-level strafe/turn/fire/retreat action running; use async ViZDoom and short Jev refreshes. |

…(truncated; full text in prior/strategies.md)

| Taking a health/armor item too early | Track current and projected damage; leave it when safe for a return, but prefer survival over hoarding. |
| Treating a visible label as a guaranteed target | Labels are visible objects, not full map truth; use bbox/world position and line-of-sight context, and re-observe after turning. |
| Exposing all objects/sectors to the reasoning model | Keep omniscient buffers disabled in fair play; use them only for debugging or a declared oracle condition. |
| Mixing Doom 1 and Doom II assumptions | Key the route/weapon policy by IWAD and map; Doom II's Super Shotgun is not a Doom 1 universal opening. |

## What belongs where

### Code: exact rules, search, bookkeeping, and safety

Implement these deterministically rather than asking a model:

- map/skill/IWAD/seed/episode bookkeeping and terminal detection;
- conversion of the configured ViZDoom button list into vectors, one-tic pulses, and held actions;
- action watchdog: never leave the player motionless while a model call is pending; keep a safe strafe/turn or retreat action;
- `ATTACK_READY`, weapon ownership/ammo checks, weapon-switch cooldown, `USE` pulse/cooldown, and no-ammo fallback;
- health/armor/ammo deltas, damage spikes, kill/item/secret counters, elapsed tics, and timeout/death/exit logging;
- angle normalization, distance/relative-bearing arithmetic, visible-label tracking by `object_id`, nearest-threat and line-of-fire estimates;
- emergency rules: break hitscan sight, avoid near splash, back away from body-blocking enemies, stop on terminal state;
- route graph bookkeeping and deterministic search over **known** doors/keys/switches; use A*/BFS only on observed or explicitly permitted map data;
- score calculation and JSON validation.

Code must not claim that a label is hostile solely from a name without a declared mapping; keep a configurable per-IWAD object taxonomy. It should also preserve the raw state for Astra while giving Jev a compact fixed schema.

### Astra prompts: planning and adaptation

Give Astra a recent state window plus route memory and ask it to:

- select the next room/key/exit objective and a conservative route;
- decide whether to bypass, fight, bait infighting, or retreat from a group;
- allocate shells/cells/rockets/BFG charges across known upcoming arenas;
- choose risk versus speed versus optional 100% kills/items/secrets;
- revise a map route after a door, teleport, ambush, death, or unexpected item;
- create/update a small map-specific plan and explain why a branch is safe.

Astra should return a bounded policy and typed intentions, not raw per-tic booleans. Its plan must include a fallback when the next label/state is absent. Do not ask Astra to calculate elapsed tics, distances, or ammo totals that code can provide exactly.

### Jev questions: fast recognition and local choice

Ask many independent typed questions against the same JSON state, for example:

- `choice`: which visible threat is most urgent: `hitscanner`, `projectile`, `body_blocker`, `Archvile/PainElemental`, `routine`, `none`;
- `choice`: `strafe_left`, `strafe_right`, `retreat_to_cover`, `advance`, `hold_angle`;
- `choice`: `shotgun`, `chaingun`, `rocket`, `plasma`, `BFG`, `chainsaw`, `no_fire` among legal weapons;
- `yes_no`: is it safe to fire a rocket now? is the visible target probably blocking the route? is `USE` warranted? should the agent abandon a 100% detour?;
- `choice`: `fight`, `bait_infighting`, `bypass`, `search_for_key`, `search_for_exit`, `retreat`;
- `score`: a bounded threat/safety label such as `safe`, `uncertain`, `dangerous`, not a damage or distance calculation.

Jev should recognize situations and choose among code-provided options; it should not plan a campaign, perform arithmetic, infer hidden map layout, or invent an unavailable weapon. Code filters its choice through legal-action and emergency rules.

## Recommended two-speed loop

1. At every 4--7 tics, serialize state and apply code safety checks.
2. Send the compact state to Jev; if the result is late, continue the previous safe action and shorten the next hold in combat.
3. Astra runs on a slower cadence (roughly 1--5 seconds, or at room/key/ambush/terminal events), updating route and resource policy while Jev continues acting.
4. On a major state transition—new weapon, key, door, arena, damage spike, or visible exit—ask Jev immediately and queue Astra replanning if needed.
5. Stop on exit/death/timeout, record the terminal reason and all counters, and never let a late model response act in a new episode.

This preserves the tier-2 property even though ViZDoom also offers synchronous `PLAYER` mode. The [ViZDoom enum docs](https://vizdoom.farama.org/api/python/enums/) state that async modes progress at the configured tic rate (default 35 tics/s), while player modes wait for an action.

### study/brief.md
# Doom (1993) study brief

## Scope and identity

This study means the classic 1993 Doom ruleset (Ultimate Doom/Doom II-compatible play through ViZDoom), not Doom (2016), Doom Eternal, or a multiplayer shooter. ViZDoom can run original Doom maps when the user supplies the corresponding IWAD; its scenario set also includes compatible Freedoom IWADs. The original game assets are not distributed by ViZDoom, so an exact original-Doom run needs a legally obtained `doom.wad` or `doom2.wad`; Freedoom is the no-account/offline fallback. Sources: [ViZDoom: playing original Doom levels](https://vizdoom.farama.org/environments/original_doom_levels/), [official `doom.cfg`](https://github.com/Farama-Foundation/ViZDoom/blob/main/scenarios/doom.cfg), [official `doom2.cfg`](https://github.com/Farama-Foundation/ViZDoom/blob/main/scenarios/doom2.cfg).

## Rules and objective

- The player is a space marine moving through a real-time first-person level. The primary objective is to reach that map's exit switch, exit door, or teleporter alive. Some maps have a secret exit; it is optional unless the scenario explicitly scores it.
- Monsters, hitscan attacks, projectiles, crushers, pits, toxic floors, locked doors, and ambushes are hazards. Combat is normally necessary, but killing every monster is not required to finish a map.
- Explore for colored keys, switches, weapons, ammunition, health, armor, power-ups, and optional secret areas. `USE` opens/activates doors and switches; keys are picked up automatically when walked over.
- The original control model has no jump or crouch. ViZDoom's classic configs expose running, strafing, forward/back/left/right movement, binary turning, attack, use, and weapon slots 1--7. The [classic control reference](https://en.wikibooks.org/wiki/Doom/Controls) lists move, run, strafe, fire, use, and map controls; the [ViZDoom config](https://github.com/Farama-Foundation/ViZDoom/blob/main/scenarios/doom.cfg) is the authoritative action ordering for the adaptor.
- Doom 1/Ultimate Doom has episodes E1--E4 with maps E1M1--E4M9. Doom II has MAP01--MAP32, including two secret maps. The exact map set depends on the IWAD supplied; ViZDoom's registration source lists the supported map identifiers in [`gymnasium_wrapper/__init__.py`](https://github.com/Farama-Foundation/ViZDoom/blob/main/gymnasium_wrapper/__init__.py).
- Skill is 1--5 in ViZDoom (`set_doom_skill`); use S3/Normal for the baseline and make skill a scenario dimension rather than silently changing it between trials. The Gymnasium wrapper documents the 1--5 range in its constructor [here](https://github.com/Farama-Foundation/ViZDoom/blob/main/gymnasium_wrapper/base_gymnasium_env.py).

## What counts as progress and score

Completion is the main objective: report `completed = true` only when the episode ends by taking the map exit, not merely because a large number of monsters died. A death or timeout is a failed map attempt. ViZDoom's official Doom configs set `map_exit_reward = 1` and a 60-minute timeout (`126000` tics); see [`doom.cfg`](https://github.com/Farama-Foundation/ViZDoom/blob/main/scenarios/doom.cfg).

Also report, at terminal state or timeout:

1. map identifier, skill, seed, terminal reason, and elapsed tics/seconds;
2. `KILLCOUNT` and, where the map exposes totals, kill percentage;
3. `ITEMCOUNT` and item percentage;
4. `SECRETCOUNT` and secret percentage;
5. final health and armor, remaining ammo, and whether the player died.

Use exit success as the dominant score. Kills, items, and secrets are useful secondary quality signals, while elapsed time breaks ties in favor of efficient play. Do not let an agent farm monsters or secrets indefinitely at the expense of reaching the exit. ViZDoom defines the counters in its [GameVariable enum documentation](https://vizdoom.farama.org/api/python/enums/); the classic intermission conventions are summarized by [DoomWiki's intermission page](https://doomwiki.org/wiki/Intermission).

## Controls exposed to the adaptor

The exact vector order must be read from `game.get_available_buttons()` rather than hard-coded. The official classic configs currently declare, in this order:

| Function | ViZDoom buttons | Notes |
|---|---|---|
| Fire | `ATTACK` | Hold for repeated shots; obey `ATTACK_READY` and weapon cadence. |
| Run/strafe modifiers | `SPEED`, `STRAFE` | Combine with movement; running is normally desirable. |
| Movement | `MOVE_FORWARD`, `MOVE_BACKWARD`, `MOVE_LEFT`, `MOVE_RIGHT` | Binary held movement. |
| Turn | `TURN_LEFT`, `TURN_RIGHT` | Binary turn; delta turn buttons are an optional alternative. |
| Interaction | `USE` | Open doors and activate switches; tap rather than hold when possible. |
| Weapon slots | `SELECT_WEAPON1` ... `SELECT_WEAPON7` | One-tic selection actions; only select weapons actually owned and supplied with ammo. |
| Weapon cycling | `SELECT_NEXT_WEAPON`, `SELECT_PREV_WEAPON` | Fallback when a slot policy is not available. |

The complete button list and the `doom.cfg` ordering are in the [official config](https://github.com/Farama-Foundation/ViZDoom/blob/main/scenarios/doom.cfg). ViZDoom binary buttons are pressed when nonzero; its delta buttons can instead encode turn or movement magnitude, as described in the [enum docs](https://vizdoom.farama.org/api/python/enums/).

## Observation needed by the harness

Prefer a JSON projection of a ViZDoom `GameState`, not a screenshot-only controller:

- `state.tic` and `state.number`;
- named game variables: `HEALTH`, `ARMOR`, `DEAD`, `ON_GROUND`, `ATTACK_READY`, `SELECTED_WEAPON`, `SELECTED_WEAPON_AMMO`, `AMMO0`--`AMMO9`, `WEAPON0`--`WEAPON9`, `KILLCOUNT`, `ITEMCOUNT`, `SECRETCOUNT`, `DAMAGE_TAKEN`, `POSITION_X/Y/Z`, `ANGLE`, and velocity where allowed;
- visible `state.labels`, each with `object_id`, `object_name`, `object_category`, bounding box, world position/velocity, and its `labels_buffer` value;
- optionally a grayscale screen buffer and a normal (explored-only) automap buffer for a visual fallback, never a visible window.

The variables are defined in [ViZDoom's enum documentation](https://vizdoom.farama.org/api/python/enums/). `GameState` timing and label metadata are documented [here](https://vizdoom.farama.org/api/python/game_state/), and the official [labels-buffer example](https://github.com/Farama-Foundation/ViZDoom/blob/main/examples/python/labels_buffer.py) explains that labels expose visible enemies, pickups, and barrels with unique IDs and bounding boxes. Do not enable all-object/all-sector information in a fair evaluation unless it is explicitly allowed: those buffers reveal actors or geometry outside the current view and are described as optional information in the [objects/sectors example](https://github.com/Farama-Foundation/ViZDoom/blob/main/examples/python/objects_and_sectors.py).

## Pace and tier

Doom's game logic runs at 35 tics/second; one tic is 1/35 second (about 28.6 ms). Monster thinking, hitscan attacks, projectile motion, weapon cooldowns, and doors continue at that cadence. ViZDoom's `PLAYER` modes wait for an action, while `ASYNC_PLAYER` modes continue at a constant rate; this distinction is explicit in the [ViZDoom mode docs](https://vizdoom.farama.org/api/python/enums/).

This is **tier 2**. A Jev action should be refreshed roughly every 4--7 tics (about 0.11--0.20 s) in exposed combat, with code-level emergency rules holding the previous action between model calls. A 0.5 s Jev call spans about 18 tics and can be fatal against hitscanners or close projectiles, so the adaptor must keep acting asynchronously rather than blocking the engine. Astra may spend seconds planning routes and resource policy, but it must not be the only real-time controller. A practical nominal decision budget is **0.20 s**; the safe budget is shorter in a line of fire and longer while traversing an empty corridor.

## Offline setup note

Install the Linux wheel into the evaluator's Python 3.12.3 environment with `vizdoom==1.3.2`, hide the window, disable sound, and use the software renderer/buffers. No display, GPU, account, server address, API key, or online service is needed. Exact setup and alternatives are detailed in [interfaces.md](interfaces.md).

### adaptor/ADAPTOR.md
# Doom adaptor

## Connection and lifecycle

This is a tier-2 adaptor for classic single-player Doom. It uses the official
`vizdoom==1.3.2` Python binding directly, with `DoomGame` configured for
headless software rendering and `Mode.ASYNC_PLAYER`. No display, GPU, server,
account, or online service is used.

One adaptor instance owns one `DoomGame`. `reset(seed)` stops and closes the
previous native episode, selects one map and skill, sets ViZDoom's native RNG
seed, initializes a fresh episode, and starts one controller thread. That
thread is the only caller that advances the native game. It repeatedly submits
the current action and calls `advance_action(1)`, so the world continues at
Doom's 35-tic/s cadence while the model is deliberating. `tick(action)` only
updates the action mailbox and returns; `ticks` counts accepted action posts.
`close()` stops and joins the controller and closes the native game, and is
idempotent.

`VIZDOOM_WAD_PATH` may point at a legally obtained `doom.wad`, `doom2.wad`, or
compatible Freedoom IWAD. When it is unset, ViZDoom's own bundled
`freedoom2.wad` is copied into this adaptor's `runtime/` directory as an
explicit no-account fallback. The observation and evaluator-only `info()`
identify the family and whether the bundled fallback was used. A supplied
IWAD whose filename is ambiguous must be paired with
`VIZDOOM_IWAD_FAMILY=doom1|doom2|freedoom1|freedoom2`; map families are never
silently mixed.

## Scenarios and seeds

Each episode is one map, not an implicit campaign. The mapping is:

- train seeds `1001..1008, 1011..1018`: early maps E1M1-E1M4 or MAP01-MAP04,
  alternating skills 2 and 3;
- held-out seeds `2001..2008, 2011..2018`: a mixed early/unseen set through
  E1M8 or MAP08 at skill 4 and new ViZDoom RNG seeds;
- `VIZDOOM_SKILL=5` is a separate stress evaluation override, not part of the
  train/held-out split;
- an explicit `VIZDOOM_MAP` fixes the map for every reset and is checked against
  the selected IWAD family.

Doom 1 families allow E1M1-E4M9; Doom II families allow MAP01-MAP32. Original
Doom and Freedoom results remain separately labeled. The declared `SEEDS` are
non-empty, disjoint, and contain 16 seeds per group. Custom integer seeds use
the held-out map mix at skill 3 unless environment overrides are supplied.

## Actions

The action vocabulary is native Doom buttons translated using
`get_available_buttons()` rather than relying on hard-coded vector positions.
Movement/fire actions are held until a replacement arrives. `use`, weapon
selection, and the weapon pulse actions are cleared after one engine tic.
Legal fixed names are exposed in `action_spec`; in addition, the parser accepts
`move <forward> <turn>` with each value `-1`, `0`, or `1`, and `select <1-7>`.
Unknown or malformed actions are ignored and appear as `action_error` in the
next observation rather than raising from `tick()`.

The initial and missed-tick action is `noop`; after the first accepted action,
the last sustained action remains held. `tick_s` is exactly `1/35` seconds.

## Observations

Observations are strict, compact JSON dictionaries. They contain the IWAD
family, filename, and source, map, skill, seed, tic/elapsed time, engine and
controller tick counts, health, armor, death/ground/attack readiness, selected
weapon and ammo, all AMMO0-AMMO9 and WEAPON0-WEAPON9 values, KILLCOUNT/ITEMCOUNT/
SECRETCOUNT/DAMAGE_TAKEN, player position/angle/velocity, latest and total
ViZDoom reward, terminal state/reason, and visible `state.labels` entries with
object name/category, label value, pixel bounding box, world position, and
velocity. Labels are view-visible only. Screen pixels, whole-map automap,
sector information, and omniscient object lists are intentionally not exposed;
the primary interface is symbolic and stays well below 200 kB.

## Score and terminal metrics

A non-death, non-timeout ViZDoom terminal episode is recorded as
`completed_by_exit`. Exit success contributes 1000 points and therefore
strictly dominates optional exploration. Secondary score components are bounded
kill/item/secret count progress, final health/armor, remaining ammo, a small
time tie-breaker, and the native total reward. Kill/item/secret counts are
reported in `info()` and in the observation. ViZDoom exposes current counts but
not map totals through this API, so exact percentages and map totals are
reported as `null`; the bounded count references are documented rather than
pretending to be completion percentages. `score_breakdown()` exposes every
numeric component, including `vizdoom_reward`.

`info()` records the IWAD family, filename/source, map, skill, seed, scenario
group, completed-by-exit/death/timeout, elapsed tics/seconds, native total reward,
progress counts, final health/armor/ammo, and trial validity. Native engine or
connection failures set `valid_trial=False` and `invalid_reason` so the
orchestrator can retry the seed. Bad play, death, timeout, and malformed
actions are not infrastructure failures.

## Limits

This adaptor does not provide an exact route planner and it does not chain
maps; a campaign harness must save each map's terminal metrics before resetting
to the next map. Exact commercial map behavior requires the corresponding
lawfully obtained IWAD. Bundled Freedoom is a faithful offline engine
integration but its maps and assets are not identical to commercial Doom.
`VIZDOOM_MODE=PLAYER` is rejected because its synchronous stepping would violate
the tier-2 real-time contract.

### study/interfaces.md
# Doom programmatic interfaces

## Recommendation

Use the **official ViZDoom Python API directly**, with `DoomGame` in `ASYNC_PLAYER` mode, hidden-window software rendering, the labels buffer, and named game variables. This is the only option here that gives the adaptor low-latency native actions plus structured Doom state without requiring a display, GPU, account, or server. Keep a synchronous `PLAYER`/`make_action(action, tics)` path as a deterministic test fallback, but do not use it as the production tier-2 loop because it makes the world wait for the adaptor.

The API is mature, maintained by the Farama Foundation, and maps closely to the engine. Its documented lifecycle is `DoomGame` -> configure -> `init()` -> `get_state()`/`set_action()` or `make_action()` -> terminal checks -> `new_episode()`/`close()`. See the [DoomGame API](https://vizdoom.farama.org/api/python/doom_game/), [Python quick start](https://vizdoom.farama.org/introduction/python_quickstart/), [GameState API](https://vizdoom.farama.org/api/python/game_state/), and the official [basic Python example](https://github.com/Farama-Foundation/ViZDoom/blob/main/examples/python/basic.py).

## Comparison of practical interfaces

| Interface | Maturity and latency | What it exposes/control surface | Legal / ToS risk | Cost and verdict |
|---|---|---|---|---|
| **ViZDoom Python API (chosen)** | High; native C++ engine behind Python bindings. A direct `set_action` in async mode avoids screenshot/OCR overhead. | `GameState` tic/time, reward, configured game variables, screen/depth/automap/audio/labels buffers, visible `Label` metadata, and optionally objects/sectors. Sends binary or delta button vectors; supports map, skill, seed, timeout, and reward configuration. | ViZDoom is open source. It does not redistribute commercial IWADs; using `doom.wad`/`doom2.wad` requires a legally obtained copy. Freedoom avoids that asset issue. No multiplayer anti-cheat or online ToS is involved. | Free software; only asset acquisition may cost money. **Best choice.** |
| **ViZDoom Gymnasium wrapper** | High and maintained in the same project; `step()` calls `make_action` with configurable `frame_skip`, so it is low overhead but synchronous. `reset(seed=...)` maps a Gym seed to ViZDoom's seed. | A Gymnasium observation dictionary containing screen, optional depth/labels/automap/audio/notifications, and `gamevariables`; binary/continuous/combined action spaces. `VizdoomScenarioEnv` covers bundled scenario configs. Native `env.state.labels` remains available, but custom full-Doom async behavior is less direct. | Same IWAD/licensing considerations; Gymnasium itself adds no online risk. | Free, but requires Gymnasium and its wrapper dependencies. **Good test harness; not the production async control path.** See the [wrapper source](https://github.com/Farama-Foundation/ViZDoom/blob/main/gymnasium_wrapper/base_gymnasium_env.py) and [wrapper API](https://vizdoom.farama.org/api/python/gymnasium_wrapper/). |

…(truncated; full text in prior/interfaces.md)

- **Variable availability:** Variables are only present if configured, and some world-space values disappear under `+viz_nocheat`. The adaptor must validate the configured list before play.
- **Termination:** `get_state()` may be `None` after termination. Read terminal flags before dereferencing state and always close the game.
- **No universal map route:** Doom maps are deterministic but map-specific. Astra needs a route memory keyed by IWAD/map/skill, while code handles local movement, target tracking, and resource bookkeeping.

## Sources

- [ViZDoom documentation](https://vizdoom.farama.org/)
- [DoomGame Python API](https://vizdoom.farama.org/api/python/doom_game/)
- [Enums: modes, buttons, variables](https://vizdoom.farama.org/api/python/enums/)
- [GameState and labels](https://vizdoom.farama.org/api/python/game_state/)
- [Original Doom level setup](https://vizdoom.farama.org/environments/original_doom_levels/)
- [Official Doom and Doom II configs](https://github.com/Farama-Foundation/ViZDoom/tree/main/scenarios)
- [Official Python basic example](https://github.com/Farama-Foundation/ViZDoom/blob/main/examples/python/basic.py)
- [Official labels-buffer example](https://github.com/Farama-Foundation/ViZDoom/blob/main/examples/python/labels_buffer.py)
- [Official objects/sectors example](https://github.com/Farama-Foundation/ViZDoom/blob/main/examples/python/objects_and_sectors.py)
- [Official Gymnasium wrapper source](https://github.com/Farama-Foundation/ViZDoom/blob/main/gymnasium_wrapper/base_gymnasium_env.py)


## Measured model latency (at archive creation, 2026-10-30 17:11)
- System-2 (Astra), one-word answer: 60 s
- System-2, an answer of ~2432 output tokens: 56 s — latency grows with the length of the answer; size timeouts for the answers your prompts actually request
- System-1 (Jev), one request with the game's initial observation and a Choice over its actions: median 169 ms (= 5.9 ticks), p90 265 ms, max 302 ms over 12 warm requests; first request on a new connection 214 ms — a timeout below the tail turns slow answers into failed calls
Later measurements per candidate are on the leaderboard (latency columns, a latency fit).

## Jev model context and usage for Meta-Harness

Snapshot: 2026-10-29 (docs.typesafe.ai: API, models, primitives, confidence, patterns, cookbooks, request
rules and the jev-1.13 known-failure page; plus the design of deployed open-source Jev agents). This is design context for the Astra proposer and for runtime Astra
when it designs Jev questions. The Python interface below is GamingMaster's wrapper, which differs
from the provider SDK. The model profile and episode budgets in the experiment manifest are
authoritative.

### What Jev supplies

Jev is TypeSafe's System One model for fast, focused judgments. It produces
typed answers and probability distributions, not prose, code, plans or reasoning
traces. It reads text and structured JSON. (The provider has an experimental image
input; this wrapper does not send images, so a visual game needs a separate perception
step — code, or Astra via `ctx.think(images=[...])` — before Jev can judge its state.)
Calibration describes aggregate predictions, not correctness of each answer.
See [System One](https://docs.typesafe.ai/concepts/system-one).

Model ids: `jev-latest` and `jev-preview` both resolve to jev-1.14.0 at this snapshot (released
2026-10-14; jev-1.13.0 remains available). An alias can move between calls; every answer records
the version that produced it, and the leaderboard warns when an archive mixes versions. Limits:
64k tokens per request, 32k per question; English is the primary training language.

In this architecture, Astra can interpret unfamiliar rules, choose subgoals and
design the decision interface. Jev can recognize situations, assess whether a
subgoal's preconditions still hold, or choose among well-described alternatives.
Code should handle exact arithmetic, legal-action checks, coordinate transforms,
bookkeeping and execution. Do not ask Jev to invent an unknown numerical action
mapping or perform an entire planning algorithm in one classification.

### Known failure modes (provider's list for jev-1.13) and what to do instead

* **Literal reading.** It answers the question written, not the one meant: state the
  exact condition in `instructions` and put boundary cases in the criteria.
* **Math, counting, comparisons of numbers and dates.** Unreliable: compute them in
  code and send the result or a named bucket ("adjacent", "two cells away"), never ask
  Jev to count, add or compare magnitudes.
* **Numeric representations.** Semantic words beat raw numbers: colour names over RGB/hex
  or palette indices, "wall to the left" over coordinates. Translate in code first.
* **Indirection.** Several hops (a property of a property, double negatives) cost
  accuracy: name the relevant state field directly, e.g. `threats.nearest`.
* **Large states with irrelevant detail.** Unrelated content distracts and costs accuracy
  ("context rot"): filter in code and send only the fields this decision needs.
* **Contradictory instructions and criteria.** Keep them aligned; a Noul whose `true`
  means "no" performs worse.
* **Choice option order.** It leans toward the option listed first: do not always list a
  default/no-op first; when a decision matters, ask the same question twice with the
  options reordered (one request, two question ids) and act only if both agree.
* **Generation.** It is not a generator: turn extraction into a Choice over candidates
  that code (or Astra) produced.

### Context that each request must carry

Jev does not inherit the proposer's prompt, Astra's conversation, `ctx.memory`,
previous requests, game description or observations automatically. This wrapper
sends only the supplied `state` and `questions` with the configured model.
Construct the necessary context explicitly on every call.

Prefer a named JSON object containing the current observation summary, the
relevant rule/field meanings, Astra's current subgoal and constraints, and any
short recent evidence needed for this decision. Include candidate descriptions
when asking Jev to compare candidates. Supply actual values, not variable names
or a reference to an inaccessible file. Keep irrelevant history out; preserve
evidence that distinguishes the options. English instructions are preferred for
this model. A request must fit 64k tokens, and the state plus any single question
32k. See [State](https://docs.typesafe.ai/concepts/state).

Batch independent judgments about that state into one call. They share the
input but do not see each other's answers. If a question really requires an
earlier answer, put that answer in the next call's state. Otherwise ask both
speculatively and combine them in code. Put the complete judgment in
`instructions`: question IDs are invisible to Jev. Refer to input fields
explicitly, e.g. `plan.preconditions`. See
[Questions](https://docs.typesafe.ai/primitives).

### Patterns that deployed Jev systems share (provider cookbooks, open-source agents)

* **Code computes facts, Jev judges, code decides.** Successful controllers send a small JSON
  state of *derived facts* (distances as named buckets, what is adjacent, time to contact,
  the last action and its measured outcome, the measured age of the observation) rather than
  raw grids or numbers, then ask atomic questions and combine the probabilities in code.
* **One request, several atomic questions.** A per-tick request typically carries 3–8
  independent questions (e.g. a Choice over the legal actions, a Score of immediate danger, a
  Noul per precondition or per candidate), including speculative ones whose answers code may
  ignore; a request with many questions costs about the same latency as one with one.
* **Structured criteria.** Option descriptions as objects (`what`, `not_for`, `examples`) and
  instructions that name the state fields (`` `threats.nearest` ``) measurably beat prose.
* **Freshness and fallbacks.** Every answer is tagged with the observation / plan version it
  judged; a late answer is dropped, and an executable fallback (the last safe action, a
  conservative rule) runs while waiting. Deadlines per question are short and explicit.
* **Confidence gates per consequence.** Irreversible or lethal actions need a higher top
  probability than cheap ones; a top probability under ~0.6 is treated as "undecided" (route to
  the fallback or to System-2), not as a weak yes. Ask high-stakes Choices twice with the options
  reordered and act only when both agree.
* **Large option sets** (more than 255, or many near-identical options) are handled in two
  stages: Score or Noul each candidate independently, then one Choice over the shortlist.
* **System-2 authors, System-1 runs.** The slow model designs the state schema, the questions,
  the thresholds and the fallback policy — and re-authors them when the evidence says the
  questions are wrong — rather than being asked per tick. Harnesses that let System-2 revise
  the System-1 interface on observed failures improved most across iterations.

### Exact local call and answer contract

Import `Choice`, `Score`, `Noul` from `metaharness.models.jev`. Call
`ctx.decide(state, questions, timeout_s=5.0)`; it returns
`dict[str, System1Answer]` directly, keyed by your question IDs.

| Question constructor | Meaning of `answer.value` | Other local fields |
|---|---|---|
| `Choice(instructions, criteria={option: description, ...})` | Selected option key, a string | `.probabilities`, `.confidence` |
| `Score(instructions, criteria=[low_description, ..., high_description])` | Fractional level index, from 0 to number of levels minus 1 | `.probabilities` with string index keys, `.confidence` |
| `Noul(instructions, criteria={"true": description, "false": description})` | Probability of yes, between 0 and 1 | `.probabilities` contains `"true"`/`"false"`; `.confidence` is normally `None` |

All answers also have `.type` and `.top(k)`. There is no `.answers` container,
`.choice`, `.score`, `.noul`, generated explanation, or Score `.legend` on the
local return objects. Keep the Score level descriptions in your own code.
`Noul` criteria are optional. JSON-serializable question dictionaries with
`type`, `instructions` and appropriate `criteria` also work. Credentials and
the HTTP client are managed outside the harness; do not instantiate `JevClient`
or bypass `ctx.decide`. See the [HTTP API](https://docs.typesafe.ai/api) for the
provider's different wire format.

For Choice, define distinguishable options and include an explicit unknown or
no-match option when needed; there can be at most 255 options. Structured
descriptions can spell out inclusion, exclusion and examples when labels are
ambiguous. See [Choice](https://docs.typesafe.ai/primitives/choice).
For Score, use 2–10 concrete ordered levels for one dimension (the API rejects more
than 10; the constructor raises); the returned number is an expected level, not
necessarily an integer or a probability — use it for thresholds, not to interpolate
magnitudes. See [Score](https://docs.typesafe.ai/primitives/score).

Confidence summarizes the distribution and is not interchangeable with the
winning option's probability. Inspect both when useful. A Noul near 0.5 is
uncertainty about a yes/no proposition, not medium severity. Handle absent
confidence explicitly; never turn `None` into certainty. Choose thresholds
using training evidence and action consequences, with a fallback or Astra
escalation for ambiguity. See [Confidence](https://docs.typesafe.ai/confidence).

What `ctx.decide` raises: `TimeoutError` when the answer (including retries of
rate-limited or transient server errors) does not arrive within `timeout_s`;
`RuntimeError` for any other failed call (e.g. a malformed question rejected by
the API); `BudgetExceeded` / `EpisodeOver` when the System-1 budget is used up or
the episode is over; `ModelUnavailable` when the key is rejected or the account is
persistently throttled (the run stops; catching it does not hide it).

### Small usage example, not a game adapter

This helper illustrates one batched request about an Astra plan and an
observation summary. The proposer still designs the summaries, questions,
thresholds, action mapping and scheduling for its candidate.

```python
from metaharness.models.jev import Choice, Noul, Score


def assess_plan(ctx, plan, observation_summary):
    state = {"plan": plan, "observation": observation_summary}
    answers = ctx.decide(state, {
        "condition": Choice(
            "Do the facts in `observation` support `plan.preconditions`?",
            {
                "supported": "All required preconditions are supported by the observations.",
                "contradicted": "An observed fact contradicts a required precondition.",
                "unknown": "The observations cannot establish all required preconditions.",
            },
        ),
        "progress": Score(
            "How much of `plan.goal` is visibly achieved in `observation`?",
            ["No visible progress.", "Some progress, but the goal is incomplete.",
             "The stated goal is visibly complete."],
        ),
        "finished": Noul(
            "Does `observation` explicitly establish completion of `plan.goal`?"
        ),
    }, timeout_s=5.0)
    condition = answers["condition"]
    return {
        "condition": condition.value,
        "condition_probability": condition.probabilities.get(condition.value, 0.0),
        "condition_confidence": condition.confidence,
        "progress_level": answers["progress"].value,
        "completion_probability": answers["finished"].value,
    }
```

### Timing and the generated bridge

One admitted `ctx.decide` request consumes one System-1 call, even with several
questions. Requests are synchronous unless submitted via `ctx.spawn`. Batching
reduces round trips but does not promise constant latency. Measure actual
latency from traces and the task's calibration: Jev can take longer than a game
tick, and the first request on a new connection (each thread keeps its own) pays
the connection setup. Keep an executable fallback while waiting, limit outstanding
requests, and associate each response with the observation and plan version that
produced it. Recheck relevant preconditions and legality on the latest observation
before acting. A cached semantic interpretation may outlive an individual action
decision; define its invalidation rule rather than treating every answer as
permanently reusable.

The context caps each request's timeout by the remaining wall budget. Handle
transient failures within that budget, let `ModelUnavailable` propagate, and stop
background work when `ctx.done`. The standard inference calls already log request,
response, latency and token usage. Add harness events explaining when a Jev
answer changes a chosen action, invalidates a plan, or triggers Astra; call
counts alone do not establish useful cooperation.

In `proposal.md`, specify the Jev state schema, each question's role, how its
answer affects control, freshness/confidence handling, and the hypothesis about
when Astra should revise the interface. These choices belong to the generated
harness; this guide supplies no environment-specific adapter.

When runtime Astra is asked to create or revise Jev questions, include this
guide (or its relevant interface sections) in that `ctx.think` prompt too.
The proposer context is not automatically forwarded to runtime Astra. A harness
can obtain the exact same bundled text with
`from metaharness.models.jev import usage_guide` and append `usage_guide()` to
the interface-design prompt, together with public observations and action
specifications. Do not send the full guide to Jev on every tick; send the
decision-specific state and questions it describes.
