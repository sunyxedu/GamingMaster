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
| Walking in a straight line toward a Cyberdemon/Revenant | Zig-zag or circle with a clear lane; use cover and never hug a wall during rocket fights. |
| Firing rockets at point-blank range | Switch to shotgun/chaingun/plasma or retreat; compute splash-risk from range/geometry. |
| Chasing every monster before finding the exit | Treat exit and required keys as primary; clear only blockers unless 100% is requested. |
| Letting hitscanners remain in a doorway | Break line of sight or kill the closest one first; do not focus on a distant high-HP target while taking instant bullets. |
| Spending cells/rockets on routine imps | Use shotgun/chaingun when safe; reserve heavy ammo for clustered/heavy targets. |
| Selecting a weapon with no ammo or while it is not ready | Code checks `WEAPONn`, ammo, `SELECTED_WEAPON_AMMO`, and `ATTACK_READY`; Jev chooses only among legal weapons. |
| Holding `USE` and walking through an interaction | Pulse `USE`, observe a message/door/position change, then route; repeated use can waste time and obscure state. |
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
