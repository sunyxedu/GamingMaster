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
