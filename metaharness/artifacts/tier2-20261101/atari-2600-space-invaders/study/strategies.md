# Atari 2600 Space Invaders strategies and model allocation

## What a strong player is optimizing

The Atari manual makes the priority ordering clear: kill all 36 invaders before they reach Earth, do not take three bomb hits, and maximize score. It also imposes a port-specific firing constraint: only one player laser may be active, so shots must be deliberate. A strong agent therefore optimizes **survival-adjusted wave clears**, not just immediate points or UFO attempts. The primary score is the sum of ALE rewards; lives remaining, invaders killed, wave resets, and terminal cause are useful secondary diagnostics.

The sources below distinguish facts from heuristics:

- **Manual facts:** [AtariAge HTML manual](https://atariage.com/manual_html_page.php?SoftwareLabelID=460) and [scanned Atari manual PDF](https://atarionline.org/wp-content/uploads/2021/11/space-invaders-1978-atari-2600-game-instructions-manual.pdf).
- **Environment facts:** [ALE SpaceInvaders](https://ale.farama.org/environments/space_invaders/) and [ALE environment specification](https://ale.farama.org/env-spec/).
- **Port/disassembly validation:** [Computer Archaeology code](https://www.computerarcheology.com/Atari2600/SpaceInvaders/Code.html) and [RAM page](https://www.computerarcheology.com/Atari2600/SpaceInvaders/RAMUse.html).
- **Player heuristics:** [AtariAge strategy discussion](https://forums.atariage.com/topic/15341-space-invaders-strategies/) and [StrategyWiki walkthrough](https://strategywiki.org/wiki/Space_Invaders/Walkthrough). These are community advice, not a formal specification.

## Core tactics

### 1. Open safely, then remove an outer column

Start near the middle rather than at a movement boundary. Pick a visible bottom/lowest invader whose column can be reached without crossing an imminent bomb. Community Atari-2600 players commonly clear a far-left or far-right column first, sometimes two columns, rather than randomly shooting the middle. The reason is geometric: the formation's outermost surviving column determines when it reaches a side and steps down; removing an outer column makes the remaining block narrower and buys horizontal travel time. This is a useful prior, not an inviolable rule: abandon it when a bomb lane, shield gap, or low invader is immediately dangerous.

**Put in code:** represent columns explicitly, order a safe outer-column candidate first, and keep a fallback target. Do not hard-code a column if the decoder says the formation has already shifted or a variation uses invisible/moving objects.

**Put in Astra prompts:** ask Astra to choose whether the edge-column time benefit is worth the current bomb risk and wave state.

**Put in Jev questions:** ask a typed `choice` such as `{"choice":"LEFT_OUTER|RIGHT_OUTER|LOWEST_THREAT|UFO|DODGE"}` from the compact object JSON; Jev should recognize danger, not calculate the geometry.

### 2. Fire at a living invader in a selected column, preferably the lowest one

A laser travels vertically and the single-shot rule means a miss costs the time until the beam exits. Align the cannon under a living column and fire when the shot path is not blocked by a target that is less useful. The lowest surviving invader in a column is a practical target: it is the immediate Earth threat and usually exposes the next target above it after the hit. Once the shot is in flight, move to the next target line or to safety; do not repeatedly press FIRE as if the game allowed a magazine.

For score, upper rows are worth more (5/10/15/20/25/30 from bottom to top), but a high-value upper alien should not outrank a low alien that is about to advance the formation or a bomb that can cost a life. A full wave is only 630 points, so one avoidable death is generally worse than a few lost 30-point opportunities.

**Put in code:** maintain `player_shot_active`/fire lock, map row/column to x, select the lowest living target in a safe column, and track the shot until hit/exit. Bookkeep score and wave transitions from reward/formation changes. Enforce cannon bounds.

**Put in Astra prompts:** resolve tradeoffs between immediate lowest-threat kills and row-value score when several columns are safe.

**Put in Jev questions:** ask `{"fire_now":true|false}`, `{"target_column":0..5}`, and `{"move":"LEFT|HOLD|RIGHT"}`. These are recognition/choice questions based on supplied coordinates; Jev should not be asked to add scores or predict long plans.

### 3. Dodge bombs while preserving a firing lane

Continuously watch bomb x/y and time-to-cannon. Move sideways rather than waiting under a bomb's vertical lane; use a shield as temporary cover but do not assume its center is invulnerable. Avoid parking against the left/right bars: the manual explicitly warns that being caught at the movement limits can leave the cannon hemmed in by enemy fire. A short oscillation or a move toward the next target is better than random jitter, provided it does not enter the predicted bomb lane.

**Put in code:** estimate nearest bomb time-to-impact, identify a safe horizontal interval, clamp movement to the legal lane, and use hysteresis so the controller does not alternate left/right every frame. Treat a shield as a mask with remaining holes, not a binary wall.

**Put in Astra prompts:** ask for a policy change when the wave is low, bombs are fast/zigzagging, a shield has a gap, or the cannon has just respawned.

**Put in Jev questions:** ask `{"danger":"NONE|LEFT|CENTER|RIGHT"}`, `{"safe_side":"LEFT|HOLD|RIGHT"}`, or `{"use_shield":"YES|NO"}`. Jev can recognize a near-term collision from JSON much faster than Astra.

### 4. Use shields, but budget them

The three fixed shields are the opening defensive resource. The manual says player beams and enemy bombs wear them away and eventually pass through. Fire from a protected lane when possible, but rotate away when the local shield mask is damaged or a bomb is descending through a gap. In late waves, a shield may be less useful than maintaining lateral room for dodging.

**Put in code:** expose per-shield health/occupancy masks and nearest threat distance; preserve a shield-health history between observations. Do not make `shield_present` a permanent boolean.

**Put in Astra prompts:** decide when to sacrifice a shield for a high-value shot and when to abandon it for open movement.

**Put in Jev questions:** classify `{"shield_status":"GOOD|DAMAGED|OPEN"}` and `{"cover_available":true|false}`.

### 5. Treat the UFO as opportunistic bonus, never the main plan

The command alien crosses the top periodically, is harmless, and is worth 200 points in normal one-player games. The manual directly warns not to focus on it so much that an enemy bomb hits the cannon. Shoot it only when the cannon is already in a safe horizontal position, the player laser is available, and no low invader/bomb requires an immediate response. The current Atari-2600 score table is not the arcade table: do not use a variable 50–300-point UFO model.

**Put in code:** detect `ufo.active`, calculate whether the current x can hit it without a dangerous detour, and disable the bonus target under a danger threshold.

**Put in Astra prompts:** decide the risk/reward threshold for a particular wave and life count.

**Put in Jev questions:** ask `{"safe_bonus_shot":true|false}` and `{"target":"UFO|INVADER|NONE"}`.

### 6. Prepare for speed and wave resets

As invaders are destroyed, the remaining formation becomes faster; when the last alien is removed, a new formation starts closer to Earth. The late-wave controller should shorten its planning horizon and prioritize bottom threats and bomb avoidance. At a new wave, reset target bookkeeping, preserve only durable strategy information, and re-center or move to a safe firing lane rather than assuming the previous x is safe.

**Put in code:** infer a wave counter from full-formation resets and formation y; maintain speed/step interval statistics; shrink the danger horizon as invader count falls. This is bookkeeping and exact timing, not an LLM task.

**Put in Astra prompts:** choose curriculum/variation policy and revise priorities after a wave or death.

**Put in Jev questions:** classify `{"phase":"OPENING|MIDWAVE|LATEWAVE|RESPAWN"}` and `{"urgency":"LOW|MEDIUM|HIGH"}`.

### 7. Handle variations explicitly

The normal one-player game is the baseline. The manual's optional variations introduce moving shields, zigzagging laser bombs, fast bombs, and invisible invaders, as well as two-player modes. If the adaptor deliberately selects an ALE `mode`/`difficulty` variation, include those settings and a variation indicator in JSON. Do not apply the normal fixed-coordinate/straight-bomb assumptions blindly to invisible or zigzagging variants.

**Put in code:** configuration and feature flags, mode/difficulty in episode metadata, robust target representation, and a mode-specific decoder/test fixture.

**Put in Astra prompts:** identify which rule changed and switch the high-level plan.

**Put in Jev questions:** recognize `{"variant":"NORMAL|MOVING_SHIELDS|FAST_BOMBS|ZIGZAG_BOMBS|INVISIBLE|UNKNOWN"}` from state features. Do not ask Jev to discover a new variant or write a decoder.

## A practical rotation / control loop

1. **Observe:** obtain RAM-derived JSON plus score, lives, frame, wave estimate, object positions, active shot, bombs, shield masks, and UFO.
2. **Emergency check:** if a bomb is on a collision course or the cannon is boxed at an edge, dodge first; do not fire for points.
3. **Target:** if no shot is active, choose the safest living lowest invader in the preferred outer column; otherwise select the next safe column. Consider the UFO only if the safety predicate passes.
4. **Move/fire:** use `LEFTFIRE`/`RIGHTFIRE` to align and shoot, otherwise use `LEFT`/`RIGHT`/`NOOP`. Keep action holds short late in a wave.
5. **Verify:** use reward and object changes to determine whether the shot hit, missed, or is still active; release the fire lock only when the ROM allows another shot.
6. **Adapt:** after a life loss, wave reset, shield collapse, or mode change, have Astra revise the horizon and let Jev resume short typed choices.

This is a good division of labor: exact collision prediction, shot cooldown, boundary clamping, score/wave bookkeeping, and target geometry belong in code; strategic risk tradeoffs and adaptation belong in Astra; near-term recognition and typed left/hold/right/fire choices belong in Jev.

## Common mistakes and counters

| Mistake | Why it loses | Counter |
|---|---|---|
| Chasing every UFO | A harmless bonus distracts from bombs and low invaders | Require a safe-bonus predicate and prioritize survival |
| Firing continuously | The ROM permits only one active player beam; wasted shots create long gaps | Maintain a fire lock and align before each shot |
| Shooting high rows while a low column survives | Score value is irrelevant if the formation reaches Earth | Clear the lowest/front threat first unless a safe high shot is already in flight |
| Clearing only the center | It leaves the formation wide and gives the edges more opportunity to step down | Prefer an outer-column opening, then re-evaluate danger |
| Hugging a boundary | The manual warns the cannon can be trapped at its movement limit | Keep an escape margin and predict bomb lanes |
| Treating shields as permanent | Both sides' fire damages them and gaps let projectiles through | Track shield masks and rotate cover |
| Importing arcade rules | This port has 36 aliens, fixed row values, and a fixed normal UFO bonus | Use the Atari-2600 manual and ROM-specific calibration |
| Using raw Gym action numbers with direct ALE | Gym's compressed action index 2 is RIGHT, while full ALE enum values have UP at 2 | Keep a single explicit action map per API |
| Ignoring sticky actions/frameskip | An apparently correct dodge may be repeated or advance four frames | Set them explicitly and test robustly at the evaluation setting |
| Asking Jev to plan a wave or do arithmetic | Jev is a fast recognizer, not a planner/calculator | Send normalized JSON and ask independent typed near-term questions |

## What to measure during strategy search

Record at least: total score/reward, terminal reason, lives lost, waves cleared/inferred, invaders killed by row, shot hit/miss/timeout rate, UFO attempts and success, bomb-collision rate, shield damage at death, time spent at movement boundaries, and action latency/frame count. A strategy that scores a little less per wave but preserves lives and clears more waves is stronger. Evaluate held-out seeds with the same mode/difficulty and with sticky-action behavior documented; use separate tests for each optional variation.
