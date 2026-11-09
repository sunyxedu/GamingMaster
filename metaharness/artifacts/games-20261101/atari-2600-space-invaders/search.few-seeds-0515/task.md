Design a harness for a **real-time** game (tier 2).

Game: `Atari 2600 Space Invaders` — Atari 2600 Space Invaders running locally in ALE/Stella, canonical mode=0, difficulty=0. The world is real-time: each tick advances 4 NTSC frames (0.0667 s), even if the controller repeats or misses a decision. Clear the six-by-six invader formation, survive with three lives, and maximize the ALE score. Invader values are 5, 10, 15, 20, 25, and 30 points from top row to bottom row; a normal UFO is worth 200 and a complete 36-invader wave is 630 points. The game ends when all lives are lost.

Actions are strings from action_spec: noop, fire, right, left, right_fire, and left_fire. Combined actions move and fire on the same tick. The player has one laser in flight at a time; firing while it is active does not create a second beam. The formation moves sideways, descends at an edge, and drops gray bombs; shields absorb some hits but are destructible. A late action cannot rewind the emulator.

Observation coordinates are the 160x210 NTSC screen coordinate system (x=0..159, y=0..209), not a screenshot. Fields are versioned JSON: mode, difficulty, frame, score, lives, game_over, terminal_cause, wave, waves_cleared, invaders_killed, player, player_shot, 36 invaders with row/col/x/y/alive/value, bombs, three shields, ufo, and any malformed action warning. Shield health_mask has nine integer rows; each row is an 8-bit left-to-right occupancy mask. Bomb and laser positions are decoded from the ROM's gray projectile pixels. The primary Gym observation is RAM; RGB is read internally only for this symbolic decoder.

reset(seed) selects ALE's seeded emulator/sticky-action RNG. The benchmark uses explicit frameskip=4 and repeat_action_probability=0.25 with mode and difficulty fixed at zero; changing cartridge modes or difficulty can alter palette, shields, bomb rules, or multiplayer behavior and is not covered by the canonical decoder.

Actions:
- noop: Release the joystick and fire button; the formation and shots continue moving.
- fire: Press FIRE without moving; only one player laser can be in flight.
- right: Hold the joystick RIGHT for this emulator tick.
- left: Hold the joystick LEFT for this emulator tick.
- right_fire: Move RIGHT and press FIRE on the same tick.
- left_fire: Move LEFT and press FIRE on the same tick.
(real-time games: ticks apply 'noop' until the first action is posted; afterwards a tick without a fresh action repeats the last posted action)
Each tick advances four NTSC Atari frames by default (~66.7 ms). A tick with no fresh valid action repeats the last valid action; before the first action it uses noop. Reversing is allowed, but bombs and the descending formation continue during every tick.

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

The game ticks every 0.06666666666666667s **on its own clock**; `ctx.act(a)` posts the current action (instant).
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
- strategy_summary: Start near the center, preserve an escape route, and usually clear a safe outer column before working inward; treat this as a heuristic rather than a hard rule. Fire only when aligned with a living invader, prefer the lowest surviving threat in a column, and maintain a one-shot/fire lock. Dodge bombs and use shields as temporary cover, prioritizing survival and low invaders over upper-row points or the harmless UFO. As the formation shrinks and speeds up, shorten Jev's action holds and let Astra revise the wave-level risk policy. Put geometry, collision prediction, fire cooldown, boundary clamping, score bookkeeping, and wave inference in code; use Jev for typed near-term choices and Astra for adaptation and tradeoffs.
- scoring: Report the sum of ALE rewards as the Atari score: invaders award 5, 10, 15, 20, 25, or 30 points by initial row, a complete 36-invader wave is 630, and the normal one-player UFO is 200. Also report lives remaining, waves cleared/inferred, invaders killed, and terminal cause because score alone can reward risky play that ends early.
- interface.observation: Primary Gym observation is a 128-byte uint8 Atari console RAM array. The adaptor must decode it into object-level JSON; ALE has no official Space Invaders object schema. During development, obs_type=rgb provides a 210x160x3 uint8 array for ROM-specific geometry/palette calibration, and info provides lives and frame counters. Reward is the score delta; terminated/truncated report episode completion. Direct ALEInterface additionally exposes raw RAM, RGB/grayscale screens, legal actions, and cloneable emulator state.
- interface.actions: Gym Discrete(6): 0 NOOP, 1 FIRE, 2 RIGHT, 3 LEFT, 4 RIGHTFIRE, 5 LEFTFIRE. Use short holds and explicit fire-lock handling because only one player beam may be active. Direct ALEInterface should use getMinimalActionSet()/Action enums rather than treating Gym's compressed numeric indices as full ALE enum values.

### study/strategies.md
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

### study/brief.md
# Atari 2600 Space Invaders

## Scope and canonical game

This study targets the Atari VCS/2600 cartridge version of **Space Invaders** (the ROM bundled by current ALE identifies itself as `Space Invaders (1978) (Atari) [!]`, 4 KiB, NTSC). It is not the arcade ROM and not an Atari 8-bit port. The cartridge's scanned Atari manual is dated 1980, while the game itself is the 1978 Atari release. The practical emulator target is `ALE/SpaceInvaders-v5`.

Primary sources:

- [Atari manual, HTML transcription](https://atariage.com/manual_html_page.php?SoftwareLabelID=460)
- [Atari manual, scanned PDF](https://atarionline.org/wp-content/uploads/2021/11/space-invaders-1978-atari-2600-game-instructions-manual.pdf)
- [ALE SpaceInvaders environment](https://ale.farama.org/environments/space_invaders/)
- [ALE installation and interfaces](https://ale.farama.org/getting-started/)
- [Computer Archaeology Atari 2600 technical pages](https://www.computerarcheology.com/Atari2600/SpaceInvaders/)

## Objective and termination

The player controls a laser cannon at the bottom of the screen. Destroy the formation before the lowest invader reaches Earth/the bottom, and avoid losing the cannon to enemy laser bombs. A normal wave contains **36 invaders in six rows of six**. When all 36 are destroyed, another set appears; the manual says each reset set starts closer to Earth, so surviving waves become progressively more urgent. The game ends when the invaders reach Earth or the player has taken the three enemy hits/lost all lives. In Gymnasium/ALE this is reported as `terminated=True`; `info` also exposes a `lives` count.

The long-term objective is the score, but survival and clearing waves are the useful progress measures for an agent. Keep total reward/score, waves cleared or inferred from formation resets, invaders killed, lives remaining, and terminal cause where the adaptor can obtain them.

## Screen, controls, and actions

A one-player game uses the joystick in the left controller port:

- Joystick left/right moves the cannon horizontally along the bounded bottom lane.
- The red controller button fires the laser beam.
- Only one player shot can be in flight: after firing, the cannon cannot fire again until that beam hits a target or leaves the top of the screen.
- Game Select chooses one of 112 cartridge variations; Game Reset starts the selected game.
- The difficulty switch changes cannon size: the manual describes the `B` position as a smaller cannon and `A` as a larger cannon that is more exposed to enemy fire.

The recommended ALE Gymnasium action space is `Discrete(6)`:

| Gym action | Meaning |
|---:|---|
| 0 | NOOP |
| 1 | FIRE |
| 2 | RIGHT |
| 3 | LEFT |
| 4 | RIGHTFIRE |
| 5 | LEFTFIRE |

Do not confuse this compressed Gym mapping with the numeric values of the full ALE `Action` enum. If using `ALEInterface` directly, use `getMinimalActionSet()`/the enum values rather than assuming Gym's indices.

## Formation, fire, shields, and UFO

The invader block traverses horizontally and steps downward at the sides. Destroying aliens makes the remaining block move faster (a standard and observed Atari-2600 pressure mechanic); consequently the end of a wave is more dangerous than its opening. Enemy bombs fall toward the cannon, and the cannon must dodge while maintaining firing alignment.

Three shields sit between the formation and the cannon. They provide temporary cover, but both player fire and enemy bombs wear them away. Gaps eventually let both kinds of beam/bomb through, and the shields disappear as the formation gets close. The command alien/UFO periodically crosses the top: it is harmless and does not drop bombs, but is a bonus target. In the normal one-player games it is worth 200 points; the manual specifies 100 in the two-player competitive variants C and D.

The manual's helpful hints specifically warn not to concentrate so hard on the UFO that an enemy bomb hits the cannon, and warn that the bottom movement bars are the cannon's limits, where it can be trapped by fire.

## Scoring

The value is fixed by the invader's **initial row**, not by how far it has descended:

| Initial row (bottom to top) | Points |
|---:|---:|
| 1 | 5 |
| 2 | 10 |
| 3 | 15 |
| 4 | 20 |
| 5 | 25 |
| 6 | 30 |

A complete 36-invader wave is 630 points. The four-digit display tops out at 9999 even though scoring can continue. ALE reward is the score delta, so summing rewards is the primary score. The adaptor should not substitute the arcade game's variable UFO scoring for this fixed Atari-2600 table.

## Variations and difficulty

The cartridge includes 112 game variations, including moving shields, zigzagging bombs, fast bombs, invisible invaders, and one-/two-player competitive or partnership modes. The normal single-player game is the best canonical benchmark. ALE exposes `mode` values 0–15 and `difficulty` values 0–1; document the selected values in an episode record because these settings change the task. `mode=0`, `difficulty=0` should be the baseline unless the evaluator explicitly asks for a variation.

## Pace and tier

This is **tier 2**: the game is a continuously animated action game, not a turn-based world. NTSC Atari video is nominally about 59.92 Hz ([Atari 2600 specifications](https://problemkaputt.de/2k6specs.htm)). ALE v5's default `frameskip=4` advances four emulator frames per `step` (about 66.8 ms of game time), and its default sticky-action probability is 0.25 ([ALE environment specification](https://ale.farama.org/environments/)). The formation, bombs, player shot, and UFO can all change while the agent is deciding.

Use a tight low-level control loop: Jev should choose a short action/hold and be queried again at roughly one ALE step or a small multiple of it; Astra should plan between batches rather than blocking the loop. A practical safe reaction budget is about **0.25 s** of game time; at a dangerous bomb or late-wave alignment, even 0.1 s can matter. The JSON deliverable uses 0.25 seconds as the conservative model-level decision budget, while the adaptor may use `frameskip=1` for calibration and `frameskip=4` for normal operation.

## References

The manual is the authoritative source for the 36-invader wave, controls, one-shot limitation, hits, shields, UFO, scoring, and cartridge variations. The [AtariAge strategy discussion](https://forums.atariage.com/topic/15341-space-invaders-strategies/) and [StrategyWiki walkthrough](https://strategywiki.org/wiki/Space_Invaders/Walkthrough) provide player heuristics such as clearing outer columns; treat those as community strategy rather than additional ROM specification. ALE's [SpaceInvaders page](https://ale.farama.org/environments/space_invaders/) supplies the machine interface, action mapping, observation shapes, frame skip, reward description, modes, and difficulty values.

### adaptor/ADAPTOR.md
# Atari 2600 Space Invaders adaptor

## Connection and setup

`game.py` uses the maintained local `ale-py`/Gymnasium `ALE/SpaceInvaders-v5`
environment.  Stella runs in-process and headlessly; no display, GPU, account,
server, or credential is needed.  The current `ale-py` package supplies the
verified `space_invaders.bin` ROM.  If an evaluator intentionally keeps ALE ROMs
elsewhere, it may set the optional `ALE_ROMS_DIR` shown in `env.example`.

The adaptor's default benchmark configuration is explicit:

- `mode=0`, `difficulty=0` (canonical one-player rules)
- `frameskip=4` (four NTSC frames per action, about 66.7 ms)
- `repeat_action_probability=0.25` (ALE v5 sticky-action RNG)

`reset(seed)` calls Gymnasium's seeded reset on a fresh episode.  The initial
cartridge opening is normally identical, but the seed controls ALE's seeded
random/sticky-action stream and therefore bomb/shot/action trajectories.  The
module declares train seeds `0,1,2,3` and held-out seeds `100,101`; they are
scenario RNG seeds, not level numbers.  Optional constructor kwargs can expose
other ALE modes/difficulty settings for experiments, but those are not covered
by the canonical decoder and are recorded in `info()`.

## Interface

The returned object is a tier-2 `RealtimeGame`.  Each `tick(action)` posts a
valid action and synchronously advances exactly one configured ALE step.  If the
action is missing, the last valid action is repeated (and `noop` is used before
the first action).  A malformed action is ignored and reported in the next
observation rather than raising an evaluator-fatal exception.  `ticks` counts
successful emulator steps.  `close()` closes the Gym/ALE instance and is
idempotent.

Legal action names are `noop`, `fire`, `right`, `left`, `right_fire`, and
`left_fire`.  Gym's six-action mapping is used: NOOP, FIRE, RIGHT, LEFT,
RIGHTFIRE, LEFTFIRE.  The combined actions move and fire on the same tick.

## Observation and decoder

Gym receives the 128-byte RAM observation, while the adaptor reads ALE's current
RGB buffer internally to build object-level JSON.  It does not expose a binary
screenshot or raw RAM to the harness.  The compact observation contains:

- mode, difficulty, frame, ALE score, lives, game-over/terminal cause, inferred wave,
  waves cleared, and invaders killed;
- player x/y and one decoded player laser;
- all 36 invader row/column entries, current lattice x/y, alive flag, and row
  value (5 through 30 from top to bottom);
- gray bomb positions, three shield positions/occupancy masks, and the purple
  UFO position when present;
- a malformed-action warning field when applicable.

The decoder is calibrated to the packaged NTSC ROM: invader sprites are matched
against their known 6x6 lattice, shield masks are read from their fixed regions,
and projectile motion between observations distinguishes rising lasers from
descending bombs.  Coordinates are 160x210 screen coordinates.  RGB decoding is
ROM-specific; PAL, altered palettes, invisible-invader modes, moving shields,
and multiplayer modes may make object fields approximate.  `info()` always
records `mode`, `difficulty`, `frameskip`, sticky-action probability, terminal
cause, and validity diagnostics.

## Scoring and limits

`score` is the sum of ALE rewards, i.e. the Atari score: invaders award 5, 10,
15, 20, 25, or 30 by initial row, a full wave is 630, and the normal UFO is 200.
`score_breakdown()` additionally reports lives remaining, inferred waves cleared,
invaders killed, UFO hits, and survival ticks.  `info()` reports the terminal
cause (`all_lives_lost`, `game_over`, `time_limit`, or an infrastructure error).

The adaptor starts no child process and writes no run-time files.  ALE's packaged
ROM remains a local dependency; this adaptor does not redistribute it.

### study/interfaces.md
# Programmatic interfaces for Atari 2600 Space Invaders

## Recommendation

Use the **local ALE emulator through `ale-py` and Gymnasium, with `obs_type="ram"` as the primary observation and RGB only as a calibration/fallback channel**. The evaluator's Python 3.12.3 was tested with `gymnasium[atari]`: current packages installed were `ale-py 0.12.1` and `gymnasium 1.4.0`; `space_invaders.bin` was present in the package and the environment reset successfully with no display. ALE is native Stella, fast, deterministic when configured, supports seeds, raw RAM, score/lives/termination, and clone/restore snapshots. It has no server, account, display, GPU, anti-cheat, or game-website ToS dependency.

Important limitation: ALE does **not** publish a game-specific object API for this ROM. `obs_type="ram"` is the 128 raw console bytes, not a JSON list of aliens. The Computer Archaeology RAM page currently labels locations generically (`m80` through `mFF`) and does not provide a complete supported semantic map ([RAM page](https://www.computerarcheology.com/Atari2600/SpaceInvaders/RAMUse.html)). The adaptor must therefore normalize RAM into JSON itself, using a decoder validated against RGB frames and score/life transitions. That still avoids screenshots in the Astra/Jev interface: the models receive a compact object state, while the emulator remains local and headless.

## Install and smoke test

Run all commands with the evaluator interpreter, not the system Python:

```bash
PY=/home/hemlix/GamingMaster-mh-tier2/runs/atari-2600-space-invaders/.venv/bin/python
"$PY" -m pip install --upgrade 'gymnasium[atari]'
```

The current ALE README says that `ale-py` packages the Atari ROMs and that `gymnasium[atari]` installs the required modules and ROMs ([official README](https://github.com/Farama-Foundation/Arcade-Learning-Environment#quick-start)). There is no need for AutoROM or an online ROM download with the current package. If an older locked environment is used, follow that release's ROM/license instructions rather than silently downloading a ROM.

Minimal **interface setup** (not a policy) is:

```python
import gymnasium as gym
import ale_py

gym.register_envs(ale_py)
env = gym.make(
    "ALE/SpaceInvaders-v5",
    obs_type="ram",
    frameskip=1,                 # use 4 for the standard v5 step size after calibration
    repeat_action_probability=0.0, # set 0 for repeatable calibration
    mode=0,
    difficulty=0,
)
obs, info = env.reset(seed=seed)
# adaptor calls env.step(one of 0..5), then closes with env.close()
```

For the direct interface, which is useful when Astra wants snapshot-based planning or when Gym wrappers are undesirable:

```python
from ale_py import ALEInterface, roms
ale = ALEInterface()
ale.setInt("random_seed", seed)
ale.setFloat("repeat_action_probability", 0.0)
ale.setInt("frame_skip", 1)
ale.loadROM(roms.get_rom_path("space_invaders"))
ale.reset_game()
ram = ale.getRAM()
frame = ale.getScreenRGB()
state = ale.cloneState(include_rng=True)
```

Settings that affect emulation must be set before `loadROM`. The direct API and its state methods are documented in the [ALE Python interface](https://ale.farama.org/python-interface/) and [environment specification](https://ale.farama.org/env-spec/).

## Chosen interface details

### ALE/Gymnasium RAM environment — chosen

- **Maturity:** Production/stable ALE package, widely used Atari RL interface; native Stella emulator. Current official package metadata supports CPython 3.11–3.15, so the supplied 3.12.3 interpreter is in range.
- **Latency:** Native emulator step is far faster than real time on a CPU. With `frameskip=1`, one action advances one Atari frame (~16.7 ms NTSC); v5's standard `frameskip=4` advances ~66.8 ms. Python observation allocation and an optional NumPy decoder are the main overhead. `repeat_action_probability=0` gives deterministic calibration; v5's documented default is 0.25 for robust evaluation.
- **Observation:** `obs_type="ram"` is `Box(0,255,(128,),uint8)`. Gym's `info` includes at least `lives`, `episode_frame_number`, and `frame_number` for this environment. A parallel `obs_type="rgb"` environment returns `(210,160,3)` `uint8`; grayscale returns `(210,160)`. `reward` is the score increment, and `terminated` covers game over; `truncated` is an adaptor/time-limit condition.
- **Control:** Gym `Discrete(6)` is exactly `NOOP=0`, `FIRE=1`, `RIGHT=2`, `LEFT=3`, `RIGHTFIRE=4`, `LEFTFIRE=5`. Use combinations so movement and firing can happen on the same step. `full_action_space=True` exposes the 18-action Atari set, but the extra up/down actions are ineffective here and should not be sent.
- **Snapshots:** Direct `ALEInterface.cloneState()`/`restoreState()` support planning and regression tests. The default clone omits the sticky-action RNG; use `include_rng=True` when exact future replay is required. Serialized states are not guaranteed compatible across ALE versions.
- **Cost/legal risk:** No account or service cost. `ale-py` is GPL-2.0-only; the packaged Atari ROM is copyrighted game data, so do not redistribute the ROM or assume the package license grants independent commercial ROM rights. Use the packaged ROM under the package/evaluator's permitted terms and keep it local. There is no online anti-cheat or game-server ToS issue.

### How to produce the adaptor's object JSON

The model-facing observation should be a compact, versioned JSON object, for example:

```text
{
  "frame": integer,
  "score": integer,
  "lives": integer,
  "game_over": boolean,
  "wave": integer_or_null,
  "player": {"x": integer, "y": integer},
  "player_shot": {"active": boolean, "x": integer, "y": integer_or_null},
  "invaders": [{"row": 1..6, "col": 0..5, "x": integer, "y": integer, "alive": boolean}],
  "bombs": [{"x": integer, "y": integer}],
  "shields": [{"x": integer, "y": integer, "health_mask": ...}],
  "ufo": {"active": boolean, "x": integer_or_null}
}
```

Build it in two stages:

1. **Calibration/development:** run a second `obs_type="rgb"` stream with `frameskip=1` and no sticky actions. The initial canonical frame has six evenly spaced columns and six rows; in the current ALE RGB coordinates invader boxes begin around x=22,38,54,70,86,102 and y=31,49,67,85,103,121, three shield regions are around x=42,74,106 and y=157–174, and the player is near the bottom around y=185–194. These are calibration observations, not a promise that every PAL/variation frame has identical coordinates. Detect the known palette/object geometry and account for sprite flicker.

…(truncated; full text in prior/interfaces.md)

| **RetroArch/libretro Stella core** | Mature frontend/core ecosystem; headless operation is possible but frontend configuration adds latency/fragility | Video frames, configured controller input, saves/states; memory access depends on core/frontend bindings and is not a stable object API | Local/free frontend; ROM rights remain; no account | A viable generic emulator fallback when ALE cannot load a supplied ROM, but worse than ALE for RAM and Python state access. See [Stella libretro](https://github.com/libretro/stella-libretro). |
| **Javatari / browser emulator** | Good visual/manual play and preservation demos; browser/JS event latency | Canvas pixels, keyboard/joystick events, emulator UI; no standard Python object/RAM API | Local if assets are vendored; browser/DOM dependency; ROM copyright; free | Useful for human validation only. The Computer Archaeology page links a playable Javatari instance, but it is not headless. |
| **Screen capture + input injection** (`Xvfb`, SDL/RetroArch/Stella, `ffmpeg`/OpenCV, `xdotool`/`evdev`) | Lowest maturity and highest latency; image parsing and virtual display can drop frames | Screenshots/video and synthetic left/right/fire events; no score/RAM unless OCR/pixel inferred | No online ToS, but many moving pieces; free; requires a virtual display and more CPU | Last-resort fallback. It violates the preference for symbolic state and makes a 0.1–0.25 s reaction budget unreliable. |

## Environment variables and credentials

There is no server address, account, API key, or other credential. The normal chosen setup needs no environment variables and no `DISPLAY`; do not request or write credentials.

- `ALE_ROMS_DIR` is an **optional** override if an evaluator deliberately keeps ROMs outside the installed package. Put a directory path there, never a ROM or credential value. With current `ale-py`, omit it and use `roms.get_rom_path("space_invaders")`.
- A Stella/RetroArch GUI fallback may need `DISPLAY` or `SDL_VIDEODRIVER=dummy`/Xvfb configuration, but the chosen ALE path does not.
- Use exactly `/home/hemlix/GamingMaster-mh-tier2/runs/atari-2600-space-invaders/.venv/bin/python`; do not install into system Python.

## Reset, seeds, and reproducibility

`env.reset(seed=n)` is the Gym entry point. For direct ALE, set `random_seed` before `loadROM`; ALE uses this for emulator randomness such as sticky actions. For deterministic calibration set `frameskip=1`, `repeat_action_probability=0.0`, and a fixed seed. For evaluation, test both the normal v5 sticky setting (0.25) and a documented deterministic setting if the benchmark requires repeatability. Keep `mode` and `difficulty` fixed within a comparison. After `terminated`, ALE does not continue emulating until reset.

Recommended adaptor setup sequence:

1. Verify the evaluator Python and import `ale_py`, `gymnasium`, and NumPy.
2. Register ALE environments and make `ALE/SpaceInvaders-v5` with `obs_type="ram"`.
3. Set `mode=0`, `difficulty=0`, and explicit frameskip/sticky settings; do not rely on changing package defaults.
4. On reset, record seed, mode, difficulty, frame number, lives, and initial RAM.
5. Decode each RAM observation into the model JSON; during decoder development compare against an RGB environment/fixture.
6. Map only the six supported Gym actions; step until `terminated or truncated`, summing rewards and recording terminal reason/lives.
7. Close the environment and discard snapshots between episodes.


## Measured model latency (at archive creation, 2026-10-30 05:09)
- System-2 (Astra), one-word answer: 30 s
- System-2, an answer of ~1851 output tokens: 68 s — latency grows with the length of the answer; size timeouts for the answers your prompts actually request
- System-1 (Jev), one request with the game's initial observation and a Choice over its actions: median 148 ms (= 2.2 ticks), p90 267 ms, max 269 ms over 12 warm requests; first request on a new connection 231 ms — a timeout below the tail turns slow answers into failed calls
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
