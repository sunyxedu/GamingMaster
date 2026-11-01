Design a harness for a **real-time** game (tier 2).

Game: `Atari 2600 Ms. Pac-Man` — Ms. Pac-Man on the Atari 2600 CX2675 ROM runs continuously in a local headless ALE emulator. Your objective is to maximize score while eating pills and power pills, collecting fruit when safe, surviving, and clearing mazes. A normal pill is worth 10 points, a power pill 50, fruit and ghost chains add score, and losing all lives ends the episode. The opening maze uses ALE ghost mode 0 (4 configured ghosts); mode 0 is the main four-ghost evaluation and modes 1/2/3 are one-/two-/three-ghost curriculum strata. 

Actions are the nine names in action_spec: noop, up, right, left, down, up_right, up_left, down_right, and down_left. Each tick applies the latest posted action and advances 4 Atari frames (tick_s is 0.066667 seconds at 60 Hz); before the first action, noop applies, and a missed decision repeats the last action. Cardinal actions are safest. The emulator does not pause while the model reasons, and the game may need its automatic opening sequence before Ms. Pac-Man begins moving. 

Observations are symbolic JSON: episode frame, cumulative ALE score and last reward, lives, terminal state, mode, player position/direction/velocity, four ghost slots, fruit and decoded fruit kind, power pills, remaining pills with row/column coordinates, HUD score, maze-progress counters, a raw 128-byte RAM snapshot, and a maze-map note. RAM does not encode wall geometry, so the maze map is explicitly marked unverified; use object coordinates and movement to infer legal corridors. Ghost RGB is their normal colour and is not a reliable frightened-state flag. The primary score is cumulative ALE reward; HUD score and survival/progress components are reported separately.

Actions:
- noop: Keep the joystick neutral for this action quantum.
- up: Hold the joystick up.
- right: Hold the joystick right.
- left: Hold the joystick left.
- down: Hold the joystick down.
- up_right: Hold up and right (ALE diagonal; cardinal movement is recommended).
- up_left: Hold up and left (ALE diagonal; cardinal movement is recommended).
- down_right: Hold down and right (ALE diagonal; cardinal movement is recommended).
- down_left: Hold down and left (ALE diagonal; cardinal movement is recommended).
(real-time games: ticks apply 'noop' until the first action is posted; afterwards a tick without a fresh action repeats the last posted action)
Each tick advances the fixed ALE frame skip (normally four Atari frames, about 67 ms at 60 Hz). The game continues between decisions; if no new action is posted, the previous action repeats. Ms. Pac-Man can reverse, but turns must be taken at legal maze corridors.

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
- strategy_summary: Strong play clears dots through a maze-specific legal-corridor graph while keeping a tunnel or power pill as an escape, rather than importing an arcade map or chasing every fruit. Code should handle wall legality, pill/life/maze bookkeeping, ghost projection, score arithmetic and a short action queue; Astra should choose the maze-wide risk policy and adapt routes; Jev should make fast typed local choices among already computed legal directions and pill/escape options. Lure clustered normal ghosts toward a power pill when safe to chain 200/400/800/1600 points, but take the pill immediately when it is the only survival route. Use side tunnels as wraparound escapes, learn each layout's different tunnel locations, and clear the reported third-maze inverted-T pocket early while few ghosts roam. Reduce ghost-chain risk as frightened time shortens after repeated maze clears, and evaluate score together with mazes completed and lives preserved.
- scoring: Report cumulative ALE reward/final decoded HUD score as the primary score because ALE reward is the score delta for dots, pills, fruit, pretzels and ghost chains. Also report mazes cleared, visible pills/dots eaten, lives remaining, deaths, mode and terminal state: score alone can reward risky ghost/fruit hunting while maze completion and survival better measure strong play.
- interface.observation: A local JSON adaptor should expose frame number, reward delta, cumulative score, ALE total lives, game_over, mode, player x/y and direction if available, visible ghosts with x/y/dx/dy, fruit, remaining power pills, visible pills keyed by grid cell, maze-map confidence and raw_ram[128]. OCAtari REM supplies the dynamic object positions and HUD score/life objects; getRAM supplies the original 128 bytes. render_mode=rgb_array optionally supplies a 210x160x3 frame for map calibration or visual frightened-state checks, but normal Jev input should be symbolic JSON.
- interface.actions: The adaptor sends ALE Discrete(9) actions: 0 NOOP, 1 UP, 2 RIGHT, 3 LEFT, 4 DOWN, plus optional diagonals 5 UPRIGHT, 6 UPLEFT, 7 DOWNRIGHT and 8 DOWNLEFT. Use cardinal actions for the four-direction joystick, batch them for a few frames to hide model latency, and maintain the last direction because the game never waits for a decision.

### study/strategies.md
# Strong-play strategy for Atari 2600 Ms. Pac-Man

## Strategic premise

This is a continuous maze-navigation game, not a puzzle in which Astra can deliberate between moves. Strong play is a repeated loop:

1. keep Ms. Pac-Man moving on a legal corridor and preselect the next turn;
2. clear dots in a route that leaves an escape corridor and a power pill available;
3. use a pill deliberately when ghosts are close enough to chain, or immediately when survival requires it;
4. use side tunnels to break pursuit;
5. update a route/map model after every turn, death, maze flash and fruit event.

The Atari 2600 port has its own maze geometry and timing. Arcade ghost-target formulas and arcade route tables are not safe priors. The manual's rules and the CX2675/ALE state should dominate; the Robert Guerra Atari 2600 guide supplies useful but version-sensitive route advice.

## Opening and route rotation

### Opening priorities

* **First objective: establish motion and map alignment.** At reset, do not spend a slow model call deciding whether to move. Start a safe cardinal direction, observe the first few RAM/RGB/object states, and confirm coordinate orientation, ghost count and which of the two decoder pill-grid/map hypotheses is active.
* **Clear dots before hunting ghosts.** The manual explicitly recommends practising dot clearing before trying to eat ghosts. The first route should sweep a connected corridor region and preserve a power pill or tunnel as an exit, rather than crossing the ghost chamber to chase a low-value target.
* **Use early lives to learn, not to gamble.** A first-life plan should record legal turns, tunnel entrances, pill positions and ghost approach directions. Once the map is known, repeat the safe sweep with a small risk budget for fruit and ghost chains.
* **Third-maze warning.** The Atari 2600 guide calls out an inverted-T-shaped island below the start in its third-maze situation: clear it early while fewer ghosts are out, before it becomes a trap. Treat this as a high-value prior for the Astra prompt, but verify the visual/map identity because colour names and level counting can differ across presentations.

### Rotation between mazes

The manual says play starts at the cherry level and progresses through the fruit levels to banana; after banana, mazes alternate between maze three and maze four. It also says that after every two dot-cleared mazes fruit selection becomes random and frightened-ghost time decreases. Therefore:

* keep a `level/maze hypothesis` in code, updated on a maze-clear flash and fruit/colour observation;
* use the first visits to each layout to learn its tunnel locations and safe power-pill approach;
* on later maze-3/maze-4 repeats, rotate the previously learned dot sweep but shorten the ghost-hunting window as the frightened duration declines;
* do not treat fruit value as the only objective late in a maze: a 5,000-point banana is not worth a life if its route crosses a ghost with no escape.

The manual's exact tunnel warning is important: different mazes have different tunnels. A route planner must use a maze-specific graph, not a single arcade map.

**Delegation:** corridor legality, pill bookkeeping, maze-change detection and route repetition belong in **code**; Astra should select and revise a maze-level sweep; Jev should recognise the current local junction/escape situation and choose among the few safe cardinal directions.

*Sources:* [AtariAge manual, game play/variations/hints](https://atariage.com/manual_html_page.php?SoftwareID=1169), [Robert Guerra Atari 2600 guide](https://gamefaqs.gamespot.com/atari2600/584919-ms-pac-man/faqs/2632).

## Movement and local navigation

### Build a verified maze graph

OCAtari REM gives dynamic pill locations but not the wall geometry. During an adaptor calibration phase, use `rgb_array` frames or a verified CX2675 map to make a graph whose nodes are corridor cells/junctions and whose edges are legal cardinal moves. Store for each node:

* legal directions and the next turn coordinate;
* side-tunnel wrap edges and their direction of travel;
* nearby power pills and likely escape branches;
* whether the node is a dead end, a pocket, or a crossing likely to be entered by a ghost;
* the layout/map identifier and confidence.

This is **code**, not a Jev question. Do not repeatedly ask a foundation model whether a pixel corridor is open. Use the static graph plus current object positions in the JSON state.

### Pre-turn and direction hysteresis

Ms. Pac-Man continues moving while a model is thinking. Strong play commits to a turn before reaching the junction, because a late direction can leave her running into a wall or entering the wrong branch. Maintain the current direction and planned next turn in code; ask Jev only when the local state changes materially or a hazard invalidates the plan.

Avoid unnecessary NOOPs. Use a cardinal input for the next short action chunk, retain the current direction through a safe corridor, and queue a replacement direction early enough to reach the junction. With default ALE frameskip, a step is about 67 ms; with frameskip 1, it is one native 60-Hz frame. The adaptor—not Jev—should choose the chunk length and interrupt it on a near collision, tunnel entrance or junction.

**Code:** map legality, coordinate-to-cell conversion, turn timing, action queue, last direction and collision horizon. **Jev:** typed choice of `UP/RIGHT/LEFT/DOWN` at a supplied junction with illegal actions already removed. **Astra:** decide how conservative the queue should be and when to abandon a planned sweep.

### Safe shortest path is not always the best path

A plain nearest-pill BFS can drive into a corridor whose exit is occupied by a ghost. Give each candidate route a risk cost based on projected ghost distance and available escape branches. Prefer a slightly longer connected sweep if it keeps a tunnel or power pill reachable. Conversely, do not oscillate forever to avoid every ghost: the score/progress objective requires eventually entering exposed corridors.

Do the arithmetic in code. Jev is a recogniser/chooser and should receive features such as `legal_actions`, `ghost_distance_steps`, `ghost_approach`, `escape_count`, `nearest_pill_distance`, `power_pill_distance`, `fruit_safe`, and `route_options`, not a raw RAM dump and a request to calculate path lengths.

## Ghost avoidance and counters

### Normal ghosts

When ghosts are their normal colour, contact is a life loss. Track each visible ghost's position and recent velocity from consecutive REM states; map each to the nearest corridor cell. Use a short collision horizon rather than only current Manhattan distance: a ghost two cells away and approaching the same junction is more dangerous than one four cells away moving away.

At a junction, reject actions that:

* enter the ghost's projected cell before Ms. Pac-Man can turn;
* move into a dead end with no pill/tunnel escape;
* leave two ghosts on the only two exits;
* require reversing after the safe turn window has passed.

The strategy guide reports that the Atari 2600 remake's ghosts do not accelerate with progression in the same way as the arcade version. Treat that as a useful hypothesis, not an invariant: estimate actual relative speed from the emulator and keep a safety margin. Do not import arcade scatter/chase schedules unless the ROM observations prove them.

### Tunnels

The manual says Ms. Pac-Man exits the right side and re-enters on the left, and vice versa; it recommends the openings as escape tunnels. The Atari 2600 guide and community advice further recommend frequent tunnel use for survival because ghosts seldom use them to get around. The exact tunnel locations change by maze. A tunnel is therefore a counter to a trailing ghost, but not a guarantee: check the exit cell and the second ghost before committing.

**Code:** represent wrap edges, detect entrance/exit, and score the exit's projected danger. **Jev:** answer a fast yes/no/choice such as `is the nearest tunnel safer than the next corridor?` given precomputed risk. **Astra:** decide whether a maze's tunnel is a routine rotation or an emergency-only escape.

### Energy pills and ghost chains

A power pill is both a 50-point object and a temporary reversal of threat. The manual advises waiting near one, teasing ghosts closer, then eating it and taking as many ghosts as possible. The chain values are 200, 400, 800 and 1,600 on one pill. Since frightened time becomes shorter after later maze clears, the chain opportunity must be judged locally, not assumed from an arcade timer.

Use two modes:

* **Score mode:** if two or more ghosts are close, the next route has a clean exit, and the pill is not needed as the only escape, take the pill and intercept the nearest safe chain. Keep the chain bounded by the remaining frightened-time estimate; do not turn into a normal ghost after a greedy fourth target.
* **Survival mode:** if a normal ghost is inside the collision horizon, consume the nearest reachable pill immediately and route to the largest safe corridor/tunnel. A 50-point pill and a saved life are better than waiting for a perfect four-ghost chain.

The released OCAtari REM decoder exposes remaining power-pill positions but not a reliable `frightened` boolean/timer; its normal ghost RGB values are hard-coded. Use a calibrated RGB/Vision check or a code-maintained temporal state inferred from the pill event and ghost visual change. Do not ask Jev to infer a timer from a raw 128-byte vector.

…(truncated; full text in prior/strategies.md)

| Sending a direction after the junction is already passed | Pre-turn at the graph's turn coordinate; retain direction through safe corridors | Code |
| Taking every fruit | Require a safe path and an escape branch; otherwise continue dot sweep | Code + Astra risk policy + Jev TAKE/IGNORE |
| Eating a pill too early with no threat | Preserve it for a cluster, unless it is the only safe exit | Astra policy; Jev local recognition |
| Waiting for a four-ghost chain while a ghost approaches | Use survival mode and take the pill now | Jev yes/no plus code hazard horizon |
| Chasing a fourth frightened ghost after the timer is nearly gone | Cap target distance and abort the chain before normal colour returns | Code |
| Treating every visible ghost slot as active | Ignore `NoObject`/invisible slots and verify ghost count from RAM byte 19/mode | Code |
| Treating OCAtari `Life.value` as total lives | Use ALE `info["lives"]`/`ale.lives()`; the HUD value is spare icons | Code |
| Reading OCAtari's third/fourth step flags by name | Normalize the known swapped return tuple or query `ale.game_over()` | Code |
| Asking Jev to calculate paths or score | Supply distances, risk classes and point values already computed | Prompt design/code |
| Using NOOP while near a turn | Keep a cardinal direction and queue the next turn | Code + Jev choice |
| Ignoring late-level frightened-time reduction | Lower chain-risk tolerance after repeated maze clears | Astra policy; code timer |
| Assuming a tunnel always saves the player | Check the exit and other ghost positions before wrap | Code + Jev recognition |

## Delegation plan: code, Astra, Jev

### Put in deterministic code

* ALE/OCAtari setup, seeds, frame-skip and sticky-action configuration.
* Action integer mapping and cardinal-action filtering.
* JSON serialization of objects, raw RAM, reward, score and lives.
* `NoObject` visibility handling, coordinate transforms and ghost slot identity.
* Pill-set/life/maze/fruit bookkeeping and maze-clear detection.
* The verified wall/tunnel graph, BFS/A* route search, legal-turn detection and action queue.
* Ghost velocity/projection, collision horizon, reachable power-pill/fruit tests and expected score arithmetic.
* Terminal-state/termination-flag normalization and evaluation metrics.

### Put in Astra prompts

* Select a risk budget: completion, survival, or score/ghost-chain mode based on lives and level.
* Plan the next maze-wide dot sweep after code supplies the map and safe branches.
* Decide when the observed map/decoder conflicts with the prior and request calibration or a conservative fallback.
* Adapt the third-maze/inverted-T opening and tunnel rotation after observing actual CX2675 behaviour.
* Interpret repeated failures and change the policy; do not make Astra answer every 67-ms direction.

Astra's prompt should include the manual facts (side tunnels, scoring, progression, shrinking frightened time) and explicitly warn that arcade patterns do not transfer automatically.

### Put in Jev questions

Send a compact, named JSON state and ask independent typed questions that require recognition, not planning or arithmetic:

* `CHOICE`: which legal cardinal action among the supplied candidates is safest for the next short chunk?
* `YES_NO`: is a normal ghost inside the collision horizon?
* `CHOICE`: `TAKE_PILL_NOW`, `WAIT_FOR_CLUSTER`, or `IGNORE_PILL`?
* `YES_NO`: is the visible fruit path safe under the supplied risk features?
* `CHOICE`: `ESCAPE_TUNNEL`, `DOT_SWEEP`, or `GHOST_CHAIN`?
* `CHOICE`: which of the already computed escape branches has the lowest risk?
* `LABEL`: `NORMAL`, `FRIGHTENED`, `RETURNING_EYES`, `DEATH/RESPAWN`, or `MAZE_FLASH` when a visual/temporal detector supplies the candidate evidence.

Have code remove illegal actions and precompute distance/score quantities. Jev should not be asked to discover the maze, perform BFS, add the 200/400/800/1,600 scores, or remember a multi-maze route. Astra should not be blocked waiting for those fast recognitions.

## Sources and confidence

* **High confidence / authoritative for this port:** AtariAge manual for controls, lives, points, tunnels, variations and progression; ALE source for action/mode/terminal wrapper; OCAtari source for the released RAM decoder fields.
* **High confidence / implementation limitation:** OCAtari paper's finding that maze geometry is absent from RAM.
* **Medium confidence / human strategy prior:** Robert Guerra's Atari 2600 guide and AtariAge community tunnel advice. Use them to seed route learning, then validate on the bundled CX2675 ROM.
* **Low confidence if imported from elsewhere:** arcade ghost personalities/schedules, arcade maze maps, browser emulator behaviour, and generic Pac-Man RAM maps. Do not use these as Atari 2600 facts.

### study/brief.md
# Atari 2600 Ms. Pac-Man — game brief

## Identity and objective

The target is the Atari 2600 cartridge **Ms. Pac-Man (Atari CX2675, 8-KB ROM)**, not the arcade/MAME Ms. Pac-Man ROM. The AtariAge manual describes the home version as having multiple mazes, floating fruit and pretzels, four ghosts, and a short end-of-game cartoon. The Internet Archive entry identifies the 1982 Atari/GCC cartridge as CX2675; the ALE ROM used by the recommended local interface is an 8,192-byte ROM with MD5 `87e79cd41ce136fd4f72cc6e2c161bee` (the checksum is declared by ALE's MsPacman settings).

The player steers Ms. Pac-Man through the maze, eating every dot. Clearing all dots flashes the maze and starts a new maze. The score-maximising objective also rewards energy pills, moving fruit/pretzels, and ghosts eaten while frightened. A normal ghost kills Ms. Pac-Man on contact; the game consumes a life and restarts her after the death animation unless the game is over.

*Sources:* [AtariAge Atari 2600 manual](https://atariage.com/manual_html_page.php?SoftwareID=1169), [Internet Archive cartridge entry](https://archive.org/details/atari_2600_ms._pac-man_1982_atari_-_gcc_mark_ackerman_glenn_parker_cx2675), [ALE MsPacman settings and checksum](https://github.com/Farama-Foundation/Arcade-Learning-Environment/blob/master/src/ale/games/supported/MsPacman.hpp).

## Rules and scoring

The manual's scoring table is the authoritative scoring prior for this port:

| Object/event | Points |
|---|---:|
| Ordinary dot | 10 |
| Energy pill | 50 |
| Cherry | 100 |
| Strawberry | 200 |
| Orange | 500 |
| Pretzel | 700 |
| Apple | 1,000 |
| Pear | 2,000 |
| Banana | 5,000 |
| First ghost eaten on one energy pill | 200 |
| Second ghost on that pill | 400 |
| Third ghost on that pill | 800 |
| Fourth ghost on that pill | 1,600 |

The four ghost awards are consecutive on the **same** energy pill; the multiplier resets for a later pill. Ms. Pac-Man starts with three lives and receives one bonus life after the first 10,000 points. The Atari manual says energy pills turn ghosts blue only briefly; uneaten ghosts blink before returning to normal, while eaten ghosts' eyes return to the centre chamber.

The variation selected on the cartridge is a ghost-count choice:

* **Cherries / mode 0:** standard/arcade-style variation with four ghosts.
* **One Bear / mode 1:** one ghost, intended for beginners.
* **Two Bears / mode 2:** two ghosts.
* **Three Bears / mode 3:** three ghosts.

ALE's source confirms four available modes, maps mode 0 to the standard variation, and maps modes 1–3 to one, two, and three ghosts. All variations start at the easiest cherry level and advance through the fruit levels to banana. After banana, the manual says mazes alternate between maze three and maze four; after every two maze clears, fruit selection becomes random and frightened-ghost time decreases.

*Sources:* [AtariAge manual, sections 2, 5 and 6](https://atariage.com/manual_html_page.php?SoftwareID=1169), [ALE MsPacman C++ settings](https://github.com/Farama-Foundation/Arcade-Learning-Environment/blob/master/src/ale/games/supported/MsPacman.cpp).

## Controls and action semantics

On the physical Atari 2600, use the joystick in the **left controller port**. Up, down, left and right steer Ms. Pac-Man. The joystick also enters the warp tunnels. Pressing GAME RESET or the red joystick button starts the selected game; GAME SELECT cycles the variation. The difficulty switches are unused and TV TYPE only selects colour/black-and-white display.

The local ALE interface exposes the minimal 9-action set:

| ALE action | Meaning |
|---:|---|
| 0 | NOOP |
| 1 | UP |
| 2 | RIGHT |
| 3 | LEFT |
| 4 | DOWN |
| 5 | UP+RIGHT |
| 6 | UP+LEFT |
| 7 | DOWN+RIGHT |
| 8 | DOWN+LEFT |

Use the five cardinal/no-op actions in the adaptor. The diagonal actions exist because they are part of the ALE minimal set, but the game is a four-direction joystick game and there is no need to plan diagonals. There is no FIRE action required for normal Ms. Pac-Man play. The full 18-action Atari set can be requested, but it adds irrelevant buttons.

Ms. Pac-Man moves continuously once started: this is not a turn-based game. A direction is held through the emulator frames until another direction is sent. At a junction, a turn must be selected before the character reaches the turn; waiting for a slow model response can therefore cause a wall hit, a missed escape turn, or a ghost collision.

*Sources:* [AtariAge manual, sections 3 and 4](https://atariage.com/manual_html_page.php?SoftwareID=1169), [official ALE MsPacman environment page](https://ale.farama.org/environments/ms_pacman/).

## Maze, tunnel, and ghost behaviour

The manual says the tunnels are on the two **sides** of the maze: leaving on the right re-enters on the left and vice versa. It explicitly warns that different mazes have different tunnel locations. Treat a tunnel as a wrap edge, not as a normal dead-end corridor.

A strong default is to clear dots safely rather than chase ghosts. Stay near (but not directly on) an energy pill, draw ghosts toward it, then consume it when several are close enough to chain the 200/400/800/1,600 awards. If survival matters more than score, take the pill earlier and use a side tunnel to break pursuit. The manual recommends using the openings as escape tunnels and practising dot-clearing before attempting ghost hunts.

The published Atari 2600 strategy guide by Robert Guerra describes four distinct maze layouts with different tunnel placements and dangerous passages. Its useful port-specific priors include: learn the tunnel locations rather than assuming the arcade map; use tunnels for escape because ghosts seldom choose them; and clear the vulnerable lower/inverted-T section early in the third-maze situation before all ghosts are roaming. The guide also warns against waiting beside a lower energy pill without watching the lower corners. These are priors, not facts to hard-code blindly: confirm them against the exact CX2675 ROM and the selected mode.

Do not import arcade ghost target algorithms, arcade level-speed tables, or an arcade maze map without validation. Atari 2600 Ms. Pac-Man has its own layouts, timing, ghost-count variations, and port-specific quirks. The AtariAge manual and the ALE ROM are the appropriate rules sources; the strategy guide is a useful human-play supplement.

…(truncated; full text in prior/brief.md)

## Progress and evaluation signals

Use these signals together rather than treating a single pixel or object count as the whole objective:

1. **Primary score:** cumulative ALE reward, which is the score delta, or the decoded HUD score. It reflects dots, pills, fruit, pretzels, and ghost chains.
2. **Survival:** remaining lives, deaths, and whether the episode is terminal.
3. **Maze progress:** number of remaining visible pill objects / dots eaten and completed maze transitions. A maze completion is more meaningful than a brief score spike.
4. **Level/maze context:** current fruit/maze colour and the learned map identifier. Do not assume an object decoder's two pill-grid templates are a four-maze index.
5. **Risk-adjusted performance:** score at game over, mazes completed, and lives remaining. Ghost/fruit hunting can increase score while reducing expected completion, so report both score and progress.

ALE's 128-byte RAM and its reward wrapper are sufficient for score, lives, actor positions, fruit, power-pill presence, and dot bitfields. However, OCAtari's own paper notes that Pac-Man/Ms. Pac-Man maze geometry is not encoded in RAM. A symbolic adaptor must therefore retain a calibrated wall/turn map (learned once from RGB frames or supplied as a verified CX2675 map) while using RAM for dynamic objects. OCAtari's MsPacman decoder is useful but should be cross-checked: its `Life.value` is the displayed spare-life count, while ALE's `info["lives"]` includes the life currently being played; its ghost objects do not expose a reliable frightened-state flag.

*Sources:* [OCAtari paper](https://arxiv.org/html/2306.08649v2), [OCAtari MsPacman RAM decoder](https://github.com/k4ntz/OC_Atari/blob/master/ocatari/ram/mspacman.py), [ALE Python interface](https://ale.farama.org/python-interface/).

## Pace and tier

Atari emulation runs at 60 frames per second. The documented `ALE/MsPacman-v5` default repeats an action for four frames, so one environment step represents about 4/60 = **0.067 seconds** of game time; v5 also has 0.25 sticky-action probability unless overridden. With `frameskip=1`, the adaptor can observe every 0.0167 seconds. Real decisions are most urgent near an intersection, tunnel entrance, or approaching ghost; a safe corridor can tolerate a held action for longer.

Use **tier 2**. The world ticks while Astra reasons, and a slow call cannot block the emulator. A practical danger horizon is roughly **0.2 seconds** (about 12 native frames, or three default ALE steps): a nearby ghost or missed turn can punish a stale direction within that interval. Jev should issue typed cardinal-direction choices rapidly from the current JSON state, while the harness keeps a short action queue running and Astra replans asynchronously. A 0.5-second Jev call is too slow for unbuffered frame-by-frame control, so the adaptor should batch a direction for a few frames and interrupt the batch on a hazard/turn event.

*Sources:* [ALE MsPacman defaults](https://ale.farama.org/environments/ms_pacman/), [ALE environment specification](https://ale.farama.org/env-spec/).

### adaptor/ADAPTOR.md
# Atari 2600 Ms. Pac-Man adaptor

## Connection and setup

`game.py` runs a local, headless Atari Learning Environment (ALE) emulator using the
Ms. Pac-Man ROM bundled by `ale-py==0.12.1`. It does not contact a server, require an
account, or read credentials. Install `requirements.txt` into the evaluator's Python
3.12 environment. `SDL_VIDEODRIVER=dummy` is an optional headless setting; normal
operation requests no display and no RGB frames.

OCAtari 2.2.1 is used in RAM Extraction Mode. It decodes the raw 128-byte Atari RAM
into Ms. Pac-Man, ghost, fruit, power-pill, pill, HUD score, and life objects. The
adaptor retains the raw RAM in every observation for cross-checking reverse-engineered
fields.

## Interface and timing

This is a Tier 2 `RealtimeGame`. Each `tick(action)` advances one fixed ALE action
quantum, and the evaluator's clock calls it at `tick_s`. The default is four Atari
frames at 60 Hz (`frameskip=4`, `tick_s=0.066666...` seconds), with the documented
v5 sticky-action probability fixed at `repeat_action_probability=0.25`. ALE's seeded
sticky-action draws make otherwise identical train and held-out seeds diverge while
`reset(seed)` remains reproducible. The emulator runs faster than wall time; `tick()`
does not sleep. Before the first action, `noop` is current; a missing action repeats the
previous action. Malformed actions are ignored and the current action is applied, so a
late command cannot crash the episode.

The action vocabulary is:

- `noop`, `up`, `right`, `left`, `down`
- `up_right`, `up_left`, `down_right`, `down_left`

Cardinal directions are recommended. The Atari action is a joystick direction rather
than a turn-based move, and Ms. Pac-Man and the ghosts continue moving between model
decisions. The ROM's automatic opening sequence can consume the first several action
quanta before movement begins.

## Observations

Observations are strict JSON-sized dictionaries. They include:

- episode frame, tick count, cumulative ALE reward (`score`), decoded HUD score,
  last reward, lives, deaths, terminal state, and last action;
- player position, dimensions, velocity, and direction (RAM byte 56);
- four stable ghost slots with position/velocity/colour/visibility;
- fruit position/visibility and a best-effort fruit kind decoded from RAM byte 123;
- four stable power-pill slots and currently visible ordinary pills with OCAtari
  `row`/`col` grid coordinates;
- visible collectible count, eaten count, maze counters, configured ghost mode, and
  the original 128-byte RAM array;
- a maze note. Dynamic objects are in RAM, but wall geometry is not. `maze.map_confidence`
  is therefore `0.0` and `known_walls_or_turn_graph` says that geometry must be inferred
  or calibrated from RGB rather than pretending that OCAtari's pill grid is a wall map.

`score` is cumulative ALE reward, which is the score delta for dots, pills, fruit and
ghost chains in this ROM. `hud_score` is reported as a separate cross-check. The
optional `score_breakdown()` reports score, visible progress, mazes cleared, lives,
deaths, mode, and terminal state so a score-only strategy is not mistaken for strong
survival or maze completion.

## Seeds and scenarios

`DETERMINISTIC` is true. `reset(seed)` reseeds ALE and starts the fixed Atari opening
state; it does not select a later maze. The declared train seeds are `0..7` and held-out
seeds are `100..103`, disjoint reproducible scenario seeds. With the main
`EVAL_KWARGS`, mode `0` is the scored four-ghost stratum. The constructor accepts
`ghost_mode=1`, `2`, or `3` for separate one-, two-, and three-ghost curriculum/
robustness strata. ALE's mode mapping is:

| `ghost_mode` | configured ghosts | meaning |
|---:|---:|---|
| 0 | 4 | main evaluation |
| 1 | 1 | curriculum |
| 2 | 2 | curriculum |
| 3 | 3 | robustness |

Later mazes are reached only by actually clearing the current maze (or by an explicitly
verified emulator checkpoint, which this minimal adaptor does not silently create).
The `maze_number` and `mazes_cleared` fields are bookkeeping for such progression.
The Atari RAM/OCAtari decoder does not reliably expose frightened/blue ghost state;
normal ghost colour is retained but should not be interpreted as frightened status.

## Limits and legal asset note

This adaptor uses the ROM supplied by the installed ALE package and does not copy or
redistribute ROM bytes. Use it only under the applicable Atari/Namco/ALE terms. The
object decoder is version-pinned and its RAM coordinates are reverse engineered; raw
RAM is included so downstream consumers can detect decoder changes. There is no
screenshot path in normal observations and no GUI, GPU, server, or account dependency.

### study/interfaces.md
# Programmatic interfaces for Atari 2600 Ms. Pac-Man

## Recommendation

Use **ALE/ale-py running the bundled Atari 2600 ROM, wrapped by OCAtari's RAM Extraction Mode (REM)**. The base emulator is local, deterministic when configured so, headless, fast, and exposes the Gymnasium action/reward/reset contract plus the 128-byte Atari RAM. OCAtari adds a maintained MsPacman decoder for Ms. Pac-Man, ghosts, fruit, power pills, ordinary pills, score and lives, which is much closer to the requested object-level JSON state than a screenshot pipeline.

The recommendation is not to trust OCAtari as the sole source of truth. Keep the raw 128-byte RAM and (optionally) the 210x160 RGB frame alongside the serialized objects. OCAtari cannot recover maze walls from RAM; its own paper calls this out for Pac-Man/Ms. Pac-Man. Calibrate a wall/turn graph from RGB once, or retain a verified CX2675 map, then use REM/RAM for the dynamic state. This still avoids screenshots during normal action selection.

## Chosen stack: ALE + OCAtari REM

### Software and compatibility

The evaluator is Python 3.12.3. The following versions were installed and exercised in an isolated target using that interpreter on a headless Linux host:

```text
/home/hemlix/GamingMaster-mh-tier2/runs/atari-2600-ms-pac-man/.venv/bin/python
  gymnasium==1.4.0
  ale-py==0.12.1
  ocatari==2.2.1
```

Install explicitly into the evaluator's venv, rather than relying on the system Python:

```text
$PY -m pip install "gymnasium==1.4.0" "ale-py==0.12.1" "ocatari==2.2.1"
```

where `$PY` is the evaluator path above. OCAtari's package declares Python versions above 3.9 and depends on Gymnasium, NumPy below 2, OpenCV, scikit-image, termcolor, seaborn, pandas, scikit-learn, keyboard, tqdm, pygame and pyfiglet. Its PyPI metadata does not declare all of those dependencies reliably, so let pip resolve them and explicitly install `ale-py` as well. `ale-py 0.12.1` publishes a CPython 3.12 manylinux x86-64 wheel and requires Python >=3.10.

The current ale-py package contains `ale_py/roms/ms_pacman.bin`; its size is 8,192 bytes and its MD5 matches ALE's supported MsPacman checksum. This avoids asking the user for an external ROM. If a future package stops bundling ROMs, follow the official ALE/Gymnasium ROM-install instructions and obtain a legally usable CX2675-compatible ROM; do not put a ROM in this study directory or commit one.

*Sources:* [OCAtari README/install instructions](https://github.com/k4ntz/OC_Atari), [OCAtari PyPI](https://pypi.org/project/ocatari/), [ale-py PyPI](https://pypi.org/project/ale-py/), [ALE getting started](https://ale.farama.org/getting-started/), [ALE ROM/checksum declaration](https://github.com/Farama-Foundation/Arcade-Learning-Environment/blob/master/src/ale/games/supported/MsPacman.hpp).

### Construction and reset

The adaptor phase should:

1. Use the evaluator's Python 3.12.3 and install the pinned packages above.
2. Import `ale_py` and register its Gymnasium environments (`gymnasium.register_envs(ale_py)` for current releases).

…(truncated; full text in prior/interfaces.md)

o official Atari/Namco Ms. Pac-Man gameplay API, account service, or mature bot library for the CX2675 Atari 2600 ROM. Arcade Ms. Pac-Man bots and MAME libraries target a different arcade ROM, maze, timing and memory layout; they cannot be substituted without changing the game. Browser sites expose a canvas/emulator for human play, not a stable local state/action API, and the user has no display or account. These options have high maintenance/ToS risk and should not be selected.

## Chosen interface contract for the next adaptor phase

The adaptor should expose a JSON observation containing:

```text
frame: native ALE frame number
score: ALE HUD/decoded score
reward_delta: reward from the last ALE step
lives: ALE info lives (not OCAtari spare-icon value)
game_over: ALE terminal truth
mode: 0..3 ghost-count variation
player: {x, y, direction_if_available, dx, dy}
ghosts: [{slot, x, y, dx, dy, colour, visible}]
fruit: {visible, x, y, kind_if_decoded}
power_pills: [{x, y, visible}]
pills: [{row, col, x, y, visible}]
maze: {map_id, known_walls_or_turn_graph, map_confidence}
raw_ram: [128 unsigned bytes]
```

The adaptor controls one of the nine ALE action integers and should retain the last direction/action, because the world continues between model calls. It should provide a short action queue so Jev/Astra latency does not pause emulation, stop or replan the queue at a collision/turn hazard, and keep Astra's slower strategic state separate from Jev's per-state typed direction choice. No credentials, server address, account, or API key are required for this local design.


## Measured model latency (at archive creation, 2026-10-30 04:13)
- System-2 (Astra), one-word answer: 29 s
- System-2, an answer of ~1114 output tokens: 89 s — latency grows with the length of the answer; size timeouts for the answers your prompts actually request
- System-1 (Jev), one request with the game's initial observation and a Choice over its actions: median 176 ms (= 2.6 ticks), p90 189 ms, max 195 ms over 12 warm requests; first request on a new connection 226 ms — a timeout below the tail turns slow answers into failed calls
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
