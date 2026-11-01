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

**Code:** detect pill consumption, maintain a conservative frightened timer estimate, compute chain feasibility and score gain, and select safe reachable targets. **Jev:** fast classify `NORMAL_THREAT`, `TAKE_PILL_NOW`, `CHAIN_GHOSTS`, or `ABORT_CHAIN`, then choose a local direction. **Astra:** set the risk policy (score hunting versus completion) and adjust it as lives/maze difficulty change.

*Sources:* [AtariAge scoring and power-pill rules](https://atariage.com/manual_html_page.php?SoftwareID=1169), [OCAtari MsPacman RAM decoder](https://github.com/k4ntz/OC_Atari/blob/master/ocatari/ram/mspacman.py).

### Fruit and pretzels

Moving fruit and pretzels are bonus opportunities, but they can cross the maze through a tunnel and pull Ms. Pac-Man away from a safe dot sweep. Take them when the fruit path is reachable without entering the ghost collision horizon or consuming the only emergency pill. Astra can choose a level-dependent fruit-risk policy; code should calculate the path and expected point value; Jev can answer a typed `TAKE/IGNORE` recognition question from those features.

The fruit value order in the manual is cherry, strawberry, orange, pretzel, apple, pear, banana with increasing values. Do not assume the fruit is stationary or that it always appears at a fixed arcade location.

## Mode-specific strategy

Mode 0 with four ghosts is the main evaluation condition because it matches the manual's standard/arcade-style variation. Modes 1–3 are useful curriculum and robustness scenarios, not interchangeable scores:

* **One Bear:** learn the map and dot sweep with generous survival margin; use it to calibrate walls, pill coordinates, and score/life decoding.
* **Two/Three Bears:** retain the same map but shrink the safe corridor threshold and test tunnel/pill escape decisions.
* **Four ghosts:** use the full risk policy, prioritise an escape branch at every junction, and reserve pills for a meaningful cluster or imminent threat rather than eating them automatically.

Keep mode in the serialized state and report scores by mode. A 20,000 score in One Bear is not directly comparable to 20,000 in the four-ghost standard mode.

## Bookkeeping and progress

Maintain these exact code-side variables:

* cumulative score and reward delta;
* current and maximum observed lives, with a death counter;
* visible/remaining pill set keyed by maze/map and grid coordinate;
* maze-clear count and level/fruit hypothesis;
* power-pill availability and last consumption frame;
* current direction, previous player cell, ghost cells/velocities and projected danger;
* route plan, next turn, escape branches, fruit target and confidence;
* terminal/game-over status.

Use score for the primary leaderboard signal, but report `mazes_cleared`, `pills_eaten`, `lives_remaining`, `deaths`, and terminal state as secondary metrics. A policy that chases a high-value fruit and dies early can score well in one seed while playing worse than a policy that completes additional mazes.

The ALE reward is already a score difference, so the harness should sum it rather than ask Jev or Astra to add per-object values. OCAtari's HUD `Score.value` is useful for cross-checking but should not replace the ALE reward or raw score bytes when the decoder disagrees.

## Common mistakes and counters

| Mistake | Counter | Where it belongs |
|---|---|---|
| Treating this as turn-based and waiting for Astra | Run a short queued cardinal action; query Jev asynchronously; interrupt only at hazards | Code/harness |
| Using an arcade map or ghost schedule | Calibrate the CX2675 wall/tunnel graph and validate every map ID | Code first, Astra validation |
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
