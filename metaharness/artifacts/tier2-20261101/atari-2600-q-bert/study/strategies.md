# Strong-play strategy for Atari 2600 Q*bert

## What a strong player optimizes

The immediate objective is to finish pyramids without losing Q*berts; score is a useful secondary objective because cubes are worth 25, Coily 500, Sam 300, Green Ball 100, and a completed round 3,100. The manual's five-level, four-round structure and colour cycles mean that “jump on every cube once” is only correct for Level 1. A good agent maintains the cube phase, routes to unfinished cubes, preserves an escape route, and treats enemies as moving constraints rather than chasing every bonus ([Parker Brothers manual](https://archive.org/details/Q-bert_1988_Atari)).

The port has a six-row/21-cube board, not the arcade 28-cube board. Build the graph for this ROM and validate it from the first reset; do not reuse arcade route tables. The board is a triangular graph with four possible directional edges at interior nodes and missing edges at boundaries. The controller's physical diagonal orientation makes an initial action calibration important ([Atari 2600 review](https://carstairsconsiders.blogspot.com/2013/07/video-game-review-qbert-for-atari-2600.html), [ALE action mapping](https://ale.farama.org/environments/qbert/)).

## Opening and route heuristics

1. **Start from the top and clear the top four cubes early.** A contemporary Atari 2600 review specifically recommends clearing the top four immediately; this reduces dangerous traffic through the central upper routes and gives Q*bert more room to recover. Treat this as a prior, not an inviolable script: if a red ball/Coily blocks the route, survive first ([VGMuseum review](https://www.vgmuseum.com/reviews/2600/qbert/)).
2. **Never enter a dead end without a plan.** Before every hop, keep at least two legal next directions when possible. The review's practical rule is to leave two movement choices when an enemy approaches; at a boundary, bias toward a route with a known disc or an uncompleted neighbour rather than the outer edge ([VGMuseum review](https://www.vgmuseum.com/reviews/2600/qbert/)).
3. **Use a coverage route, not random wandering.** On Levels 1–2, prefer an adjacent cube that still needs a landing and that does not force a completed cube to be undone. On Levels 3–5, select a target whose phase transition is useful, then plan the return/revisit path. A route search should penalize edge falls and enemy collision much more heavily than the small immediate 25-point gain.
4. **Exploit safe movement macros.** A hop is an animation, not an instantaneous teleport. Queue only a short sequence whose intermediate nodes are safe; re-observe before committing through an intersection or when a hazard enters the same branch. This lets Jev act at sub-hop cadence while Astra plans the next coverage segment.
5. **Finish deliberately.** When one or two cubes remain, stop following the opening route and search for a path that reaches exactly those phases while retaining a disc/escape option. Completing the round is worth 3,100, so do not risk the last Q*bert for an optional enemy bonus.

## Exact cube bookkeeping

The manual supplies these transition systems; this belongs in deterministic code and should be included in Astra's planning context:

| Level | Per-cube transition on successive landings | Planning consequence |
|---|---|---|
| 1 | `S -> D`, `D -> D` | Greedy adjacent uncovered-cube coverage is usually sufficient. |
| 2 | `S -> I -> D`, `D -> D` | Track each cube's one/two landing status; do not count a visit as completion. |
| 3 | `S -> D -> S` | A landing on `D` undoes progress; route revisits only when they are needed to reach another cube or to fix an `S`. |
| 4 | `S -> I -> D -> I` | A `D` cube is not permanently safe; schedule the second pass to return `I` to `D`. |
| 5 | `S -> I -> D -> S` | Repeating Level 5 requires a cycle-aware route; mark exact phase, not just “visited”. |

At each state, code should maintain `cube_phase[21]`, `needed_landings`, and a set of legal graph edges. The phase can be updated from the decoded RAM/object state; until that decoder is complete, action-trace bookkeeping plus the 25-point reward events is a useful cross-check. Sam invalidates naïve bookkeeping by reverting cubes, so detect Sam's landing/reversion and resynchronize rather than assuming the internal model is still correct.

## Enemy and disc play

### Red and purple balls

The expert difficulty includes Red Balls; novice omits them. A Red Ball is lethal, as is a Purple Ball on contact. Purple Ball reaching the bottom produces Coily. The manual says enemy speed and appearance frequency increase with rounds, so an early-round route should not be treated as safe at the same route later. Code should classify a collision path and reserve an emergency action; Jev should answer which of the four directions is safe from the current local hazard; Astra should decide whether to abandon the target cube.

### Coily

Coily is not merely another moving obstacle: removing him is worth 500 and frees the board. The reliable counter is:

1. identify a disc and its one valid exit/boarding cube;
2. approach/occupy that exit without stepping off the pyramid;
3. wait until Coily commits toward the exit;
4. board the disc immediately so Q*bert returns to the top while Coily falls.

Do not board from an arbitrary edge cube. The manual explicitly warns that each disc has one correct exit cube and that the wrong one causes Q*bert to fall ([manual](https://archive.org/details/Q-bert_1988_Atari)). Store disc exits in code after calibration. Jev is suitable for the fast question “is the currently visible disc exit safe to board now?”; Astra should choose whether sacrificing route progress to lure Coily is worthwhile.

### Sam and Green Ball

Sam reverses cube progress but cannot catch Q*bert; running into Sam earns 300. Green Ball also cannot catch Q*bert; catching it freezes all other characters briefly and earns 100. These bonuses are attractive only when the intercept does not enter a lethal branch. Code should track the temporary freeze and update cube phases; Jev should recognize “Sam/Green present” and choose between intercept, continue, or flee; Astra should set the risk threshold. Do not waste a life chasing either character for a 100/300 bonus while a round completion is available.

## Difficulty and progression

* Start evaluation on difficulty 0/ALE's default, which corresponds to the manual's novice/B setting; use difficulty 1 for the expert/A setting with Red Balls. Confirm the switch mapping in the adaptor rather than relying on a label.
* Level 5 repeats after the first five levels. The agent must not assume the episode ends after 20 rounds.
* Track extra-life events: one after the first five rounds, then one every four rounds. This is code bookkeeping, not a model judgment.
* Use round/level and elapsed frames to adapt the hazard margin. Higher rounds require shorter action queues and more conservative edges.

## Common mistakes and counters

| Mistake | Countermeasure | Locus |
|---|---|---|
| Treating the Atari port as arcade Q*bert | Use a validated 21-node graph and Atari enemy roster; reject 28-node/arcade-only Ugg, Wrong-Way, and Slick assumptions. | **Code** |
| Holding a direction until Q*bert falls off an edge | Enforce graph-boundary legality and stop/turn before the edge; every macro has a maximum hop/frame horizon. | **Code** |
| Reacting only after seeing a collision | Predict occupied branches from current positions and choose a safe alternate early; movement is slow enough that last-second reactions fail. | **Jev questions + code** |
| Counting every landing as permanent progress | Apply the level-specific S/I/D finite-state machine; resync after Sam. | **Code** |
| Boarding any flying disc | Store its correct exit and require a positive safe-exit decision. | **Code + Jev** |
| Chasing a bonus through danger | Use a survival-first threshold; a lost life costs more future cube/round score than a small bonus. | **Astra prompt** |
| Letting a slow model block the world | Astra emits a rolling route/macro plan; Jev returns the next typed choice; keep the last safe action while a call is in flight. | **Harness/code** |
| Treating a stale model answer as current | Attach frame number to every Jev answer and reject it after a short horizon or hazard change. | **Harness/code** |
| Assuming a seed creates a new puzzle | Q*bert starts from the same pyramid; seeds mainly alter ALE sticky-action realization. Evaluate held-out seeds as robustness tests, not as different board layouts. | **Astra explanation + code** |
| Reading only total reward | Report score, lives, completed rounds, cube phases, and terminal/truncation separately. | **Code** |

## Division between code, Astra, and Jev

### Deterministic code

* Implement the 21-node triangular board, four directional actions, boundary legality, and calibrated action-to-edge mapping.
* Decode ALE's score/lives bytes and keep cumulative reward; detect round completion, life loss, game over, and evaluator truncation.
* Maintain the S/I/D transition table for all five levels, `cube_phase[21]`, needed landing count, extra-life bookkeeping, and Sam reversion.
* Track frame numbers and enemy/object positions once the ROM-specific RAM decoder is calibrated. Estimate short-horizon collision/edge risk and reject illegal or stale actions.
* Maintain a short action queue, clone/restore states for offline route search, and choose the best candidate by a fixed lexicographic objective: survive, complete the current pyramid, then capture safe bonuses.
* Record per-event score deltas so a high score can be audited as cube progress, round bonuses, Coily, Sam, or Green Ball.

### Astra prompts

Give Astra a compact symbolic snapshot, recent event history, current level/round, cube phases, legal exits, hazard trajectories, and the action queue—not raw pixels alone. Ask it to:

* choose a coverage order and safe return path for the current colour cycle;
* decide when to abandon an unfinished cube to preserve a life or reach a disc;
* plan a Coily lure and decide whether a Sam/Green interception is worth the risk;
* adjust the macro horizon as round speed rises and explain recovery after a decoder mismatch;
* maintain a long-horizon policy across level transitions and the repeating Level 5.

Astra should not be asked to answer every frame or do routine score arithmetic; deterministic code owns those tasks.

### Jev questions

Batch independent typed questions over the latest JSON state, each with a small action/choice domain:

* `choice`: one of `{UP, RIGHT, LEFT, DOWN, NOOP}` that keeps Q*bert on a legal safe edge for the next hop;
* `choice`: which adjacent unfinished cube is safest to target now;
* `yes_no`: is a Red/Purple/Coily collision imminent on the proposed action horizon?
* `yes_no`: is the current disc exit the validated safe exit, and is boarding safe now?
* `choice`: `continue_target | flee_to_disc | lure_coily | intercept_sam | catch_green`;
* `label`: `normal | life_lost | round_complete | level_transition | game_over | stale_state`.

The prompt should include precomputed legal actions and hazard distances. Jev recognises the situation and selects among typed options; it should not discover the graph, calculate score totals, or invent a multi-minute route.

## Suggested planning objective

Use a lexicographic or heavily risk-weighted score rather than raw immediate reward:

1. no edge fall or enemy contact;
2. complete the current pyramid and preserve a Q*bert;
3. maximize future rounds/levels;
4. safely collect Coily/Sam/Green bonuses;
5. maximize points among otherwise equivalent safe routes.

This reflects the manual's 3,100-point round bonus and finite lives better than greedily pursuing every 25-point cube event. Keep the default ALE sticky-action setting for robust evaluation, but run a deterministic no-sticky calibration episode before trusting a decoder.

## Sources

* [Parker Brothers Atari 2600 Q*bert manual, including colour cycles and scoring](https://archive.org/details/Q-bert_1988_Atari)
* [AtariAge HTML manual](https://atariage.com/manual_html_page.php?SystemID=2600&SoftwareID=1224&itemTypeID=HTMLMANUAL)
* [ALE Q*bert actions, observations, rewards, and variants](https://ale.farama.org/environments/qbert/)
* [ALE Q*bert score/lives/terminal implementation](https://github.com/Farama-Foundation/Arcade-Learning-Environment/blob/main/src/ale/games/supported/QBert.cpp)
* [Atari 2600 port review with 21-cube board and strategy observations](https://carstairsconsiders.blogspot.com/2013/07/video-game-review-qbert-for-atari-2600.html)
* [Atari 2600 control and enemy/response review](https://www.vgmuseum.com/reviews/2600/qbert/)
