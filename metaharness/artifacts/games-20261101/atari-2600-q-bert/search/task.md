Design a harness for a **real-time** game (tier 2).

Game: `Atari 2600 Q*bert` — Atari 2600 Q*bert is a local NTSC Parker Brothers cartridge emulated by ALE/Stella; no display, account, server, GPU, or network service is used. The player hops on a six-row, 21-cube pyramid.  A cube landing changes its colour according to the current level's finite cycle; finish the required destination phase on all cubes while preserving Q*bert lives.  Level 1 uses S->D, Level 2 S->I->D, Level 3 S->D->S, Level 4 S->I->D->I, and Level 5 S->I->D->S; Level 5 repeats after the first five levels.  The port has four rounds per level.  Avoid Red Balls (expert difficulty), Purple Balls and Coily; Sam reverses cube progress and is worth 300, Green Ball is worth 100, and luring Coily off a disc is worth 500.  Cube destination changes are 25 points and a completed round is 3,100 points.  Survival and completing a round are more important than optional bonuses.

This is a real-time interface: call tick(action) once per tick_s.  The default is ALE/Qbert-v5 with frameskip=4 (four emulated NTSC frames, about 66.7 ms) and repeat_action_probability=0.25.  A late tick with None repeats the last posted action, or applies NOOP before the first action; malformed names or indices are ignored, a NOOP is applied, and action_error is reported in the next observation.  Legal names are exactly NOOP, FIRE, UP, RIGHT, LEFT, DOWN; integer indices 0..5 are accepted as a convenience.  FIRE is not a normal movement action.  UP/RIGHT/LEFT/DOWN are the four joystick edges, and an edge can be lethal if it is absent from the current node's safe_moves.  Do not use the 28-node arcade board.

The JSON observation includes frame/tick counters, lives and other resources, cumulative ALE score, decoded BCD cartridge score and score_match, level/round progress, qbert_node, rendered Q*bert position/motion, hop/fall phase, and named cartridge timers/scheduler states.  It contains cube_state and needed_landings for all 21 fixed nodes plus destination/intermediate/incomplete counts; red_ball/purple_ball/coily/sam/green_ball objects when visible with kind, position and motion; hazard distances/risks; available disc_exits; graph legal_moves and conservative hazard-blocked/safe_moves; last reward/event/action, raw 128-byte RAM, and terminal/truncation flags.  The decoder cross-checks RAM sprite slots against the current ALE RGB palette internally so stale animation placeholders are not shown as live enemies; the RGB frame itself is never exposed.  reset performs a bounded NOOP bootstrap by default so the first observation is the playable Level 1 Round 1 opening rather than the cartridge's attract-screen blank frames; those bootstrap frames are not counted as agent ticks.  Seeds select ALE's seeded sticky-action RNG stream, not a new board layout.  Set frameskip=1 and repeat_action_probability=0 only for a separate deterministic decoder calibration run.

Actions:
- NOOP: Hold the joystick neutral for one ALE quantum; use this only for a deliberate wait.
- FIRE: Press FIRE. It is not a movement direction and may restart/reset the cartridge in this ROM.
- UP: Hop toward the upper-right edge of the diamond; from a left child this returns toward its parent.
- RIGHT: Hop down/right to the right child of the current cube when that edge exists.
- LEFT: Hop up/left to the parent of a right child when that edge exists.
- DOWN: Hop down/left to the left child of the current cube when that edge exists.
(real-time games: ticks apply 'NOOP' until the first action is posted; afterwards a tick without a fresh action repeats the last posted action)
Official minimal ALE indices are NOOP=0, FIRE=1, UP=2, RIGHT=3, LEFT=4, DOWN=5.  One tick holds the chosen action for the configured frameskip (default four NTSC frames, about 66.7 ms); v5 sticky actions can repeat the previous joystick action with probability 0.25.  Do not use the arcade 28-cube map: this Atari port has 21 cubes.

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
- strategy_summary: Maintain a 21-cube graph and exact per-level S/I/D colour phase, rather than treating each landing as permanent progress. Clear the top region early when safe, preserve at least two escape choices, and prioritize survival and round completion over optional bonuses. Use validated disc exits to lure Coily, intercept Sam or Green Ball only when safe, and shorten action queues as rounds accelerate. Astra plans coverage routes and adaptation; Jev answers typed local danger/next-move questions while code owns legality, bookkeeping, score, seeds, and stale-action rejection.
- scoring: Report cumulative ALE reward, which tracks the in-game score delta, and cross-check it against the decoded score. The manual awards 25 per cube changed to destination, 300 for Sam, 100 for Green Ball, 500 for luring Coily off, and 3,100 per completed round. Also report lives, completed rounds/levels, cube phases, and whether the episode terminated or was truncated because terminal-step reward is deliberately zero in ALE's Q*bert wrapper.
- interface.observation: A uint8[128] Atari 2600 RAM array plus Gymnasium info with lives, episode_frame_number, and frame_number; step also returns score-delta reward and terminated/truncated flags. ALE's Q*bert wrapper reads CPU-style $88 for lives and $DB,$DA,$D9 for BCD score, corresponding in the 128-byte view to indices 0x08 and 0x5B,0x5A,0x59. The adaptor must decode named Q*bert/cube/enemy objects itself; optional rgb_array is 210x160x3 for calibration, not the primary channel.
- interface.actions: Discrete(6): 0=NOOP, 1=FIRE, 2=UP, 3=RIGHT, 4=LEFT, 5=DOWN. Use the four directional actions for hops, NOOP only for deliberate waits, and FIRE only for an intentional reset. Each Gymnasium action is repeated for frameskip frames and may be made sticky by repeat_action_probability; keep a short action macro and replan at hazards. Native ALEInterface.act, getRAM, getScreenRGB, lives, game_over, cloneState, and restoreState are the lower-level fallback.

### study/strategies.md
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

### study/brief.md
# Atari 2600 Q*bert — playing brief

## Scope and version

This study is for the Atari 2600 Parker Brothers cartridge, not the original arcade game or the later Atari-branded release. The ALE-supported `qbert.bin` is the 4 KiB Atari 2600 ROM identified by MD5 `484b0076816a104875e00467d431c2d2`; the ALE source binds its Q*bert reward/life wrapper to that checksum ([ALE QBert settings](https://github.com/Farama-Foundation/Arcade-Learning-Environment/blob/main/src/ale/games/supported/QBert.cpp)). The scanned Parker Brothers manual is dated 1983 ([manual scan/OCR](https://archive.org/details/Q-bert_1988_Atari)); use the ROM supplied by ALE rather than silently substituting an arcade ROM.

## Objective and board

Q*bert starts on the top cube of a pyramid. Hop onto every cube until the whole pyramid reaches its destination colour; completing a pyramid ends the round and produces a new destination colour. The home conversion has a six-row pyramid (21 cubes), rather than the arcade game's 28-cube pyramid; this port-specific count is documented by a contemporary play review ([Carstairs, Atari 2600 review](https://carstairsconsiders.blogspot.com/2013/07/video-game-review-qbert-for-atari-2600.html)). The manual itself states the objective and progression but does not give the row count ([Parker Brothers manual](https://atariage.com/manual_html_page.php?SystemID=2600&SoftwareID=1224&itemTypeID=HTMLMANUAL)).

There are five levels, four rounds per level. After Level 5, Level 5 Round 1 repeats. Q*bert, the enemies, and the frequency of enemy appearances become faster/more frequent as rounds advance. The game starts with four Q*berts (lives); one extra is awarded after round 5 and another after every four rounds thereafter. The game ends when the supply is exhausted ([manual, progression and scoring pages](https://archive.org/details/Q-bert_1988_Atari)).

## Cube colour rules

Let `S` be the starting colour, `I` an intermediate colour, and `D` the destination colour. A jump applies the following state transition to the cube landed on:

| Level | Rounds | Transition when landing | Consequence |
|---|---:|---|---|
| 1 | 1–4 | `S -> D`, `D -> D` | Each cube normally needs one useful landing. |
| 2 | 1–4 | `S -> I -> D`, `D -> D` | Each cube needs two useful landings. |
| 3 | 1–4 | `S -> D -> S` | A landing on an already-complete cube undoes it; revisit deliberately. |
| 4 | 1–4 | `S -> I -> D -> I` | A complete cube cycles back to `I`; revisit to finish it. |
| 5 | 1–4 and repeat | `S -> I -> D -> S` | A complete cube cycles back to `S`; revisit until all are `D`. |

This table is transcribed from the manual's “Reaching the Destination Color” page ([manual scan/OCR](https://archive.org/details/Q-bert_1988_Atari)). Do not assume the arcade version's colour schedule.

## Characters and hazards

* **Red Ball:** rolls down the pyramid and kills Q*bert on contact. Red Balls do not appear in the manual's novice setting.
* **Purple Ball:** a lethal rolling hazard; if it reaches the bottom it hatches Coily.
* **Coily:** the snake pursues Q*bert. The reliable counter is to lure him toward a flying-disc exit, then board the disc as Coily reaches the exit so Coily falls off. Each disc has one correct exit cube; boarding from the wrong cube kills Q*bert.
* **Sam:** cannot catch Q*bert, but changes landed cubes back toward their original colour. Run into Sam to catch him and score a bonus.
* **Green Ball:** cannot catch Q*bert. Catching it freezes every other character (not Q*bert) briefly and awards a bonus.
* **Flying discs:** emergency transport to the pyramid top. Their number and placement vary by level/round; they are also the main Coily trap. These behaviours and the one-correct-exit warning are in the manual ([manual pages 3–5](https://archive.org/details/Q-bert_1988_Atari)).

The Atari port omits some arcade-only enemies and has a simplified pyramid; do not import arcade Ugg/Wrong-Way/Slick behaviour into the state model. A port review confirms the omitted enemies and notes that the home controls require anticipating rather than reacting to the slower/unresponsive movement ([VGMuseum review](https://www.vgmuseum.com/reviews/2600/qbert/)).

## Score and progress

| Event | Points |
|---|---:|
| Change a cube to its destination colour | 25 |
| Catch Sam | 300 |
| Catch Green Ball | 100 |
| Lure Coily off the pyramid | 500 |
| Complete a round | 3,100 |

The manual gives these values and the extra-life schedule ([manual scoring page](https://archive.org/details/Q-bert_1988_Atari)). The adaptor should report:

1. cumulative score (the sum of ALE reward deltas, or the decoded in-game score),
2. current lives, level and round when decoded,
3. number of cubes at `D` and the colour state of each cube, and
4. whether the episode ended by game over or by an evaluator truncation.

ALE's Q*bert wrapper reads the score from the game RAM and returns the score delta as reward; it deliberately returns zero on the terminal step ([QBert.cpp](https://github.com/Farama-Foundation/Arcade-Learning-Environment/blob/main/src/ale/games/supported/QBert.cpp)). Thus the cumulative non-terminal reward is the meaningful score, not the final terminal-step reward. Score is a better primary measure than survival alone, while lives and completed rounds prevent a high-score report from hiding reckless play.

## Controls and actions

The manual says to hold the joystick with its four corners making a baseball diamond and the fire button at the top; Q*bert moves in the four diagonal board directions. In ALE, the physical joystick directions are represented by these action IDs:

| ALE action | Meaning | Q*bert use |
|---:|---|---|
| 0 | `NOOP` | Hold position; normally only for an intentional wait. |
| 1 | `FIRE` | Fire/reset input; do not use during play unless the adaptor explicitly wants a reset. |
| 2 | `UP` | One board-diagonal direction. |
| 3 | `RIGHT` | One board-diagonal direction. |
| 4 | `LEFT` | One board-diagonal direction. |
| 5 | `DOWN` | One board-diagonal direction. |

The mapping is the official ALE mapping ([Q*bert environment](https://ale.farama.org/environments/qbert/), [action specification](https://ale.farama.org/environments/#action-space)). The board's “up-right”/“down-left” names depend on the manual's rotated joystick orientation; calibrate the four action IDs once from the top cube and retain that mapping. ALE's minimal Q*bert set is exactly these six actions; the 18-action full Atari set is unnecessary.

A reset switch starts a new manual game. The Gymnasium adaptor should use `reset()` rather than injecting `FIRE`; the native ALE API exposes `reset_game()`.

## Pace and tier

The Atari 2600 world advances continuously at roughly one NTSC frame per 1/60 second. The game samples joystick input while Q*bert and enemies animate, and the manual explicitly says both speeds increase with rounds. With `ALE/Qbert-v5`, one environment action is repeated for four emulated frames and is therefore about 66.7 ms of game time; ALE documents the fixed four-frame skip and 25% sticky-action default ([ALE environment defaults](https://ale.farama.org/environments/qbert/), [stochasticity settings](https://ale.farama.org/environments/)). A bad direction can be fatal in a single hop, so the practical safe synchronous decision budget is about **0.10 s at fast rounds** (roughly one or two four-frame decisions), not the seconds Astra may take to reason.

This is **tier 2**: the world does not wait for a model response. Jev should run the short-horizon action queue and answer recognition/choice questions quickly; Astra should plan routes, colour schedules, and adaptation asynchronously. A 0.5-second Jev call cannot be placed directly on every hop without buffering several safe actions or running the emulator faster than wall-clock. For exact calibration/search, `frameskip=1` exposes every frame; for the normal harness, fixed `frameskip=4` is a useful compromise.

## References

* [ALE Q*bert environment specification](https://ale.farama.org/environments/qbert/)
* [ALE Python interface: RAM, ROMs, state cloning](https://ale.farama.org/python-interface/)
* [Parker Brothers Atari 2600 Q*bert manual](https://atariage.com/manual_html_page.php?SystemID=2600&SoftwareID=1224&itemTypeID=HTMLMANUAL)
* [Searchable manual scan/OCR](https://archive.org/details/Q-bert_1988_Atari)
* [Atari 2600 review and port-specific board details](https://carstairsconsiders.blogspot.com/2013/07/video-game-review-qbert-for-atari-2600.html)
* [Atari 2600 control/response review](https://www.vgmuseum.com/reviews/2600/qbert/)

### adaptor/ADAPTOR.md
# Atari 2600 Q*bert adaptor

## Connection and setup

`game.py` owns one local `ALE/Qbert-v5` Gymnasium environment backed by
ALE/Stella. It is headless (`obs_type="ram"`, `render_mode=None`); no display,
GPU, account, server, or credential is needed. `requirements.txt` installs the
pinned ALE/Gymnasium pair and OCAtari 2.2.1's maintained Q*bert RAM decoder.
The constructor verifies that `ale_py.roms.get_rom_path("qbert")` is a regular
file with MD5 `484b0076816a104875e00467d431c2d2`. A private legal ROM can be
selected with the optional `ALE_ROMS_DIR` environment variable shown in
`env.example`; the adaptor never downloads or accepts an unverified ROM.

The default is the study's robust evaluation configuration:

- `mode=0`, `difficulty=0` (novice), `frameskip=4`;
- `repeat_action_probability=0.25` (ALE v5 sticky actions);
- minimal six-action ALE space, no full action space;
- `tick_s = frameskip / 60`, so the default tick quantum is about 66.7 ms.

`difficulty=1` is supported separately for expert/A play and adds the Red Ball
hazard. `frameskip=1, repeat_action_probability=0.0` is supported for a
separate deterministic decoder calibration run, not the robust default.

The Atari cartridge spends about 150 emulated frames in its attract/boot
sequence. `warm_start=True` (the declared evaluator default) sends bounded
NOOPs during reset until Q*bert's playable Level 1 Round 1 opening is present.
Those bootstrap frames do not increment the public `ticks` counter and do not
change the score. Set `warm_start=False` to expose the literal frame-zero
cartridge state instead.

## Interface

The returned object is Tier 2 `RealtimeGame`. The evaluator calls
`tick(action)` on its own clock; `tick` never sleeps or starts a worker thread.
One call advances one ALE `frameskip` step and increments `ticks` once. `None`
means a missed decision: it repeats the last valid posted action, or uses
`NOOP` before the first action. A malformed action is converted to `NOOP`, does
not replace the posted action, and is reported in `action_error` rather than
raising. Names and integer conveniences are:

```
NOOP=0, FIRE=1, UP=2, RIGHT=3, LEFT=4, DOWN=5
```

`FIRE` is deliberately exposed because it is part of ALE's legal minimal set,
but it is not a normal movement action and can restart the cartridge. The
four directional names are joystick edges on the Q*bert diamond. `safe_moves`
is derived from the ROM's 21-node graph and excludes pyramid edges that do not
exist at the estimated `qbert_node`; a hop is animated, so the state may be
between nodes and the estimate should be rechecked before committing a long
macro.

## Observation and decoding

Every observation is strict JSON and contains the complete 128-byte RAM view as
`raw_ram` plus compact decoded fields:

- `frame`, `global_frame`, `tick`/`ticks`, `lives`, the `resources` map (life losses,
  extra lives, and next scheduled extra-life round), `difficulty`, `level`, `round`,
  `round_in_level`, `completed_rounds`, and `completed_levels`;
- cumulative `score`/`cumulative_ale_reward`, the BCD `decoded_score`, and
  `score_match`;
- `qbert` rectangle and pixel motion, estimated `qbert_node`, `qbert_in_hop`,
  `qbert_motion` (`standing`/`hopping`/`falling`/`not_visible`), and `timers`
  containing the verified hop-animation timer, object scheduler, enemy spawn
  scheduler bytes, falling flag, and sprite-state bytes;
- 21 `cube_state`/`cube_phase` labels (`S`, `I`, `D`, or `unknown`), raw cube
  palette bytes, fixed-node coordinates, color names, `destination_cubes`,
  `intermediate_cubes`, `incomplete_cubes`, and `needed_landings`;
- named `red_ball`, `purple_ball`, `coily`, `sam`, and `green_ball` objects
  with kind, position and motion when visible; absent objects are `null`;
- `hazards` with lethal/non-lethal classification, estimated node, pixel
  distance to Q*bert, and a conservative risk label; `legal_moves` is the
  geometry-only graph set, while `blocked_moves` and `safe_moves` account for
  an immediately occupied hazard node;
- `disc_exits` with rendered position/availability, plus `last_reward`, event
  counters, `last_action`, malformed-action diagnostics, `terminal`/`terminated`,
  `truncated`, `done`, and termination reason.

The observation is normally about 5--6 kB, far below the 200 kB contract.

OCAtari supplies the ROM-specific 21-cube/player/disc/Purple Ball/Coily/Sam/
Green Ball extraction from RAM. Its Player/character slots can retain a
placeholder during an animated edge fall, so the adaptor cross-checks the
current native RGB frame internally against Atari palette components and uses
the rendered component to correct Q*bert/enemy visibility, position, and
motion. OCAtari's current Q*bert list has no Red Ball slot, so expert mode also
uses the yellow/red-ball palette component and exposes only its rectangle/motion,
never the binary frame. If a sprite is between palette frames it is reported
as absent rather than as a stale RAM object. The two disc entries report
availability but intentionally leave `validated_exit` unset: the Atari port's
correct Coily lure exit must be learned/calibrated rather than guessed from the
arcade game.

The cube tracker follows the manual's five finite cycles. It uses the actual
RAM palette transition and keeps raw bytes beside the named phase, so a Sam
reversion or Level 3-5 reversible cycle does not get flattened into a boolean
"visited" flag. `needed_landings` is the number of transitions to the next D
phase (or `-1` when the cartridge is in an uncalibrated/blank state).

## Seeds and scenarios

`SEEDS` provides eight train and eight disjoint held-out integer seeds. All
seeds use the same fixed Level 1 Round 1 opening and 21-cube board; Q*bert has
no normal randomized board layout. The seed selects ALE's seeded stochastic
stream, principally the v5 sticky-action realization, so held-out episodes
measure control robustness rather than puzzle-layout transfer. The same seed
mix is available for `difficulty=1` by constructing with that keyword; novice
and expert results should be recorded separately.

The adaptor's ordinary scenario is a fresh Level 1 Round 1 reset. It tracks
progress into Levels 1-5, four rounds per level, the repeating Level 5, lives,
colour cycles, Coily/discs, Sam, Green Ball, and life loss as they occur during
play. It does not use RAM writes, cheats, or a save state during ordinary play.
Advanced later-level experiments should use a separately recorded,
ALE-version-matched safe action prefix or an RNG-inclusive ALE clone/restore
calibration harness; no unverified warm-start state is silently substituted by
this adaptor. `warm_start` only skips the cartridge attract screen and never
claims to select a later level.

## Score and limits

`score` is cumulative unmodified ALE reward. `decoded_score` independently
uses the Q*bert wrapper's verified BCD bytes (`RAM[0x5b]`, `[0x5a]`, `[0x59]`)
and `score_match` is the consistency check. `score_breakdown()` reports ALE
reward, decoded score, cube/Sam/Green/Coily/round event points, completed
rounds, lives, life losses, extra-life awards, and terminal/truncation flags. ALE deliberately
returns zero on Q*bert's terminal game-over step; the observation and
`info()` expose `terminal_step_reward_zero` so a final life loss is not
mistaken for a negative score event.

`info()["valid_trial"]` is false only for an emulator/connection/decoder
infrastructure failure and includes `invalid_reason`; bad actions are handled
inside the episode. `close()` closes the Gymnasium/ALE instance, releases its
native emulator state, and is idempotent. The adaptor starts no external
process. No screenshots, logs, caches, or generated code are written while it
runs.

### study/interfaces.md
# Programmatic interfaces for Atari 2600 Q*bert

## Recommendation

Use the **local Arcade Learning Environment (ALE) through `ale-py`/Gymnasium**, with `obs_type="ram"`, no render mode, and the bundled Q*bert ROM. It is the only option found that combines a maintained Atari 2600 emulator, a standard reset/step API, a 128-byte console-RAM observation, extracted Q*bert reward/lives, seeding, and emulator state cloning. It is local and headless: no display, GPU, server, account, API key, or network connection is needed at run time.

The recommended public environment is `ALE/Qbert-v5`. For the final adaptor, use Gymnasium's wrapper for reset/seed/step and reach through `env.unwrapped.ale` only for clone/restore or the native RAM/screen methods. The native `ALEInterface` is a good lower-overhead alternative if the wrapper is unavailable.

The ALE package is GPL-2.0-only and its current Python wheel packages Atari ROMs, including `qbert.bin`; the supported ROM is identified by MD5 `484b0076816a104875e00467d431c2d2` ([ALE package](https://pypi.org/project/ale-py/), [ALE README](https://github.com/Farama-Foundation/Arcade-Learning-Environment), [Q*bert settings](https://github.com/Farama-Foundation/Arcade-Learning-Environment/blob/main/src/ale/games/supported/QBert.cpp)). The emulator/library license does not automatically resolve every jurisdiction's rights to redistribute Atari/Parker Brothers ROM data. Keep the bundled evaluator copy private and do not download or redistribute an unverified ROM.

## Installation and setup

Use the evaluator interpreter, not the system interpreter:

```text
/home/hemlix/GamingMaster-mh-tier2/runs/atari-2600-q-bert/.venv/bin/python -m pip install "ale-py==0.12.1" "gymnasium>=1.1.0,<2"
```

`numpy>1.20` is an ALE dependency. If the evaluator pins a newer compatible ALE, verify the wheel has Python 3.12 support and that `ale_py.roms.get_rom_path("qbert")` resolves to the expected ROM. The official setup is [`pip install ale-py`](https://ale.farama.org/getting-started/); the package's Gymnasium interface requires registering ALE environments (the current package may auto-register them, but explicit registration is harmless):

```text
import gymnasium as gym
import ale_py
gym.register_envs(ale_py)
env = gym.make(
    "ALE/Qbert-v5",
    obs_type="ram",
    frameskip=4,
    repeat_action_probability=0.25,
    full_action_space=False,
    mode=0,
    difficulty=0,
    render_mode=None,
)
```

Do not request `render_mode="human"`; it enables SDL/display and is unnecessary on the headless server. `render_mode="rgb_array"` is an optional diagnostic fallback, not the primary observation.

### Environment variables and credentials

There is no server address, account, token, or API key. Leave `DISPLAY` unset and do not require X11. `ALE_ROMS_DIR` is optional: if set, it must be a directory containing a legal, checksum-matching `qbert.bin`; if unset, `ale_py` uses its packaged ROM directory. Never put a credential or a ROM URL/value in the study files.

### Adaptor setup checklist

1. Install the two packages with the evaluator's Python and record their versions.

…(truncated; full text in prior/interfaces.md)

## Failure handling

If RAM decoding is not ready, retain the raw 128 bytes and use `rgb_array` only for a short calibration phase; do not silently switch to screenshot-only play. If the packaged ROM is missing, fail with an actionable “install/verify `ale-py` or set `ALE_ROMS_DIR`” error rather than accepting a different Q*bert release. If a model call is late, continue the last validated short action macro and mark the observation stale; never block the emulator waiting for Astra.

## Sources

* [ALE getting started](https://ale.farama.org/getting-started/)
* [ALE Q*bert environment](https://ale.farama.org/environments/qbert/)
* [ALE Python interface](https://ale.farama.org/python-interface/)
* [ALE environment specification](https://ale.farama.org/environments/)
* [ALE Q*bert reward/lives implementation](https://github.com/Farama-Foundation/Arcade-Learning-Environment/blob/main/src/ale/games/supported/QBert.cpp)
* [ALE Python Gymnasium wrapper](https://github.com/Farama-Foundation/Arcade-Learning-Environment/blob/main/src/ale/python/env.py)
* [ALE PyPI package](https://pypi.org/project/ale-py/)
* [Stella debugger](https://stella-emu.github.io/docs/debugger.html)
* [Libretro Stella 2023](https://docs.libretro.com/library/stella2023/)
* [Xvfb manual](https://linux.die.net/man/1/xvfb)


## Measured model latency (at archive creation, 2026-10-31 08:42)
- System-2 (Astra), one-word answer: 14 s
- System-2, an answer of ~937 output tokens: 45 s — latency grows with the length of the answer; size timeouts for the answers your prompts actually request
- System-1 (Jev), one request with the game's initial observation and a Choice over its actions: median 152 ms (= 2.3 ticks), p90 166 ms, max 168 ms over 12 warm requests; first request on a new connection 213 ms — a timeout below the tail turns slow answers into failed calls
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
