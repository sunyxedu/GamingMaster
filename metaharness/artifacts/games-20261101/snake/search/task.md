Design a harness for a **real-time** game (tier 2).

Game: `Snake (local classic)` — This is a local, headless classic Snake on a bounded rectangular grid. The reference profile is 12 rows by 12 columns, zero-based [row, column] coordinates, no wall wrapping, one food item, initial length three near the centre, tick_s=0.10 seconds, and max_ticks=4000. The configured rows, cols, tick_s, and max_ticks are fixed for an episode and are shown in every observation. The objective is to eat food: score is exactly the cumulative food count and each food adds one body cell. Survival without food earns no score. The episode ends on wall_collision, self_collision, or filled_board; a still-live snake at max_ticks is truncated with terminal_reason=time_limit. A filled board wins even on the limit tick.

The snake list is head-first and contains [row, column] cells. On every tick the head advances exactly one cell in the current absolute cardinal direction. UP is row-1, RIGHT is column+1, DOWN is row+1, and LEFT is column-1. An action request in the immediate opposite direction is ignored (the snake keeps going); a same direction, NOOP, missing action, malformed action, or stale state-tick request does not stop the clock. On a non-food move the current tail vacates simultaneously, so entering that exact tail cell is legal. On a food move the tail stays and the body grows; the next food is sampled uniformly from row-major empty cells using a private random.Random(seed). No obstacles, opponents, power-ups, acceleration, starvation timer, or implicit collision avoidance exists.

action_spec names are UP, RIGHT, DOWN, LEFT, and NOOP. Integer aliases 0, 1, 2, 3 map to the four directions. For the optional low-level API, a dictionary such as {'action': 'RIGHT', 'state_tick': 12} carries a request derived from observation tick 12 and is rejected if it is stale; set_direction(direction, state_tick) queues the same kind of request for the next tick. The normal runner calls tick(action) at its own wall/virtual-clock cadence; tick_s is metadata for that cadence and tick() does not sleep. Each applied call increments ticks, records action_tick, and returns quickly. observe() only snapshots state and never advances time. Observations also include food, length, free_cells, max_score, score_fraction, nonreverse directions, one-step safe directions using exact tail-vacating semantics, terminal flags, last reward/event/error, attempted head, and timing fields. info() adds final score, length, ticks, terminal reason, completion, and rejected-action diagnostics.

Actions:
- UP: Steer the head one cell toward decreasing row on the next tick.
- RIGHT: Steer the head one cell toward increasing column on the next tick.
- DOWN: Steer the head one cell toward increasing row on the next tick.
- LEFT: Steer the head one cell toward decreasing column on the next tick.
- NOOP: Do not change heading; the snake continues in its current direction.
(real-time games: ticks apply 'NOOP' until the first action is posted; afterwards a tick without a fresh action repeats the last posted action)
Actions are absolute names (or integer aliases 0=UP, 1=RIGHT, 2=DOWN, 3=LEFT); a dict {'action': name, 'state_tick': t} is also accepted. NOOP and a missing action retain the current heading. An immediate reverse is ignored, never converted into a death. Malformed or stale actions are ignored and reported in the next observation. One call to tick advances exactly one game cell.

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

The game ticks every 0.1s **on its own clock**; `ctx.act(a)` posts the current action (instant).
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
- strategy_summary: Use deterministic code for legality, collision/tail-vacating simulation, seeded RNG, BFS/A*, flood-fill area, head-to-tail reachability, score bookkeeping, and tick deadlines. Pursue food with a short route only after simulating the post-growth state and passing a tail-access/reachable-area safety filter; this is not an indefinite-survival proof unless the full route and invariant are checked. Otherwise chase the tail or maximize connected free space. On boards with a validated Hamiltonian cycle, follow it as the late-game safety fallback and permit shortcuts only when cycle ordering, growth reserve, and tail reachability remain valid; disable shortcuts as occupancy becomes high. Because Jev's response can span several ticks, it should choose among a small set of code-validated directions or FOOD/TAIL/CYCLE modes at upcoming turns, while Astra sets conservative thresholds, handles unusual boards, and reviews failures without blocking the clock. Cite [John Flux's cycle algorithm](https://johnflux.com/2015/05/02/nokia-6110-part-3-algorithms/) and the [Safe A*/Hamiltonian comparison](https://github.com/shashank11yadav/Snake-game) in the implementation notes.
- scoring: Report cumulative food count as score (one point per food), final length, ticks survived, terminal_reason, and filled_board success. Primary performance is final score/cumulative reward because it measures actual food eaten; survival ticks and completion are diagnostics, and survival without food must not be rewarded as strong play.
- interface.observation: Fresh JSON containing game/version, seed and scenario, rows/cols, tick/sim_time_s/tick_s/deadline_in_s, head-first snake coordinates, head and tail, current direction, food or null, score, length, free_cells, nonreverse_directions, one-step safe_directions_now, terminated/truncated/terminal_reason, last_reward, input age, and action tick. Keep the complete body ordering; a screenshot or head-only image is insufficient for safe late-game play.
- interface.actions: Absolute UP, RIGHT, DOWN, LEFT strings, optionally aliases 0=UP, 1=RIGHT, 2=DOWN, 3=LEFT. A same-direction request is harmless; an immediate reverse is ignored and logged; malformed or stale actions are rejected. set_direction queues the next tick without blocking, while deterministic step(action) advances exactly one tick for replay tests. The scheduler continues moving during model calls.

### study/strategies.md
# Snake — strategy prior

## What strong play is optimizing

In single-player Snake there is no opponent to outmaneuver. Strong play balances two competing objectives:

1. eat the next food quickly enough to gain score; and
2. preserve a connected, sufficiently large escape region so the growing body never seals itself in.

A Manhattan-shortest route to food is not necessarily a safe route. Every food item changes the body geometry, and the tail only frees a cell on a non-food tick. The strategy should therefore treat “can reach food” and “will still be able to survive after eating” as separate predicates.

The strongest general fallback is a Hamiltonian cycle on a rectangular grid: visit every cell once in a cyclic order, while keeping every body segment in tail-to-head cyclic order. The body need not occupy every intermediate cycle cell after a shortcut; the safety condition is that the head does not overtake the tail and the body remains ordered on the tour. John Flux’s description of the Nokia 6110 bot uses a cycle, a one-dimensional tail/head ordering invariant, and safe shortcuts; his implementation generated a maze-based cycle under severe memory constraints ([Part 3: Algorithms](https://johnflux.com/2015/05/02/nokia-6110-part-3-algorithms/)). A modern open implementation compares Safe A*, tail reachability, and Hamiltonian-plus-shortcuts and documents its assumptions and benchmarks ([shashank11yadav/Snake-game](https://github.com/shashank11yadav/Snake-game)). Those are algorithmic references, not guarantees for every board or ruleset.

## State invariants and bookkeeping

These are mechanical facts and should be enforced by the adaptor, not left to a language model:

- The body is an ordered list from head to tail; no cell appears twice.
- Every body coordinate is in bounds, and the head is adjacent to the second body cell except for the initial one-cell state.
- Direction is one of four cardinal vectors and cannot be its own opposite after an accepted steering update.
- A non-food transition removes the old tail before judging a head move into that exact cell; a food transition does not.
- Food is outside the body, unless the episode has just ended in a full-board win.
- `score == foods_eaten`; under the base rules `length == initial_length + score`.
- The cycle index, if used, is modulo the number of board cells. The body must not overtake its tail in cycle order.
- A model response is attached to the state tick it observed. Stale responses are ignored, never applied to a later state.

**Placement:** exact validation, collision rules, score/length accounting, coordinate arithmetic, tick deadlines, and deterministic RNG belong in **code**. Astra and Jev should see the validated state, not repair corrupted bookkeeping.

## Strategy ladder

Use a layered controller rather than one universal heuristic. Each layer has a clear entry condition and a safer fallback.

### 1. Immediate legality and emergency steering

Before any food planning, remove directions that reverse into the neck or collide with a wall/body on the next tick. If the current direction is already a one-tick collision, choose a legal turn with the largest reachable region; do not continue a stale plan merely because it was shortest before the last tick. If no move is legal, accept termination and report the diagnostic rather than inventing a move.

**Code:** compute `nonreverse_directions`, one-step occupancy after tail-vacating, and a flood-fill count for each candidate. The code should never ask a model whether an out-of-bounds coordinate is in bounds.

**Jev:** choose among already checked alternatives only when there is enough lead time. Never wait for Jev to resolve a one-tick emergency; legality and the emergency choice belong in code.

**Astra:** only handle a surprising state (corrupt/partial observation, changing rules, obstacle introduced) and decide whether to abandon a longer plan.

### 2. Food path with a post-eating safety test

When the food is reachable without an immediate collision, search for a short path to it. Simulate every step of the candidate path with the actual moving body, including which tail cells vacate. At the simulated food transition, retain the tail and grow. Then test whether the resulting head can reach the resulting tail through cells that will eventually become free. Also measure the number of reachable free cells and whether the path creates a one-cell pocket.

Accept a food path only if it passes the safety test. A shortest path is a preference, not a safety criterion. Among paths with the same safety status, prefer fewer turns and a larger post-route reachable region; a slightly longer route that lets the tail advance is often better than a direct corridor.

**Important limitation:** a static flood-fill or ordinary BFS from the post-eating head to the current tail is only a heuristic; it ignores when intervening body cells become free and does not prove that the snake can survive forever. A time-expanded simulation/search can establish that a specific finite route is collision-free under its modeled actions, but the controller must replan at the endpoint. Treat tail access plus reachable area as filters/ranking features unless every intervening move has actually been simulated.


**Code:** BFS/A* over grid cells, dynamic-body simulation, head-to-tail reachability, flood fill, path length/turn scoring, and rejection of plans that enter a sealed component. A* is appropriate for the shortest candidate; a safety simulation is still required afterward. The source project’s “Safe A*” simulates the candidate food path and checks whether the head can still reach its tail ([Snake-game repository](https://github.com/shashank11yadav/Snake-game)); its own documentation describes this as a heuristic and reports stalls, so do not treat a static reachability test as an indefinite-survival proof.

**Astra:** choose the risk policy when several safe paths differ: conservative tail access at high occupancy versus faster food at low occupancy. Astra may reason about an unusual pattern, but should not do cell-by-cell arithmetic that code can verify.

**Jev:** classify a precomputed candidate as `FOOD_SAFE`/`FOOD_TRAPS`, identify whether the board is in a “food now” situation, or choose among a small list of code-validated routes. Do not ask Jev to discover a path or count cells.

### 3. Tail-chasing fallback

If no safe food route exists, follow a route toward the current tail while it moves, or choose the legal move with the greatest future tail access and reachable area. Chasing the tail is a way to buy time for the body to unwind; it is not simply “move to the cell currently occupied by the tail.” Recompute after each tick because the tail position and obstacles change.

Do not chase a tail through a move that will eat food unless the simulation proves that growth remains safe. Do not enter a corridor whose length is smaller than the body’s remaining time-to-clear it. If all tail routes are blocked, maximize open area and preserve a turn rather than greedily reducing distance to food.

**Code:** search over the moving-body transition, treat the current tail as a future target, rank by tail reachability, connected free area, and distance only after safety. A flood fill is a useful fast safety filter.

**Astra:** select `TAIL_FALLBACK` when a food route is unsafe, and decide how conservative to be when the head/tail reachability test is borderline.

**Jev:** answer a typed mode choice (`FOOD`, `TAIL`, `CYCLE`) from a state with code-provided safety labels, or choose the best of a few legal tail-following candidates.

### 4. Hamiltonian-cycle safety mode

On a rectangular grid with at least one even dimension, both dimensions greater than one, and no internal obstacles, precompute a verified Hamiltonian cycle. Verify that each consecutive pair, including the final-to-first pair, is orthogonally adjacent and that every cell occurs exactly once. Assign each cell a cycle index.

Following the cycle makes the body advance in cyclic order. If all body indices remain in tail-to-head cyclic order and the head does not overtake the tail, the head cannot collide with its own body; eventual food consumption fills the board under the base rules. This is the high-confidence late-game fallback and the safest way to survive when the snake occupies a large fraction of cells.

The invariant must be checked, not assumed: an unsafe shortcut or an irregular start can break the tail-to-head order. Gaps between occupied cycle indices are permitted; requiring contiguity would wrongly reject valid shortcuts. If order is broken, do not blindly switch to the cycle. Use dynamically simulated search, find an alternative compatible tour, or execute a checked entry route; none of those entry procedures has a general completion guarantee. The cycle source by John Flux emphasizes that collision safety follows from maintaining tail/head ordering on the one-dimensional cycle ([John Flux](https://johnflux.com/2015/05/02/nokia-6110-part-3-algorithms/)).

**Code:** generate or load the cycle, validate it, maintain indices, determine whether a candidate shortcut preserves tail-before-head ordering, and reserve enough cycle distance for pending growth. A simple fixed cycle is suitable for the reference 12×12 board; a validated generator is safer for arbitrary dimensions.

…(truncated; full text in prior/strategies.md)

- **Greedy food:** direct route enters a pocket; reject after post-eating tail test.
- **Manhattan-only planning:** ignores body order and walls; use BFS/A* over the current grid and simulation.
- **Tail collision bug:** treats the current tail as permanently occupied on every move; implement tail vacating exactly.
- **Growth collision bug:** allows the tail to vacate when eating; simulate food transitions separately.
- **Immediate reversal:** model requests opposite direction; ignore it and record the event.
- **Stale action:** Jev answers an old state after a tick; attach `state_tick` and reject stale responses.
- **Screenshot lag/focus loss:** GUI receives no key or key goes elsewhere; selected symbolic API avoids this; a GUI fallback must log accepted input.
- **Cycle overconfidence:** fixed cycle is applied after a shortcut or odd initial body breaks ordering; validate invariant and fall back to safe search.
- **Shortcut late in the game:** head overtakes tail after growth; reserve cycle distance and disable shortcuts at high occupancy.
- **Infinite tail loop:** the snake survives but never eats; use cycle/food progress and score as the objective, not survival alone.
- **Odd dimensions or obstacles:** a simple rectangular Hamiltonian cycle may not exist; use safe dynamic search or a generated cycle that is explicitly validated. The [Snake-game repository](https://github.com/shashank11yadav/Snake-game) notes the checkerboard parity issue when both dimensions are odd.
- **Terminal ambiguity:** time limit is confused with collision or full-board completion; retain distinct `terminated`, `truncated`, and `terminal_reason` fields.

## What belongs in each component

### Code / deterministic adaptor

Put all exact, repeatable work here:

- state parsing, coordinate/vector conversion, legal-action mask, reverse rejection;
- wall/body collision and tail-vacating transition;
- seeded food RNG, score, length, terminal/truncation bookkeeping;
- tick scheduler, deadlines, stale-action rejection, and latency trace;
- BFS/A*, path simulation, flood fill, tail reachability, occupancy;
- cycle construction/validation, cycle-index ordering, shortcut proof checks;
- deterministic tie-breaks and replay logs;
- reporting final score, length, ticks, reason, and completion.

No model should be asked to count Manhattan distances, remember body order across calls, or prove an action is in bounds.

### Astra prompts

Astra is slow and strong; use it for low-frequency supervisory reasoning:

- choose the active mode (`FOOD`, `TAIL`, `CYCLE`) when code provides candidate analyses;
- choose board-specific risk thresholds and whether to abandon a shortcut policy;
- interpret a failed trace, identify which invariant was violated, and propose a policy change;
- handle a new board shape, obstacle variant, wrap-rule variant, or nonstandard initial state;
- plan a high-level route/cycle orientation and set conservative versus aggressive play;
- review held-out seed results and keep rules that improve score without increasing terminal collisions.

Astra should not be in the per-tick blocking path. Ask it for a policy/threshold/route before the next several expected turns, then let deterministic code execute and verify that plan.

### Jev questions

Jev is fast and typed; feed it the current JSON plus code-computed candidate labels and require a small finite answer:

- `choice`: which of the already legal candidate directions is safer now;
- `choice`: `FOOD`, `TAIL`, or `CYCLE` when each candidate is annotated with safety/area/food data;
- `yes_no`: whether the food route is visibly entering a dead end, whether a turn is urgent, or whether a cycle invariant still appears intact;
- `score`: a qualitative risk class only if code supplies the measurable features; do not ask for arithmetic or long plans.

Good prompt shape: “Given this state and these prevalidated candidates, return exactly one of `UP|RIGHT|DOWN|LEFT`.” Bad prompt shape: “Find a path to the food, count the reachable cells, and guarantee it is safe.” The latter should be deterministic code.

## Source notes

- [John Flux — Nokia 6110, Part 3: Algorithms](https://johnflux.com/2015/05/02/nokia-6110-part-3-algorithms/)
- [Snake-game — A*, Safe A*, Hamiltonian cycle and shortcuts](https://github.com/shashank11yadav/Snake-game)
- [MarkMonnin SnakeAI](https://github.com/MarkMonnin/SnakeAI)
- [Snake Hamiltonian cycle discussion](https://kychin.netlify.app/snake-blog/hamiltonian-cycle/)
- [Playing Snake on a Graph](https://arxiv.org/pdf/2506.21281)
- [AlphaSnake paper](https://arxiv.org/abs/2211.09622)
- [Jumanji Snake environment](https://instadeepai.github.io/jumanji/environments/snake/)
- [The Original Snake — scoring and speed](https://theoriginalsnake.com/blog/snake-game-scoring)
- [Snake (video game genre)](https://en.wikipedia.org/wiki/Snake_(video_game_genre))

### study/brief.md
# Snake — game brief

## Scope and version choice

“Snake” is a family of games rather than one canonical software release. The genre began with *Blockade* (1976) and has hundreds of later implementations; the familiar Nokia version is influential but is not a universal specification. The genre-level invariant is that a continually moving head is steered on a grid, food increases length, and collision ends the run. [Snake (video game genre), Wikipedia](https://en.wikipedia.org/wiki/Snake_(video_game_genre))

For this study, the adaptor should target **classic single-player, bounded-grid Snake**:

- rectangular discrete board;
- no wrap-around: leaving the rectangle hits a wall;
- one snake and one food item;
- one cell of forward movement per game tick;
- four absolute cardinal steering directions;
- the snake may not reverse directly into its neck;
- food is placed on an unoccupied cell, and eating it grows the snake and increments score;
- hitting a wall or the snake's body terminates the run;
- no power-ups, portals, enemies, obstacles, or multiplayer unless a scenario explicitly declares one.

This is a deliberately explicit local contract, not a claim that every commercial Snake clone uses these exact dimensions, speed, reverse-input behavior, or scoring rules. The historical ancestor *Blockade* was a two-player light-cycle game in which trails—not apples—were central; this study targets the later single-player food-growing genre, not a faithful *Blockade* emulation ([Arcade Museum: Blockade](https://www.arcade-museum.com/Videogame/blockade)). The [Coolmath Snake instructions](https://www.coolmathgames.com/0-snake) and [SnakeGa.me rules](https://snakega.me/how-to-play/) are representative later implementations: arrow keys steer a continuously advancing snake, apples grow it, and wall/self collision loses. SnakeGa.me also documents the common “no immediate reverse” rule.

## Objective and end states

The primary objective is to **eat as many food items as possible while remaining alive**. The score is the number of food items eaten; each item adds one point and one body cell in the base contract. A run is a success if the snake fills every playable cell (or the selected implementation declares its maximum-length condition); otherwise it terminates on collision or on an external time/step limit. There is no opponent in the base game.

The terminal reason must be explicit in the observation:

- `wall_collision` — the proposed head is outside the board;
- `self_collision` — the proposed head intersects body that remains occupied;
- `filled_board` — no empty cell remains after eating (success);
- `time_limit` — an evaluator-imposed truncation, not a collision;
- optionally `invalid_action` only if a chosen third-party backend treats an invalid action as terminal. The recommended local contract ignores an immediate reverse and continues straight, as keyboard Snake commonly does.

When a non-food move enters the current tail cell, the tail vacates during that same tick, so that move is legal. When the move eats food, the tail does **not** vacate and the body grows; collision checking must use that distinction. This is an implementation detail that strongly affects a safe adaptor and should be tested directly.

## Board, state, and food

The recommended reference scenario is a **12 × 12** grid, matching the documented default of the headless [Jumanji Snake environment](https://instadeepai.github.io/jumanji/environments/snake/). The local adaptor should accept `rows` and `cols` so the same policy can be evaluated on other rectangular boards. Use zero-based coordinates `[row, column]`, with row 0 at the top and column 0 at the left.

Recommended initial state for the reference scenario:

- snake length 3, represented head-first;
- head near the centre, travelling in one cardinal direction;
- one food sampled uniformly from currently empty cells;
- a seeded, local pseudorandom generator for all subsequent food locations;
- no food can spawn under the snake;
- `score = length - initial_length` for the standard one-food/one-cell rule.

A reset seed must determine the initial placement and food sequence reproducibly. The exact random-number algorithm is less important than using one documented generator consistently and exposing the seed in `info`.

The recommended symbolic observation is JSON, not pixels. It should contain at least:

- board dimensions and current tick;
- `snake`, a head-first list of occupied `[row, col]` cells;
- `head`, `tail`, current direction, and food position (or `null` after a fill);
- score, length, free-cell count, terminal flags/reason;
- the next-tick deadline or remaining time, when run by the real-time wrapper;
- non-reverse legal directions and one-step collision-safe directions, calculated by the adaptor;
- seed and scenario metadata.

The raw body list is important: a head-only or 3×3 local image loses the tail ordering and free-space information required for reliable late-game play.

## Actions and controls

Use absolute actions `UP`, `RIGHT`, `DOWN`, and `LEFT`. The adaptor should accept their string names in the JSON protocol and may expose integer aliases `0, 1, 2, 3` in the order `UP, RIGHT, DOWN, LEFT`, which is the documented ordering in Jumanji's Snake environment ([environment page](https://instadeepai.github.io/jumanji/environments/snake/), [API page](https://instadeepai.github.io/jumanji/api/environments/snake/)).

At each tick, the latest valid requested direction is applied. A request opposite to the current direction is ignored for that tick rather than allowing a 180-degree suicide; repeated current-direction requests are harmless. A malformed direction is rejected and recorded in `info`; it must not silently become a random action. Between direction changes the snake continues straight without requiring an action every tick.

For a GUI version, arrow keys are the canonical controls; WASD is a common alternative. A real-time adaptor must not use pause as an agent action. It should have a separate reset/stop control and should record when an input arrived relative to the tick deadline.

## Pace and tier

Snake is **tier 2** for this harness: the snake advances on its own at a fixed tick rate while Astra is reasoning, and a stale action can leave the head moving into a wall or body. This remains true even though a symbolic test API may offer a synchronous `step` function for reproducibility; the real-time wrapper must keep a scheduler running independently of model calls.

…(truncated; full text in prior/brief.md)

For the local reference runner, use `tick_s = 0.10` (10 moves/second), expose it in the observation, and permit a scenario to test another fixed rate. The action deadline is one tick, approximately **0.10 seconds**; a controller that does not send a new direction retains the previous direction. A policy should decide before the next turn, not wait for a screenshot after the head has moved. At 80 ms, the same policy has only 0.08 s per tick. Jev's nominal ~0.5 s response time spans about five reference ticks, so Jev should be asked at anticipated junctions with a short code-validated route already queued; code should hold straight and execute the route between junctions. Astra should supply a longer-horizon mode/route plan and must never block the tick loop.

The score is not time-based, so a survival loop without eating is not progress. The evaluator should report final score, final length, ticks survived, terminal reason, and whether the board was filled. Use food count/final score as the primary performance measure, with survival and completion as diagnostics.

## Reference backend versus local contract

[Jumanji](https://instadeepai.github.io/jumanji/environments/snake/) is a useful documented reference: its default is 12×12, actions are Up/Right/Down/Left, observations include a 5-channel grid, `step_count`, and an action mask, reward is +1 for fruit and 0 otherwise, and `Snake-v1` has a 4000-step limit. It is an RL environment whose normal `step` call is synchronous; it is not a real-time commercial game. The selected adaptor below uses a small local symbolic API so it can preserve the tier-2 clock without depending on JAX or a GUI.

The distinction between a synchronous test step and a wall-clock run matters. The official [Gymnasium environment contract](https://gymnasium.farama.org/api/env/) says an environment advances when `step()` is called and returns `(observation, reward, terminated, truncated, info)`. The local real-time wrapper must add an independent tick scheduler; it must not claim that an ordinary Gymnasium `step` automatically advances while a model thinks.

## Sources and further reading

- [Wikipedia — Snake (video game genre)](https://en.wikipedia.org/wiki/Snake_(video_game_genre))
- [Arcade Museum — Blockade](https://www.arcade-museum.com/Videogame/blockade)
- [Coolmath — Snake instructions](https://www.coolmathgames.com/0-snake)
- [SnakeGa.me — How to play](https://snakega.me/how-to-play/)
- [Jumanji — Snake environment](https://instadeepai.github.io/jumanji/environments/snake/)
- [Jumanji — Snake API](https://instadeepai.github.io/jumanji/api/environments/snake/)
- [Gymnasium — Env API](https://gymnasium.farama.org/api/env/)
- [John Flux — Nokia 6110, Part 3: Algorithms](https://johnflux.com/2015/05/02/nokia-6110-part-3-algorithms/)

### adaptor/ADAPTOR.md
# Snake adaptor

This adaptor implements the study-selected **local symbolic Snake API**. There is
no external game binary, display, account, server, browser, or network connection:
`game.py` is the faithful local reference implementation used by the evaluator.

## Interface and timing

`make_game(**kwargs)` returns a `RealtimeGame` named `Snake (local classic)`.
The runner owns the real-time clock and calls `tick(action)` at the declared
cadence; `tick()` does not sleep and advances exactly one grid cell. The default
cadence is `tick_s = 0.10`. `observe()` is a side-effect-free deep snapshot.
`step(action)` is also provided for deterministic replay/unit tests, but does not
turn the game into a turn-based benchmark.

`ActionSpec` names the five advertised actions:

- `UP`, `RIGHT`, `DOWN`, `LEFT`: absolute steering directions.
- `NOOP`: keep the current heading.

Integer aliases `0..3` map to `UP, RIGHT, DOWN, LEFT`. A dictionary with
`action` and optional `state_tick` is accepted too. A stale `state_tick`, malformed
input, or immediate reverse is ignored, recorded in diagnostics, and the snake
continues moving; malformed input never raises from `tick()`.

## Rules and observations

The reference scenario is a 12x12 zero-based `[row, column]` bounded board, with
no wrapping, one food cell, a valid length-three opening selected near the centre,
and a private `random.Random(seed)` stream. Food is uniformly selected from the
row-major list of currently empty cells, including after every successful eat.
The body in `observe()["snake"]` is head-first. A non-food move may enter the
current tail because it vacates on that transition; a food move retains the tail
and grows by one. Wall/self collision terminates. Filling the board terminates
successfully without sampling another food. A live episode at `max_ticks` is
truncated as `time_limit`.

The symbolic observation contains the complete body, head/tail, food, direction,
score, length, free cells, legal one-step safety aid, timing metadata, terminal
flags/reason, last reward/event/error, and the attempted head. It is strict JSON
and contains no screenshots or binary assets.

## Seeds and scenarios

`SEEDS` declares training seeds `0..7` and held-out seeds `101..107`. A seed
selects both the valid central opening and all future food placements; it is not a
hard-coded level or a policy. Same seed, profile, and applied actions reproduce
the same trajectory on a fresh instance. Held-out seeds use the same rules but
unseen RNG streams/routes.

The default `EVAL_KWARGS` is the reference profile:

```json
{"rows": 12, "cols": 12, "tick_s": 0.10, "max_ticks": 4000}
```

For explicitly declared generalisation experiments, `make_game` accepts fixed
constructor options such as:

- `rows=8, cols=8, tick_s=0.08, max_ticks=4000`
- `rows=10, cols=14, tick_s=0.15, max_ticks=6000`
- `rows=14, cols=10, tick_s=0.25, max_ticks=6000`
- `rows=11, cols=11, tick_s=0.10, max_ticks=5000` (odd-by-odd stress case;
  no Hamiltonian-cycle assumption is made)

These options alter only board dimensions, cadence metadata, and the explicit
external tick budget. Wall wrapping, growth, reverse handling, food sampling,
and score semantics never vary with seed or profile. Dimensions must be at least
4 and the maximum tick budget prevents an infinite tail loop.

## Score and limits

`score` is exactly the cumulative number of food items eaten, one point each.
`score_breakdown()` and `info()` report food count, final length, ticks survived,
`terminal_reason`, `filled_board`, `max_score`, and score fraction. Survival
without eating is diagnostic only and does not increase score. The reference
maximum is `12*12-3 = 141`; other profiles report their own `rows*cols-3`.

The adaptor is process-free and resource-free. `close()` is idempotent, clears
pending input, and releases the in-process state. No environment variables or
credentials are required.

### study/interfaces.md
# Snake — programmatic interfaces

## Decision summary

There is no single official network API or canonical package for the game named “Snake.” It is a genre with many incompatible rulesets. The standard Gymnasium project supplies an environment interface but does not ship a canonical Snake game; community packages implement their own variants. For this offline, headless, no-account run, the chosen interface is:

> **Chosen: `api` — a local, Gymnasium-shaped symbolic JSON API with an independent real-time tick scheduler.**

The next phase should implement the small reference rules in [brief.md](brief.md), rather than launch a GUI. It needs no package, display, server, account, API key, browser, or ROM. It gives the model exact state with low, deterministic in-process observation/action overhead, makes seeds reproducible, and avoids the ambiguity and fragility of pixel reconstruction. A synchronous `reset(seed)`/`step(action)` mode is useful for tests, but the adaptor must also run a clock-driven mode in which ticks continue while Astra or Jev is thinking.

The chosen symbolic API is a **recommended adaptor contract to implement**, not an installed third-party service or an existing official Snake endpoint. No credentials are needed unless the scope later changes to a named external game.

## Chosen API: exact adaptor contract

### Lifecycle

The adaptor should expose these conceptual operations (HTTP, a subprocess JSON-lines protocol, or an in-process class are all acceptable; in-process is simplest):

- `reset(seed, options)` → `(observation, info)`. `info` contains the resolved seed, scenario/options, initial length and rules version. Reference defaults: `rows=12`, `cols=12`, `tick_s=0.10`, `max_ticks=4000`, initial length 3, no wrapping. Seed is mandatory in evaluation. Reset stops any previous scheduler and clears pending input before preparing tick 0; `run_realtime()` starts the clock after the controller is initialized.
- `observe()` → a fresh snapshot; it must not return a mutable object that the controller can accidentally change.
- `set_direction(direction, state_tick)` → enqueue the latest `UP|RIGHT|DOWN|LEFT` request for the next tick. The adaptor should reject a stale `state_tick` rather than applying a decision to a different state. The current direction is retained if no request arrives.
- `step(direction)` → deterministic test-mode advance by exactly one tick, returning `(state, reward, terminated, truncated, info)` in the Gymnasium shape. It must not be used as evidence that the real-time game waits for Astra.
- `run_realtime()` / a background scheduler → advance at `tick_s` independently of model calls. Every tick publishes a state and records `input_age_s`, `deadline_s`, and whether the action was on time. The harness can replace wall-clock sleep with a deterministic virtual clock while retaining the same deadline semantics.
- `close()` → stop the scheduler and release resources. A terminal episode must require `reset` before another step.

…(truncated; full text in prior/interfaces.md)

- [Gymnasium Env API](https://gymnasium.farama.org/api/env/)
- [Jumanji Snake](https://instadeepai.github.io/jumanji/environments/snake/)
- [Jumanji Snake API](https://instadeepai.github.io/jumanji/api/environments/snake/)
- [Jumanji on PyPI](https://pypi.org/project/jumanji/)
- [Jumanji source `Snake` environment](https://github.com/instadeepai/jumanji/blob/main/jumanji/environments/routing/snake/env.py)
- [snake-environment on PyPI](https://pypi.org/project/snake-environment/)
- [Python snake environment source](https://github.com/LPengulin/Python_snake_environment)
- [snake-gym source](https://github.com/vivek3141/snake-gym)
- [snake-gym on PyPI](https://pypi.org/project/snake-gym/)
- [gymnasium-snake-game on PyPI](https://pypi.org/project/gymnasium-snake-game/)
- [gymnasium-snake-game README/source](https://github.com/lychanl/Gymnasium_Snake_Game)
- [Playwright Python introduction](https://playwright.dev/python/docs/intro)
- [Playwright keyboard API](https://playwright.dev/python/docs/api/class-keyboard)
- [Playwright screenshot API](https://playwright.dev/python/docs/api/class-page#page-screenshot)
- [Xvfb manual](https://xorg.freedesktop.org/archive/X11R7.7/doc/man/man1/Xvfb.1.xhtml)
- [PyAutoGUI documentation](https://pyautogui.readthedocs.io/en/latest/)
- [snake-io example](https://github.com/puterazarul/snake-io)
- [Battlesnake webhook API](https://docs.battlesnake.com/api/webhooks)
- [Battlesnake Game object](https://docs.battlesnake.com/api/objects/game)
- [Python monotonic clock documentation](https://docs.python.org/3.12/library/time.html#time.monotonic)


## Measured model latency (at archive creation, 2026-10-30 11:42)
- System-2 (Astra), one-word answer: 32 s
- System-2, an answer of ~1220 output tokens: 161 s — latency grows with the length of the answer; size timeouts for the answers your prompts actually request
- System-1 (Jev), one request with the game's initial observation and a Choice over its actions: median 136 ms (= 1.4 ticks), p90 157 ms, max 158 ms over 12 warm requests; first request on a new connection 184 ms — a timeout below the tail turns slow answers into failed calls
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
