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

**Concrete cycle construction (our derivation, no playing code):** for an even number of columns, reserve the top row as the return path. Start at `(0,0)`, descend column 0 to the bottom, move into column 1 and ascend only to row 1, then move into column 2 and descend to the bottom. Continue alternating through columns, always excluding row 0. The final column is odd-indexed, so this traversal ends at `(1, cols-1)`; move up to `(0, cols-1)` and left along the top row back to `(0,0)`. This visits every cell exactly once before closure. For an even number of rows instead, transpose the construction. Code must validate the resulting list and the initial body's order; a centre opening is not automatically aligned with this tour.

**Astra:** choose when to switch from food pursuit to cycle mode, especially when occupancy is high or a food shortcut looks attractive but hard to prove. It may choose a cycle family/rotation in a new board scenario.

**Jev:** recognize a cycle-safe state or choose `FOLLOW_CYCLE` versus `FOOD_SEARCH` from precomputed safety flags. It should not invent a cycle coordinate sequence.

### 5. Safe shortcuts from a cycle

A cycle is safe but can be slow. A shortcut toward food is acceptable only if all of these are true:

- every edge in the shortcut is in bounds and free at the time it is traversed;
- the shortcut does not reverse or intersect the simulated body;
- after including any food growth, the head will not overtake the tail in cycle order;
- enough free/cycle distance remains for the body to advance and the tail to escape;
- the body can still reach its fallback cycle or tail route after the shortcut;
- the shortcut does not pass through food in a simulation that assumes the food remains ahead, nor make the planner’s food index stale.

The cited Nokia algorithm uses a growth/space buffer and disables opportunistic shortcuts once occupancy is above roughly half the board; the cited TypeScript project also documents a 50% shortcut cutoff as an empirical safety policy. Treat “50%” as a conservative prior, not a theorem: the exact threshold should be configurable and safety simulation remains mandatory. At high occupancy, follow the cycle exactly.

**Code:** enumerate only local cycle-respecting shortcuts, simulate them, and reject any that violate ordering or tail reachability. Never let a model output an unverified arbitrary multi-step shortcut.

**Astra:** set or revise the shortcut threshold after observing held-out runs; explain a failure trace in terms of ordering/growth rather than merely lowering a score.

**Jev:** classify `shortcut_safe` versus `cycle_only` and choose among prevalidated candidates. It is not responsible for proving a Hamiltonian invariant.

## Openings and common board situations

### Safe opening

If the initial snake is centred and points along an open row, continue straight until a turn is needed; early food routes have abundant space. If the initial state is random, first use immediate legality and area maximization rather than committing to the nearest food. Establish a cycle index and body ordering before enabling shortcuts.

**Code:** detect the first forced turn, walls, and cycle validity.

**Astra:** prefer `CYCLE` on dense layouts only if code has verified body order on that cycle; otherwise request a checked entry/escape strategy. Density alone does not make cycle entry safe.

**Jev:** choose the next turn only when a route fork occurs.

### Low occupancy / abundant free space

At low length, a safe A* food route is often faster than blindly following the cycle. Prefer a route with few turns and enough exit space. Do not mistake a large free area for proof of safety: a narrow articulation corridor can still trap the head after the food grows the body.

**Code:** compare post-path reachable area and tail access; keep a fallback route.

**Astra/Jev:** select among code-labelled `FOOD_SAFE` and `CYCLE_SAFE` options; the model should not be optimizing raw Manhattan distance.

### Medium occupancy / emerging corridors

As the snake grows, avoid cutting across the board merely to shorten food distance. Preserve a loop, keep the head and tail in the same navigable component, and prefer routes that let the tail move before the next growth. Reduce shortcut use as occupancy approaches half the board.

**Code:** compute occupancy, articulation-like narrow cells, tail ordering, and reachable area.

**Astra:** decide whether a small score gain is worth leaving the known cycle.

**Jev:** recognize “food is reachable but dangerous” and select the tail/cycle mode.

### High occupancy / endgame

Switch to cycle-following or a formally verified tail-safe route. Food will eventually be encountered along the cycle; a short-term detour is not worth losing a nearly complete board. Test the full-board transition separately, because there is no next food after the last empty cell.

**Code:** use exact cycle indices, growth reserve, and terminal handling. Report completion rather than treating `food=null` as an error.

**Astra:** may explain a cycle repair if the board dimensions or initial body do not align with the fixed cycle.

**Jev:** only make local, code-constrained direction choices.

## Latency-aware execution

The reference clock is 0.10 s, while Jev is nominally about 0.5 s per call and Astra can take seconds to a minute. A model cannot safely choose a new action at every tick. Code should execute a prevalidated sequence of straight moves and turns, ask Jev at least several ticks before the next fork, and attach every answer to the state tick it observed. If a Jev answer misses its deadline, retain the current direction or use the deterministic safety fallback; never wait at the head of a corridor for a response. Astra should update mode, threshold, or route policy only between episodes or well before a planned junction.

**Code:** maintain the active route, validate it against each fresh state, schedule the next query before its latency budget, and provide an emergency legal action without a model. **Jev:** choose one of a small set of already checked routes at a predicted fork, not a per-tick path. **Astra:** choose the high-level risk policy and analyze failure traces, not immediate steering.


## Heuristics and priority order

When more than one action is safe, use this priority order as a baseline:

1. never collide or reverse;
2. preserve a path to the tail after the simulated transition;
3. preserve enough reachable free space and an exit from bottlenecks; treat connected-area size as a relative heuristic rather than a hard safety proof (a static flood fill may miss the timing of tail movement);
4. preserve a verified cycle/loop and avoid cutting off its entrance;
5. take food if the post-food simulation remains safe;
6. minimize path length/turns and then maximize distance from walls or narrow dead ends;
7. use deterministic tie-breaking so seeded runs are reproducible.

“Maximize reachable area” is a safety heuristic, not a guarantee. A flood fill can count a large region while the body blocks its only exit; head-to-tail reachability and dynamic simulation must be checked first. The research/community references discuss flood fill, tail reachability, A*, and Hamiltonian fallbacks; see [Snake-game algorithms](https://github.com/shashank11yadav/Snake-game), [MarkMonnin SnakeAI](https://github.com/MarkMonnin/SnakeAI), and [Playing Snake on a Graph](https://arxiv.org/pdf/2506.21281).

## Failure modes and counters

There is no enemy counter-strategy in the base game. The effective “counters” are the board boundary, the snake’s own body, random food placement, growth, and a stale controller. Handle these explicitly:

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
