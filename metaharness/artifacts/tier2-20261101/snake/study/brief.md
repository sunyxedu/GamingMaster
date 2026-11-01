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

There is no genre-wide speed constant. One modern “classic” implementation reports a starting interval around 167 ms that accelerates toward an 80 ms floor, while other Nokia-era and web implementations use different curves. See [The Original Snake: scoring/speed](https://theoriginalsnake.com/blog/snake-game-scoring), [The Original Snake: how to beat Snake](https://theoriginalsnake.com/blog/how-to-beat-snake), and the implementation-specific [SnakeGa.me instructions](https://snakega.me/how-to-play/). Therefore speed is a scenario parameter, not a rule Astra should guess from the name.

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
