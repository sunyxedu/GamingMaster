# Snake — programmatic interfaces and adaptor specification

Research date: 2026-09-21. **Recommendation: implement a small local, headless Snake environment with a real-time JSON API**, using `classic-walled-v1` below. This is within the user's explicit fallback authorization. It requires no third-party game server or credentials. Everything in the proposed contract is a requirement for the next phase, not an already available API.

## Interface survey

Latency estimates below are engineering expectations, not measurements on this machine. A library being mature does not imply its particular Snake integration is maintained. There is no single publisher API for the Snake genre; the searches found community environments and version-specific integrations.

| Route | Exposure and control | Maturity and latency | Cost, rights, and fit |
|---|---|---|---|
| **Local implementation — chosen** | Exact ordered body, food, direction, score, tick; absolute direction submissions. Seeded reset and replay. | New implementation must be verified. In-process/IPC updates should be far below a 200 ms tick; target p99 under 10 ms for controller work and measure it. | No game service charge/account or third-party service ToS. Use original implementation/assets. Small, explicit rule surface; deterministic and offline. |
| [helpingstar/gym-snakegame](https://github.com/helpingstar/gym-snakegame) | Gymnasium reset/step; numbered grid preserves body order; four absolute actions. Human/RGB rendering. | Small project; [metadata calls it Alpha](https://raw.githubusercontent.com/helpingstar/gym-snakegame/main/pyproject.toml). Local synchronous steps wait for caller; rendering is separate from an autonomous clock. | MIT, free; Python >=3.10 with NumPy, Gymnasium, MoviePy, Pygame. Closest off-the-shelf candidate, but inspected correctness issues below prevent recommending it unchanged. |
| [telmo-correa/gym-snake](https://github.com/telmo-correa/gym-snake) | Classic square-grid environments, tile image observations, relative forward/left/right actions. Also hex and multiplayer variants. | Research-era OpenAI Gym project; older API/dependencies, not native current Gymnasium. Local steps wait. Default episode limit is 1,000 steps. | BSD-3-Clause project, free; Python/Gym/NumPy and PyQt5 for graphics. Migration, body-order extraction, seeding, terminal behavior, and time-limit audit needed. |
| [chynl/snake](https://github.com/chynl/snake) bot/game project | Source contains a game and graph-search/RL players; could expose internal state and replace the built-in player. | Established community example, but not a standardized environment API; integration requires source changes. Local controller timing can be fast. | MIT, free; Python >=3.10 and project dependencies; PyTorch only for optional RL. Useful reference, but do not accidentally benchmark its bundled AI as the generated harness. |
| Browser instrumentation + [GoogleSnakeModLoader](https://github.com/DarkSnakeGang/GoogleSnakeModLoader) | Mod loader offers a modded game site. Custom hooks could expose internal state; keyboard events steer. Mod availability is not an observation/reset API. | Community, version-specific; minification/updates can break hooks. In-page logic can be fast; external browser calls add variable delay. | Browser and mods generally accessible without game payment; automation permissions and reused assets need version/site review. Not selected: reproducible seeding/state access unverified. |
| Browser screen/input or own-page JS bridge | [Playwright evaluation](https://playwright.dev/python/docs/evaluating) can retrieve deliberately exposed state; [keyboard API](https://playwright.dev/python/docs/api/class-keyboard) sends arrow events. Without hooks, capture canvas and infer cells/body order over time. | Mature browser tooling, custom game bridge. DOM snapshots alone do not describe a canvas board. Capture, rendering and browser scheduling add latency/jitter; benchmark rather than assume a fixed number. | Free tooling plus browser. An own local page avoids third-party service dependence; public sites have separate terms. Requires focus/input and background throttling management. |
| Desktop screenshot + input injection | [PyAutoGUI](https://pyautogui.readthedocs.io/en/latest/) provides screenshots and key presses; add pixel classification/OpenCV and track body motion. | Mature generic tool, fragile game perception. Documentation quotes ~100 ms for a 1080p screenshot and 1–2 s for image locating; default call pause is 100 ms, plus macOS input delay. | Free packages; macOS screen-recording/accessibility permissions may be required. Site/app rules still apply. Latency and ambiguous body order make this a poor primary route. [Timing source](https://pyautogui.readthedocs.io/en/latest/screenshot.html). |
| Emulator hooks | For a compatible Snake ROM/homebrew, [BizHawk Lua](https://tasvideos.org/Bizhawk/LuaFunctions) offers frame advance, controller injection, memory reads, and state tooling. | Mature emulator tooling; game-specific memory map and platform/core support need work. Frame-stepping can wait indefinitely; real-time emulation retains deadlines. | Emulator tooling is available freely; user must supply a lawfully usable game image where necessary. No particular ROM/core or native macOS setup was verified. Not a ready Nokia emulator recommendation. |
| [Battlesnake API](https://docs.battlesnake.com/api/introduction) | JSON HTTP webhooks with board/body state and move responses. Local rules engine also exists. | Purpose-built bot ecosystem; request/response deadlines and possible network latency. | Free public game with hosting/account considerations; local engine possible. **Different game:** health drain, rivals, simultaneous moves and delayed growth. Reject for this classic-Snake task. [Rules](https://docs.battlesnake.com/rules). |

### Why the nearest ready-made environment is not selected

Static inspection of [helpingstar's `snake_game.py`](https://raw.githubusercontent.com/helpingstar/gym-snakegame/main/gym_snakegame/envs/snake_game.py) found: collision is checked against the old tail before removal; full-board length is checked before appending the newly grown head, so the winning move is not recognized; returned `prev_action` is updated after building `info` and records the request rather than necessarily the executed heading; the four-channel implementation order disagrees with the README. These are source findings, not reproduced runtime failures. Its `render_fps=20` throttles human rendering; it does not advance the world independently. Patching and wrapping is possible, but a tiny original environment avoids those ambiguities and unused dependencies. Do not generalize these defects to all Gym environments.

If revisiting that option, the documented ID is `gym_snakegame/SnakeGame-v0`, with `board_size`, `n_channel=1`, `n_target=1`, and `render_mode=None`. Actions are 0 down, 1 right, 2 up, 3 left. The one-channel array shape is `(1, size, size)`, with 0 empty, 1 head, increasing body indices, and `size²+1` food. Pin the retrieved revision and repair/test before trusting scores. [README](https://github.com/helpingstar/gym-snakegame). Never infer coordinates from variable names: the source uses row/column indexing.

## Installation, credentials, and dependencies

For the chosen route, **no account, API key, server address, ROM, browser, GPU, or environment variable is required**. `study.json` therefore uses an empty `env` array. Model-provider credentials belong to the surrounding Meta-Harness, not this game adaptor; never put credential values into these files or logs.

Required runtime: Python 3.11+ is recommended, using only the standard library for core state, JSON, queues, scheduling, and seeded RNG. This machine's `python3 --version` returned 3.9.6, so the adaptor phase should select/install a newer interpreter if following that recommendation. Nothing was installed during study. Optional: Pygame for human display; optional Gymnasium/NumPy only if an offline compatibility wrapper is useful. The primary interface must work headlessly without them. Lock any installed dependency versions in the adaptor phase.

Create an isolated virtual environment in the project. Implement the game process and a thin transport wrapper; then implement the independent player/harness. Keep game transitions and policy decisions separate so the environment never secretly avoids a collision for the player. The environment's only missing-input behavior is continuing straight.

## Complete chosen contract

### Board, transition order, and terminal states

Use integer `[x,y]` cells, origin at top left, x rightwards and y downwards. The body array is **head first, tail last**. Direction displacements: up `[0,-1]`, right `[1,0]`, down `[0,1]`, left `[-1,0]`. Default board 10 × 10; length 3; score 0; exactly one food; fixed tick 200 ms; no wrap, obstacles, acceleration, or starvation. Ruleset version: `classic-walled-v1`.

At each movement tick:

1. Consume the accepted command for that tick, if any. A reverse-direction command is ignored, using the current heading instead. No command also keeps the heading. Never process two turns within one movement tick.
2. Calculate the neighboring head cell and whether it contains the current food.
3. Check the boundary. On a non-food move, collision occupancy excludes the old tail; on a food move it includes the entire old body. A collision terminates without changing body or score; record attempted destination separately.
4. On success, prepend the new head. Remove the old tail only on non-food moves; otherwise increment food score once.
5. After growth, if length equals width × height, terminate with `won=true`, `food=null`, and reason `board_full`. Otherwise choose replacement food uniformly from empty cells in row-major order using the episode RNG.
6. Increment tick once for every attempted movement, including the fatal move. Publish a complete immutable snapshot. Freeze all terminal states until explicit reset.

This transition order is the study's normative contract. Never spawn food on the body, sample an empty set, allow instant reversal through two queued turns, or return success before committing the final growth.

### Lifecycle and transport

Use newline-delimited JSON over a local child process's stdin/stdout; diagnostics go to stderr. No network service is necessary. Give every request a `request_id`; every response carries that ID, `episode_id`, and current `tick`. State events are distinguishable by `type=state`; responses by `type=response`. Use a background transport reader so a slow model call or a full output pipe cannot block the simulation. Coalesce live observation events when consumers are slow; record the complete applied-action replay to a local file independently.

| Operation | Required semantics |
|---|---|
| `reset(seed, options)` | Validate integer seed/profile, cancel pending inputs, create a new episode ID, reset RNG and counters. Return initial state in `ready` status; clock not started. Options are evaluator configuration, not in-episode actions. |
| `start(episode_id)` | Start monotonic clock; first tick due 200 ms later. Already-running calls are idempotent and cannot postpone a deadline. Initial policy may be prepared before start. |
| `observe(episode_id)` | Return latest snapshot immediately; does not move, pause, restart or extend time. Also emit an observation after every tick when transport can keep up. |
| `act(episode_id, target_tick, direction, sequence)` | Accept only one of four direction strings for the **next** tick, received strictly before its deadline. Highest sequence received for that tick wins. Acknowledge queued/ignored/rejected and reason. Never silently apply a late command to a later tick. |
| `close` | Stop process and release resources. Mid-episode evaluator cancellation is truncation, not game victory. |

Only `act` is a gameplay action; reset/start are evaluator lifecycle operations. No pause or speed mutation is exposed during scored play. Validate episode IDs to prevent old replies acting after reset. Reject unknown fields/types that would alter semantics, and return errors without changing state. Terminal-state actions return `episode_finished`.

The game must own a monotonic clock in a process independent from model calls. Set deadlines from episode start and tick count; do not schedule each next deadline from when a slow call finishes. [Python's monotonic clock](https://docs.python.org/3/library/time.html#time.monotonic) is suitable for elapsed time. If the scheduler wakes late, resolve elapsed ticks in order; a newly arrived input cannot affect a deadline already passed. Preserve arrival timestamps and applied-command records. Host suspension or severe engine lag should produce a separately flagged infrastructure failure, not an invisible pause that improves the score.

Suggested observation fields (JSON-compatible values):

| Field | Meaning |
|---|---|
| `schema_version`, `ruleset_id`, `episode_id`, `tick` | Version and freshness identifiers. |
| `status` | `ready`, `running`, `won`, `dead`, or `truncated`. |
| `width`, `height`, `initial_length` | Board configuration. |
| `body`, `direction`, `food` | Ordered cell list, executed heading, cell or null. |
| `score`, `length`, `food_eaten` | Score equals food count; explicit consistency checks. |
| `tick_interval_s`, `server_time_s`, `next_tick_deadline_s`, `time_to_next_tick_s` | Engine clock information; deadline null outside running state. Do not assume a remote clock shares its origin. |
| `last_applied_action`, `last_action_sequence` | Executed action and accepted command provenance, null where inapplicable. |
| `terminated`, `truncated`, `won`, `end_reason` | Distinguish wall/body death, board completion, external tick cap, and infrastructure failure. |

Expose the current board completely but never future food, RNG state, or future random draws to the player. Keep seed in evaluator/replay metadata rather than model observations. Candidate paths, collision masks, and safety claims are **controller-derived fields**, not hidden assistance from the game.

### Seeds and meaningful held-out scenarios

Use a local seeded RNG, never process-global randomness or wall-clock seeding. Version and record the sampling algorithm. From the seed select a board from 8×8, 10×10, 12×12, 16×12, then a rotation/reflection of a central three-cell straight start with matching heading; transform dimensions as needed. Reject invalid configurations or starts without at least two clear forward moves. Sample the initial and subsequent foods uniformly from row-major empty-cell lists. Default no-options demonstration uses 10×10; the held-out suite uses the board mixture explicitly.

Hold timing fixed at 0.200 seconds in the primary suite. Optional 0.100/0.300-second stress suites must be separately named and scored. Odd-by-odd or obstacle boards are separate generalization suites because the standard full-board-cycle guarantee changes. There are no opponents to randomize. Do not construct resets to match a particular player's private cycle.

The same seed **and same applied tick-indexed actions** must reproduce the same game trajectory. The seed alone cannot make real-time model arrival jitter deterministic. Log requested and applied actions, receipt times, ignored/stale commands, profile, engine version, terminal flags, and score. Use disjoint development/held-out seed lists and prohibit reading future RNG state. Food draws remain dependent on which cells the policy leaves empty.

### Evaluation and acceptance checks for the adaptor phase

Report raw food count and normalized completion `food_eaten/(width×height−3)`; aggregate mean completion and full-board win rate, then ticks/food as a separate efficiency measure. Use a cap of `2×(width×height)²` attempted moves, marked truncation; allow a win/collision on the cap tick to take precedence. Do not award survival bonuses.

Before accepting the adaptor, verify: ignored reversal; one-turn-per-tick; growth exactly once; legal entry into departing tail; fatal entry into retained body; wall boundaries; food never overlaps the body; final-cell win without respawn; identical seeded replay; rejection of old-episode/late actions; and continued ticks during a one-second controller stall. Test the scheduler independently of rendering and check p99 controller runtime versus the actual available deadline. A synchronous debug mode should use identical transition semantics but cannot be advertised as tier-2 evaluation. No tests or game code have been written in this study.
