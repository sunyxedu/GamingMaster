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

The Gymnasium names and five-return convention are deliberately familiar. Official documentation specifies `reset()` returning `(observation, info)` and `step()` returning `(observation, reward, terminated, truncated, info)`, with `terminated` for an inherent game end and `truncated` for an external limit ([Gymnasium environment creation](https://gymnasium.farama.org/tutorials/gymnasium_basics/environment_creation/), [Env API](https://gymnasium.farama.org/api/env/)).

### JSON observation

Publish a complete symbolic state, with these fields (additional diagnostic fields are allowed):

```json
{
  "game": "snake",
  "version": "local-classic-v1",
  "seed": 123,
  "scenario": "reference-12x12",
  "rows": 12,
  "cols": 12,
  "tick": 37,
  "sim_time_s": 3.7,
  "tick_s": 0.1,
  "deadline_in_s": 0.1,
  "snake": [[6, 6], [6, 5], [6, 4], [6, 3], [6, 2], [5, 2], [4, 2]],
  "head": [6, 6],
  "tail": [4, 2],
  "direction": "RIGHT",
  "food": [2, 9],
  "score": 4,
  "length": 7,
  "free_cells": 137,
  "nonreverse_directions": ["UP", "RIGHT", "DOWN"],
  "safe_directions_now": ["UP", "RIGHT", "DOWN"],
  "terminated": false,
  "truncated": false,
  "terminal_reason": null,
  "last_reward": 0,
  "input_age_s": 0.012,
  "action_tick": 37
}
```

This is an illustrative, internally consistent state, not a recorded run of seed 123. The body list is head-first. `safe_directions_now` applies the exact tail-vacating rule: a non-growing move may enter the current tail. Food cannot occupy a body cell under the base rules, so a simultaneous food-at-tail example is an invalid state, not a valid growth test. It is a one-step legality aid, not a promise that a move will remain safe after several steps. The adaptor should also publish `food` as `null` after a full-board win and include `info` diagnostics rather than hide them in the observation.

The raw state is preferable to a screenshot. If the harness wants compact features for Jev, derive them without discarding the full observation. Jumanji's five-channel grid also contains body ordering through its normalized body-state channel; local JSON makes that ordering directly readable without decoding floating-point channels. Do not expose future food placements or the private RNG state to the player.

### Actions and transition semantics

- Accept only the four absolute cardinal directions, with optional integer aliases `0=UP`, `1=RIGHT`, `2=DOWN`, `3=LEFT`.
- A same-direction request is a no-op with respect to steering.
- An immediate 180-degree request is ignored and logged as `reverse_ignored`; it does not reverse the snake.
- On each tick, move the head exactly one cell in the current direction.
- Leaving the rectangle terminates with `wall_collision`.
- A non-food move may enter the cell occupied by the current tail because that tail leaves on the same transition. Any other occupied-body hit terminates with `self_collision`.
- A food move retains the tail, adds one segment, increments `score` and reward by one, and samples the next food uniformly from empty cells using the episode RNG.
- Eating the last empty cell terminates successfully with `filled_board` and no new food.
- A configured maximum tick count sets `truncated=true` and `terminated=false` unless the game already ended that tick.
- Reward is `1` exactly on a food-eating transition and `0` otherwise. The evaluator should use cumulative reward/score, not a hand-written “distance to food” reward that can teach unsafe loops.

A job queue may be a safer implementation than a concurrent mutation of the live state: enqueue the requested direction before the tick deadline, then atomically consume one request per tick. Keep `action_tick` as the tick from which the request was derived; on consumption, reject it if it is older than the latest published state. On every wall-clock transition, publish a new immutable snapshot and log the response/consumption ticks.

### Real-time scheduling

The scheduler owns the clock; model calls do not. It should publish a state before each deadline, accept a direction for the next tick, and advance even if neither model responds. Jev can choose the next turn while the snake travels straight; Astra can update a high-level mode without blocking. For reproducible evaluation, use a virtual monotonic clock and the same `tick`/deadline fields rather than sleeping in tests. In a wall-clock implementation, use Python's `time.monotonic()` or `time.perf_counter()` for elapsed deadlines: the official [Python time documentation](https://docs.python.org/3.12/library/time.html#time.monotonic) says these clocks do not go backwards, while `sleep()` may overshoot its requested duration. Schedule against the next absolute deadline and record lateness rather than accumulating sleep intervals.

Set the reference `tick_s` to `0.10` seconds, with scenario overrides for 0.08–0.25 seconds. A modern web “classic” implementation reports intervals from about 167 ms down to an 80 ms floor, but Snake versions disagree; see [The Original Snake speed/scoring guide](https://theoriginalsnake.com/blog/snake-game-scoring). The adaptor must not infer speed from a package name.

### Seeded resets, scoring, and held-out scenarios

These are **proposed local benchmark choices**, not universal Snake rules:

1. Use an episode-local `random.Random(seed)` from Python 3.12; never seed or consume the process-global RNG. Record the rules version and resolved options. Same seed + options + applied action sequence gives the same game states; wall-clock timestamps are not expected to match.
2. For the reference 12×12 opening, enumerate head cells in the central 3×3 square, row-major, and headings in `UP, RIGHT, DOWN, LEFT` order. Retain candidates whose three straight body cells and next forward cell are in bounds. Sample one candidate with the episode RNG, build the two trailing cells opposite its heading, and initialize score/tick to zero. This supplies valid length-3 starts with no unavoidable opening collision. Do not force starts to suit one solver's cycle.
3. Enumerate free cells in row-major order and select one uniformly with that RNG on reset and after each food. Never use unordered-set iteration or rejection sampling near a full board. On filling the board, stop before trying to sample food. The RNG stream is seeded, but future food coordinates also depend on the player's occupied cells; different policies need not see an identical coordinate sequence under the same seed.
4. Default `max_ticks=4000`, with no extra starvation timer. This is a time-limited score benchmark, not a promise that a slow cycle will fill the board before truncation. At 0.10 s, a full-duration run takes 400 s. For completion studies, declare a separate longer-budget profile (e.g. 21000 ticks on 12×12); do not silently extend a losing or low-score run.
5. Keep the main held-out distribution at 12×12 with unseen seeds. Declare separate generalization profiles: 8×8, 10×14, 14×10; 11×11 as a non-Hamiltonian-cycle stress case; and fixed 0.08/0.15/0.25 s clocks. Dimensions must both be at least 4 for these ordinary length-3 starts. No opponents, health, obstacles, wrapping, or accelerating speed are introduced implicitly by seed.
6. Report raw `score=foods_eaten`, `max_score=rows*cols-3`, and `score_fraction=score/max_score`; the reference maximum is 141. Compare raw scores within a fixed board/profile and normalized fractions across sizes. Also report length, terminal reason, filled-board flag, ticks, missed deadlines, and actual elapsed time. Survival alone earns no score; a full-board win takes priority over a simultaneous tick limit. Keep test seeds held out from policy/threshold tuning.

### Timing boundary between game and player

The game engine must **not contain an automatic collision-avoidance player**. If no valid input arrives, the game continues straight and may die. The active route, emergency steering, BFS, and cycle selection described in `strategies.md` belong to the generated player harness, whose actions are recorded normally.

- State at tick `t` is the state after `t` moves. Its next deadline advances it to `t+1`. Primitive `set_direction(direction, state_tick=t)` is accepted only while that state is current and before its deadline. Compare reverse inputs with the committed heading, not an earlier pending input; two key events in one tick must not manufacture a 180-degree turn. Consume the latest valid pending input once, then clear the pending slot.
- Jev's ~0.5 s answer cannot be sent as a primitive tied to the now-five-ticks-old state. Instead, ask it to choose a **candidate ID or mode** with a validity window while deterministic player code continues. When the answer arrives, player code rechecks the candidate against the newest snapshot, simulates its remaining route, and issues a fresh tick-stamped primitive. Discard candidates whose expected route prefix, food, body, or expiry no longer matches. Do not merely relabel an old direction with a new tick.
- Query ahead by at least the measured high-percentile model latency plus a margin; six to eight reference ticks is only a starting estimate for Jev, not a guarantee. A food spawn invalidates assumptions about future food. Cancel obsolete choices and let code handle tight corners. Astra supplies durable policies, not old coordinate commands.
- Run the clock/controller in an event loop or process that is independent of blocking model calls and expensive search. Use absolute monotonic deadlines, instrument scheduler lag, and account for all elapsed ticks. Never apply a newly arrived action retroactively to missed ticks or discard elapsed game time to rescue a run.
- A virtual-clock evaluation preserves tier 2 only if it advances the environment through the modeled/recorded call latency using the actions available at each deadline. An unlimited-thinking `step` loop is a useful deterministic test mode, but it is tier 1 and must not be scored as real-time Snake.
- Keep replay data: seed/options/version, initial observation, each applied action and tick, input receipt/deadline, food events, model query/response IDs, rejections, terminal reason. Policy output, game state, and reporting must remain separate.

### Acceptance checklist for the adaptor phase

No game implementation or playing policy was written or executed during this study. Before connecting models, the adaptor phase should verify these ordinary game-rule cases:

| Case | Expected result |
|---|---|
| Same seed/options/applied actions twice | Identical symbolic game trajectory; independent RNGs across episodes |
| Two non-food moves | Head advances twice, tail vacates twice, length/score unchanged |
| Food directly ahead | One added cell, +1 score/reward, old tail retained, next food on free cell |
| Adjacent tail in a bent body, no food | Moving into the vacating tail is legal if not reversing |
| Food placed on body in an imported fixture | Reject invalid fixture; not a legitimate growth/collision case |
| Move into an interior body cell or wall | Collision terminal, score unchanged; preserve last valid body and record attempted head in `info` |
| Opposite direction / two turns in one tick | Opposite input ignored; no extra movement or double-turn reversal |
| Last empty cell contains food | +1 score, full-board terminal, `food=null`, no sampling loop |
| Tick limit reached while alive | `truncated=true`, reason `time_limit`; no collision claimed |
| Collision or fill on limit tick | Game terminal reason wins; not counted as a pure time truncation |
| No model response, stale response, or malformed action | Straight movement continues; rejection is logged, no hidden rescue |
| `observe()` repeated / attempted mutation of a snapshot | No game advance, no aliasing into engine state |
| Reset while a previous episode had queued input | Old scheduler/input/query result cannot affect the new episode |

### Installation and credentials

For the chosen interface:

- **Required software to install:** none beyond the supplied `/home/hemlix/GamingMaster-mh-tier2/runs/snake/.venv/bin/python` (Python 3.12.3) and the standard library.
- **Optional software:** `gymnasium` can be installed if the next phase wants formal `spaces`/wrappers, but the reference API does not require it. The official Gymnasium package supports the contract; it does not provide the Snake rules.
- **Required environment variables:** none.
- **Credentials:** none. There is no server address, account, API key, browser profile, or ROM.
- **Display variables:** none. Do not require `DISPLAY`, X11, SDL, or Xvfb for the selected interface.

If the adaptor is exposed as a subprocess or HTTP service, its endpoint is local and should be passed by the harness rather than written as a credential. If a future external implementation is selected, keep its URL/token in the process environment or secret store; never commit values to these files.

## Alternative 1 — Jumanji Snake (headless JAX environment)

**What it is.** [Jumanji’s Snake documentation](https://instadeepai.github.io/jumanji/environments/snake/) describes a maintained, headless RL environment: default 12×12 grid, four actions `Up, Right, Down, Left`, a five-channel float grid, `step_count`, action mask, +1 reward on fruit, and a 4000-step `Snake-v1` time limit. The API page documents `Snake(num_rows=12, num_cols=12, time_limit=4000, viewer=None)` and JAX-style `reset(key)`, `step(state, action)`, `observe(state)`, and `render(state)` ([API](https://instadeepai.github.io/jumanji/api/environments/snake/)). The source currently initializes a random single-cell snake (`length=1`) and samples a random head/fruit from the supplied JAX PRNG key; that differs from the selected local reference's length-3 centre opening, so it is a reference backend rather than a drop-in rules match. It is based on the Snake environment described by Bonnet et al. ([paper](https://arxiv.org/abs/2111.00206)).

- **Maturity:** strongest documented/reusable option found; part of a broader tested JAX suite. It is not the official implementation of the entire Snake genre, and its state/action termination semantics must be checked against the selected version.
- **Latency:** no GUI or IPC; after JAX compilation, one step should be computationally cheap. The first call can pay JAX tracing/compilation overhead, so it is a poor place to measure 80–100 ms game deadlines without warm-up.
- **Observation/control:** structured JAX arrays; the 5 channels encode fruit, head, body/tail information and normalized body state, with an action mask. It does not by itself expose the local JSON body list or a real-time scheduler.
- **Legal/ToS risk:** low for the Apache-licensed open-source package; comply with its license. No account or network service.
- **Cost:** free, but installing JAX and its dependencies increases disk/startup cost and may require a compatible CPU wheel. Jumanji 1.1.2 declares Python `>=3.10` and classifiers for Python 3.10–3.13; the evaluator’s Python 3.12 is in that range. Its declared dependencies include `jax>=0.4.27`, Gymnasium, NumPy, Chex, DM-Env, Esquilax, Pillow, Matplotlib, Requests, and typing extensions; verify the exact resolved wheels in the venv.
- **Install:** use the evaluator interpreter, e.g. `python -m pip install "jumanji==1.1.2"`; use a CPU JAX build if pip does not resolve one. Do not install into the system Python. Jumanji is Apache-2.0 licensed ([PyPI metadata](https://pypi.org/project/jumanji/)).
- **Environment/credentials:** no credentials. No `DISPLAY`. A caller supplies a JAX PRNG key rather than an account secret.
- **Adaptation work:** wrap JAX arrays into the JSON state above, convert the step API to the harness contract, warm up compilation, and add an independent clock if retaining tier 2. Do not assume a normal Jumanji step keeps running while Astra thinks.

## Alternative 2 — community Gym/OpenAI-Gym implementations

### `snake-environment`

The [PyPI project](https://pypi.org/project/snake-environment/) reports version 0.3.6, MIT license, a `SnakeGame` class, an 18-element NumPy state, four actions, and `render=False` support; its reported method result is a nonstandard `(state, done, reward, score)` tuple. The repository is [LPengulin/Python_snake_environment](https://github.com/LPengulin/Python_snake_environment).

- **Maturity:** small community package; PyPI's current 0.3.6 upload dates to April 2024 and metadata requires Python `>=3.8` but does not declare a Python 3.12 classifier. The license field is blank although a PyPI classifier says MIT; verify the repository license before redistribution.

- **Latency/exposure:** in-process NumPy/Pygame logic; state is compact but not a full body list, and render behavior depends on Pygame.
- **Risk/cost:** Pygame/SDL can still create headless setup problems. PyPI's classifier claims MIT, but its license metadata field is blank; verify the repository's license before redistribution. No account is needed, and its unusual return order is an integration hazard.
- **Install:** `python -m pip install snake-environment`; test import and `render=False` in the evaluator venv. No credentials; no required environment variables documented.

### `snake-gym`

[The project README](https://github.com/vivek3141/snake-gym) documents `snake-v0` (150×150 RGB NumPy observations) and `snake-tiled-v0` (10×10 matrix: 0 empty, 1 snake, 2 food), with `env.fps` configuration. It says reset launches a GUI and `env.render()` is not implemented.

- **Maturity:** latest PyPI release is 1.8 from 2019; treat it as an older research/demo package. It is GUI-first and not a clean headless API.
- **Latency/exposure:** pixel mode costs image capture/interpretation; tiled mode is better but still may omit body ordering and exact timing. GUI reset adds startup latency.
- **Risk/cost:** MIT/free, no account; SDL/display failures are likely on this server. A local copy has no online ToS issue.
- **Install:** `python -m pip install snake-gym` (or a pinned source checkout). It imports the legacy `gym` ecosystem in its README, so version pinning and registration tests are required. A display/`DISPLAY` may be needed; the README does not promise a dummy-driver mode.

### `gymnasium-snake-game`

This is a more concrete community option than its sparse PyPI page suggests. The [MIT-licensed repository README](https://github.com/lychanl/Gymnasium_Snake_Game) registers `Snake-v1`, uses `gymnasium.make`, and documents `render_mode="human"`, `env.reset()`, and five-return `env.step(action)`. PyPI reports version 0.2.1, `pygame >=2.1.0`, `numpy >=1.21`, and `gymnasium >=0.29`; its classifiers identify the project as Alpha and list Python 3.10. The source defaults to `fps=60`, `max_step=200`, `init_length=4`, a 16×16 board, and a human-only render mode; options can change width, height, rewards, and colours. In `render_mode=None`, the wrapper does not open a window, but its core still imports Pygame. It returns a 3-channel grid array plus an `info` dictionary containing only head and food coordinates, so it does not expose ordered body state. Reset ignores the Gymnasium `seed` (food uses module-level `random.randint`), `step` is synchronous, and its action encoding differs from Jumanji (`0=up, 1=down, 2=left, 3=right`). The current source also compares the y boundary against the x width, so non-square boards need additional testing.

- **Maturity:** latest PyPI release 0.2.1 dates to March 2024; the GitHub repository is classified Alpha, and the PyPI metadata only lists a Python 3.10 classifier. Python 3.12 may work but is not established by that metadata.

- **Latency/exposure:** in-process NumPy/Pygame; with `render_mode=None`, reset/step do not pace themselves to a real-time clock. They advance synchronously only when called, so this package alone is tier 1, not a tier-2 clock. The 3-channel observation is reconstructible as occupancy/head/food, but does not identify body order or tail; rewards are configurable (`food_reward`, `dist_reward`, `living_bonus`, `death_penalty`), so pin options to compare scores.
- **Seed/rules risk:** its `reset(seed=None, options=None)` neither calls `super().reset(seed=seed)` nor otherwise seeds randomness; food uses module-level `random.randint`. Do not assume Gymnasium seed reproducibility. Its action convention is `0=up, 1=down, 2=left, 3=right`; adapt and test it. The source's wall check compares y against the x width, so non-square boards may have incorrect boundaries.

- **Risk/cost:** MIT/free, no account; Pygame/SDL/display compatibility and configurable reward risks remain. No credentials are documented.
- **Install:** `python -m pip install "gymnasium-snake-game==0.2.1"` in the evaluator venv only, then import the package before `gymnasium.make('Snake-v1', render_mode=None)`. Test the exact release; reject it as the primary backend if it opens a window or fails headlessly.

## Alternative 3 — browser game through Playwright

For a named local HTML5 implementation, [Playwright Python](https://playwright.dev/python/docs/intro) can launch Chromium headlessly, send key events using [`page.keyboard.press`](https://playwright.dev/python/docs/api/class-keyboard#keyboard-press), and capture pixels with [`page.screenshot`](https://playwright.dev/python/docs/api/class-page#page-screenshot). If the page is under our control, a much better variant is to instrument its JavaScript game object or add a test-only JSON endpoint; pixels should be the fallback, not the observation source.

- **Maturity:** high-quality browser automation, but it does not define Snake rules; page-specific selectors, canvas coordinates, and event code are brittle.
- **Latency:** keyboard IPC and browser scheduling are usually much slower and more variable than an in-process state call; screenshot plus image parsing can consume a substantial fraction of an 80–100 ms tick. Headless Chromium startup is hundreds of milliseconds to seconds, then input is generally workable if scheduled early.
- **Exposure:** DOM text, JavaScript-instrumented state, canvas screenshots, and native keyboard events. Canvas alone does not reveal exact coordinates, body order, or food RNG without image interpretation.
- **Legal/ToS risk:** low for a locally owned page; potentially significant if automating a third-party site, bypassing interaction limits, or sending traffic to an online service. A public game’s terms and robots/anti-bot rules control.
- **Cost/setup:** free open-source package; install `python -m pip install playwright`, then `python -m playwright install chromium` (browser download). A local URL/path is required; no account for a local page. Optional `GAME_URL`/browser path must be supplied out-of-band, never committed.
- **Adaptation:** use a fixed viewport and deterministic page, inject a state bridge if permitted, send one key before each tick deadline, and record browser timestamps. Do not choose this for the present offline task because it introduces an unnecessary browser and version surface.

## Alternative 4 — screen capture plus input injection

A Pygame or native GUI can be placed in a virtual display with [Xvfb](https://xorg.freedesktop.org/archive/X11R7.7/doc/man/man1/Xvfb.1.xhtml), sampled with Pillow/MSS, and controlled with [PyAutoGUI](https://pyautogui.readthedocs.io/en/latest/) or `xdotool`. PyAutoGUI documents `press`, `keyDown`, and `keyUp`; on Linux it uses X11. Xvfb is an X server for machines with no display hardware or physical input devices; its `-screen` option configures virtual screen dimensions and depth. It solves the display-server requirement, not the need to locate a game window, capture pixels, or inject input correctly.

- **Maturity:** individual tools are mature; the combined Snake observation pipeline is fragile and game-specific.
- **Latency:** worst of the options: capture, image segmentation, coordinate calibration, and X11 input can approach or exceed a short Snake tick. It is especially unsafe when the window loses focus or a frame is stale.
- **Exposure:** screenshots only unless a separate debug API exists; computer vision must infer grid, head, body, food, score, and terminal overlay. Input can send arrows but cannot guarantee the game accepted them.
- **Legal/ToS risk:** low for a local implementation; GUI botting may violate a third-party service’s terms or anti-cheat rules.
- **Cost/setup:** free packages, but likely OS packages (`xvfb`, possibly `xdotool`) and Python packages (`pyautogui`, `pillow`, or `mss`). Run a private X server such as `DISPLAY=:99` and ensure X authority/permissions. No credentials for a local binary; a third-party game may require an account.
- **Use case:** last-resort compatibility test or visual regression, not the primary adaptor. Keep a screenshot trace for debugging if it is used.

## Alternative 5 — emulator or memory/input hooks

An emulator can expose a specific historical Snake binary’s framebuffer, controller input, or memory. This is only meaningful after the user identifies a platform and exact ROM/build; generic Snake has no standard memory map. A platform emulator such as RetroArch/BizHawk may offer scripting, but the hook and legal status are version- and ROM-specific.

- **Maturity:** mature emulator tooling in general, no mature generic Snake hook.
- **Latency/exposure:** deterministic frame stepping and exact memory can be excellent once mapped; reverse engineering and emulator startup are costly.
- **Legal/ToS risk:** ROM copyright/licensing and distribution are the main risks. Do not download or include a commercial ROM. Online anti-cheat is irrelevant to a local ROM but terms still matter.
- **Cost/setup:** emulator plus a user-supplied lawful ROM and platform-specific script. Likely required variables include a user-local `ROM_PATH`; no server credential should be written here.
- **Use case:** preserve a historically specific Nokia/arcade ruleset, not the generic offline task.

## Alternative 6 — network/server/WebSocket API

Some web multiplayer Snake clones expose WebSockets or a server-side game protocol; search results include examples such as [snake-io](https://github.com/puterazarul/snake-io) and [schokotets/snake](https://github.com/schokotets/snake). These are implementation-specific, not an official Snake protocol.

- **Maturity:** varies from demo to hobby project; protocol stability and state authentication are unknown.
- **Latency:** potentially good on localhost, but network RTT, server tick rate, reconnects, and queueing make deadlines nondeterministic.
- **Exposure/control:** possibly structured JSON positions and direction messages, which is ideal if documented; otherwise reverse engineering is required.
- **Legal/ToS risk:** highest alternative: server terms, bot restrictions, account bans, and traffic limits apply. Never connect to a third-party service without user authorization.
- **Cost/setup:** local server may be free; remote service can require URL, account, token, and network access. Keep `SNAKE_SERVER_URL` and `SNAKE_API_TOKEN` out of files if ever supplied.
- **Use case:** only when the user names that exact server. It is not appropriate for this offline run.

## Alternative 7 — Battlesnake webhook API (documented, but a different game)

[Battlesnake's API](https://docs.battlesnake.com/api/webhooks) is the closest thing to an official structured Snake bot interface: a bot exposes `GET /` plus `POST /start`, `POST /move`, and `POST /end`. The `/move` request contains a structured game/turn/board/you state and the response chooses `up`, `down`, `left`, or `right`; the Game object documents a per-request `timeout` in milliseconds ([Game object](https://docs.battlesnake.com/api/objects/game)). It is a multiplayer competitive ruleset with health, multiple snakes, and server-controlled turns—not the bounded single-player food Snake specified here.

- **Maturity:** documented and designed for bot servers, unlike ad-hoc browser clones; not a rules match for this study.
- **Latency/exposure:** excellent structured JSON board state and one move per server turn; response time is constrained by the server timeout. It does not give a local continuous clock or classic Snake's single body/food semantics.
- **Legal/ToS risk:** follow the selected Battlesnake server's rules and bot policies; remote play requires network access and may involve an account or service registration.
- **Cost/setup:** requires a server endpoint reachable by Battlesnake and, depending on the platform, a public URL or deployment. Environment values such as `BATTLESNAKE_SERVER_URL` or credentials belong in a secret store, never in study files.
- **Use case:** only if the user meant Battlesnake rather than classic Snake. Do not install or connect it for this offline task.

## Comparison

| Interface | Maturity for generic Snake | State/control | Expected latency | Headless fit | Legal/ToS risk | Cost/setup |
|---|---|---|---|---|---|---|
| **Local symbolic API (chosen)** | Contract is standard; game logic is ours | Full JSON body, food, clock; direct direction action | Low, deterministic in-process overhead; measure scheduler lateness | Excellent | None beyond our code | No install, no credentials |
| Jumanji Snake | Strong documented RL package | JAX feature grid, mask, actions | Fast after JIT; compile warm-up | Excellent | Apache license | JAX/Jumanji install |
| `snake-environment` | Small community package; license metadata needs verification | 18-vector, four actions | In-process | Maybe with `render=False` | Verify package/repository license | Pygame dependency/compatibility |
| `snake-gym` | Demo/research package (2019 release) | RGB or 10×10 tiles; GUI reset | GUI/image overhead | Poor/uncertain | MIT | Pygame/display |
| `gymnasium-snake-game` | Alpha but documented | 3-channel array, head/food info; configurable rewards | In-process Pygame/NumPy | Maybe with `render_mode=None`; test | MIT; package-specific | NumPy/Pygame/Gymnasium |
| Playwright browser | Browser tooling mature; page not | JS bridge, DOM/canvas, keys | tens of ms+ and jitter | Good headless | Third-party ToS | Browser download |
| Xvfb + screen/input | Tools mature; vision custom | Pixels and keyboard | highest/jittery | Works with virtual display | Third-party ToS | OS packages/display |
| Emulator hook | Platform-specific | Frames/memory/controller | deterministic if mapped | Good | ROM licensing | Emulator + lawful ROM |
| WebSocket/server | Protocol-specific | JSON if documented | network/server dependent | Usually | highest | URL/account/token |

## Why the chosen interface wins

The local symbolic API is the simplest option that simultaneously satisfies offline operation, no display, no account, exact body/tail information, seeded resets, and a controllable tier-2 clock. Jumanji is the best external reference and a reasonable fallback if the next phase prefers a maintained package, but JAX compilation and its synchronous state API are unnecessary overhead and would still require a real-time wrapper. Pixel/browser/emulator paths throw away information that is already available in a local implementation. The chosen interface also lets the evaluator test safety strategies over held-out seeds without depending on network availability, GUI focus, browser versions, or third-party terms.

## Sources

- [Gymnasium environment creation](https://gymnasium.farama.org/tutorials/gymnasium_basics/environment_creation/)
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
