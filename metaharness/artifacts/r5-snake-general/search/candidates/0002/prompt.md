# Meta-Harness iteration -> candidate `0002`

Working directory: `<repo>/runs/r5-snake/search` (you have full read/write access here).

## Task
Design a harness for a **real-time** game (tier 2).

Game: `Snake` — Classic walled Snake. Eat food and fill the board. Coordinates are [x,y], origin top left; x increases right, y down. Body is head first, tail last. Start length is 3. Every tick moves exactly one cell. UP/RIGHT/DOWN/LEFT are absolute directions; NOOP continues straight. Reversals are ignored. The runner applies its current mailbox action each tick, even while the controller is thinking; late decisions can affect a later tick. Eating grows by one and spawns food uniformly in empty cells. Otherwise the tail moves away, and entering that departing tail is legal. Walls and the retained body kill; a fatal attempt increments tick but preserves body and score. Filling the board wins immediately. There is no wrapping, starvation, obstacle, opponent or acceleration. Terminal states freeze. Profile demo-v1; movement interval 0.200 seconds. Default demo boards are 10x10; primary boards are 8x8, 10x10, 12x12 or 16x12, with central rotated/reflected straight starts. Game.score is completion = food_eaten/(width*height-3), from 0 to 1. Observation score is raw food_eaten = length-3; completion is normalized. Surviving without eating earns nothing. Evaluators cap at 2*(width*height)^2 ticks, marking truncation separately and preserving earned score. Observations expose schema/ruleset/scenario versions, profile, tick, status, width/height/initial_length, body/direction/food (null on victory), length, score/food_eaten/completion, tick_interval_s, last_requested_action, last_applied_action, reverse_ignored, attempted_destination (null before first tick), terminated/truncated/won and end_reason. No future food or RNG state is exposed. Ready means reset with no tick yet; running means at least one tick applied. No clock deadlines are owned by the adaptor.

Actions:
- UP: Request one cell upward (y - 1).
- RIGHT: Request one cell rightward (x + 1).
- DOWN: Request one cell downward (y + 1).
- LEFT: Request one cell leftward (x - 1).
- NOOP: Continue in the current heading for this tick.
(no decision in time => 'NOOP')
Absolute directions, one movement per tick; immediate reversal is ignored.

Models available through `ctx`:
* `ctx.think(prompt, session_id=...)` — Astra (System-2, slow: seconds). Deep reasoning, can plan,
  can explain rules. Returns `.text` and `.session_id` (pass it back to keep conversation context).
  Budget: 40 calls per episode. Astra never sees the game unless you show it.
* `ctx.decide(state, questions)` — Jev (System-1, fast typed judgments). Budget: 4000 calls per
  episode. Read the complete Jev model context and local usage contract below before designing
  its state, questions and answer-to-action bridge. Actual latency must be measured.
* `ctx.spawn(fn)` — background thread, so Astra can think while Jev keeps acting.

The game ticks every 0.2s **on its own clock**; `ctx.act(a)` posts the current action (instant).
A tick with no fresh action re-applies the last one. Evaluation metric: mean score over seeds
(primary), then fraction of stale ticks (lower is better), then System-2 calls (fewer is better).
Clock mode: wall.

What the harness must design (this is the whole point): how Astra's slow output (a plan, a rule
summary, a policy sketch, a subgoal) is turned into *state + questions* for Jev when needed, how
Jev's typed answer is turned into a game action, when to go back to Astra, and what to do when a
decision is late.

**Generality requirement.** The best harness is re-evaluated on a *held-out game* with the same
interface (`game.describe()`, `game.action_spec`, dict observations, `ctx`) but different rules and
different observation keys. Do not hard-code this game's rules or observation fields; discover them
from `game.describe()` / the observation, or ask the System-2 model. Game-specific solvers score 0
on the held-out game.

## Prior knowledge from the study phase
The study phase researched this game (rules, interface, community strategies). Use it: implement exact
rules/bookkeeping/search as code, put planning and adaptation into Astra prompts, and fast recognition
into Jev questions — but verify every claim against observations, because sources can be wrong or
outdated for this build of the game.

### study/brief.md
# Snake — study brief

Research date: 2026-09-21. Scope: single-player, classic grid Snake. No account, multiplayer, power-ups, or specific commercial release was requested. This study recommends a local implementation in the adaptor phase; it does not implement the game or a player.

## Rules and objective

Steer an automatically moving snake toward food. Eating increases its length; colliding with a wall or its own body ends the run. The practical objective is to collect as much food as possible and ultimately fill the board. Arrow keys commonly select direction; direct reversal is disallowed. These shared mechanics are documented by the [Original Snake recreation](https://theoriginalsnake.com/how-to-play). Implementations disagree on board dimensions, wrapping, speed changes, and point multipliers: for example, [AppsFactory](https://www.appsfactory.dev/games/snake) distinguishes wrapping from solid-wall modes, while [1700 Games](https://1700games.com/arcade/classic-snake) uses ten points per food and accelerating levels. These are examples of variants, not authoritative specifications for every Snake release.

The following **study-selected ruleset**, `classic-walled-v1`, resolves those ambiguities:

| Property | Required behavior |
|---|---|
| Board | Empty rectangular grid; default 10 × 10; solid exterior walls. |
| Snake | Initial length 3; ordered head-to-tail body; one cell of motion per tick. |
| Food | Exactly one food in a uniformly sampled empty cell. Replenish immediately after eating unless the board is full. |
| Growth | One extra segment per food; the tail stays in place on that move. No delayed growth. |
| Direction | `UP`, `RIGHT`, `DOWN`, `LEFT`; opposite-direction requests are ignored and motion continues in the current direction. |
| Collision | Leaving the grid or entering retained body cells is fatal. Entering the old tail cell is permitted on a non-growing move because that tail leaves simultaneously. |
| Score | One point per food, starting at zero. Length = 3 + score. |
| Victory | Length equals board area; stop immediately, with no attempt to spawn another food. |
| Pace | Fixed 0.200 seconds per cell, no automatic acceleration in the primary profile. |
| Other rules | No starvation, shrinking, obstacles, rivals, wraparound, or bonus foods. |

The simultaneous tail movement and food-growth convention agrees with the formal movement model in [Graafsma, Manthey, and Skopalik, sections 2.2–2.3](https://arxiv.org/html/2506.21281v1). Their paper adds other theoretical conventions; this study does not adopt its repeated-position loss rule. The exact board size and tick duration above are benchmark decisions, not measured Nokia timings.

## Timing and tier

**Tier 2.** A running episode advances every 0.200 seconds regardless of observations or pending model calls. A snapshot taken just after a tick provides at most 0.200 seconds to affect the next move; later snapshots provide less. Missing a deadline continues the last heading, which can be fatal on that next tick. Several seconds of straight corridor may occasionally provide slack, but it is state-dependent and cannot be assumed.

Astra's seconds-to-minute calls cannot steer directly. Jev's approximately 0.5-second calls also span multiple ticks. Generated local control code must execute the current policy every tick, validate model suggestions against the current state, and remain functional when neither model responds. Jev selects among computed strategy options asynchronously; Astra designs or revises policies. This is a timing requirement, not evidence that Jev can reliably choose every turn.

A synchronous `step(action)` environment would instead be tier 1 **for that interface**, with no gameplay deadline. Provide such a mode only for debugging/replay, label it separately, and do not report its results as real-time performance. Pausing while models think changes the task.

## Score and progress reporting

Report `food_eaten` as the raw game score, `length`, `board_area`, `won`, `ticks`, and `end_reason`. For comparisons across board sizes, the primary scalar is `food_eaten / (board_area - initial_length)`, in [0, 1]. Also report win rate over held-out seeds. Among equal completion scores, fewer ticks is better; report efficiency separately rather than paying positive reward for surviving in circles. A collision does not erase food already earned.

The evaluator may stop a run after `2 × board_area²` ticks. This is an external truncation, not a classic Snake death or starvation rule. Mark it `truncated=true`, preserve the earned score, and distinguish it from collision or victory. At a 100-cell board the cap is 20,000 ticks, or about 66.7 minutes at the selected pace; ordinary successful runs can still last minutes. Faster offline evaluation is useful but belongs to a separately labeled profile.

## Confidence and limits

Rules are well supported; exact timing is selected explicitly because no platform was specified. External repositories were inspected through documentation/source, not installed or executed. Network retrieval through the web tool worked, but a shell `curl` request failed in this workspace. No software installation or account setup has occurred. See `interfaces.md` for the complete adaptor contract and `strategies.md` for the control design.


### study/interfaces.md
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

Required runtime: Python 3.11+ is recommended, using only the standard library for core state, JSON, queues, scheduling, and seeded RNG. This machine's `python3 --version` returned 3.9.6, so the adaptor phase should select/install a newer interpreter if following that recommendation. Nothing was installed during study. Optional: 
…(truncated)

### study/strategies.md
# Snake — strategy prior for Astra, Jev, and generated code

Research date: 2026-09-21. Target: the real-time, empty, walled rectangular board in `interfaces.md`, one food, growth one, immediate tail release on non-growing moves. There are no opponents, builds, weapons, rotations, or adversarial counters in this variant. The relevant counters are geometric traps, greedy mistakes, and missed deadlines.

## Evidence and the central strategy

[John Tapsell's Nokia Snake project](https://johnflux.com/2015/05/02/nokia-6110-part-3-algorithms/) uses a Hamiltonian cycle, then accelerates food collection with shortcuts that preserve cyclic tail/body/head order and growth space. He disables shortcuts at high occupancy. This is a strong practical starting point, not a claim that every shortcut implementation is safe.

[chynl/snake](https://github.com/chynl/snake) describes simulating a route to food, checking escape afterward, and otherwise taking a longer route toward the tail. Its reported graph-search success is 94% on its own 6×6 evaluation, not a universal win guarantee or a benchmark for this adaptor. The [Snake-on-graphs paper](https://arxiv.org/html/2506.21281v1) provides a formal basis for Hamiltonian strategies. Arbitrary obstacles and already-entangled states require more than simply drawing a cycle on the board.

The recommendations below are this study's synthesis and engineering design. Numeric thresholds and budgets are initial tuning choices, not experimentally optimized findings. Implement exact movement, search, counting, arithmetic, and freshness checks in code. Ask models to choose policies from supplied evidence, not to calculate whether a coordinate is occupied.

## 1. Establish a safe traversal before chasing food

**Owner: code. Astra chooses the policy family and reviews invariants. Jev is unnecessary for the baseline.**

A directed Hamiltonian cycle visits every board cell once and returns to its start. If the snake initially occupies a contiguous section in the matching order, repeatedly taking the cycle successor is collision-free until board completion under this ruleset. It also eventually encounters every possible food placement. This gives a reliable score-first baseline, albeit a slow one.

For an empty rectangle with both dimensions at least two and at least one even dimension, construct a cycle directly. Do not launch an exponential Hamiltonian search on an ordinary 10×10 rectangle. One construction when height is even: reserve the leftmost column as a return lane, traverse the top row left-to-right, traverse each later row alternating right-to-left and left-to-right across columns 1 through width−1, then enter the bottom of column 0 and travel up to close the loop. Transpose for an even width instead. This construction is a mathematical derivation for the selected geometry. Verify coverage, unique cells, adjacent edges, closure, and body order before using it.

Opening: inspect the three-cell starting body and heading, choose cycle orientation/transformation that agrees with it, and start moving. If no candidate cycle contains the body in order, use simulated safe repositioning or another candidate cycle; never snap the snake onto a cycle or assume arbitrary starts satisfy the invariant. The default symmetric central starts make this easy but the controller must still check.

An odd-by-odd rectangular grid has unequal checkerboard color classes, while a cycle alternates colors, so no cycle can cover every cell. This parity argument is a geometry limitation, not proof that every individual Snake run there is unwinnable. Treat odd grids and obstacles as separate strategy regimes.

Derived progress bound: on an N-cell cycle a persistent food is reached in at most N moves; starting from length L0, full completion takes at most N(N−L0) moves. The proposed 2N² evaluator cap therefore does not cut off a correctly initialized pure-cycle policy. This bound does not apply to a controller that loops on a smaller cycle or breaks its traversal invariant.

## 2. Add shortcuts only with an explicit safety condition

**Owner: code for cycle indexing and validation; Astra for risk/efficiency tradeoffs; Jev may select among already certified candidates.**

Assign each cell its cycle index. Maintain that body cells, read from tail to head, appear in forward cyclic order even when there are gaps from previous shortcuts. A shortcut must be one actual grid move, respect the heading rule, and land on an unoccupied cell strictly in the forward cycle interval between head and tail. Check the full post-move invariant by exact simulation during development; do not rely on a hand-waved distance-to-food metric. Equality with the old tail is a separate case and is legal only when it departs and the resulting order remains valid.

Preserving order ensures an ordinary cycle-successor fallback remains available. Growth must be included in the post-state check: a food move retains the tail, reducing slack. For a multi-step shortcut, simulate every intermediate state and food consumption, not just its endpoint. Stop the known-food route at the food; a newly sampled food invalidates predictions beyond it.

For progress, prefer forward shortcuts that do not jump past the current food's cycle position. Otherwise a safe policy can repeatedly skip food. At 50% occupancy, initially disable discretionary shortcuts and use the verified traversal; tune later only with held-out evidence. The source of this conservative starting threshold is [Tapsell's implementation discussion](https://johnflux.com/2015/05/02/nokia-6110-part-3-algorithms/); it is not a universal optimum.

Common failure: after a failed shortcut test, selecting an arbitrary locally legal neighbor breaks the cycle invariant. Keep the cycle successor as the actual fallback, and validate against the current body each tick. If the invariant is already broken, explicitly enter a recovery policy; do not claim guaranteed safety.

## 3. Faster food pursuit with escape checks

**Owner: code. Astra selects when to use it; Jev can compare summarized candidate routes.**

On an open board, BFS finds a shortest path through a static occupancy grid; A* can use Manhattan distance as a lower bound for four-direction movement. Neither alone solves Snake because the body moves. For each proposed food route, simulate the ordered body through the route, including the retained tail at eating time. Reject any intermediate collision or forced reversal. Inspect the resulting escape opportunities, not just whether the food is reachable now.

Use head-to-tail reachability and available-space calculations as filters. A static path to the tail after eating is encouraging, but **not a proof of eventual completion**: the tail moves, narrow passages can close, and a small closed loop may exclude the food. Likewise, a large flood-filled region may have an unusable bottleneck. Prefer a verified cycle/order certificate when available, then bounded exact search, then these heuristics. The simulate-food-then-tail idea comes from the [graph-search reference](https://github.com/chynl/snake); the conservative distinction between heuristic and certificate is this study's requirement.

For higher difficulty, search states containing the whole ordered body, direction, and current food, not just the head coordinate. A cell that is blocked now may be free later; a release-time estimate changes when food is eaten. A time-indexed head-only route can still be wrong if it ignores the new body deposited by its own route. Bound search by an actual local compute budget, provisionally 5 ms per 200 ms tick, and preserve a precomputed fallback if it expires. Benchmark before widening the search.

## 4. Survive traps without stalling forever

**Owner: code for metrics and candidate generation; Astra for choosing recovery rules; Jev for qualitative comparisons of supplied flags.**

If a safe food route is unavailable, use a tail-following detour that opens s
…(truncated)


## Jev model context and usage for Meta-Harness

Snapshot: 2026-09-19. This is design context for the Astra proposer and for
runtime Astra when it designs Jev questions. The Python interface below is
GamingMaster's wrapper, which differs from the provider SDK. The model profile
and episode budgets in the experiment manifest are authoritative.

### What Jev supplies

Jev is TypeSafe's System One model for fast, focused judgments. It produces
typed answers and probability distributions, not prose, code, plans or reasoning
traces. It accepts text and structured JSON, not images, audio or video. A visual
game therefore needs a separate perception step before Jev can judge its state.
Calibration describes aggregate predictions, not correctness of each answer.
See [System One](https://docs.typesafe.ai/concepts/system-one).

In this architecture, Astra can interpret unfamiliar rules, choose subgoals and
design the decision interface. Jev can recognize situations, assess whether a
subgoal's preconditions still hold, or choose among well-described alternatives.
Code should handle exact arithmetic, legal-action checks, coordinate transforms,
bookkeeping and execution. Do not ask Jev to invent an unknown numerical action
mapping or perform an entire planning algorithm in one classification.

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
this model. See [State](https://docs.typesafe.ai/concepts/state).

Batch independent judgments about that state into one call. They share the
input but do not see each other's answers. If a question really requires an
earlier answer, put that answer in the next call's state. Otherwise ask both
speculatively and combine them in code. Put the complete judgment in
`instructions`: question IDs are invisible to Jev. Refer to input fields
explicitly, e.g. `plan.preconditions`. See
[Questions](https://docs.typesafe.ai/primitives).

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
For Score, use 2–10 concrete ordered levels for one dimension; the returned
number is an expected level, not necessarily an integer or a probability.
See [Score](https://docs.typesafe.ai/primitives/score).

Confidence summarizes the distribution and is not interchangeable with the
winning option's probability. Inspect both when useful. A Noul near 0.5 is
uncertainty about a yes/no proposition, not medium severity. Handle absent
confidence explicitly; never turn `None` into certainty. Choose thresholds
using training evidence and action consequences, with a fallback or Astra
escalation for ambiguity. See [Confidence](https://docs.typesafe.ai/confidence).

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
latency from traces: Jev can take longer than a game tick. Keep an executable
fallback while waiting, limit outstanding requests, and associate each response
with the observation and plan version that produced it. Recheck relevant
preconditions and legality on the latest observation before acting. A cached
semantic interpretation may outlive an individual action decision; define its
invalidation rule rather than treating every answer as permanently reusable.

The context caps each request's timeout by the remaining wall budget. Handle
transient failures within that budget, propagate `ModelUnavailable`, and join
all spawned futures. The standard inference calls already log request,
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


## Harness contract (must be satisfied exactly)
```
The harness contract — the *only* thing a proposed harness must satisfy.

A harness is a Python module that defines::

    def build(ctx: HarnessContext) -> HarnessProgram

and ``HarnessProgram`` has one method::

    def run_episode(self, game) -> None

The harness owns the whole control loop for one episode. Everything it may
use comes through ``ctx``:

* ``ctx.system2``   — Astra (slow, reasoning). ``ctx.think(prompt, session_id=...)``
                      wraps it with tracing + budget accounting.
* ``ctx.system1``   — Jev (fast, typed decisions). ``ctx.decide(state, questions)``
                      wraps it with tracing + latency accounting. Tier 1 tasks
                      pass ``system1=None``.
* ``ctx.spawn(fn)`` — run ``fn`` on a background thread (e.g. an Astra call
                      while Jev keeps the game alive). Returns a Future.
* ``ctx.clock``     — the game clock. In *virtual* mode, model latency is
                      charged to the clock deterministically (tests); in
                      *wall* mode it is real time.
* ``ctx.memory``    — a dict reset per episode by default. Explicit shared
                      memory evaluations allow learning between seeds; the
                      evaluation manifest records which protocol was used.
* ``ctx.log(...)``  — free-form events into the trace.
* ``ctx.budget``    — remaining System-2 calls / tokens; exceeding it ends the
                      episode (score kept), so harnesses must be thrifty.

Real-time games (tier 2) expose ``game.tick_s``. The runner keeps ticking the
game on its own clock: in wall mode a thread ticks every ``tick_s``; in
virtual mode each model call's latency advances the game by
``latency / tick_s`` ticks (using the *last action the harness posted*).
The harness posts actions with ``ctx.act(action)``; posting is instant
(virtual mode charges a 1 ms overhead so a loop that decides without any
model call still lets time pass). ``ctx.wait_for_tick()`` blocks until the
next tick boundary — use it instead of spinning.
Turn games (tier 1) are stepped with ``ctx.act(action)`` synchronously and
return the next observation.

Exact signatures (use these; anything else is a contract error caught in pre-flight)::

    ctx.think(prompt: str, *, session_id: str | None = None, system: str | None = None,
              images: list[str] | None = None, timeout_s: float = 300.0) -> System2Result   # .text .session_id
    ctx.decide(state: Any, questions: dict[str, Choice | Score | Noul], *, timeout_s: float = 5.0)
              -> dict[str, System1Answer]                                                   # .value .probabilities .confidence
    ctx.act(action) -> observation | None      ctx.observe() -> observation      ctx.done -> bool
    ctx.wait_for_tick() -> None                ctx.spawn(fn) -> concurrent.futures.Future
    ctx.log(**fields) -> None                  ctx.memory: dict     ctx.clock.now() -> float (s)
    ctx.budget.remaining() -> {"system2_calls": int, "system1_calls": int, "wall_s": float}
    game.describe() -> str    game.action_spec.names / .descriptions / .noop / .notes    game.observe()
    game.tick_s (realtime only)    game.ticks    game.score    game.done
```


## Archive
# Leaderboard

| # | candidate | mean score | episodes | stale ticks | S1 calls | S2 calls | Jev→act | Astra→act | status |
|---|---|---|---|---|---|---|---|---|---|
| 0000 | 0000-tier2_jev_only | 0.000 | 3 | 80% | 6 | 0 | 100% | 0% | ok |
| 0001 | 0001-tier2_astra_only | 0.000 | 3 | 100% | 0 | 3 | – | – | ok |
| 0002 | 0002 | – | – | – | – | – | – | – | not evaluated |

`Jev→act` / `Astra→act`: fraction of posted actions preceded by a fresh System-1 / System-2 answer since the previous action (reported, not used for selection).

Best so far: `0000-tier2_jev_only` (mean score 0.000)

Prior candidates (read their code, proposals, eval.json and the JSONL traces — traces contain every
model call with request/response/latency and every game tick/step):
- candidates/0000-tier2_jev_only/ (harness.py, proposal.md, eval.json, trace_seed*.jsonl)
- candidates/0001-tier2_astra_only/ (harness.py, proposal.md, eval.json, trace_seed*.jsonl)

## What to do
1. Read the best candidate(s) and the traces of their worst episodes. Diagnose the failure mode
   concretely (quote evidence from traces).
2. Write `candidates/0002/harness.py` — a complete module with `build(ctx)`.
   It must import only the standard library and `metaharness.*`.
3. Write `candidates/0002/proposal.md`: hypothesis, what changed vs. the parent
   candidate, what evidence would confirm/refute it.
4. Append 3-6 bullet lessons to `notes.md` at the archive root (a running lab notebook: what
   generalised, what was a dead end, which contract details bit you). Read it first.
5. If you cannot use tools, reply with the full harness in ONE ```python fenced block and the
   proposal text after it.
Before the real evaluation the candidate is *pre-flighted*: one short episode with stand-in models
that return arbitrary text / first options on a virtual clock. Any exception there (a contract
misuse, an unhandled parse failure of model output) is sent back to you once for repair; a harness
must survive garbage model output without crashing.
Do not modify any other candidate. Do not run long experiments yourself; the evaluator will.
Read only this archive and the interface documentation above. Do not inspect game implementation
source, other experiment archives, credentials, or held-out games/results. Infer semantics from the
public game description and observations supplied through ctx. Do not call model backends directly:
all inference must go through ctx.think/ctx.decide so the evaluator can account for it.
ModelUnavailable must propagate. Join all ctx.spawn futures before returning; no background work
may escape the episode. The default evaluation resets both memory and module globals per episode;
only an explicit shared-memory protocol allows cross-episode caching. The manifest is authoritative.