Design a harness for a **turn-based, hidden-rule** game (tier 1, ARC-AGI-3 style).

Game: `ARC-AGI-3` — ARC-AGI-3 is a turn-based visual rule-discovery game. Infer objects, controls, constraints and each level's objective from pixels and action effects; no hidden rules, object labels or goal masks are supplied. Complete all levels to WIN. Thinking does not advance the engine. Use one action string per turn: noop, reset, up, down, left, right, action5, undo, or click X Y (integer pixel coordinates 0..63, origin top-left). Directions correspond to ACTION1..4, action5 to ACTION5, click to ACTION6, undo to ACTION7; effects are game-dependent. Only numbered actions in available_actions are legal. action_spec lists currently available forms; click 32 32 is an example of the full click X Y grammar, not the only target. Dictionary equivalents are {id: N} or {id: 6, x: X, y: Y}. reset is RESET/0: generally retries the current level after an action, but can restart the entire game at zero actions or after a level transition; inspect full_reset and levels_completed. GAME_OVER permits explicit reset recovery. WIN ends the episode. noop returns the same engine state. Every submitted request including noop, retry, undo and invalid input consumes the controller budget; initialization does not. Invalid actions raise ValueError before engine mutation. Observations: frame is the final 64x64 palette grid, indexed frame[y][x], values 0..15 (0 white, 5 black); screenshot is its enlarged PNG path. animation_frame_count counts temporal frames; animation_frames_path, when present, is a JSON file containing ALL transition grids in order, including the final grid. state, levels_completed, win_levels, available_actions, full_reset and last_action are public SDK data; last_action echoes the last SDK action (noop does not change it). legal_actions mirrors action_spec. observation_id counts submitted requests. episode_actions, numbered_actions, retries, undos, noops and invalid_actions are separate counters. score is the highest levels_completed this episode, bounded by win_levels; reward is its increase on the last request, never duplicate retry credit. completion_fraction=score/win_levels; won, done, truncated, truncation_reason and actions_remaining distinguish success from budgets/interface failure. Observe is read-only; repeated reads do not pay reward again. Elapsed wall time is in info(). A new evaluator reset creates a fresh game with no adaptor game-specific memory; retain learned mechanics between levels within that episode. The evaluator must clear its own game-specific memory between episodes and hold out entire game families. Budgets: 10000 total requests, 20 invalid requests, wall time None seconds (None means unlimited).

Actions:
- noop: Observe without advancing the engine; consumes one controller action.
- reset: Explicit in-game retry (RESET/0); may restart the entire game.
(no decision in time => 'noop')
Read the current action_spec/available_actions each turn. click 32 32 represents all click X Y commands; bare click is invalid.

Model available through `ctx`: `ctx.think(prompt, session_id=...)` — Astra (System-2). Budget: 40
calls per episode; every call may return *several* actions to save budget. `ctx.act(action)` steps the
game and returns the next observation. By default each episode has fresh `ctx.memory` and module
globals. An explicitly configured shared-memory experiment may carry procedural knowledge between
seeds; action semantics are re-drawn, so never carry over seed-specific answers.

Evaluation metric: mean levels completed over seeds (primary), then System-2 calls (fewer is better).

What the harness must design: what to store (action→effect hypotheses, visited states, level maps),
what to retrieve and show Astra each turn (full history? a diff? an ASCII map with a legend?), how to
parse Astra's reply into actions robustly, how many actions to commit per call, when to run cheap
scripted exploration instead of asking Astra, and how to detect that a hypothesis was refuted.

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
# ARC-AGI-3 study brief

Studied 2026-09-21. Target: the eight locally cached environments, using the evaluator's Python 3.14.5 and official `arc-agi` package in OFFLINE mode. This is an interactive puzzle benchmark, not one game with one fixed ruleset. No gameplay policy or adaptor was implemented.

## Objective and rules

Discover each environment's controls, causal rules, and success condition from its visual responses; complete its sequence of increasingly demanding levels. Natural-language instructions and explicit goal descriptions are generally absent. Objects, movement, transformations, pattern relationships, and resource constraints must be inferred. Earlier levels teach mechanics that later levels combine. A familiar-looking board is not sufficient evidence that familiar game rules apply. [Official introduction](https://arcprize.org/blog/arc-agi-3-launch)

The observation is a sequence of palette-index grids plus metadata. Grids have at most 64 × 64 cells with values 0–15. Coordinates start at the top left: x increases rightward, y downward. A response can contain several animation frames; the last is the decision frame. The four state labels are `NOT_PLAYED`, `NOT_FINISHED`, `WIN`, and `GAME_OVER`. [Game schema](https://docs.arcprize.org/game-schema)

All eight cached games successfully initialized during a reset-only smoke check. Their observed total level counts sum to 60:

| Game/version | Levels | Initially available numbered actions |
|---|---:|---|
| cd82-fb555c5d | 6 | 1, 2, 3, 4, 5, 6 |
| lf52-271a04aa | 10 | 1, 2, 3, 4, 6, 7 |
| ls20-9607627b | 7 | 1, 2, 3, 4 |
| sb26-7fbdac44 | 8 | 5, 6, 7 |
| sp80-589a99af | 6 | 1, 2, 3, 4, 5, 6 |
| tn36-ef4dde99 | 7 | 6 |
| tu93-0768757b | 9 | 1, 2, 3, 4 |
| vc33-5430563c | 7 | 6 |

These are runtime observations, not promises about future versions or later action availability. All initial frames were 64 × 64 `int8`; LF52 returned two frames. See [runtime-check.json](runtime-check.json).

## Controls

`RESET` is ID 0. Actions 1–4 conventionally correspond to up/down/left/right, but their actual effects depend on the environment. Action 5 is a game-specific operation such as interact, rotate, or execute. Action 6 requires integer x,y coordinates in 0–63. Action 7 is undo, when offered. Read `available_actions` after every response; it does not reveal valid click locations. Human UI equivalents are W/S/A/D or arrow keys, Space/F for action 5, mouse click for action 6, and Z for action 7. [Actions](https://docs.arcprize.org/actions)

Do not confuse a player-issued `RESET` with evaluator `reset(seed)`: the former may retry the current level; the latter must construct a fresh game instance. After `GAME_OVER`, recovery requires RESET. After `WIN`, finish the episode. Details and version-specific caveats are in [interfaces.md](interfaces.md).

## Pace and tier

**Tier 1. No finite in-game decision deadline.** The engine runs synchronously only when an action is submitted. It resolves that action's internal animation before returning. Waiting seconds or minutes for Astra does not consume moves or advance hazards. Apparent clocks or energy bars can be action-dependent; metadata `default_fps` governs presentation, not autonomous simulation. This follows both the documented turn-based contract and inspection of installed `ARCBaseGame.perform_action`. [Game schema](https://docs.arcprize.org/game-schema), [engine source](https://github.com/arcprize/ARCEngine/blob/main/arcengine/base_game.py)

`study.json` uses **-1.0 as the explicit sentinel for an unbounded game decision latency**, because standard JSON cannot encode infinity. Interpret this as no game-imposed timeout, not negative seconds. An evaluator may separately cap wall time, actions, or model spend; no such cap was supplied here. The optional REST server has scorecard cleanup settings; the chosen in-process path does not start that cleanup thread.

## Score and progress

Use the greatest observed `levels_completed` during one evaluator episode as its primary score, bounded by `win_levels`. Each new level adds one; replaying an already completed level adds nothing. Record actions, retries, elapsed time, and completion fraction separately. Reset all episode bookkeeping on evaluator `reset(seed)`. This implements the user's requested natural score without conflating SDK scorecards with level counts.

Official ARC-AGI-3 scoring is different: RHAE compares actions against human baselines, squares the ratio, caps individual level credit at 1.15, weights later levels more heavily, and limits game credit by completion. It is useful optional telemetry, not the primary scalar for this harness. [Scoring methodology](https://docs.arcprize.org/methodology)


### study/interfaces.md
# ARC-AGI-3 programmatic interfaces and adaptor specification

Studied 2026-09-21. **Choose the official in-process Python SDK in OFFLINE mode.** The installed API and eight cached versions were inspected and initialized with the exact evaluator interpreter. No policy, solver, or game-specific source analysis was performed. Runtime checks only constructed environments and obtained their initial RESET observations.

## Practical interface inventory

Latency estimates below describe transport/processing, not model latency. Only local construction times were measured; no steady-state gameplay benchmark was run.

| Interface | Maturity and observation/control | Latency | Cost and risks | Decision |
|---|---|---|---|---|
| Official `arc-agi` in-process OFFLINE SDK | Supported Python API; raw NumPy frames, progress/state/action metadata; enum actions and coordinate clicks; local scorecards/recordings | No network round trip. Docs advertise about 2,000 FPS headless; actual action time depends on animation count. Measured first constructions after import: 6–124 ms | Local CPU and model costs; no ARC account or API credential needed. MIT toolkit; retain license notices if redistributing | **Chosen:** already installed, cached, fastest and simplest observation path |
| Official SDK NORMAL mode | Same wrapper; discovers/downloads games through ARC service, then executes locally | Startup needs network; subsequent local execution fast | Needs server connectivity even if some games are cached; may fetch anonymous key | Incompatible with stated no-server-network constraint |
| Official SDK ONLINE mode / direct ARC REST API | Remote sessions, JSON frames, actions, scorecards and online replay support; `game_id`, `guid`, `card_id` identify state | Network RTT plus server processing; documented cap 600 requests/minute | Server connectivity and registered/anonymous API access; model charges separate. Service rules/rate limits apply; current commercial pricing was not established | Unavailable here; no reason to request credentials |
| Official local REST server, `Arcade.listen_and_serve` | Flask server exposes the SDK through documented REST routes; useful for non-Python clients or process isolation | Adds local HTTP and frame serialization overhead; unmeasured | Same local compute plus process management. Bind loopback. Optional scorecard idle cleanup can close sessions | Practical alternative if the next harness needs HTTP; unnecessary for Python |
| Official ARC-AGI-3-Agents / benchmarking repositories | Agent runners, logging, examples, model integration; use underlying SDK | Local/remote backend latency plus framework/model overhead | Extra dependencies; examples may assume keys, remote services, older APIs or competition configuration | Read as examples; do not install the full stack for this adaptor |
| ARCEngine directly / custom game subclass or wrapper | Open engine has sprites, level objects, camera, action processing; supports custom environments and rendering | In-process | Bypasses SDK bookkeeping. Internal state/source access leaks information unavailable through ordinary play and undermines held-out evaluation | Technically possible; use SDK public observations instead |
| Browser gameplay, screen capture + keyboard/mouse injection | Human-facing canvas and controls; screenshot observations; key/mouse input | UI animation, capture, input focus and scaling overhead; unmeasured | Hosted page needs network; fragile coordinate mapping and duplicate key presses. Automation of play is the benchmark's purpose, but external service rules still apply | Inferior to direct pixels/API; unavailable as an offline hosted-page solution |
| Community Pygame human console | Local desktop client wraps SDK and adds controls/local convenience | GUI event/display overhead; unmeasured | Additional `pygame-ce`/environment manager dependencies; compatibility with this Python untested | Optional debugging UI, not an agent backend |
| Replay JSONL / replay viewer | Historical action/frame/state sequence; useful for diagnostics | File reads | Storage can grow substantially. Replays do not control a live game | Supplement only |

Sources: [local vs online](https://docs.arcprize.org/local-vs-online), [SDK repository](https://github.com/arcprize/ARC-AGI), [local REST server](https://docs.arcprize.org/toolkit/listen_and_serve), [REST workflow](https://docs.arcprize.org/full-play-test), [official agents](https://github.com/arcprize/ARC-AGI-3-Agents), [benchmark runner](https://github.com/arcprize/arc-agi-3-benchmarking), [ARCEngine](https://github.com/arcprize/ARCEngine), [Pygame console](https://github.com/aashen1/arc-agi-3-local-play), [recordings](https://docs.arcprize.org/recordings), [toolkit MIT license](https://github.com/arcprize/ARC-AGI/blob/main/LICENSE).

No emulator/ROM interface is needed: these environments are Python engine programs. No separately verified Gymnasium environment, native MCP game endpoint, or game-specific bot mod is required. The SDK wrapper is not itself a Gymnasium `Env`.

## Exact software and configuration

Interpreter: `<repo>/.venv/bin/python` — **Python 3.14.5**. Do not use the system Python to judge compatibility.

Installed and verified: `arc-agi==0.9.9`, `arcengine==0.9.3`, `numpy==2.4.2`, `pydantic==2.13.5`. Both ARC distributions declare `Requires-Python: >=3.12`; 3.12 is a minimum, not an exact version requirement. All cached games initialized on 3.14.5. No install or upgrade is currently needed. If reconstructing the environment, retain these versions and install the toolkit's declared dependencies: Flask, Matplotlib, Pydantic, python-dotenv, Pillow, requests, and arcengine; arcengine requires NumPy. Full dependency constraints are in [runtime-check.json](runtime-check.json). Torch, a GPU, SciPy, pygame, and browser automation are not prerequisites.

| Variable | Requirement and meaning |
|---|---|
| `OPERATION_MODE` | Required; already set to `offline`. Also explicitly pass `OperationMode.OFFLINE` and verify the resulting mode. A `competition` environment value overrides even a constructor choice in this installed version |
| `ENVIRONMENTS_DIR` | Required; already set to the existing cache directory. Read from the process; do not replace it with an empty relative directory |
| `RECORDINGS_DIR` | Optional; a writable directory under the run workspace if SDK recordings are enabled |
| `MPLCONFIGDIR` | Recommended in this sandbox; create a writable directory such as `<study>/.cache/matplotlib` before importing `arc_agi` |
| `XDG_CACHE_HOME` | Recommended in this sandbox; use `<study>/.cache`, allowing Fontconfig to cache fonts |
| `MPLBACKEND` | Recommended `Agg` for headless use; set before imports |
| `ARC_API_KEY` | **Not required.** Leave absent. Never log any key or serialize a configuration object containing one |
| `ARC_BASE_URL` | Not used by the chosen offline execution path; no server address needed |
| `ONLY_RESET_LEVELS` | Engine behavior override, not a required user setting. Leave unset for default semantics; document it if inherited. Fresh `make` remains the evaluator reset mechanism |

The first import attempted to create Matplotlib/Fontconfig caches outside the writable workspace and printed many warnings. It still completed. A second run with writable cache variables and `MPLBACKEND=Agg` completed without those permission warnings; its first font-cache build still took startup time. `arc_agi` imports rendering support even with `render_mode=None`. Thus headless mode alone does not avoid first-import cache setup.

No account, paid game license, ARC credential, Kaggle token, model-provider credential, or server endpoint must be provided for the **environment adaptor**. Astra/Jev access belongs to the surrounding model harness.

## Construction and reset contract

API signatures verified in the installed package:

- `Arcade(operation_mode=OperationMode.OFFLINE, environments_dir=..., recordings_di
…(truncated)

### study/strategies.md
# ARC-AGI-3 strategy prior for Astra, code, and optional Jev

Studied 2026-09-21. This document distinguishes published evidence from proposed harness design. The recommendations have not been evaluated as a playing agent in this study. There is no universal build order, fixed opening move, opponent, or metagame counter: the central task is learning unfamiliar rules efficiently.

## What the evidence supports

| Evidence | Finding useful to this harness | Scope / limitation |
|---|---|---|
| [Official preview retrospective](https://arcprize.org/blog/arc-agi-3-preview-30-day-learnings) | Strong humans move from exploration to efficient execution. Preview games sometimes rewarded brute force, prompting harder future designs | Historical 2025 preview; scores and level counts are not current cache specifications |
| [Graph-based exploration paper](https://arxiv.org/html/2512.24156v1) | Connected components reduce click candidates; a directed graph records tested transitions and routes to unexplored frontiers. The paper reports a median 30 of 52 preview levels solved in its extended runs | Large state spaces and partial observability are limitations. A reset-bookkeeping bug caused repeated fatal exploration |
| [StochasticGoose author's repository](https://github.com/DriesSmit/ARC3-solution) | The preview winner learned which actions change frames. It predicts action type and click position separately, using a state/action experience buffer | Frame change is an exploration signal, not proof of progress. Historical implementation requires extra training dependencies |
| [FrugalExplorer author's repository](https://github.com/shloksah/arc-agi-3-agent) | A lighter effect model uses action/color/region statistics, component-based clicks, and return-to-frontier navigation. It describes frozen UI masks and death attribution | Author-reported prototype, not proof of optimality. Its scoring discussion includes a cutoff that must not be imported into this local level-count evaluator |
| [Official analysis of 160 model replays](https://arcprize.org/blog/arc-agi-3-gpt-5-5-opus-4-7-analysis) | Repeated failures: recognizing a local effect without composing rules, importing an incorrect familiar-game analogy, and winning an easy level without learning a transferable mechanic | Qualitative analysis of specific model runs, not a controlled test of this proposed harness |
| [Workspace Optimization / DreamTeam paper](https://arxiv.org/html/2605.09650v1) | An executable learned model is updated against observed transitions; revisions are checked against earlier evidence. Short action batches can stop at transitions and unexpected events | Authors report public-set gains; architecture and compute budget differ from Astra/Jev. Borrow the evidence discipline, not the entire multi-agent system |
| [Official human study](https://arcprize.org/blog/arc-agi-3-human-dataset) | Public human replays allow study of exploration and recovery; first-run testing measures learning, not memorized solutions | Do not train on a held-out game's replays and then describe its evaluation as unseen |

## Recommended control loop

The following is a proposed adaptation to the two-model system, rather than a claim that these exact settings are established optimal play.

**Code owns exact measurements and execution. Astra owns rule discovery, goal inference, planning, and revision. Jev is optional semantic recognition.** Keep tier 1: no model needs to act while Astra is thinking. A cheaper second model is useful only when it can answer a bounded question more economically than Astra; it is not necessary for real-time safety.

### 1. Inspect before spending a move

**Code:** Parse the current frame into palette counts, connected components, bounding boxes, shape masks, candidate grid spacing, and spatial relations. Preserve raw pixels and intermediate animation frames. Calculate frame differences exactly. Split the display into tentative board, legend/target, and status regions, but keep these labels reversible.

**Astra prompt:** Identify a few possible controllable objects and target relationships. State each as a hypothesis, with observable evidence and an uncertainty. Ask what one interaction would distinguish the leading alternatives. Do not describe an object as a key, avatar, goal, or wall merely because of its color.

**Jev question:** Given already measured objects, choose a tentative visual role from `target_example`, `control`, `movable_piece`, `status_indicator`, `unknown`. Treat this as a suggestion; code cannot infer mechanics from that label.

Why: the official benchmark requires learning without instructions, while its replay analysis shows that misleading analogies can dominate action selection. [Benchmark introduction](https://arcprize.org/blog/arc-agi-3-launch), [model replay analysis](https://arcprize.org/blog/arc-agi-3-gpt-5-5-opus-4-7-analysis)

### 2. Use small, discriminating opening probes

**Astra:** Start with one action whose plausible outcomes separate hypotheses. For a directional interface, a single move can identify the affected object, movement step, or rotation. For clicks, select a visually distinct candidate on its actual occupied pixels. In a mixed interface, determine whether a click selects an object for later actions. Probe action 5 near relevant objects only when it is available. There is no justified universal ACTION1 spam opening.

**Code:** Compare before/after positions, shapes, palette changes, selection markers, and status changes. Record no-ops as well as successes. A reverse action is a probe, not an assumed inverse: up/down could rotate a piece or shift a whole board. If a selection or mode changes, previously ineffective actions deserve reconsideration in that new state.

**Jev:** Optionally classify a supplied difference as `translation`, `rotation`, `recolor`, `selection_change`, `multiple_effects`, or `unclear`. Exact displacement and shape transforms should already be computed in code.

This adapts the action-effect learning idea from [StochasticGoose](https://github.com/DriesSmit/ARC3-solution). The proposed opening must remain sensitive to context; it is not a fixed script.

### 3. Compress click exploration into meaningful candidates

**Code:** Generate representatives from visible components, buttons, corners, outlines, and grid cells. Choose an occupied pixel near a component's center, not an arithmetic centroid that might land in a hole. Keep several candidates for a large or multi-part object. Deduplicate coordinates, not every object of one color. Start with high-salience objects; expand to alternate points and a coarse spatial grid if no effect is found. Keep global exploration coverage so a mistaken segmentation does not make part of the screen unreachable.

**Astra:** Decide whether the next uncertainty concerns object identity, click location, an order dependency, or a missing selection mode. Do not keep clicking adjacent pixels when a different action type is the more informative test.

**Jev:** Rank a supplied finite list of candidate object IDs for “looks like a control.” Code maps the chosen ID to coordinates. Jev does not invent x/y values.

The published graph explorer prioritizes connected visual regions over exhaustive pixel search. FrugalExplorer independently describes component-snapped candidates. These support a candidate hierarchy, not a guarantee that each component has one uniform response. [Graph exploration](https://arxiv.org/html/2512.24156v1), [FrugalExplorer](https://github.com/shloksah/arc-agi-3-agent)

### 4. Maintain explicit, falsifiable rules

**Astra:** Maintain a compact rule table: condition, action, predicted effect, supporting observations, counterexamples, and confidence. Distinguish observation (“this press changed orientation”) from inferred mechanism (“orientation determines which face is painted”). Prefer two or three live alternatives to either absolute certainty after one observation or endless en
…(truncated)

### adaptor/ADAPTOR.md
# ARC-AGI-3 adaptor

This Tier 1 adaptor executes the **real cached games** through the official
`arc-agi==0.9.9` OFFLINE SDK and `arcengine==0.9.3`. It reads public wrapper
responses and submits public enum actions. It does not inspect game source,
hidden level data, private engine state, or solutions. No server, GUI, account,
credential, or download is needed with the supplied cache.

## Setup and use

Export `OPERATION_MODE=offline` and the existing `ENVIRONMENTS_DIR`; see
`env.example`. In this run the supplied cache is
`$ENVIRONMENTS_DIR`. Do not replace it with an
empty directory. Run `./setup.sh` to install requirements in the evaluator's
Python 3.14.5 environment and initialize all eight games. It bootstraps pip if
necessary. Missing exact versions fail explicitly; no online fallback occurs.
A nonempty `ONLY_RESET_LEVELS` is rejected to retain the studied retry semantics.
The adaptor configures writable plotting caches before importing the SDK.

```python
from game import make_game

game = make_game(max_actions=10000, max_seconds=None,
                 max_invalid_actions=20, screenshots=True)
try:
    observation = game.reset(0)
    observation = game.step("click 32 32")
    print(game.score, game.info())
finally:
    game.close()
```

Constructor options are explicit keyword arguments. The default 10,000-request
cap is an evaluator safety budget, not a game rule. `max_seconds=None` imposes
no wall-clock deadline. The world advances synchronously only on SDK actions;
thinking and `noop` do not move it. Time budgets are checked between calls;
they cannot interrupt a blocked engine call. Instances are sequential, not
thread-safe. `close()` releases the wrapper and closes its local scorecard.

## Scenarios and isolation

`reset(seed)` accepts a nonnegative integer (not bool), chooses `seed % 8`,
and constructs a **fresh** wrapper with engine seed **0**. The constructor's
initial RESET is the only initialization action; counters start at zero.

| Residue | Exact game ID | Levels | Suggested split |
|---:|---|---:|---|
| 0 | cd82-fb555c5d | 6 | development |
| 1 | lf52-271a04aa | 10 | development |
| 2 | ls20-9607627b | 7 | development |
| 3 | sb26-7fbdac44 | 8 | development |
| 4 | sp80-589a99af | 6 | held out |
| 5 | tn36-ef4dde99 | 7 | held out |
| 6 | tu93-0768757b | 9 | held out |
| 7 | vc33-5430563c | 7 | held out |

There are 60 levels total. Seed 8 repeats CD82, not a new scenario. All seeds
sharing a game prefix must remain in the same split. The adaptor retains no
maps, hypotheses, policies or learned click statistics. The surrounding
harness **must clear its own game-specific memory between evaluator episodes**;
it should retain learned mechanics across levels inside an episode. SDK class
caching is process-wide; use a fresh worker process per episode for stronger
RNG/global-state isolation. `DETERMINISTIC=False`: repeatable initial frames
are not a claim of full-trajectory determinism.

## Actions and terminal handling

`action_spec` is refreshed after every SDK response. It lists `noop`, `reset`,
and the available numbered actions: `up` (1), `down` (2), `left` (3), `right`
(4), `action5` (5), `click 32 32` (6), and `undo` (7). Effects are game-dependent.
`click 32 32` is a concrete legal sample for the generic validator; the full
parameter grammar is **`click X Y`**, integer pixel coordinates 0..63, top-left
origin. Bare `click` is rejected. Dictionary forms are `{"id":N}` and
`{"id":6,"x":X,"y":Y}`. Booleans, noninteger dict coordinates, out-of-range
coordinates, unavailable actions, and extraneous dictionary fields are rejected
before SDK mutation.

`reset` invokes the game's RESET/0, usually a level retry after an action.
At zero actions (including immediately after a level transition), it can fully
restart the game. It never resets adaptor score or budget. `GAME_OVER` is
recoverable via explicit `reset`, so it does not alone set `done`. Numbered
actions are rejected there. `WIN` sets `done`; later step calls return the
retained state without contacting the SDK. No automatic retry occurs. A `None`,
empty or malformed SDK response raises `InterfaceError`, retains the last good
observation/score, and truncates with `interface_failure` because mutation may
already have occurred. The caller must start a new evaluator episode.

## Observation and score

Observations are JSON dictionaries, normally about 15 KB, always with only the
final **64x64 palette-index grid** inline. `frame[y][x]` is 0..15; the SDK palette
uses 0 white and 5 black. `screenshot` is an absolute 512x512 nearest-neighbor
PNG path under `observations/`, or null with `screenshots=False`. When a response
has multiple temporal frames, `animation_frames_path` points to a JSON list of
**all** original grids, in order; otherwise it is null.
`animation_frame_count` includes the final frame. These content-addressed files
are immutable and can be shared across repeated identical observations. They
are not hidden engine state. Retain them while logs refer to them; delete the
`observations/` directory between evaluation runs to reclaim storage. Long
runs can accumulate substantial animation data. The inline observation remains
below the contract's 200 kB limit regardless of animation length.

Public metadata: `state`, current `levels_completed`, `win_levels`, integer
`available_actions`, `full_reset`, and `last_action` (last SDK action ID/data).
`noop` retains that SDK echo. `legal_actions` mirrors the current action spec.
`observation_id` increments for each controller request during an active episode.
`episode_actions` counts all such requests, including noop, reset, undo, and
invalid requests; initialization and calls after `done` are excluded.
`numbered_actions`, `retries`, `undos`, `noops`, and `invalid_actions` distinguish
request types. Numbered actions include undo. Invalid requests raise ValueError,
consume the budget, and truncate after `max_invalid_actions` total failures.

**Primary score = max observed levels_completed since evaluator reset**, bounded
by the pinned `win_levels`. `reward` is the increase of this high-water mark on
the last request. Repeated observation reads expose the same reward; they are
not extra reward events. Retries/full restarts never award duplicate progress.
`completion_fraction=score/win_levels`. `won` records actual WIN; `done` is WIN
or truncation. `truncated`, `truncation_reason`, and `actions_remaining` separate
budget exhaustion from game success. Wall time, seed, game ID, split, engine seed,
session GUID, package versions, SDK call count, and budgets appear in `info()`.
No SDK RHAE/scorecard aggregate is used as the requested score. Scorecards are
local diagnostics only; SDK recordings are disabled. No credentials/configuration
objects are exposed in observations or logs.

## Verification

Run the required generic validator:

```sh
<repo>/.venv/bin/python -c "from metaharness.games.adaptor import validate_adaptor; print(validate_adaptor('<repo>/runs/r6-arc3/adaptor').to_markdown())"
```

`test_adaptor.py` performs live cache initialization, seeded random development
play, click/direction/undo/retry checks, serialization and artifact checks, and
budget/validation checks. Focused injected **public response fixtures** additionally
exercise progress regression/duplicate credit and terminal/error branches without
claiming that a random policy solved a game. No held-out gameplay policy is trained.

Verified in the evaluator environment on 2026-09-21:

- Required `validate_adaptor`: **PASS** (both seeds, all listed actions, JSON,
  score/done types and scenario variety).
- All eight pinned real games initialized with expected level totals; initial
  observations were 13.1–16.9 kB. Seeds 0/8 reproduced the same initial observation.
- 280 seeded random requests across development games (40/40/160/40) changed
  249 final frames; animation sequences re
