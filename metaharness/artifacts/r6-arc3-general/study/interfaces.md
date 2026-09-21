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

Interpreter: `/Users/hemlix/Documents/GamingMaster/.venv/bin/python` — **Python 3.14.5**. Do not use the system Python to judge compatibility.

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

- `Arcade(operation_mode=OperationMode.OFFLINE, environments_dir=..., recordings_dir=..., logger=...)`.
- `arc.get_environments()` returns a list of `EnvironmentInfo` objects.
- `arc.make(game_id, seed=0, scorecard_id=None, save_recording=False, include_frame_data=True, render_mode=None, renderer=None)` returns a wrapper or `None`.
- `env.observation_space` is the latest response, not a Gymnasium space descriptor.
- `env.action_space` is a list of `GameAction` values inferred from that response.
- `env.step(action, data=None, reasoning=None)` and `env.reset()` return an observation or `None`.

[Wrapper reference](https://docs.arcprize.org/toolkit/environment_wrapper). For version-sensitive implementation details, the installed sources take precedence over changing web examples: `arc_agi/base.py`, `local_wrapper.py`, `wrapper.py`, and `arcengine/enums.py`, under `/Users/hemlix/Documents/GamingMaster/.venv/lib/python3.14/site-packages/`.

The adaptor phase should follow this sequence:

1. Configure writable caches before imports and validate the two supplied offline variables. Use a logger that writes to stderr or a file if stdout carries a JSON protocol. Some SDK scorecard messages use module/root logging separately, so configure those too.
2. Construct `Arcade` in OFFLINE mode. It scans recursively for `metadata.json`; no anonymous-key lookup or server discovery occurs in this mode. A cache entry needs valid metadata plus the corresponding local Python game class file. Missing data must produce an explicit setup error, never a fallback to online mode.
3. Resolve all eight IDs against the fixed manifest below. Pin complete versioned IDs; a bare prefix selects the most recently downloaded matching version. Do not let filesystem traversal order select scenarios.
4. On evaluator `reset(seed)`, close the previous episode's scorecard if used, discard its wrapper, clear score/action/retry counters and per-game agent memory, select the mapped game, and call a **fresh `arc.make`**. Pass a distinct scorecard per episode or allow a new default after closing the old one.
5. `make` automatically performs initial RESET. Read `env.observation_space` immediately. Do not call another RESET just to fetch an initial observation. Reject a `None` wrapper, `None` observation, or initial observation with no frame as a setup failure.
6. Run one sequential action at a time on a wrapper. Retain the returned observation and final frame. Do not call private game methods, load hidden level data, or mutate source to make scenarios easier.
7. End on `WIN` or evaluator budget exhaustion. A `GAME_OVER` can be recovered by an explicit in-game RESET while the evaluator episode continues, subject to retry/action budgets. After final scoring, close the optional scorecard and release references. No public wrapper `close()` or generic snapshot/restore API was found.

### Fixed seed mapping and holdouts

Use this immutable order (indices 0–7):

| Index | Full ID | Initial actions | `win_levels` |
|---:|---|---|---:|
| 0 | cd82-fb555c5d | 1,2,3,4,5,6 | 6 |
| 1 | lf52-271a04aa | 1,2,3,4,6,7 | 10 |
| 2 | ls20-9607627b | 1,2,3,4 | 7 |
| 3 | sb26-7fbdac44 | 5,6,7 | 8 |
| 4 | sp80-589a99af | 1,2,3,4,5,6 | 6 |
| 5 | tn36-ef4dde99 | 6 | 7 |
| 6 | tu93-0768757b | 1,2,3,4 | 9 |
| 7 | vc33-5430563c | 6 | 7 |

Require a nonnegative integer evaluator seed. Select index `seed % 8`. Prefer a fixed engine seed of 0: scenario diversity here comes from the game ID. Optionally pass `seed // 8` as the engine seed only if later validation establishes useful within-game variation. The loader only forwards a seed when the game constructor supports it. Two independent seed-0 constructions and a seed-1 construction produced identical final initial-frame bytes for each of the eight games. This proves initial-frame repeatability in that probe, not full-trajectory determinism or absence of later randomness.

Suggested split: development residues 0–3 (`cd82`, `lf52`, `ls20`, `sb26`); held-out residues 4–7 (`sp80`, `tn36`, `tu93`, `vc33`). Keep **every seed sharing a game prefix in the same split**: seeds 0 and 8 are repetitions of CD82, not train/test novelty. Hold out game families, not later levels of a trained game. Preserve general algorithmic priors across episodes, but clear game-specific maps, hypotheses, solutions, and learned click statistics on a new episode. Log the seed, selected full ID, engine seed, and package versions with each result. Missing a pinned version is an error rather than an invitation to reshuffle indices.

## Observation serialization

The installed `FrameDataRaw` has these public fields:

| Field | Meaning and adaptor handling |
|---|---|
| `frame` | List of NumPy 2D arrays in temporal order. Use last frame for current state; preserve earlier frames for transition evidence |
| `state` | `GameState`; serialize its string value |
| `levels_completed` | Integer authoritative progress; do not use old field name `score` |
| `win_levels` | Total levels needed to win; do not use old name `win_score` |
| `available_actions` | List of integer IDs, re-read every turn |
| `action_input` | Echo of enum action, data, and optional reasoning; serialize action ID explicitly |
| `full_reset` | Whether a full restart occurred; not a general indication of every level retry |
| `game_id`, `guid` | Versioned game ID and session identifier; useful for evaluation logs |

**Critical:** `frame` is a Pydantic private-backed property and is omitted from `model_dump()` and `model_dump_json()`. Assemble the payload explicitly, converting each frame with `.tolist()`. A naïve model dump yields metadata without the board. `GameAction.value` in this installed enum is the integer action ID; use explicit names/IDs for robust JSON rather than serializing enum internals. Source: installed [enums.py](/Users/hemlix/Documents/GamingMaster/.venv/lib/python3.14/site-packages/arcengine/enums.py).

Recommended adaptor JSON: `state`, `levels_completed`, `win_levels`, `available_actions`, `full_reset`, `frame` (last 2D grid), `animation_frames` (all grids or selected event grids), `last_action`, `episode_actions`, `retries`, `score`, and a monotonically increasing `observation_id`. Keep the game ID in evaluator logs; the agent can solve from public observations without an ID lookup table. Raw pixels remain accessible even if the agent usually sees a compressed object summary.

Array indexing is `[y][x]`; action payloads are `{x,y}`. Pixel coordinates are not automatically logical tile coordinates. Derive any logical cell centers from visible grid geometry. If generating images for Astra, use the installed `arc_agi.rendering.COLOR_MAP` and nearest-neighbor enlargement. This palette differs from conventional ARC-AGI-1 colors: here 0 is white and 5 black. Preserve the original 0–15 values in JSON. Supply coordinates/labels as an optional separate overlay; do not replace the underlying grid.

The observation contains no authoritative object names, collision map, goal description, clickable-area mask, or full hidden simulator state. Metadata tags are not an action mask: LF52 is tagged `click` yet also exposes directions and undo; TU93 is tagged `keyboard_click` yet initially exposes only four directions. Use runtime actions.

## Action contract and failure handling

Suggested agent-facing forms are `{"id":1}` for a simple action, `{"id":6,"x":32,"y":20}` for a click, and `{"id":0}` for an explicit retry. Convert IDs using `GameAction.from_id`; invoke `env.step` with the enum. Pass click coordinates in `data={"x":...,"y":...}`. Do not pass a raw dictionary as the action argument. Directional actions need no data.

Validate integers, supported IDs, and 0–63 click coordinates before calling the SDK. RESET is a separate control and need not appear in the numbered action mask. Reject missing click coordinates instead of silently defaulting to (0,0). Local execution does not necessarily enforce all REST validators: the installed `ActionInput.data` is an arbitrary dictionary. Optional reasoning must be JSON-serializable and at most 16 KiB; omit it unless required for concise decision logs.

Use the final frame of every successful response; the list is animation time, not RGB channels or board layers. An action may advance a level and finish with the next level already displayed. Check metadata before interpreting frame differences. If a response is `None`, the wrapper caught an exception: report an interface failure with context. Do not substitute a blank frame or count it as a lost level. Preserve the last good observation for diagnosis. Avoid automatic retries of a failed mutating call, because the game may have partly advanced before an exception.

Handle `WIN` and `GAME_OVER` before sending numbered actions. The installed engine has a terminal fast-return branch with an empty frame and legacy progress field arguments; a post-terminal extra action can yield misleading/default progress. Stop on WIN, and save progress before recovery from GAME_OVER. An empty response must not erase the last valid frame or high-water score.

### Player RESET is not a fresh evaluator episode

In installed `ARCBaseGame.handle_reset`, after at least one action in the current level, RESET ordinarily retries that level. At action count zero, or after WIN, RESET ordinarily performs a full reset. Setting a level zeros its action count. Therefore a duplicate RESET, or RESET immediately after a level transition, can erase earlier progress. `ONLY_RESET_LEVELS=true` changes part of this behavior. `env.reset()` calls the same RESET action; it accepts no seed. These details come from installed [base_game.py](/Users/hemlix/Documents/GamingMaster/.venv/lib/python3.14/site-packages/arcengine/base_game.py), not an assumption based on Gym APIs. Game subclasses may customize reset behavior, so always inspect the returned metadata.

## Score, budget, and logging contract

Initialize episode score to zero. After each valid response, set it to the maximum of its previous value and `levels_completed`; retain `win_levels` as the denominator. Incremental reward, if needed, is the increase in that maximum. Never sum positive progress increments across repeated full restarts: replaying levels 1–2 must not earn four levels. Distinguish raw current progress from high-water episode score in logs.

Maintain a controller-issued action counter independent of SDK scorecard internals. Count explicit retries/undo and all submitted gameplay actions toward the harness budget, including no-ops; exclude the initial constructor RESET. Track invalid requests separately and cap repeated protocol failures. The SDK tracks resets separately from numbered actions, so label your counter clearly. Do not declare a universal built-in action limit: individual levels may have resource limits, and the evaluator must choose an external episode cap. On cap exhaustion return a truncation reason plus earned levels; do not report a false WIN.

SDK scorecards are optional diagnostics: `arc.get_scorecard()` / `arc.close_scorecard()` expose normalized performance, not the requested level-count score. Do not scrape their aggregate score as reward. Save an explicit allowlist of metadata and frames; avoid dumping entire SDK objects that could contain credentials. `save_recording=True` with `include_frame_data=True` writes JSONL under the selected recordings directory; cap storage for long searches.

For repeated evaluator use, a fresh wrapper is the minimum reset isolation. Prefer a fresh worker process per episode if strict RNG/global-state isolation matters: the installed loader caches game classes process-wide, although it creates new instances. Do not concurrently mutate one wrapper or rely on deep copies of its private engine as a supported search API. Search should operate on an agent-built model, or on counted legal interactions.

## Evidence and remaining verification

[runtime-check.json](runtime-check.json) records package constraints, all loaded versions, initial frames' dimensions/dtypes, action masks, level totals, construction times, and seed comparisons. No gameplay action, win sequence, death recovery, later level, or full trajectory was tested during this study. Before adaptor completion, perform targeted API checks on development games: one valid simple action, one valid click, LF52 multi-frame serialization, illegal-coordinate rejection in the adaptor, retry semantics, terminal handling, and accounting across restart. No additional package is currently blocking implementation.
