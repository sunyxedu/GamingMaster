# ARC-AGI-3 adaptor

This Tier 1 adaptor executes the **real cached games** through the official
`arc-agi==0.9.9` OFFLINE SDK and `arcengine==0.9.3`. It reads public wrapper
responses and submits public enum actions. It does not inspect game source,
hidden level data, private engine state, or solutions. No server, GUI, account,
credential, or download is needed with the supplied cache.

## Setup and use

Export `OPERATION_MODE=offline` and the existing `ENVIRONMENTS_DIR`; see
`env.example`. In this run the supplied cache is
`/Users/hemlix/Documents/arc3-wm/environment_files`. Do not replace it with an
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
  249 final frames; animation sequences reached 42 frames. LS20 reached a real
  GAME_OVER and recovered through explicit RESET.
- Live LF52 retry returned `full_reset=false`; a consecutive RESET returned
  `full_reset=true`. Direction, click, undo and retry dispatch all succeeded.
- Random play earned zero levels; no full solution or natural WIN is claimed.
  Public-response fixtures checked high-water scoring through progress
  2→0→1→2→3, terminal WIN protection, and missing/empty response failures.
- Tests blocked `requests.Session.request` throughout, confirming no HTTP was
  needed. Budget/invalid-input checks, PNG palette verification, multi-frame
  serialization, Python compilation, and `pip check` passed.

Machine-readable evidence is in `verification.json`; validator output is in
`validation.md`. Held-out games were initialized only, not used for gameplay
search or rule learning.
