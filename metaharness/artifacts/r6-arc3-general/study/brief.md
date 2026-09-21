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
