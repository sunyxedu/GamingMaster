Design a harness for a **turn-based** game (tier 1: the world waits for the agent).

Game: `2048` — 2048 is a deterministic seeded 4x4 turn-based puzzle. The board starts with two tiles (2 or 4). On each action, all tiles slide in that direction; equal adjacent tiles merge once, and the resulting value is added to the merge score. A changed move then places one random tile in an empty cell (90% 2, 10% 4). No-op and malformed actions leave state unchanged. The objective is to maximize merge score and the largest tile; reaching 2048 does not end play. Observations contain board (four rows of four integers), score, moves, max_tile, game_over, seed, changed, spawn, and legal_actions. The scalar evaluator score is merge score plus max_tile, with both components available in score_breakdown().

Actions:
- up: slide all tiles toward the top edge, merging equal neighbors once
- down: slide all tiles toward the bottom edge, merging equal neighbors once
- left: slide all tiles toward the left edge, merging equal neighbors once
- right: slide all tiles toward the right edge, merging equal neighbors once
Each changed move spawns one tile in an empty cell: 2 with probability 0.90 or 4 with probability 0.10. A no-op is accepted, consumes no move, and spawns nothing. The game continues after creating 2048.

Model available through `ctx`: `ctx.think(prompt, session_id=...)` — Astra (System-2). Budget: 40 calls
per episode; one call may return a plan of several actions. `ctx.act(action)` applies one action and
returns the next observation. Each episode has a wall-clock deadline of 1800 s, including model
latency (`ctx.budget.remaining()["wall_s"]`); when it passes, or the call budget is used up, the episode
ends with the score reached. Each episode starts cold: fresh `ctx.memory` and fresh module globals.

Evaluation: mean score over the training seeds (the game's own score, see the description), then fewer
System-2 calls, then fewer actions. A candidate can only be selected if none of its training episodes
ends with an error (see the contract: e.g. a TimeoutError from a model call that the harness does not
catch).

What the harness must design: how observations are shown to Astra (full history, a diff, an ASCII map
with a legend, only what changed), what to remember between calls (rules learned, states visited,
plans), how to parse Astra's reply into actions robustly, how many actions to commit per call, when
cheap scripted search or bookkeeping replaces a model call, and how to notice that a plan or belief was
wrong. If the description says that rules or action effects are not given, discovering them by
experiment is part of the task.

A plan must be able to fail and recover. Whatever executes without the model (a compiled build order, a
convention policy, a solver) must check its own effects after each step — `ctx.progress()` counts the
actions since the score and the observation last changed and the actions the game rejected — and, when
the evidence says the plan has stopped working, hand the situation with that evidence to Astra for
re-planning, then verify the new plan the same way. Scenarios the training seeds never show are where a
scripted plan breaks: a candidate with identical scores on every training seed is a scenario-independent
script and will be judged on the unseen validation and held-out seeds.

**Evaluation scope.** The harness is selected on the training seeds of this game and then reported on
unseen seeds of the same game; there is no held-out game. Rules, strategies and search that the game
description or the study establish may be implemented as code once they are verified against
observations — but nothing specific to the training seeds.

## Prior knowledge from the study phase
The study phase researched this game (rules, interface, community strategies). Use it: implement exact
rules, bookkeeping and search as code and put planning and adaptation into Astra prompts — but verify
every claim against observations, because sources can be wrong or outdated for this build of the game.

The full documents are readable in this archive under `prior/` (`prior/strategies.md`, `prior/brief.md`, `prior/interfaces.md`, `prior/study.json`, `prior/ADAPTOR.md`). Below: excerpts within a size budget; every cut is marked `…(truncated; full text in prior/<file>)`.

### study/study.json (summary)
- strategy_summary: Strong play fixes a corner anchor and builds a monotone snake of increasing tiles toward it while preserving empty cells. Code should implement exact transitions, seeded chance spawns, legal-action simulation, and shallow expectimax with emptiness, monotonicity, smoothness, anchor position, and merge-potential features. Astra should select the anchor, diagnose damaged layouts, and adapt search priorities. Jev should answer typed, fast rankings and safety checks over precomputed candidate boards. Reject random cycling and short-term merges that break the corner stack; continue playing after 2048 to maximize score and tile.
- scoring: Report cumulative merge score and maximum tile after the fixed move budget. Score is the sum of values created by merges, while maximum tile measures progress toward the 2048 objective and discourages farming small merges.
- interface.observation: A JSON object containing a 4x4 integer board (zero means empty), cumulative score, move count, maximum tile, terminal flag, seed/replay metadata, and optionally the RNG state.
- interface.actions: The adaptor sends one of up, down, left, or right. The engine compresses and merges each line once, spawns one random tile only when the board changed, updates score, and returns the next JSON observation; legal_actions reports directions that would change the board.

### study/strategies.md
# 2048 strategy prior

## Core human strategy

Strong play preserves empty cells and keeps the largest tile in a fixed corner, commonly the bottom-left or bottom-right. Arrange values in a monotone “snake”: values increase toward the anchor corner along one edge and then along the next row. This makes merges predictable and leaves a safe direction for new tiles. Avoid moving away from the anchor unless necessary; an upward move in a bottom-corner plan is often dangerous because it breaks the ordered stack. These principles are discussed in the [2048 strategy section of Wikipedia](https://en.wikipedia.org/wiki/2048_(video_game)#Strategy) and illustrated by the community [ov3y/2048-AI](https://github.com/ov3y/2048-AI).

Keep at least one or two empty cells when possible, and prefer moves that create merges while preserving a smooth gradient between neighboring tiles. A new 2 or 4 can appear in any empty cell, so never rely on a particular spawn location. Do not make a move solely because it increases the immediate score: sacrificing the corner anchor or filling the board can make the position unrecoverable. No-op moves should be rejected because they consume a turn in a harness only if the adaptor chooses to count them, and in the reference game they do not spawn a tile.

## Search and evaluation

For code, implement exact board transitions, legal-action generation, score bookkeeping, and seeded chance tiles. A shallow expectimax search is a strong baseline: maximize over the four player directions, then take the expectation over every empty-cell spawn (2 at 0.9, 4 at 0.1). Cache transpositions by board and depth. Evaluate leaves with a weighted combination of (a) number of empty cells, (b) monotonicity along the chosen snake order, (c) smoothness, penalizing large neighboring log2 differences, (d) maximum tile in the anchor corner, and (e) immediate merge potential. The open-source [nneonneo/2048-ai](https://github.com/nneonneo/2048-ai) is a useful reference for expectimax and heuristic weighting; its implementation should be treated as inspiration and checked for license and rule assumptions.

Use Astra prompts for strategic adaptation: select an anchor corner, explain the current monotone order, identify whether a move would strand a high tile, and choose a safe fallback when all moves reduce emptiness. Astra can adjust heuristic weights or search depth after observing repeated traps, but should receive exact JSON state and candidate transitions so it does not do hidden arithmetic from prose.

Use Jev for fast typed judgments over the same JSON state. Good independent questions are: `choice`: which legal move best preserves the anchor/snake; `yes_no`: does this move break monotonicity or lose the corner tile; `score`: rate each candidate's free-cell/merge safety on a small fixed scale; and `choice`: which move leaves the most empty cells after one-step simulation. Jev should recognize board patterns and rank precomputed candidates; the harness should perform all transitions and arithmetic.

## Opening, endgame, and mistakes

In the opening, prioritize creating a stable edge and merging small pairs while retaining several empty cells. Once a 128/256 or larger chain is established, fill the edge from the corner outward and use side-to-side moves to merge rows. In the endgame, avoid a full board with alternating high tiles; use a controlled “setup” move to align equal values before taking a merge. If no move preserves the anchor, choose the move with the best monotonicity and maximum emptiness rather than chasing a single merge.

Common mistakes are random direction cycling, repeatedly moving in the unsafe direction, breaking a corner stack for a short-term merge, and treating the 2048 target as an automatic terminal condition. The game continues after 2048 in the reference implementation, so maximize score and tile value after reaching it. These pitfalls and the corner/monotonicity ideas are consistent with [Wikipedia](https://en.wikipedia.org/wiki/2048_(video_game)#Strategy) and the search behavior in [nneonneo/2048-ai](https://github.com/nneonneo/2048-ai).

## What belongs where

* **Code:** rules, seeded RNG, legal moves, exact score, candidate simulation, expectimax/beam search, transposition cache, and numerical heuristic features.
* **Astra:** choose and maintain the anchor/snake plan, diagnose traps, change search depth or weights, and explain a recovery plan from damaged boards.
* **Jev:** quick typed ranking of already-simulated legal candidates, corner-break and monotonicity recognition, and yes/no safety checks. Do not ask Jev to implement merges or perform long arithmetic.

### study/brief.md
# 2048 brief

## Rules and objective

2048 is a single-player, turn-based puzzle on a 4x4 grid. A move slides every tile as far as possible in one of four directions (up, down, left, right); adjacent equal tiles on that line merge once, in the direction of travel, into a tile whose value is their sum. A tile cannot merge twice during one move. After a move that changes the board, one empty cell receives a new tile: 2 with probability 0.90 or 4 with probability 0.10 in the reference implementation. The game starts with two randomly placed tiles. The objective is to create a 2048 tile, while continuing to maximize score and the largest tile when possible. The original implementation and its rules are in the open-source repository: [gabrielecirulli/2048](https://github.com/gabrielecirulli/2048), especially [`js/grid.js`](https://github.com/gabrielecirulli/2048/blob/master/js/grid.js) and [`js/game_manager.js`](https://github.com/gabrielecirulli/2048/blob/master/js/game_manager.js). A concise rules overview is also available at [Wikipedia: 2048](https://en.wikipedia.org/wiki/2048_(video_game)).

The score starts at zero. Every merge adds the value of the newly created tile to the score (2+2 adds 4, 4+4 adds 8, and so on); ordinary slides and spawned tiles add nothing. The reference score bookkeeping is in [`game_manager.js`](https://github.com/gabrielecirulli/2048/blob/master/js/game_manager.js). Play ends when no empty cell exists and no adjacent equal pair can merge. Reaching 2048 does not force termination in the original game: the player may continue.

## Controls and state changes

The browser game accepts arrow keys and, on supported devices, swipe gestures. The canonical keyboard mapping is Up/Down/Left/Right; its input manager is documented in [`js/keyboard_input_manager.js`](https://github.com/gabrielecirulli/2048/blob/master/js/keyboard_input_manager.js). For an adaptor, represent actions as the four strings `up`, `down`, `left`, and `right` (optionally reject an action if it would be a no-op). An observation should contain the 16 integer cells, current score, move count, game-over flag, and RNG state/seed when deterministic replay is needed.

This is tier 1: the world waits between moves. A move is processed synchronously, then the random tile is spawned and the next decision is available. There is no real-time clock or timeout in the normal game, so a decision can take seconds or minutes without changing the board. The evaluator may cap wall-clock time externally, but game quality is determined by the selected move sequence over a fixed number of turns, not reaction speed.

## Evaluation-relevant progress

Report both the cumulative score and the maximum tile. Score rewards making many high-value merges; maximum tile captures the main objective and prevents a strategy that farms small merges from appearing strong. For held-out seeds, reset should reproduce the same initial board and random stream so that runs are comparable while still testing generalization.

Sources: [official 2048 repository](https://github.com/gabrielecirulli/2048), [Game Manager source](https://github.com/gabrielecirulli/2048/blob/master/js/game_manager.js), [Grid source](https://github.com/gabrielecirulli/2048/blob/master/js/grid.js), and [Wikipedia](https://en.wikipedia.org/wiki/2048_(video_game)).

### adaptor/ADAPTOR.md
# 2048 adaptor

This adaptor embeds a small Python implementation of the MIT-licensed 2048 rules. It runs fully
offline and needs no browser, server, account, credentials, or external package. `make_game()`
returns a Tier 1 `TurnGame`; `step()` applies one complete move synchronously.

`reset(seed)` creates a fresh `random.Random(seed)` stream, clears the 4x4 board, and places two
tiles. Every changed move uses the same stream to select an empty cell and then chooses 2 with
probability 0.90 or 4 with probability 0.10. Rows/columns are compressed toward the action and
merged from the travel edge, so `[2, 2, 2, 2]` becomes `[4, 4, 0, 0]` toward the left. No-op and
malformed actions are harmless and do not spawn a tile. Play continues after a 2048 tile.

The observation is strict JSON with the board, raw cumulative merge `score`, move count,
`max_tile`, terminal status, seed, last-move metadata, and currently legal actions. The scalar
`score` property is `merge_score + max_tile`, while `score_breakdown()` and `info()` expose the
individual components so evaluation can distinguish merge quality from tile progress. Episodes
default to 200 changed moves (`EVAL_KWARGS`), leaving room for policies to differ; callers may pass
`max_moves` to `make_game`.

Scenario seeds are declared in `SEEDS`; train, held-out, development, and validation groups are
disjoint and use the same seed-stream rules. `close()` is idempotent and only clears local state,
since this adaptor starts no process or connection.

### study/interfaces.md
# Programmatic interfaces

The reference game is a client-side JavaScript application, not a hosted service with a supported gameplay API. The source is MIT-licensed ([repository](https://github.com/gabrielecirulli/2048)); the practical interfaces below are therefore local and do not require an account.

## Options

| Interface | Maturity / latency | Observation and control | Legal/ToS and cost |
|---|---|---|---|
| **Standalone Python rules engine (chosen)** | Small, deterministic, very mature rules; microseconds to milliseconds per move | Full board as a 4x4 integer matrix, score, move count, terminal flag, and seeded RNG. Call `step("up"|"down"|"left"|"right")`; return a JSON-serializable state and whether a tile spawned. | No website, account, anti-cheat, or ToS issue. Free; Python standard library only. Must faithfully reproduce merge order and 90/10 spawn distribution. |
| Browser DOM + JavaScript source (`gabrielecirulli/2048`) under Playwright/Selenium | Mature browser automation, usually 10–100 ms per key but browser startup is heavier | Read rendered tile elements (`tile-inner` classes encode value and position), score DOM, and game-over state; inject Arrow keys or call the game's JS event path. | Local copy is permitted by its MIT license; automating the public website can violate site policy and is fragile when markup changes. Requires Chromium/driver and extra packages; no GUI is needed with headless mode. |
| Browser screenshot + input (`xdotool`, `pyautogui`, OpenCV/OCR) | Mature desktop tooling but 50–500+ ms and resolution/theme dependent | Screenshot pixels and recognize tile colors/text; inject arrows. | Highest fragility and no need on a server without GUI. X server, fonts, and image/OCR packages add cost; website automation may have ToS concerns. |
| Existing AI/bot implementations (for example [nneonneo/2048-ai](https://github.com/nneonneo/2048-ai), [ov3y/2048-AI](https://github.com/ov3y/2048-AI)) | Community code is useful as a strategy reference; integration quality varies | Usually consume a board array and emit a direction using expectimax/heuristics. Wrap their engine or port the search; inspect licenses and Python/runtime assumptions. | Free/open source when license terms are followed. They are not a stable API, may target the browser DOM, and may not provide seeded reset or exact evaluator telemetry. |
| Reinforcement-learning environment ports (search terms include [Gymnasium](https://gymnasium.farama.org/) and “2048 environment”) | APIs vary and many ports are stale; local step latency is good | Usually `reset(seed)` and `step(action)` with a board observation and reward. Verify dimensions, spawn RNG, merge semantics, and licenses against the reference game. | No network/account after installation, but dependency installation can be difficult offline. Environment mismatch can invalidate scores. |

## Chosen adaptor contract

Use the standalone engine. It is the only option that is fully offline, GUI-free, deterministic under an explicit seed, and independent of browser markup. The adaptor can be implemented with `/home/hemlix/GamingMaster-mh-tier1/runs/venvs/2048/bin/python` and the standard library; no credentials or environment variables are required.

Suggested module contract (the next phase may choose names):

* `reset(seed: int | None) -> dict`: initialize a 4x4 board, score 0, move count 0, and a seeded pseudo-random generator; spawn exactly two starting tiles.
* `observe() -> dict`: return `{"board": [[...], ...], "score": int, "moves": int, "max_tile": int, "game_over": bool, "seed": ...}`. Keep zeros for empty cells. Including the RNG state is useful for exact replay but can be omitted from model prompts.
* `step(action: str) -> dict`: apply one direction. If the direction is a no-op, do not spawn a tile and mark `changed: false`; otherwise merge, add the merge values to score, spawn one tile, increment `moves`, and return the new observation plus `changed` and `spawn` fields.
* `legal_actions() -> list[str]`: directions that change the board. Return an empty list at terminal state.

The engine must preserve the reference line-processing order: compress toward the direction, merge each eligible adjacent pair once, then pad with zeros. A deterministic test suite should compare known boards against the JavaScript implementation's behavior. Do not install packages or fetch assets at runtime; if an adaptor elects to use NumPy, install it into the evaluator venv ahead of execution and record that choice.

## Credentials and setup

No server address, account, API key, or secret is needed for the chosen interface. Setup is: (1) create or vendor the Python engine in the generated harness; (2) run it with the evaluator's Python; (3) expose JSON observations and the four action strings to Astra/Jev; (4) seed reset for each scenario. Browser alternatives would require a local game checkout plus Chromium/Playwright, but they are deliberately not prerequisites.

## Risks and compatibility checks

The main risk is an implementation mismatch: spawning after no-op moves, allowing a tile to merge twice, using the wrong 90/10 distribution, or seeding Python differently from the harness can change scores. Validate merge order with cases such as `[2,2,2,2] -> [4,4]` toward one side and `[2,2,4] -> [8]` after compression. Community bots may carry incompatible licenses or assumptions; browser/screenshot approaches add GUI, latency, and markup risks.


## Measured model latency (at archive creation, 2026-10-30 19:58)
- System-2 (Astra), one-word answer: 34 s
- System-2, an answer of ~932 output tokens: 102 s — latency grows with the length of the answer; size timeouts for the answers your prompts actually request
Later measurements per candidate are on the leaderboard (latency columns, a latency fit).