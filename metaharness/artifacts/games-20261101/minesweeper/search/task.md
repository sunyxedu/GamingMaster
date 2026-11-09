Design a harness for a **turn-based** game (tier 1: the world waits for the agent).

Game: `Minesweeper (offline deterministic)` — Minesweeper is a finite turn-based puzzle. Reveal every safe cell to win; revealing a mine loses. The board uses eight-neighbour mine counts, and a zero count automatically reveals its connected zero region and its border. The first reveal is guaranteed safe. Each turn accepts reveal/flag/chord or noop. Use zero-based coordinates in strings such as 'reveal 3 4', or objects with type,row,col. Flag toggles a hidden cell and is only an annotation. Chord on a revealed number opens adjacent hidden cells when the number of adjacent flags exactly matches that number. Observations expose cells, status, safe_revealed, safe_total, flags, and remaining_mines; mine locations stay hidden until terminal. The score is safe_revealed / safe_total, with a separate won indicator in info and score_breakdown.

Actions:
- reveal: Reveal a cell: use the string 'reveal ROW COL' (zero-based).
- flag: Toggle a flag on a hidden cell: use 'flag ROW COL'.
- chord: On a revealed number, reveal neighbours when adjacent flags equal it: 'chord ROW COL'.
- noop: Do nothing for one turn.
(real-time games: ticks apply 'noop' until the first action is posted; afterwards a tick without a fresh action repeats the last posted action)
Actions may also be JSON objects {'type':'reveal'|'flag'|'chord', 'row': ROW, 'col': COL}. Rows and columns are zero-based. Invalid or illegal actions are ignored and reported in the next observation. A first reveal is always safe; zero cells flood-fill their neighbours.

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
- strategy_summary: Strong play repeatedly applies local constraint rules: residual zero means all adjacent unknowns are safe, and residual equal to the unknown count means all are mines. Store frontier equations, use subset differences and exact component enumeration to derive probabilities, and choose the safest move when a guess is unavoidable. Open near the center in first-click-safe games, exploit zero flood fills, and chord only when confirmed flags satisfy a number. Put bookkeeping, search, and probability in code; use Astra for planning and ambiguous cases and Jev for fast typed pattern or chord checks.
- scoring: Report safe_revealed divided by (width*height - mine_count) as the progress score, plus a separate won boolean. The fraction rewards every safe cell opened even on a loss, while won records the objective of revealing all safe cells.
- interface.observation: JSON containing width, height, mine count, status, a grid of unknown/revealed:n/flagged cells, safe_revealed, and safe_total; mine coordinates are hidden while playing.
- interface.actions: One JSON action per turn: reveal(row,col), flag(row,col) to toggle a hidden-cell flag, or chord(row,col) on a revealed number when its adjacent flag count is satisfied. Return the next observation and terminal reason.

### study/strategies.md
# Strong-play strategies and where to implement them

The key skill is constraint bookkeeping. A revealed number `n` gives an equation over adjacent hidden cells: `sum(mines) = n - flagged_neighbors`. If the residual is zero, every adjacent unknown is safe; if it equals the number of unknown neighbors, all of them are mines. Marking a mine (in code's internal knowledge and optionally with `flag`) enables more deductions. These rules and the eight-neighbor coordinate bookkeeping belong in **code** because they are exact and repetitive. Sources: [Minesweeper Wiki, Strategy](https://minesweeper.fandom.com/wiki/Strategy), [Minesweeper Wiki, Basic solving](https://minesweeper.fandom.com/wiki/Strategy#Basic_strategy).

When basic rules stall, retain frontier equations and perform subset inference: if unknown set A is contained in B, subtract A's mine count from B's to constrain `B-A`. Build connected components of the frontier and enumerate or backtrack assignments for small components; compute each cell's mine probability conditioned on the global remaining-mine count. Reveal the lowest-risk cell, with a deterministic tie-break such as the cell that borders the most unrevealed constraints. Exact enumeration, consistency checks, and probability arithmetic belong in **code**; Astra should choose a search depth and explain a genuinely ambiguous guess. The [Minesweeper Wiki strategy guide](https://minesweeper.fandom.com/wiki/Strategy) documents the same subset and probability reasoning used by competitive solvers.

Open deliberately. In first-click-safe variants, click near the center or a location likely to expose a large zero region; after a blank expansion, process the entire new frontier before guessing. Do not spend turns flagging cells unless the flag supports a chord or bookkeeping. These are planning/adaptation choices for **Astra prompts**; the engine still performs flood fill and exact counts. First-click safety and opening behavior are described in [Minesweeper Wiki, First click](https://minesweeper.fandom.com/wiki/First_click).

Use chording to clear a number's neighbors in one action once the number of confirmed adjacent mines equals the number. Chording is an exact **code** action and a good **Jev question** (“is this revealed cell fully flagged, and which neighboring cells are safe?”) when many candidate numbers are available. Never treat a speculative flag as confirmed: an incorrect flag can make a chord reveal a mine in clients that trust flags.

For unavoidable guesses, compare all consistent assignments rather than choosing a visually attractive square. Prefer the cell with the highest conditional safety probability; if tied, prefer a move that reveals more frontier information or a zero. Keep a log of remaining global mines so disconnected components are coupled by the total. Probability ranking and tie-breaking belong in **code**; Astra can inspect exceptional global-coupling cases, while Jev can answer independent typed choices from a supplied JSON frontier.

Common errors are off-by-one edges/corners, forgetting diagonals, counting flags outside a number's neighborhood, double-applying a flood fill, assuming every flag is correct, and continuing after `won`/`lost`. Put these invariants and terminal handling in **code**. Ask Jev for quick recognition of standard local patterns (1-1, 1-2, edge/corner configurations) and for a yes/no “safe under the supplied constraints?” classification; do not ask Jev to invent a plan or perform long arithmetic. Astra should review Jev's uncertain answers and decide when to invoke exact enumeration.

Efficiency-oriented human play tracks 3BV (the minimum number of clicks in an ideal no-guess solution) and minimizes clicks/time, but this evaluator's score is safe-cell fraction plus wins. Use 3BV only as an optional tie-break among equally safe actions, never as a reason to risk a mine. See [Minesweeper Wiki, 3BV](https://minesweeper.fandom.com/wiki/3BV) and [Minesweeper Wiki, Probability](https://minesweeper.fandom.com/wiki/Probability).

Suggested prompt split:

* **Code:** board generation from seed, visibility/flags, flood fill, neighborhood equations, subset/enumeration solver, probability and terminal checks.
* **Astra:** choose opening and component order, interpret unusual rule variants, decide whether a low-probability guess is justified, and recover after a mistaken assumption.
* **Jev:** typed safe/mine/unknown classifications for explicitly supplied cells, detect chordable numbers, rank a small list of already-scored candidates, and flag uncertainty for Astra.

### study/brief.md
# Minesweeper brief

## Rules and objective

Minesweeper is a finite, turn-based deduction puzzle. A rectangular board contains hidden mines and safe cells. Revealing a mine ends the game (loss). Revealing a safe cell shows the count of mines in its eight neighboring squares (orthogonal and diagonal). A zero count recursively opens adjacent zero/number cells (the usual flood fill). The player wins after every safe cell is revealed; flags are annotations and are not required for the win condition. Most modern implementations make the first reveal safe, and many regenerate the board so that the first reveal also has useful expansion. The classic presets are Beginner 9x9/10 mines, Intermediate 16x16/40, and Expert 30x16/99 (Microsoft's historical game uses these sizes).

Sources: [Wikipedia, “Minesweeper (video game)”](https://en.wikipedia.org/wiki/Minesweeper_(video_game)); [Minesweeper Wiki, Rules](https://minesweeper.fandom.com/wiki/Minesweeper).

## State, controls, and actions

The adaptor should represent coordinates as zero-based `(row, column)`, with rows increasing downward. A cell has one of `unknown`, `revealed` (with an integer adjacent-mine count 0–8), or `flagged`; a revealed mine is only present in a terminal loss state. The practical action set is:

* `reveal(r,c)` (left click): reveal a safe cell or lose on a mine; a zero may reveal a whole connected region in one transition.
* `flag(r,c)` (right click): toggle a flag on an unrevealed cell. A flag is bookkeeping and can be wrong.
* `chord(r,c)` (simultaneous/second click on a revealed number): if the number of adjacent flags equals the number, reveal all remaining adjacent unknown cells. Implementations differ on whether an incorrect flag can make a chord lose; the local API should specify and test this explicitly.
* `reset(seed, width, height, mines, first_click_safe=true)`: create a deterministic board for a scenario.

The JSON observation should include dimensions, mine count, status (`playing`, `won`, `lost`), visible cell values, flags, and counts of safe cells revealed. Do not expose mine locations before loss; exposing them after a terminal state is useful for debugging and scoring.

## Scoring and progress

For this study's evaluator, report `safe_revealed / (width*height - mine_count)` as the primary score and a separate boolean `won`. A loss normally stops further actions, so the fraction measures useful progress even when a guess is wrong; a win means all safe cells were found. Traditional clients also show elapsed time and sometimes a 3BV/efficiency score, but wall-clock time should not affect this offline turn-based benchmark. See [Minesweeper Wiki, 3BV](https://minesweeper.fandom.com/wiki/3BV) and [Minesweeper Wiki, scoring](https://minesweeper.fandom.com/wiki/Scoring).

## Pace and tier

The board changes only after an action. There is no opponent and no real-time deadline: Astra may spend seconds to a minute deriving constraints between actions without a game penalty. This is tier 1. Set an adaptor RPC timeout (for example 60 seconds per decision) for infrastructure safety, while treating timeout as a harness failure rather than a game rule. A fast Jev call is useful for repeated local pattern classification, but it is not required to keep up with a ticking world.

### adaptor/ADAPTOR.md
# Minesweeper adaptor

This is a dependency-free, offline Tier 1 `TurnGame` implementing the local deterministic JSON interface selected in the study. It starts no process, server, GUI, account, or network connection. `game.py:make_game()` returns a fresh instance and `reset(seed)` begins an episode.

The seed selects one of six board families: classic 9x9/10, 16x16/40, and 30x16/99, plus 12x10/18, 20x12/45, and 10x10/15. Selection uses a stable modulo-six mapping, and mine coordinates are sampled with `random.Random(seed)`. The declared train, validation, and held-out groups each contain two examples of every family. A first reveal is excluded from the mine candidates, so it is always safe; flags made before the first reveal do not affect layout generation. Supplying `width`, `height`, and `mines` to `make_game` fixes a custom family while retaining seeded placement.

The observation is a JSON dictionary containing `width`, `height`, `mines`, `status`, `cells`, `safe_revealed`, `safe_total`, `flags`, `remaining_mines`, `turn`, `last_action`, `error`, `won`, and `terminal_reason`. During play, cells are `unknown`, `flagged`, or `revealed:N`; after loss or win, mine cells are shown as `mine` for terminal feedback. The mine map is never exposed during play. `reveal` flood-fills zero regions, `flag` toggles hidden cells, and `chord` opens neighbours only when adjacent flags equal the revealed number. Invalid or illegal actions are ignored and described in `error`.

Actions can be compact strings (`reveal ROW COL`, `flag ROW COL`, `chord ROW COL`, or `noop`) or objects such as `{"type":"reveal","row":3,"col":4}`. Coordinates are zero-based. The public score is `safe_revealed / (width*height - mines)`, so progress remains useful after a loss. `score_breakdown()` reports progress, counts, and a numeric `won` component; `info()` records the seed, board family, and terminal state.

`SEEDS` declares disjoint train, dev, validation, and held-out seed groups. The adaptor is deterministic and needs no environment variables or third-party packages.

### study/interfaces.md
# Programmatic interfaces

The board is a local deterministic puzzle, so a network account is unnecessary. Interfaces below are ordered from most suitable for this Linux, headless, offline benchmark to least suitable.

## 1. Local deterministic JSON API (chosen)

**Maturity/latency.** This is a small adaptor-facing protocol around a pure-Python engine (the benchmark permits an own implementation). It is deterministic, unit-testable, and has sub-millisecond transitions; model latency is the only meaningful delay. No package beyond the evaluator's Python 3.12 standard library is needed. **Exposure.** `reset(seed,width,height,mines,first_click_safe)`, `observe()`, and `act({"type":"reveal|flag|chord","row":r,"col":c})`; observations contain the visible grid, flags, status, remaining mine count, and safe-revealed count. The engine must keep the mine map private until `lost`/`won`. **Risk/cost.** No service, credentials, GUI, license, or ToS risk; the adaptor author must document the exact first-click and chord semantics.

Use this JSON shape so `interfaces.md` is sufficient to build the adaptor:

```json
{"width":9,"height":9,"mines":10,"status":"playing",
 "cells":[["unknown", "revealed:1", "flagged"]],
 "safe_revealed":12,"safe_total":71}
```

Actions are one JSON object per turn. `reveal` on an unknown cell performs flood fill for zeros; `flag` toggles only hidden cells; `chord` is valid on a revealed number and applies the flag-count rule. Return the next observation and a terminal reason. A seed must deterministically select the mine layout; if first-click safety is enabled, derive the layout after the first coordinate while retaining reproducibility.

**Install/setup.** Use `/home/hemlix/GamingMaster-mh-tier1/runs/venvs/minesweeper/bin/python`; no `pip install`, server, account, or environment variable is required. The adaptor phase should (1) implement/choose a documented PRNG such as `random.Random(seed)`, (2) generate exactly `mines` unique coordinates, (3) enforce first-click safety, (4) calculate eight-neighbor counts and flood fill, (5) expose the JSON calls above, and (6) run deterministic reset/action smoke tests. Keep mine coordinates out of model observations.

## 2. Open-source solver/bot libraries

Repositories such as [avineshwar/minesweeper-solver](https://github.com/avineshwar/minesweeper-solver), [AdityaMhatre/Minesweeper](https://github.com/AdityaMhatre/Minesweeper), and [microsoft/minesweeper](https://github.com/microsoft/minesweeper) illustrate board representations and constraint solvers. Maturity varies (many are demos, not maintained libraries); latency is local and usually milliseconds. They expose an in-memory grid or UI-specific callbacks rather than a stable benchmark API. Code licenses must be checked per repository; copying code can create license obligations. They have no service cost, but often assume a browser or desktop UI and therefore are not sufficient alone for this headless run. Use their algorithms as references, not as required dependencies.

## 3. Terminal curses clients

[Simon Tatham's Portable Puzzle Collection](https://www.chiark.greenend.org.uk/~sgtatham/puzzles/) includes a Mines variant with a text UI and source code. A pseudo-terminal can send keystrokes and parse text, and it works without a physical display. This is mature free software, but screen parsing is slower and brittle (terminal dimensions, redraws, and keymaps); the exact random seed and board export controls vary by puzzle. It exposes text/characters rather than structured JSON. License is the collection's free software license; check package metadata before redistribution.

## 4. Browser automation

For a web-hosted clone, [Playwright for Python](https://playwright.dev/python/) or [Selenium](https://www.selenium.dev/documentation/) can click cells and read DOM/accessibility labels. Mature, but needs a browser binary and usually network access; canvas-only games require coordinate clicks and screenshot/OCR. Latency is tens of milliseconds to seconds. Automating a third-party site can violate its Terms of Service, trigger anti-bot systems, or change unexpectedly. There is no reason to use it for this offline benchmark.

## 5. Screenshot plus input injection

[Pillow](https://pillow.readthedocs.io/), [OpenCV](https://docs.opencv.org/), and [PyAutoGUI](https://pyautogui.readthedocs.io/) can capture a clone and synthesize mouse clicks. On a server, [Xvfb](https://www.x.org/releases/X11R7.6/doc/man/man1/Xvfb.1.xhtml) supplies a virtual display. This is the least reliable option: template/OCR errors, focus and scaling problems, and roughly 50–200 ms or more per action. It exposes pixels only and may violate a site's anti-cheat/ToS. It requires GUI packages and has no advantage here.

## 6. Native Windows game hooks/emulation

Old Windows Minesweeper clones can be run under [Wine](https://www.winehq.org/) and inspected with process/debugger hooks, or driven through Wine's X11 window. This is version-specific, fragile, and potentially disallowed by the game's license or anti-cheat. It adds Wine and a GUI dependency and cannot guarantee offline reproducibility. Do not select it.

### Selection

Select the local deterministic JSON API. It is the only option that is headless, offline, deterministic under `reset(seed)`, and free of credentials and external ToS. The benchmark needs game semantics, not a particular commercial client; an in-process API also gives Astra complete structured observations and makes Jev's typed questions cheap.

### Credentials and environment

No server, account, API key, or secret environment variable is needed. If the harness wants configurable defaults, it may define non-secret `MINESWEEPER_WIDTH`, `MINESWEEPER_HEIGHT`, `MINESWEEPER_MINES`, and `MINESWEEPER_SEED`, but the canonical `reset` arguments must override them. Never put credential values in this study.


## Measured model latency (at archive creation, 2026-10-30 15:22)
- System-2 (Astra), one-word answer: 45 s
- System-2, an answer of ~970 output tokens: 89 s — latency grows with the length of the answer; size timeouts for the answers your prompts actually request
Later measurements per candidate are on the leaderboard (latency columns, a latency fit).