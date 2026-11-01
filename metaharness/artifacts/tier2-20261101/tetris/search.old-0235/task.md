Design a harness for a **real-time** game (tier 2).

Game: `Tetris (Guideline-compatible marathon)` — A local, headless Guideline-compatible 10x20 Tetris marathon. The settled board is exposed as 20 rows top-to-bottom of integers (0 empty, 1..7 I/J/L/O/S/T/Z), while active and ghost cells use bottom-origin x,y coordinates. Pieces use deterministic seven-bag randomization seeded by reset(seed), standard SRS rotations/kicks, hold once per piece, gravity, and a 0.5 second lock delay. Level starts at 1 for the standard scenario and rises every 10 cleared lines; the optional medium/high/recovery scenarios are selected when constructing the game.

The objective is to clear lines while keeping the stack low and avoiding holes/top-out. Scoring is 100/300/500/800 times level for single/double/triple/Tetris; T-spin is 400/800/1200/1600 and mini T-spin is 100/200 for zero/one line. Soft drop gives one point per cell and hard drop gives two. Combo adds 50*combo*level after the first consecutive clear; a consecutive difficult clear (Tetris or line-clearing T-spin) receives a 1.5x back-to-back bonus. There is no perfect-clear bonus or garbage/multiplayer mode.

Each tick advances exactly tick_s seconds (0.05 by default), applies the supplied action before gravity, and increments ticks. The runner should call tick on its own clock while thinking; the world never waits for a model. Use an action name exactly as listed in action_spec, or {'action': name}. Malformed actions are ignored and counted in illegal_actions. Observations also contain hold/queue, timers, score components, lines, level, combo, back-to-back state, stack metrics, events from the last tick, seed, scenario, terminal reason, and survival duration. Episodes end at top-out, 200 placed pieces, or 20,000 ticks by default.

Actions:
- left: Move the active tetromino one column left if legal.
- right: Move the active tetromino one column right if legal.
- soft_drop: Move down one cell if legal; awards one point per cell moved.
- rotate_cw: Rotate clockwise using the Guideline SRS wall-kick tests.
- rotate_ccw: Rotate counter-clockwise using the Guideline SRS wall-kick tests.
- hard_drop: Drop to the ghost position, award two points per cell, and lock immediately.
- hold: Swap with hold (once per falling piece), or store the current piece if hold is empty.
- noop: Take no input; gravity and lock delay still advance.
(real-time games: ticks apply 'noop' until the first action is posted; afterwards a tick without a fresh action repeats the last posted action)
Actions are strings. A dict {'action': name} is also accepted. Each tick is 0.05 s; the current action is applied before gravity. Invalid actions are ignored and counted.

Models available through `ctx`:
* `ctx.think(prompt, session_id=...)` — Astra (System-2, slow). Deep reasoning, can plan, can explain
  rules. Returns `.text` and `.session_id` (pass it back to keep conversation context).
  Budget: 40 calls per episode. Astra never sees the game unless you show it.
* `ctx.decide(state, questions)` — Jev (System-1, fast typed judgments). Budget: 4000 calls per
  episode. Read the complete Jev model context and local usage contract below before designing
  its state, questions and answer-to-action bridge. Actual latency must be measured.
* `ctx.spawn(fn)` — background thread, so Astra can think while Jev keeps acting.

Model economics (facts, not instructions): one Jev request answers *many independent typed questions
about one state in parallel* for the price and latency of one, and the Jev budget is two orders of
magnitude larger than Astra's; an Astra call takes far longer and its latency grows with the length
of its answer. Design timeouts and pipelining from the measured latencies (below, if calibrated, and on
the leaderboard). Whatever the design, the evaluator reports how many posted actions were preceded by a
fresh Jev / Astra answer, so the division of labour is visible.

The game ticks every 0.05s **on its own clock**; `ctx.act(a)` posts the current action (instant).
A tick with no fresh action re-applies the last one. Each episode has a wall-clock deadline of
900 s (`ctx.budget.remaining()["wall_s"]`); the game's own end, the deadline or a used-up call
budget end the episode with the score reached. Each episode starts cold: fresh `ctx.memory` and fresh module globals. Clock mode: wall.

Evaluation: mean score over the training seeds (primary), then fraction of stale ticks (lower is
better), then System-2 calls (fewer is better); real-time ties favour longer survival. A candidate can
only be selected if none of its training episodes ends with an error (see the contract). The idle
baseline (never acts) shows what doing nothing scores.

What the harness must design (this is the whole point): how Astra's slow output (a plan, a rule
summary, a policy sketch, a subgoal) is turned into *state + questions* for Jev when needed, how
Jev's typed answer is turned into a game action, when to go back to Astra, and what to do when a
decision is late.

**Evaluation scope.** The harness is selected on the training seeds of this game and then reported on
unseen seeds of the same game; there is no held-out game. Rules, strategies and search that the game
description or the study establish may be implemented as code once they are verified against
observations — but nothing specific to the training seeds.

## Prior knowledge from the study phase
The study phase researched this game (rules, interface, community strategies). Use it: implement exact
rules, bookkeeping and search as code, put planning and adaptation into Astra prompts, and fast
recognition into Jev questions — but verify every claim against observations, because sources can be
wrong or outdated for this build of the game.

The full documents are readable in this archive under `prior/` (`prior/strategies.md`, `prior/brief.md`, `prior/interfaces.md`, `prior/study.json`, `prior/ADAPTOR.md`). Below: excerpts within a size budget; every cut is marked `…(truncated; full text in prior/<file>)`.

### study/study.json (summary)
- strategy_summary: Keep the stack low, flat, flexible, and hole-free; use a side well for Tetrises only while it does not create dangerous dependencies. Enumerate exact SRS-legal placements and evaluate holes, hole depth, aggregate/max height, bumpiness, wells, lines, landing height, combo/B2B state, and top-out margin in code. Astra should choose conditional openings such as 9-0, 6-3, TKI, or DT only while the queue and board support them, and should switch to downstack/recovery when an opener becomes unsafe. Jev should make fast typed choices among precomputed candidates, recognize buried holes, T-spin/B2B status, opener feasibility, and lowest top-out risk, and never plan long sequences or perform arithmetic. Gravity continues during Astra calls, so code executes short verified finesse sequences and Jev or a deterministic fallback acts before the lock deadline.
- scoring: Report exact Guideline-compatible score, cumulative lines, level, pieces placed, and survival duration. Use 100/300/500/800 times level for single/double/triple/Tetris, 100/200 for mini T-spin with zero/one line, 400/800/1200/1600 for T-spin zero/one/two/three lines, one point per soft-dropped cell, and two per hard-dropped cell; enable and document combo and back-to-back policy (commonly 50*combo*level and a 1.5x difficult-clear back-to-back multiplier). Also report lines, pieces, holes, aggregate height, bumpiness, top-out height, illegal actions, and terminal reason because score conventions vary and drop points alone do not measure quality.
- interface.observation: JSON state derived from the environment's symbolic dictionary observation: a settled 20x10 board (with any buffer rows represented separately), active tetromino type/orientation/cells and position, ghost landing cells, hold piece and can_hold, next queue, score, cumulative lines, level, combo, back-to-back status, gravity interval, lock timer, elapsed time, seed, ruleset, and game-over flag. The base tetris-gymnasium API documents board, active_tetromino_mask, holder, queue, and lines_cleared; do not use its RGB wrapper for agent decisions.
- interface.actions: Named JSON actions: left, right, soft_drop, rotate_cw, rotate_ccw, hard_drop, hold, and noop, optionally with dt_s supplied by the adaptor's monotonic clock. Map names to the pinned environment action mapping, execute sequences only after legality checks, verify each resulting state, and return events for movement, lock, line clear, score, hold, illegal action, and top-out. A grouped placement wrapper may enumerate x*4+rotation candidates and an action_mask, but the adaptor must validate that behavior against the installed release.

### study/strategies.md
# Tetris strategy prior

## First principle: identify the ruleset

Before selecting a move, code must carry a `ruleset` field.  Guideline play uses seven-bag, SRS, hold, next queue, hard drop, and lock delay; NES play does not.  Mixing these assumptions is a high-impact failure ([Guideline](https://tetris.wiki/Tetris_Guideline), [Random Generator](https://tetris.wiki/Random_Generator), [NES mechanics](https://tetris.wiki/Tetris_(NES))).  The recommendations below target the selected Guideline-compatible marathon engine.

## What strong play looks like

Strong play keeps the stack low, flat, and flexible, avoids buried holes and unnecessary overhangs, and uses the queue/hold to preserve future options.  A practical default is a 9-0 stack: leave one side column open for I-piece Tetrises when the surface is safe.  A 6-3 stack can be easier to keep stable and gives more room for T-spin structures; neither is a commandment.  The [Tetris Effect beginner/community guide](https://tetriseffect.game/beginners-community-guide/) emphasizes clean, accommodating flat stacking and discusses 9-0 and 6-3 approaches.

A hole is an empty cell with occupied cells above it.  Holes are expensive because future pieces must clear the overstack before reaching them.  Downstacking means using the next pieces to remove lines above a hole and expose it; prioritize a placement that both clears and uncovers a hole over a superficially larger clear.  Spread residue rather than building a tall clean-looking tower.  The [Tetris Concept downstacking guide](https://www.tetrisconcept.com/2015/02/downstacking.html) recommends lowering surface height, avoiding blocks above holes, and finding placements within the existing upstack.

A useful placement evaluator should minimize holes, hole depth, aggregate/max height, surface bumpiness, inaccessible wells, row/column transitions, and top-out risk, while rewarding lines cleared, eroded cells, a safe landing height, and flexible future surfaces.  The classic four-feature Tetris AI formulation is `a*aggregate_height + b*complete_lines + c*holes + d*bumpiness`; its published coefficients are only a starting point because they were tuned for one ruleset ([Codemyroad heuristic](https://codemyroad.wordpress.com/2013/04/14/tetris-ai-the-near-perfect-player/)).  The code should search legal placements and score the resulting board; neither model should do arithmetic that the engine can do exactly.

## Openings and transitions

Openers are conditional plans, not scripts to force regardless of the queue:

* **9-0 / Tetris stacking:** preserve one side well and use I pieces for four-line clears when the stack stays low.
* **6-3 freestyle:** use a three-wide side structure when it leaves a smooth surface and provides T-spin/downstack options.
* **TKI:** an early T-spin opener; attempt only while the required queue and safe placements exist.
* **DT Cannon:** a T-spin double into Tetris route; it is queue- and surface-dependent.
* **4-wide / perfect-clear openers:** high-variance patterns; use only when the seed/queue supports them and a fallback is available.

The [Hard Drop opener reference](https://harddrop.com/wiki/Opener) is a catalog rather than a guarantee.  Astra should choose an opener from the observed queue, estimate whether it remains feasible, and abandon it immediately when it raises top-out risk, creates holes, or consumes the only recovery piece.  Code should enumerate and validate candidate placements; Astra should choose the multi-piece plan and transition point.

## T-spins, B2B, and combos

The engine should recognize a T-spin exactly: the last maneuver of the T was a rotation and three of the four diagonal cells around its center are occupied; pointing-side occupancy distinguishes a proper T-spin from a mini ([T-spin rules](https://tetris.wiki/T-Spin)).  T-spin doubles are often a safer repeatable objective than elaborate triples.  Astra may identify a promising setup, but code must verify legality and scoring.

Track **back-to-back** and **combo** as separate state machines.  A Tetris or line-clearing T-spin is a difficult clear and normally continues B2B; a normal single/double/triple breaks it, while a no-line T-spin is neutral.  Modern Guideline scoring commonly uses a 1.5x B2B action multiplier and `50 * combo * level` for successive line-clearing pieces, but attack tables and combo conventions vary ([TetrisWiki scoring](https://tetris.wiki/Scoring), [Hard Drop B2B](https://harddrop.com/wiki/Back-to-Back)).  Code owns counters and score.  Jev can answer “does this candidate preserve B2B?” or “is this clear difficult?” but should not calculate points.

## Rotation and finesse

Use the actual SRS state/kick tables: four orientation states (0, R, 2, L), five kick tests, a separate I table, and no O kicks ([SRS](https://tetris.wiki/SRS)).  Code should precompute legal placements and shortest input sequences, including wall/floor kicks, rather than asking Jev to invent a rotation sequence.  Finesse is an input-generation/bookkeeping problem: send the minimum legal moves and avoid late corrections.  The [movement finesse reference](https://tetris.wiki/Movement_finesse) and [DAS reference](https://tetris.wiki/DAS) explain why wall-adjacent tapbacks and repeat timing matter.

Astra should reason in placements and board outcomes, not in frame-by-frame keypresses.  Jev should choose among named candidates such as `{"candidate":"B"}` or `{"sequence_id":"p17"}`.  Code should execute and verify every input, rejecting a sequence if the world has advanced or the target is no longer legal.

## Real-time policy for tier 2

The engine’s monotonic clock continues while Astra is planning.  At level 1, a cell takes about one second to fall; by level 10 it is about 0.064 seconds, and at high levels gravity can exceed one cell per frame ([Marathon](https://tetris.wiki/Marathon)).  A grounded piece has roughly 0.5 seconds of lock delay, not a guaranteed model-call budget.  Hard drop removes that slack.  Jev therefore acts on every active piece using a short, typed choice; Astra periodically supplies a longer plan and fallback policy.

A practical control loop is:

1. Code obtains the current symbolic state and enumerates every legal final placement reachable with SRS, hold choices, and the current queue.
2. Code computes exact features and filters placements that top out, create fatal holes, violate the chosen opener, or waste a needed recovery piece.
3. Jev answers independent typed questions: best low-risk candidate, whether a candidate creates a buried hole, whether an opener remains feasible, whether a T-spin/B2B condition is present, and which precomputed sequence is shortest.
4. Code executes the selected sequence immediately, using ghost/lock timing and checking each action result.
5. Astra reviews the resulting board every few pieces or on events (new level, high stack, lost B2B, failed opener, garbage if a future multiplayer mode is enabled), then updates the plan and fallback.

Jev recognizes and chooses; it does not plan a 20-piece strategy or perform arithmetic.  Astra plans openings, transition/recovery, downstack priorities, and risk tradeoffs.  Code owns simulation, search, timing, collision, exact scoring, and bookkeeping.

## What belongs in code, Astra, and Jev

| Knowledge/behavior | Code | Astra prompt | Jev typed question |
|---|---|---|---|
| Board collision, spawn, top-out, line removal | Exact implementation and tests | Never outsource | No |
| Seven-bag, queue, hold permission | Randomizer and state machine | Explain only if diagnosing | “Is hold the safer candidate?” |
| SRS rotations and wall kicks | Complete tables and legality search | No frame-level improvisation | “Which legal sequence reaches C?” |
| Gravity, lock delay, DAS/ARR, elapsed time | Monotonic clock and event loop | Choose a safer fallback at high speed | “Is target still viable?” |
| Aggregate height, holes, hole depth, bumpiness, wells | Exact feature calculation | Interpret tradeoffs | “Which candidate has lower top-out risk?” |
| Score, level, combo, B2B, T-spin classification | Exact counters and variant policy | Decide when to pursue offense | “Full T-spin, mini, or none?” |
| 9-0/6-3 and opener feasibility | Enumerate candidate continuations | Select, abandon, or transition | “Is TKI still feasible: yes/no?” |
| Downstack/recovery | Simulate hole exposure and candidate results | Plan the recovery route | “Which candidate uncovers the next hole?” |
| Finesse/input sequence | Shortest legal sequence and verification | No arithmetic | “Choose sequence A/B/C” |
| Evaluation and replay | Seed, event log, score/lines/survival | Analyze failures | No |

## Common mistakes and safeguards

1. **Ruleset contamination:** applying seven-bag/hold/SRS/T-spins to NES play, or applying NES gravity/scoring to Guideline play. Store and display the ruleset.
2. **Building over holes:** penalize every new hole and its depth; prefer a clear that exposes the hole.
3. **Forcing an opener:** abandon TKI/DT/4-wide/perfect-clear plans when the queue or board no longer supports them.
4. **Rigid well worship:** do not preserve a side well or I piece if the stack becomes unsafe.
5. **Chasing Tetrises while drowning:** downstack and reduce height before optimizing B2B/offense.
6. **Breaking B2B accidentally:** code tracks the chain; Jev only recognizes whether a candidate preserves it.
7. **Ignoring hold reset:** hold is once per falling piece, not unlimited swaps.
8. **Non-finesse late input:** use the ghost and precomputed sequence, then hard-drop before the deadline.
9. **Relying on model time:** never pause gravity while Astra thinks; Jev or a deterministic fallback must act.
10. **Treating a pixel screenshot as ground truth:** prefer exact symbolic state; use screenshots only for diagnostics.

## Sources

* [2009 Tetris Design Guideline PDF](https://archive.org/download/2009-tetris-variant-concepts_202201/2009%20Tetris%20Design%20Guideline.pdf)
* [Guideline OCR](https://ia600405.us.archive.org/12/items/2009-tetris-variant-concepts_202201/2009%20Tetris%20Design%20Guideline_djvu.txt)
* [Tetris Guideline](https://tetris.wiki/Tetris_Guideline)
* [Random Generator](https://tetris.wiki/Random_Generator)
* [SRS](https://tetris.wiki/SRS)
* [Scoring](https://tetris.wiki/Scoring)
* [T-spin](https://tetris.wiki/T-Spin)
* [Marathon](https://tetris.wiki/Marathon)
* [Tetris Effect beginner/community guide](https://tetriseffect.game/beginners-community-guide/)
* [Downstacking](https://www.tetrisconcept.com/2015/02/downstacking.html)
* [Tetris AI heuristic](https://codemyroad.wordpress.com/2013/04/14/tetris-ai-the-near-perfect-player/)
* [Movement finesse](https://tetris.wiki/Movement_finesse)
* [DAS](https://tetris.wiki/DAS)
* [Opener reference](https://harddrop.com/wiki/Opener)

### study/brief.md
# Tetris — game brief

## Scope and ruleset choice

“Tetris” is a family of games rather than one single ruleset.  This study chooses a **Guideline-compatible single-player marathon ruleset** for the local headless game: a 10-column by 20-row visible matrix, the seven standard tetrominoes, seven-bag generation, hold, SRS rotation/kicks, a real-time gravity clock, and a finite lock delay.  The archived Blue Planet Software **2009 Tetris Design Guideline** is the primary rules reference ([archived PDF](https://archive.org/download/2009-tetris-variant-concepts_202201/2009%20Tetris%20Design%20Guideline.pdf); [OCR text](https://ia600405.us.archive.org/12/items/2009-tetris-variant-concepts_202201/2009%20Tetris%20Design%20Guideline_djvu.txt)).  TetrisWiki’s [Guideline summary](https://tetris.wiki/Tetris_Guideline) is useful for cross-checking the document.

This is deliberately **not** a claim that the game is NES Tetris.  NES Tetris uses a history-based randomizer, Nintendo Rotation System, no hold, no hard drop, no wall kicks, and different gravity and scoring ([NES mechanics](https://tetris.wiki/Tetris_(NES))).  The adaptor must keep the ruleset in its state and prompts; it must not silently mix NES rules with Guideline strategy.

## Objective and end condition

A tetromino falls from the top of the matrix.  The player translates and rotates it, then lets it lock on the stack or hard-drops it.  Every completely occupied horizontal row is removed and the blocks above it descend.  The objective in the recommended marathon scenario is to maximize score and lines cleared while surviving as long as possible; the episode ends when the next tetromino cannot spawn without colliding with the occupied stack (top-out).  A Tetris means clearing four rows with one tetromino ([TetrisWiki glossary](https://tetris.wiki/Glossary)).

There is no opponent, garbage attack, or network service in the recommended scenario.  “Progress” should therefore be reported as score, total lines, level, pieces placed, and survival time/pieces; top-out is the primary failure event.  A fixed piece horizon is useful for fair evaluation because an agent can otherwise gain score simply by running longer.

## Board, pieces, and state

* The visible matrix is 10 cells wide and 20 cells tall.  The Guideline also specifies a 20-row buffer above the visible matrix, so spawning and collision checks need not be restricted to the visible rows ([Guideline summary](https://tetris.wiki/Tetris_Guideline)).  A symbolic adaptor should expose the visible 20x10 board and, separately, active cells that may be in the buffer.
* The seven tetromino types are I, J, L, O, S, T, and Z.  A seven-bag contains exactly one of each and is shuffled before pieces are dealt ([Random Generator](https://tetris.wiki/Random_Generator)).  This bounds droughts but does not guarantee that the next piece is I.
* Guideline/SRS has four named orientations, 0, R, 2, and L.  A blocked rotation tries the appropriate five kick tests; J/L/S/T/Z share one kick table, I has a separate table, and O has no kicks ([SRS](https://tetris.wiki/SRS)).  Rotations are reversible, but a generic matrix rotation is not an adequate substitute for SRS.
* The active piece is spawned near the top in its north-facing orientation.  The next queue has one to six previews in the Guideline; the local adaptor should expose at least the next five pieces.  Hold stores one piece and may be used once per falling piece, with the permission resetting after that piece locks ([Guideline summary](https://tetris.wiki/Tetris_Guideline)).
* A board cell is either empty or occupied by a tetromino/garbage cell.  Keep the active piece separate from the settled board in observations so that Astra/Jev can distinguish a candidate placement from committed cells.

## Controls and actions

The logical action set is:

| Action | Effect |
|---|---|
| `left`, `right` | Translate one cell if legal. |
| `soft_drop` | Request faster downward movement while retaining control; award one drop point per cell in the selected scoring model. |
| `rotate_cw`, `rotate_ccw` | Apply SRS rotation and kicks if a legal kicked position exists. |
| `hard_drop` | Move to the lowest legal position and lock immediately; award two points per descended cell in the selected scoring model. |
| `hold` | Swap with the hold slot if hold is available for this piece; the replacement spawns at its normal spawn position. |
| `noop` | No player input; real-time gravity and lock timers still advance. |

The 2009 Guideline describes hard drop as immediate and soft drop as a faster fall, and describes the default-style delayed auto-shift and lock behavior in its timing sections (see the [Guideline PDF/OCR](https://ia600405.us.archive.org/12/items/2009-tetris-variant-concepts_202201/2009%20Tetris%20Design%20Guideline_djvu.txt)).  The local JSONL interface in [`interfaces.md`](interfaces.md) makes these events explicit and does not require a keyboard, X server, or screen.

## Gravity, lock, and pace

Tetris is **tier 2** here: the world advances while a model is thinking.  A wall-clock driver must accumulate elapsed time and move the active piece even when the latest model call is slow.  The Guideline/Tetris Worlds gravity curve is commonly expressed as seconds per row

```text
seconds_per_cell(level) = (0.8 - (level - 1) * 0.007) ** (level - 1)
```

for the early levels, with the usual 60 Hz G values approximately `0.01667, 0.021017, 0.026977, ...` at levels 1, 2, 3 and continuing to approximately `20.23G` by level 19 ([Marathon gravity table](https://tetris.wiki/Marathon); [Tetris Worlds timing](https://harddrop.com/wiki/Tetris_Worlds)).  Level 1 is about one cell per second; level 10 is about 0.064 seconds per cell; high-level play can be much faster.  The exact table is a ruleset parameter, not something Jev should infer from a screenshot.

A normally landed piece has about **0.5 seconds of lock delay**.  Extended/move-reset lock can reset after successful movement or rotation up to a 15-reset cap; classic/step-reset behavior only resets on a downward step.  Hard drop bypasses the delay.  The Guideline also specifies an approximately 0.2-second generation/entry delay in its timing model, although an adaptor may set it to zero if it documents that choice ([Guideline lock/timing sections](https://ia600405.us.archive.org/12/items/2009-tetris-variant-concepts_202201/2009%20Tetris%20Design%20Guideline_djvu.txt)).

The practical deadline is state-dependent rather than one constant: at low level a piece has many gravity intervals to plan, but at high level it may fall multiple cells during a 0.5-second call.  Jev must therefore choose from precomputed legal placements and emit short inputs; the engine should not wait for Astra.  Astra can plan an opening or recovery while Jev continues a safe policy.  The scalar `decision_latency_s` in `study.json` uses the explicit 0.5-second grounded lock grace as the baseline, while the adaptor must enforce the faster gravity deadlines.

## Scoring and level progression

For the recommended Guideline score, use the current level multiplier on line-clear actions:

| Event | Base points before level multiplier |
|---|---:|
| Single | 100 |
| Double | 300 |
| Triple | 500 |
| Tetris (four lines) | 800 |
| Mini T-spin, no lines / one line | 100 / 200 |
| T-spin, no lines / one / two / three lines | 400 / 800 / 1,200 / 1,600 |
| Soft drop | 1 per descended cell |
| Hard drop | 2 per descended cell |

The table is documented in the 2009 Guideline and summarized by [TetrisWiki Scoring](https://tetris.wiki/Scoring).  Modern Guideline implementations commonly add `50 * combo * level` for successive line-clearing pieces and a 1.5x action multiplier for a back-to-back chain.  A Tetris and a line-clearing T-spin are difficult clears; an ordinary single/double/triple breaks the chain, while a T-spin with no lines is neutral ([Back-to-back/search summary and Scoring](https://tetris.wiki/Scoring), [T-spin](https://tetris.wiki/T-Spin)).  Combo and perfect-clear conventions vary, so the local engine must state whether they are enabled rather than allowing the model to assume them.

The 2009 fixed-goal mode advances after ten lines per level through its standard marathon goals.  For a simple local scenario, start at level 1 and increase level after each ten cumulative lines; expose `level`, `lines`, and `lines_to_next_level` so score calculations are auditable.  If the evaluator wants NES scoring instead, use the NES table rather than this one.

A full T-spin is recognized when the T’s last maneuver is a rotation and three of the four diagonally adjacent cells around its center are occupied; the pointing-side rule distinguishes proper and mini T-spins ([T-spin](https://tetris.wiki/T-Spin)).  The engine, not Jev, should determine this from its transition history.

## What the adaptor should measure

At every decision, report enough state to reconstruct the choice:

* settled 20x10 board, active tetromino type/orientation/cells, ghost landing cells;
* hold piece and `can_hold`, next queue, current position and lock timer;
* level, cumulative lines, score, combo, back-to-back status, and last-clear classification;
* gravity interval, elapsed time since the last state, game-over flag, and deterministic seed.

For evaluation, retain the raw score but also record lines, pieces, survival duration, top-out height, holes, aggregate height, bumpiness, and illegal-action count.  Score alone can reward risky drop points or a variant-specific T-spin rule; the supplementary measures make held-out-seed comparisons interpretable.

## Sources

* [2009 Tetris Design Guideline PDF](https://archive.org/download/2009-tetris-variant-concepts_202201/2009%20Tetris%20Design%20Guideline.pdf) and [OCR text](https://ia600405.us.archive.org/12/items/2009-tetris-variant-concepts_202201/2009%20Tetris%20Design%20Guideline_djvu.txt)
* [Tetris Guideline overview](https://tetris.wiki/Tetris_Guideline)
* [TetrisWiki: Random Generator](https://tetris.wiki/Random_Generator)
* [TetrisWiki: SRS](https://tetris.wiki/SRS)
* [TetrisWiki: Scoring](https://tetris.wiki/Scoring)
* [TetrisWiki: Glossary](https://tetris.wiki/Glossary)
* [TetrisWiki: Marathon/gravity](https://tetris.wiki/Marathon)
* [Hard Drop: Tetris Worlds](https://harddrop.com/wiki/Tetris_Worlds)
* [Tetris Effect community beginner guide](https://tetriseffect.game/beginners-community-guide/)

### adaptor/ADAPTOR.md
# Tetris adaptor

## Connection and engine

This adaptor is a faithful, self-contained local implementation of a modern
Guideline-style falling-blocks game. It does not open a display, use a GPU,
connect to a server, or require an account. `game.py` owns the board, deterministic
seven-bag randomizer, SRS wall kicks, hold, gravity, lock delay, line clearing,
T-spin classification, and scoring. The evaluator's Python process is the game
client, so `close()` only closes the in-process instance and is idempotent.

The implementation intentionally exposes the symbolic state directly rather than
converting it to pixels. It uses no third-party runtime package or credentials.

## Tier and clock

This is Tier 2 (`RealtimeGame`). `tick_s` is 0.05 seconds (20 Hz). Every evaluator
call to `tick(action)` counts one applied clock tick: the named action is processed,
then gravity and lock delay advance by exactly 0.05 seconds. Thus a slow model does
not pause the game; the runner's clock decides when the next tick arrives. There is
no hidden background thread. A malformed action is ignored, counted, and the last
valid action is used for that tick; it never raises from `tick`.

The normal `standard` scenario starts with an empty 10x20 board at level 1. The
piece sequence is a deterministic seven-bag stream derived from `reset(seed)`, and
held-out seeds use the same rules without queue-specific tuning. `make_game` also
accepts `scenario="low"`, `"medium"`, `"high"`, or `"recovery"` for controlled
scenario evaluations: medium starts at level 5, high at level 10, and recovery
starts at level 3 with a fixed uneven board containing holes and a right-side well.
These optional starts do not alter the standard seed semantics.

Episodes end on top-out, after 200 placed pieces, or after 20,000 ticks by default;
`max_pieces` and `max_ticks` are constructor options. The observation reports the
terminal reason, score, cumulative lines, level, pieces, survival duration, top-out,
and board-quality metrics.

## Actions

`action_spec` contains:

- `left`, `right`: one-column movement.
- `soft_drop`: one cell downward, worth one point if it moves.
- `rotate_cw`, `rotate_ccw`: SRS rotation with wall/floor kicks.
- `hard_drop`: move to the ghost position, worth two points per cell, and lock now.
- `hold`: store or swap the piece, once per falling piece.
- `noop`: no input; gravity and lock delay still advance.

Actions are normally strings. `{"action": "left"}` is accepted as a convenience.
There are no parameterised actions, so no additional grammar is needed.

## Observation

`observe()` returns strict JSON under 200 kB. `board` is a settled 20-row board in
top-to-bottom order (`0` is empty and `1..7` identify I/J/L/O/S/T/Z). `active` and
`ghost_cells` contain exact absolute x/y cells (y increases upward), with active
piece type, orientation, x, and y. `hold`, `can_hold`, and the next five `queue`
pieces are explicit. The state also includes score components, lines, level, combo,
back-to-back status, gravity interval, lock timer, ticks, survival time, events from
the last tick, seed, scenario, illegal-action count, stack holes/aggregate height/
bumpiness, top-out height, and terminal status.

## Score policy

The score follows the requested Guideline-compatible policy: 100/300/500/800 times
level for one/two/three/four lines; 100/200 for mini T-spin with zero/one line;
400/800/1200/1600 for T-spin with zero/one/two/three lines; one point per soft-
dropped cell; two per hard-dropped cell. A combo starts at -1, becomes 0 after the
first clear, and adds `50 * combo * level` for later consecutive clears. Tetris and
line-clearing T-spins are difficult clears; consecutive difficult clears receive a
1.5x back-to-back multiplier. A non-difficult line clear breaks back-to-back. There
is no perfect-clear bonus, garbage, or multiplayer mode. `score_breakdown()` and
`info()` expose numeric components and quality metrics so drop points are not the
only evaluation signal.

## Seeds and limits

Training seeds are 0--7 and held-out seeds are 101--107; the lists are disjoint and
all use the standard empty level-1 start. `DETERMINISTIC=True`: a fresh instance
reset with the same seed has the same first observation. Controlled level/recovery
starts are opt-in constructor scenarios rather than hidden seed cases. The engine
is deliberately offline and does not need `env.example` credentials.

### study/interfaces.md
# Programmatic interfaces

## Recommendation

Use **`tetris-gymnasium` as a local, headless Gymnasium environment**, wrapped by a thin JSONL adaptor that adds an explicit real-time clock, the full symbolic state, deterministic reset/seed, and the chosen scoring/level policy.  It is the best fit for this run because it is pure local software, MIT-licensed, configurable, has no account or server, supports `render_mode=None`/`ansi`, exposes a dictionary board observation, and already implements the logical actions (movement, rotation, hard drop, hold, no-op).  Its public source is [GitHub](https://github.com/Max-We/Tetris-Gymnasium), its [installation docs](https://max-we.github.io/Tetris-Gymnasium/introduction/installation/) say `pip install tetris-gymnasium`, and PyPI metadata at [pypi.org/pypi/tetris-gymnasium/json](https://pypi.org/pypi/tetris-gymnasium/json) reports Python `>=3.10,<4.0`, MIT license, and current dependencies including Gymnasium, NumPy, JAX/Chex, and OpenCV.

The adaptor should prefer the base environment’s board dictionary over an RGB wrapper.  The documented environment API has `board`, `active_tetromino_mask`, `holder`, and `queue` arrays, `Discrete` actions for `move_left`, `move_right`, `move_down`, clockwise/counter-clockwise rotation, `swap`, `hard_drop`, and `no_op`, optional gravity, optional `ansi`/RGB rendering, `reset(seed=...)`, and an info field such as `lines_cleared` ([environment source](https://raw.githubusercontent.com/Max-We/Tetris-Gymnasium/main/tetris_gymnasium/envs/tetris.py), [action mapping](https://raw.githubusercontent.com/Max-We/Tetris-Gymnasium/main/tetris_gymnasium/mappings/actions.py)).  `GroupedActionsObservations` is useful for a fast placement adaptor: it enumerates `x * 4 + rotation` destinations and returns an `action_mask`, but it should be used only after validating its semantics against the pinned package version ([wrapper source](https://raw.githubusercontent.com/Max-We/Tetris-Gymnasium/main/tetris_gymnasium/wrappers/grouped.py)).

Do not ask Astra or Jev to control a GUI.  The engine owns legality, SRS, collision, gravity, lock delay, scoring, and shortest-input generation; Jev chooses among already legal candidates and Astra supplies longer-horizon strategy.

## Option A — local `tetris-gymnasium` (chosen)

**Maturity:** actively documented, modular Gymnasium environment with tests and a published package; newer than many small Tetris Gym packages, so pin and smoke-test the exact version.  It is a general RL environment rather than a finished agent protocol; the JSONL adaptor remains necessary.

**Latency:** in-process `step()` latency is normally milliseconds.  It has no network round trip.  Real-time pacing is the adaptor’s responsibility: run a monotonic-clock loop and call/update gravity independently of model calls.  If the selected version advances one cell per `step`, the adaptor should subdivide elapsed time into deterministic ticks or implement the clock around the engine rather than pretending one model call equals one turn.

**Observation:** symbolic NumPy arrays: a padded board, active-piece mask, hold, and queue.  Convert them to JSON lists plus explicit `piece`, `orientation`, `x`, `y`, `ghost_y`, `can_hold`, timers, score, lines, level, combo, B2B, and seed.  If exact piece identities are not recoverable from the installed package’s arrays, maintain them in the adaptor’s piece source and verify them in a reset/step smoke test.  Do not use `RgbObservation` unless debugging; it intentionally converts state to pixels ([observation wrapper source](https://raw.githubusercontent.com/Max-We/Tetris-Gymnasium/main/tetris_gymnasium/wrappers/observation.py)).

**Actions:** expose named JSON actions and map them to the package’s integer mapping: left, right, down/soft drop, rotate clockwise, rotate counter-clockwise, hard drop, swap/hold, and no-op.  Reject unknown actions before calling the environment.  If using grouped placements, map the selected `(column, rotation)` through `GroupedActionsObservations`, check `info["action_mask"]`, then emit the shortest legal low-level sequence.  The adaptor must label whether an action is rejected, buffered, or committed.

**Cost/legal risk:** free/open-source local installation; no account, API key, server, or online ToS.  The package’s own license is MIT; comply with its license and dependencies.  It is not the trademarked Tetris.com service, and the local ruleset should be described as Guideline-compatible rather than falsely calling the package official.

### Installation and setup

Run all installation commands with the evaluator interpreter, not system Python:

```bash
/home/hemlix/GamingMaster-mh-tier2/runs/tetris/.venv/bin/python -m pip install --upgrade tetris-gymnasium
```

The adaptor phase should:

1. Pin the tested `tetris-gymnasium` version in its requirements/lock file and record `python --version`.

…(truncated; full text in prior/interfaces.md)

* **Legal/cost:** emulator packages may be free/open source; ROMs are the user’s responsibility.  No cloud credentials, but legal provenance and version drift are risks.

## Option F — screen capture + input injection

A desktop/browser Tetris client can be observed with screenshots and controlled with X11/uinput/xdotool/PyAutoGUI.  This is the least suitable interface on the stated server: it needs a display or virtual framebuffer, pixel recognition, focus management, and a timing-sensitive input path.  It exposes screenshots rather than exact board state, and latency/jitter is highest.  It also risks violating a service’s rules if used online.

TETR.IO’s official [rules](https://tetr.io/about/rules/) explicitly prohibit third-party utilities, macro programs, bots, and assistance tools.  Do not use an online account or screen bot for this study.  An official Tetris.com developer/control API was not found; public bots and the [Tetris Bot Protocol MVP](https://gist.github.com/MinusKelvin/de519f7e922a1c0821c3ac55de10384c) are community interfaces, not an endorsed offline game engine.  The protocol uses JSON messages over stdin/stdout or web-worker messaging, observes board/queue/hold/combo/B2B, and requests placements, but it leaves timing/opponent details unspecified and supports Guideline-conforming frontends rather than supplying one.

## Interface contract for the next phase

The adaptor should expose a local process such as:

```text
reset(seed) -> {"state": {...}, "ruleset": "guideline-marathon-v1"}
step({"action": "...", "dt_s": ...}) -> {"state": {...}, "events": [...], "done": false}
```

A state should contain JSON-native values:

```json
{
  "board": [[0,0]], "active": {"type":"T","orientation":0,"x":4,"y":19,"cells":[[4,19]]},
  "ghost_cells": [], "hold": null, "can_hold": true, "queue": ["J","L","I","O","S"],
  "score": 0, "lines": 0, "level": 1, "combo": -1, "back_to_back": false,
  "gravity_s_per_cell": 1.0, "lock_remaining_s": null, "game_over": false,
  "seed": 0, "ruleset": "guideline-marathon-v1"
}
```

The example is schematic; the adaptor must report all 10x20 cells, exact active coordinates, and actual timers.  `dt_s` should be measured by the adaptor, not invented by Jev.  Every response should include line-clear/lock/top-out events and an `illegal_action` count.  This contract makes it possible to replay a seed without GUI, network, credentials, or hidden state.


## Measured model latency (at archive creation, 2026-10-30 02:30)
- System-2 (Astra), one-word answer: 20 s
- System-2, an answer of ~1018 output tokens: 62 s — latency grows with the length of the answer; size timeouts for the answers your prompts actually request
- System-1 (Jev), one request with the game's initial observation and a Choice over its actions: median 131 ms (= 2.6 ticks); first request on a new connection 183 ms — a timeout below the tail turns slow answers into failed calls
Later measurements per candidate are on the leaderboard (latency columns, a latency fit).

## Jev model context and usage for Meta-Harness

Snapshot: 2026-10-29 (docs.typesafe.ai: API, models, primitives, confidence, patterns, cookbooks, request
rules and the jev-1.13 known-failure page; plus the design of deployed open-source Jev agents). This is design context for the Astra proposer and for runtime Astra
when it designs Jev questions. The Python interface below is GamingMaster's wrapper, which differs
from the provider SDK. The model profile and episode budgets in the experiment manifest are
authoritative.

### What Jev supplies

Jev is TypeSafe's System One model for fast, focused judgments. It produces
typed answers and probability distributions, not prose, code, plans or reasoning
traces. It reads text and structured JSON. (The provider has an experimental image
input; this wrapper does not send images, so a visual game needs a separate perception
step — code, or Astra via `ctx.think(images=[...])` — before Jev can judge its state.)
Calibration describes aggregate predictions, not correctness of each answer.
See [System One](https://docs.typesafe.ai/concepts/system-one).

Model ids: `jev-latest` and `jev-preview` both resolve to jev-1.14.0 at this snapshot (released
2026-10-14; jev-1.13.0 remains available). An alias can move between calls; every answer records
the version that produced it, and the leaderboard warns when an archive mixes versions. Limits:
64k tokens per request, 32k per question; English is the primary training language.

In this architecture, Astra can interpret unfamiliar rules, choose subgoals and
design the decision interface. Jev can recognize situations, assess whether a
subgoal's preconditions still hold, or choose among well-described alternatives.
Code should handle exact arithmetic, legal-action checks, coordinate transforms,
bookkeeping and execution. Do not ask Jev to invent an unknown numerical action
mapping or perform an entire planning algorithm in one classification.

### Known failure modes (provider's list for jev-1.13) and what to do instead

* **Literal reading.** It answers the question written, not the one meant: state the
  exact condition in `instructions` and put boundary cases in the criteria.
* **Math, counting, comparisons of numbers and dates.** Unreliable: compute them in
  code and send the result or a named bucket ("adjacent", "two cells away"), never ask
  Jev to count, add or compare magnitudes.
* **Numeric representations.** Semantic words beat raw numbers: colour names over RGB/hex
  or palette indices, "wall to the left" over coordinates. Translate in code first.
* **Indirection.** Several hops (a property of a property, double negatives) cost
  accuracy: name the relevant state field directly, e.g. `threats.nearest`.
* **Large states with irrelevant detail.** Unrelated content distracts and costs accuracy
  ("context rot"): filter in code and send only the fields this decision needs.
* **Contradictory instructions and criteria.** Keep them aligned; a Noul whose `true`
  means "no" performs worse.
* **Choice option order.** It leans toward the option listed first: do not always list a
  default/no-op first; when a decision matters, ask the same question twice with the
  options reordered (one request, two question ids) and act only if both agree.
* **Generation.** It is not a generator: turn extraction into a Choice over candidates
  that code (or Astra) produced.

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
this model. A request must fit 64k tokens, and the state plus any single question
32k. See [State](https://docs.typesafe.ai/concepts/state).

Batch independent judgments about that state into one call. They share the
input but do not see each other's answers. If a question really requires an
earlier answer, put that answer in the next call's state. Otherwise ask both
speculatively and combine them in code. Put the complete judgment in
`instructions`: question IDs are invisible to Jev. Refer to input fields
explicitly, e.g. `plan.preconditions`. See
[Questions](https://docs.typesafe.ai/primitives).

### Patterns that deployed Jev systems share (provider cookbooks, open-source agents)

* **Code computes facts, Jev judges, code decides.** Successful controllers send a small JSON
  state of *derived facts* (distances as named buckets, what is adjacent, time to contact,
  the last action and its measured outcome, the measured age of the observation) rather than
  raw grids or numbers, then ask atomic questions and combine the probabilities in code.
* **One request, several atomic questions.** A per-tick request typically carries 3–8
  independent questions (e.g. a Choice over the legal actions, a Score of immediate danger, a
  Noul per precondition or per candidate), including speculative ones whose answers code may
  ignore; a request with many questions costs about the same latency as one with one.
* **Structured criteria.** Option descriptions as objects (`what`, `not_for`, `examples`) and
  instructions that name the state fields (`` `threats.nearest` ``) measurably beat prose.
* **Freshness and fallbacks.** Every answer is tagged with the observation / plan version it
  judged; a late answer is dropped, and an executable fallback (the last safe action, a
  conservative rule) runs while waiting. Deadlines per question are short and explicit.
* **Confidence gates per consequence.** Irreversible or lethal actions need a higher top
  probability than cheap ones; a top probability under ~0.6 is treated as "undecided" (route to
  the fallback or to System-2), not as a weak yes. Ask high-stakes Choices twice with the options
  reordered and act only when both agree.
* **Large option sets** (more than 255, or many near-identical options) are handled in two
  stages: Score or Noul each candidate independently, then one Choice over the shortlist.
* **System-2 authors, System-1 runs.** The slow model designs the state schema, the questions,
  the thresholds and the fallback policy — and re-authors them when the evidence says the
  questions are wrong — rather than being asked per tick. Harnesses that let System-2 revise
  the System-1 interface on observed failures improved most across iterations.

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
For Score, use 2–10 concrete ordered levels for one dimension (the API rejects more
than 10; the constructor raises); the returned number is an expected level, not
necessarily an integer or a probability — use it for thresholds, not to interpolate
magnitudes. See [Score](https://docs.typesafe.ai/primitives/score).

Confidence summarizes the distribution and is not interchangeable with the
winning option's probability. Inspect both when useful. A Noul near 0.5 is
uncertainty about a yes/no proposition, not medium severity. Handle absent
confidence explicitly; never turn `None` into certainty. Choose thresholds
using training evidence and action consequences, with a fallback or Astra
escalation for ambiguity. See [Confidence](https://docs.typesafe.ai/confidence).

What `ctx.decide` raises: `TimeoutError` when the answer (including retries of
rate-limited or transient server errors) does not arrive within `timeout_s`;
`RuntimeError` for any other failed call (e.g. a malformed question rejected by
the API); `BudgetExceeded` / `EpisodeOver` when the System-1 budget is used up or
the episode is over; `ModelUnavailable` when the key is rejected or the account is
persistently throttled (the run stops; catching it does not hide it).

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
latency from traces and the task's calibration: Jev can take longer than a game
tick, and the first request on a new connection (each thread keeps its own) pays
the connection setup. Keep an executable fallback while waiting, limit outstanding
requests, and associate each response with the observation and plan version that
produced it. Recheck relevant preconditions and legality on the latest observation
before acting. A cached semantic interpretation may outlive an individual action
decision; define its invalidation rule rather than treating every answer as
permanently reusable.

The context caps each request's timeout by the remaining wall budget. Handle
transient failures within that budget, let `ModelUnavailable` propagate, and stop
background work when `ctx.done`. The standard inference calls already log request,
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
