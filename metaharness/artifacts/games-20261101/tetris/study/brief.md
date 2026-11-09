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
