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
