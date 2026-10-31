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
