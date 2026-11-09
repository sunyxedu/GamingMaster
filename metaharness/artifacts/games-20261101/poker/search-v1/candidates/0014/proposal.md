parent: 0012

## Hypothesis

Candidate 0012's postflop planner is strong, but its button opening branch always raises to the minimum 200 chips (2bb). The frozen heuristic's BB continuation is price-sensitive: against a 200 open it calls 100 into a 300 pot, while a 250 open asks 150 into 350 and raises the documented call threshold from about 0.413 to 0.509. The archived 0012 traces show that this cheap price is being paid frequently: after hero's 200 button opens, the opponent called 44/51 times on seed 3301, 39/78 on 2207, 60/81 on 4409, and 60/87 on 1103. This leaves many marginal hands in postflop pots where the parent spends most of its computation. A larger, position-specific opening price should fold out the weakest BB calls while extracting more from hands that continue.

The weakest archived 0012 match is seed 2207 (269.75 chips/hand). Its trace records hand-3 as `raise_to 200` by hero with K♠5♦, followed by the BB `call` for 100 into the 300-chip pot; the hand settled at -200 chips. That result is only a diagnostic example, not proof that K5s should fold, but it shows the exact cheap-price continuation this experiment changes.

## What changed vs. the parent

I copied 0012's completion-aware anytime turn-to-river planner, exact card evaluator, public-action range updates, legal integer sizing, and wall-time reserve. The sole strategic change is the unopened button preflop branch: when the parent would raise to 200 with a hand whose canonical equity is at least 0.365, this harness raises to the legal total 250 (2.5bb), matching heuristic-v1's documented button size. Premium preflop jams, BB defense, postflop search, exposure checks, and cancellation behavior are unchanged. The target is passed through the observation's legal bounds, so it remains a current-street total and cannot create an illegal increment.

This tests a structural price/position hypothesis rather than retuning postflop thresholds. The code records how often the branch is used but never reads score, seed, hidden cards, or unrevealed engine state; all decisions depend on the public observation and canonical hand equity.

## Evidence that would confirm or refute it

Confirmation would show fewer BB calls after otherwise identical button opens, more immediate folds or larger pots against continuing hands, and equal or improved net chips per required hand with zero illegal actions and clean completion. The strongest evidence is an improvement across independent seeds together with a lower fraction of hands entering postflop after weak button opens. Refutation would be a score decline caused by heuristic-v1 continuing with the same range despite the higher price, or by the larger pot increasing losses on marginal opens; a single all-in outcome is insufficient. Traces should compare the opponent's response frequencies and the parent's postflop action distribution before attributing a score change to this preflop sizing experiment.

One permitted development run (seed 901) completed all 100 hands in 122 seconds with score 1119.0, 210 legal actions, zero model calls, and no errors. The trace contains 40 `raise_to 250` unopened-button actions and cleanly reaches `game_done`. This is runtime and contract evidence only; one independent seed cannot establish a score improvement.
