parent: 0006

## Hypothesis

The 0006 policy's weak result (82.885 chips/hand, with one seed at -227.83 and two deadline endings) is consistent with a strategy that spends computation on postflop uniform-card equity while entering too few small-blind pots to exploit the opponent's predictable preflop response. The study describes this opponent as using a wide button opening range and price-sensitive calls; its BB defense still has a positive equity margin over pot odds. This proposal tests the structural hypothesis that opening substantially wider to the minimum 2.5bb size gains more from initiative and blind capture than it loses through postflop realization.

## Change from parent

Derived from 0006, not the leaderboard leader. 0008 expands the SB opening set to every pair, ace, king, suited hand, connected or one-gap hand, and selected broadways, always at 250 chips. It replaces the parent's broader generic thresholds with a transparent preflop response model: defend cheap opens with playable equity, continue selectively versus the documented premium-only reraises, and raise limps for value. Postflop uses deterministic, visible-card Monte Carlo equity against uniform legal hands, a realization margin when calling, modest value sizing at equity above the opponent's free-bet threshold, and only sparse bluffing. A rare bounded Astra review can recommend a reusable style; it cannot post an action. All emitted actions are checked against the current candidates/bounds, and `ctx.progress()` checks execution after every action. A detected rejection or stall is recorded and handed to the review path before the passive legal recovery action.

## Evidence that would confirm or refute it

The hypothesis is supported if independent training matches complete without errors and the widened opening improves the weak-seed floor and mean chips/hand, especially through more successful blind steals and profitable single-raised pots. It is refuted if additional opens create disproportionate postflop losses, if BB's call threshold means wider hands realize too little equity, or if the change lowers net chips despite increasing hands played. Track opened hands, folds/calls, large postflop commitments, and per-seed score; a single match or hand does not establish the effect.

Preflight should confirm arbitrary model output and timeout recovery, legal action formatting, and execution progress checks. No full evaluation was run by the proposer.
