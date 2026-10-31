parent: 0015

## Hypothesis

The cached-equity latent model in 0015 still updates its opponent range using a small set of latent equity nodes and likelihoods fitted to visible betting. That can become too certain after repeated checks/calls, while the fixed opponent's value bets and mixed bluffs are generated from its own 96-sample uniform-card equity, not our range posterior. The gap may make thin bluff-catches and marginal value raises profitable on paper but costly against the opponent's threshold policy. Test a conservative evidence gate: retain the existing posterior/search, but require stronger estimated equity to take high-cost postflop actions and use call/fold price thresholds rather than posterior-driven stack commitment.

## What changed vs. parent

Keep 0015's complete standard-library harness, preflop decisions, cached-equity inference, and robust model recovery. Add a postflop exposure gate around the selected action: when facing a bet, never raise unless the posterior equity is at least 0.88; call only when posterior equity clears the exact price threshold plus a street margin (0.06 flop/turn, 0.025 river). When not facing a bet, cap a raise to 75% pot unless equity is at least 0.78, and permit a stack-sized raise only at 0.94. These thresholds are bounded by the observation's legal candidate list and are checked after each step with `ctx.progress()`.

This is a deliberate test of model-risk control, not a claim that posterior thresholds are solver-optimal. The gate only affects postflop actions and leaves free checks and preflop play alone. Astra reviews remain advisory and cannot bypass the gate.

## Trace motivating the test

In candidate 0013's seed-2207 trace, hand 57 continued with Qs6h on 4h Qc 3h 9h Kc after the opponent raised the flop. Its estimates were about 0.611 on the flop, 0.533 on the turn, and 0.539 facing the river bet; the hand settled at -20,000 against KsKd. The earlier leader 0012 lost 4,688 in the same hand after folding to the river raise. The result cannot identify the best action by itself, but it exposes the risky pattern the gate targets: one pair plus a single-street response model can preserve a large commitment even as later action updates the range.

## Evidence that would confirm or refute

Confirm if complete independent matches reduce large losses from marginal one-pair decisions without a material fall in net chips per hand, and trace replays show fewer high-cost calls/raises at posterior equity below the gate. Refute if the fixed opponent's 8% bluff branch and wide calls make the gate fold too much, or if thin value lost exceeds saved exposure. Compare independent match blocks; 100 correlated hands are not 100 independent trials.
