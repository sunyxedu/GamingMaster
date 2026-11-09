parent: 0012

## Hypothesis

Candidate 0012's range-conditioned solver is strategically strong, but its shared flop runout table spends the same expensive work late in the episode when the fixed 100-hand denominator makes completion more valuable than another marginally precise estimate. The archive shows seed 2207 ending at 85/100 hands and 1,717 seconds, with a final model call cancelled at the deadline; the same candidate completed seed 1103 in 969 seconds. Candidate 0015 demonstrates that a cheaper approximation can finish, but its mean score (691.16) is far below 0012 (1021.80). I test a middle path: retain full legal opponent-hole support and exact turn/river completion work, while thinning only the shared flop quadrature as wall time is consumed.

## What changed vs. parent

This is candidate 0012's complete-combination public-history solver with its legal-action validation, deterministic card evaluator, posterior likelihoods, aggregate response model, progress checks, and bounded Astra reviews unchanged. In `board_table`, the opponent-hole table still contains every legal combination, so rare premium holdings cannot disappear through particle resampling. Flop board completions are sampled deterministically at 64 when more than 1050 seconds remain, 40 when 650–1050 seconds remain, and 24 below 650 seconds. Turn completions remain all legal single-card rivers and river equity remains exact. The thresholds are based on remaining wall budget, so fast seeds preserve the parent's fidelity while slow seeds spend less time per later hand. Every sampled table still checks `ctx.progress()`/the wall guard and falls back through the parent's safe recovery path.

## Evidence that would confirm or refute

Confirm if the candidate completes materially more than 85 hands on slow training seeds while retaining most of 0012's chips per completed hand, and if traces show the reduced runout count only after the wall reserve falls. Refute if flop thinning changes action rankings enough to recreate the low score of particle approximations, or if slow seeds still hit the deadline because exact turn/river work dominates. A useful mechanism check is that all 990-ish opponent-hole rows remain present while the logged flop table work changes from 64 to 40/24 runouts. Compare independent match blocks and fixed-denominator scores; a profitable prefix from an incomplete episode is not evidence of improvement.

## Verification

The module is copied from the preflighted 0012 implementation and changes only the fidelity schedule. The action path remains candidate-ID based and catches model/calculation failures. Preflight passed with 149 legal actions, no exception, and no model calls answered by the stand-in.
