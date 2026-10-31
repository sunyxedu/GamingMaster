parent: 0003

## Hypothesis

The leader’s public-range exploit is useful, but its action search overvalues the largest legal size when the opponent’s continuation range is only moderately dominated. The traces motivate this hypothesis: on seed 3301 it jams `raise_to 19800` on the flop with range equity 0.7315 and called equity 0.6318; on seed 4409 it jams with 0.7123/0.6704. Those wagers can be profitable in a large model expectation, yet one river or a small posterior error swings almost an entire 200bb hand. In the weakest seed-2207 match (83 chips/hand), hand-82 holds Kc3s on Kd9s5c and selects `raise_to 19750` with equity 0.7696 and called equity 0.6173. The publicly revealed Kh2c calls and wins on the 2h river; the settlement is -20000 chips. That outcome establishes exposure, not a negative expectation: hero was ahead of that particular caller. Hand-22 also loses 20000 after a river jam with 0.9409 range equity, illustrating a risk that the proposed 0.88 exemption will deliberately retain.

I test whether an exposure-aware action constraint improves realized chips per hand: retain the leader’s exact evaluator, public-action posterior, and response-aware EV, but bound ordinary postflop raises to roughly four times the pot. A near-stack raise remains available only when range equity is at least 0.88 and the candidate is the legal maximum. Earlier-street exposure also receives a larger safety penalty and a higher called-equity gate. This should preserve the strongest range-value jams while replacing medium-equity all-ins with sizes that still exploit heuristic-v1’s loose calls.

## What changed vs. the parent

Candidate 0003 is copied unchanged for dealing, range updates, preflop policy, card-removal bookkeeping, cancellation, and legal raise-bound mapping. In the postflop target loop, sizes beyond the four-pot exposure budget are skipped unless the hand has very strong range equity and the target is the legal all-in. The EV safety deduction is increased from 5.5% to 8.5% of called exposure on flop/turn (3.5% on river), and the large-bet called-equity floor is raised from 0.61/0.56 to 0.64/0.59. No model calls or seed-specific rules are added.

## Evidence that would confirm or refute it

Confirmation would be complete, zero-error matches with fewer near-stack postflop actions, retained value extraction against the opponent’s price-insensitive calls, and improved mean chips/hand across independent seeds. Traces should show medium-equity nodes selecting bounded targets while very strong made hands can still jam. Refutation would be a material score drop because capped wagers lose too much value, or persistent losses despite the cap, indicating that the leader’s all-in EV estimate was well calibrated and variance was not the limiting issue. The four training seeds remain the selection data; any two-seed development result is diagnostic only.


## Validation and development evidence

Preflight passed: 60 actions, zero model calls, no exceptions, clean cooperative return at the short dry-run deadline. The final policy was then evaluated in the two allowed development runs, with no intervening strategy changes:

| Dev seed | Chips/hand | Completed hands | Actions | Wall seconds | Illegal actions/errors |
|---|---:|---:|---:|---:|---:|
| 901 | 635.00 | 100 | 211 | 180.1 | 0/0 |
| 902 | 393.87 | 100 | 186 | 157.2 | 0/0 |

Both ended with `game_done` and zero S1/S2 calls. Mean chips/hand is **514.435**, with sample standard error **120.565** across two independent match blocks. The parent’s archived dev results on these same seeds were 574.50 and 214.25 (mean 394.375). Both observed differences favor this candidate, but these are only two matches, and the development comparison is not a reliable significance result or a training score. Runtime is machine-load dependent; these wall times do not establish a strategy-independent speedup.

The seed-901 trace shows the intended behavior: hand-23 bets 1600 with equity 0.7632 and called equity 0.6698; hand-24 bets 2000 with equity 0.7483 and called equity 0.6479. Strong range-value exceptions remain: hand-19 jams 19800 with equity 0.9058/called equity 0.8523. Importantly, the cap is relative to the current pot and does not prevent later stack commitment. Seed-902 hand-96 bets 6000 into 1500 on the turn, faces a raise to 16125, and ultimately commits 20000 with Ts8h against publicly revealed QcQs, losing on 9d. This is a concrete limitation rather than a promise of low variance.

Reducing variance alone is not the objective and cannot justify giving up positive chip expectation. The proposed constraint is a robustness hypothesis about posterior/one-street-EV error, tested through mean chips/hand. It remains vulnerable to strong-equity estimation errors, large preflop wagers (unchanged), and future betting that grows a previously bounded pot. No official training or held-out evaluation was run.
