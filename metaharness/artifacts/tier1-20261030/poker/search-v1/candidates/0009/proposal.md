parent: 0004

## Hypothesis

The deeper turn-to-river planner is being rejected by a computational failure, not yet a demonstrated strategic failure. Its training matches all stop at 1800 seconds with only 64, 82, 83 and 78 hands complete. The leader's one-street planner misses future information value: in seed 2207 hand-22 with Js8c on Td4h9dJc, 0003 values the passive turn line at 335.4 and bets 250; 0004 values it at 1588.18 and checks. After the opponent bets, its posterior changes, and it ultimately folds to a river raise. The leader instead jams the river and loses 20000 to legally revealed JhQc. This observation supports testing deeper planning, though it does not prove the range EV is calibrated.

This candidate tests whether **amortized exact card evaluation and shared numerical work** can make 0004's existing lookahead affordable for complete matches. It uses a different parent from the current leader and targets the compute architecture rather than tuning the leader's betting parameters. Keep the parent strategy, action menu, posterior and simulation samples as controls wherever possible.

## Implementation

Start from 0004's self-contained standard-library harness. Replace repeated rank counting and list construction with an exact bit-mask evaluator and small precomputed rank-mask tables. Preserve legal-action filtering, range-wide decisions on every public branch, all river outcomes, and cooperative cancellation. No model calls, private information, seeds or external files are used during gameplay. The common five-card board is encoded once for all hole pairs. Replace six binary searches per pair with one strength-sorted sweep: if L is the number of lower ranks and T the number of tied ranks, subtract the lower/tied counts containing either opponent hole card, then add its own combination back once to the tie count. This exactly reproduces the opponent information set without removing hero cards. Cache each hypothetical river as fixed equity/pair/showdown-share columns; rebuild only the prefix weight moments for each counterfactual betting branch. This removes repeated triple construction and Python append loops. All 160 flop runouts, every turn river, preflop samples, likelihoods, bet sizes, and exposure safeguards are unchanged.

## Confirm or refute

Confirm the engineering hypothesis if evaluator/range/action differential checks agree with the parent while runtime falls enough to complete all 100 hands. Confirm the strategic hypothesis only with completed independent-match scores, not by treating individual hands as independent. Refute the implementation if exact ranks or action values differ unexpectedly, or if CPU cost still causes incomplete episodes. Refute the strategic claim if complete independent matches fail to benefit from future-street planning. Development results are exploratory, not official training scores.


## Validation so far

- Preflight passed with 72 actions, zero model calls, and no exceptions; its short deadline ended cleanly at a cooperative checkpoint.
- Exact rank comparison against the archived evaluator passed on 90,000 independently sampled five-, six-, and seven-card hands. The board-specialized evaluator matched on the same inputs.
- Full old/new equity dictionaries agree exactly for 10 river boards, three turn boards, and three flop boards: 17,722 combination entries. All turn river slices also agree exactly. Another 2,760 randomly selected prefix-moment intervals agree with zero weights included, and cancellation leaves no partially cached table.
- The hand-22 turn diagnostic produces the same check and 1588.18 future passive EV as 0004. With the same local test context, cold range setup took 7.328 seconds in the parent and 3.387 seconds here; profiled turn planning took 3.391 versus 1.825 seconds. These are local measurements, not guarantees under evaluator CPU contention. The profiles and reproducible validation scripts are in this directory.

Development evaluation has been started on the two declared dev seeds; complete results will be recorded here. No official training or held-out evaluation is run by this proposer.

The two allowed dev episodes completed all 100 hands with `game_done`, zero illegal actions, and zero model calls. Seed 901 scored 1105.50 chips/hand in 210 actions (832.5 seconds); seed 902 scored 787.25 in 192 actions (766.4 seconds). Their mean is 946.375 chips/hand, matching the parent's recorded two-seed diagnostic mean while removing the parent's completion failure. This supports the engineering hypothesis: the strategic policy was held constant and the optimized implementation reaches the terminal denominator. It does not establish a strategic improvement over 0004 or 0003. The full dev report is in `dev/20261030-022649/`.
