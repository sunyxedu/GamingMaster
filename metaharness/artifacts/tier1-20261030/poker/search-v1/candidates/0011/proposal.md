parent: 0003

## Hypothesis

Candidate 0003's public-range exploit is profitable, but its main failure mode is exposure: it repeatedly chooses 19,000–20,000 chip jams on medium-strength ranges. The archive records a 200bb turn jam with Jd7c on Jh Ks 5h 8c that was called by Ah9h and lost to a river flush, and notes that the leader's flop jams had only about 0.63–0.67 called equity. The traces show the exposure directly: seed 4409 hand-5 logs `raise_to 19800` at equity 0.7123 with called equity 0.6704, and seed 2207 hand-51 logs the same size at equity 0.7352 with called equity 0.7209. A favorable range edge can still create poor match-block outcomes when one river card dominates the score. I hypothesize that the exploit survives, and mean chips/hand improves or becomes more stable, when ordinary value and probe bets are capped while near-nut value retains the ability to use the stack.

## What changed vs. parent

This is derived from 0003's exact evaluator, public-action Bayesian range update, legal-action handling, and scripted no-model loop. The postflop bet menu is changed structurally: if posterior range equity is below 0.78 before the river (0.70 on the river), targets above roughly 1.5 pot are removed, including the all-in endpoint. The full legal range remains available for very strong ranges. Existing continuation likelihoods, range-wide response EV, future-street haircut, preflop policy, and cooperative deadline checkpoints are unchanged. Every decision is still recomputed from the current public history, and all submitted raises are mapped through current-street legal bounds.

## Evidence that would confirm or refute

Confirmation would be completed 100-hand matches with zero illegal actions, positive scores, and fewer extreme stack-sized wagers while retaining the parent's exploitative value extraction. A lower cross-seed variance at similar or better mean would support the exposure hypothesis. Refutation would be a material score drop despite completion, or traces showing that medium-strength bets were routinely capped exactly when the heuristic would have paid a larger legal wager. Two dev seeds are diagnostic only; official selection must use independent training matches.
