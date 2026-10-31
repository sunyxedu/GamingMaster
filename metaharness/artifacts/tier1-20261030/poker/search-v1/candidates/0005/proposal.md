parent: 0002

## Hypothesis

The frozen heuristic is profitable to attack through repeated medium sized value bets, but candidate 0003's range solver often converts that edge into 200bb all-in variance. Its training scores swing from 83 to 836 chips/hand, and the seed-2207 trace contains many `raise_to 19750`/`raise_to 19200` decisions. I hypothesize that capping value bets around 0.6–0.8 pot, folding bluff-catchers to large pressure, and reserving raises for high equity will retain the heuristic's calling leak while reducing catastrophic single-hand losses. This is a structural risk-control test rather than a refinement of the leader's Bayesian-jam recipe.

## What changed vs. the parent

Starting from candidate 0002's fast code-owned loop, this harness adds deterministic Monte Carlo showdown equity against legal random opponent holdings and future board cards. It owns all legality and current-street total sizing, uses wider heads-up button opens and price-aware BB defense, and treats the opponent's wager-to-pot ratio as evidence about its value branch. Postflop bets are capped at repeatable fractions of the pot; river bluffs are suppressed, large bets require an extra equity margin, and stack shoves are never a default action. It makes no Astra calls, so arbitrary stand-in model text cannot break preflight and all available time is spent on complete hands.

## Evidence that would confirm or refute it

Confirmation would be completed 100-hand matches with no illegal actions, positive chips/hand, fewer all-in pots, and a narrower dispersion of independent match scores than candidate 0003 while still extracting calls from one-pair and draw-heavy heuristic ranges. Refutation would be a materially lower mean despite completion, showing that the opponent's overcalls pay enough to justify the leader's large value jams, or repeated missed value because capped sizes fail to get paid. Hand traces should show whether losses are concentrated in unavoidable showdown variance or in threshold errors when facing a large bet.
