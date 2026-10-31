parent: 0000

## Hypothesis

The leader is limited primarily by model latency, rather than by a lack of
strategic deliberation. Its trace for seed 2207 used 5 actions in 1,547 seconds;
one Astra call took 280.7 seconds and the episode ended after only two completed
hands (score -5.75). Across the four episodes it made 28 calls and completed at
most four hands, so its displayed score is dominated by an unfinished prefix.
The action traces also show oversized, weakly justified bets (for example
`raise_to 3600` followed by `raise_to 16000` on seed 1103) and calls that do not
encode the known opponent policy. A fast conservative baseline should complete
far more hands, collect the heuristic's wide opens and value-heavy betting, and
avoid paying large river prices with marginal holdings.

## What changed vs. the parent

This candidate removes Astra from the per-turn control loop. It uses only cheap
stdlib code and the public observation: a small exact five-card evaluator,
hole-card quality, straight/flush draw detection, pot-odds thresholds, and
position-aware heads-up preflop ranges. Every action is selected from the exact
`candidates` supplied by the adapter, including current-street raise totals; a
fallback checks/calls/folds only when that choice is legal. Premium hands 3-bet,
the button opens widely to 250, the big blind checks or raises limps by value,
and postflop play bets made hands/draws while folding weak bluff-catchers to
expensive action. There are no model calls, so the 40-call budget and long Astra
latencies cannot truncate the match.

## Evidence that would confirm or refute it

Confirmation would be substantially more than the leader's 1–4 completed hands
per episode, zero illegal actions, and a positive score over the fixed 100-hand
denominator; the strongest evidence is positive mean chips/hand on the training
seeds despite using no Astra calls. Refutation would be a completed, fast match
whose score is materially below the leader, or traces showing that the policy's
preflop quality cutoffs systematically fold profitable blind/button situations.
In that case the next iteration should add selective Astra calls only at large
postflop decisions, rather than returning to one call per action.
