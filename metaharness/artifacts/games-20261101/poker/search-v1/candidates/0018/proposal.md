parent: 0013

## Hypothesis

The exposure-gated parent and leader 0014 share an inconsistent river observation model: the planner treats the opponent's river equity thresholds as exact, while `update_range` applies a sigmoid and a 0.003 likelihood floor. Thus an observed river check, call, or value bet retains holdings that cannot produce that action under the documented exact-river heuristic. This can distort subsequent value bets, bluffs and bluff catches. Candidate 0010 tested persistent latent equity across all postflop streets; this experiment isolates the exact terminal street without changing flop/turn sampling or replacing the successful completion-aware planner.

The leader's worst match (2207, 418.57 chips/hand) includes hand-22's river `raise_to 19500` at logged range equity .9402, followed by a -20000 settlement, and hand-74's river `raise_to 19650` at .9581, also -20000. Parent 0013 still loses 20000 in hand-74 after a river `raise_to 19550` at .9515 (trace line 296). These losses motivate auditing river range calibration, but are not evidence that the wagers had negative EV: neither hand has an earlier river action to condition on. The proposed change must instead be judged at nodes with actual opponent river evidence and by independent match results.

## Changes from the parent

Retain 0013's 200-chip button openings, exposure gate, exact evaluator, earlier-street posterior, anytime turn search, resource reserve, and zero-model-call loop. Replace only the likelihood of observed opponent river actions with exact threshold and integer-size matching. Reconstruct remaining stacks, street bets, the last full raise increment and whether the opponent already raised from public history. Apply minimum/maximum clipping before matching observed bet totals. Preserve the .08/.92 mixed-bet probabilities. Impossible river action/holding pairs receive zero likelihood; if an entire observation has zero support, restore the pre-event public holding posterior and log an explicit conflict. Log evidence mass and excluded mass for every supported river observation.

The deterministic river equity is computed against the opponent's own information set (its hypothetical holes plus public board); hero's holes are removed only from possible opponent holdings and hero-villain showdown comparisons. No private state, deal seed, future board, or result is read by the policy.

## Confirmation and refutation

Confirm implementation with threshold-boundary tests, clipped-size tests, public trace replay with zero unsupported observations, legal actions, preflight, and completion. Confirm the strategic hypothesis only with improved chips per required hand across independent completed match blocks. Refute it if exact river filtering contradicts observed policy actions, loses completion, or systematically worsens the decisions relative to the soft posterior. A single drawout or one development seed cannot establish superiority. The opponent's sampled earlier-street equity remains approximate and retains the parent's uncertainty floor; this proposal does not claim an exact full-hand posterior.

Validation and controlled replay results will be appended after the harness is tested.

## Validation completed

Preflight passed with 115 legal actions, zero model calls, and no exceptions. Threshold, stack-clipping, raise-availability and pot-odds boundary tests passed. Public replay of archived candidates 0013 and 0014 audited 228 opponent river actions and 43 actually shown opponent holdings with zero unsupported actions or evidence conflicts. On the parent 0013 seed-2207 replay, exact filtering removed substantial impossible posterior mass at several river nodes (up to 21.98%), while preserving every selected action; 18 exact river observations were applied and the 30 river decisions had zero action changes. Other archived replays changed at most one river action, so this is primarily a calibration and future adaptation test rather than a guaranteed action rewrite.

One permitted development evaluation completed both seeds cleanly: seed 901 scored 1153.25 chips/hand in 216 actions and seed 902 scored 912.25 in 194 actions, for a mean of 1032.75, zero illegal actions, zero model calls and zero errors. This exploratory result is not a training or held-out score and does not establish that exact river filtering beats the parent.
