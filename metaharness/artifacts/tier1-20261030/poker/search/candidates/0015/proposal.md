parent: 0010

## Hypothesis

Opponent mc-pot-odds-v2 caches one uniform-card equity estimate while its visible cards stay unchanged on a street. Its subsequent check, call, bet or raise decisions therefore share a latent equity value. Treating each action as an independent softened equity observation can overstate confidence and distort predictions of later responses. This candidate tests a joint action-sequence likelihood and posterior-predictive response model, derived from 0010 rather than extending leader 0012's full shared-runout solver.

## Archive evidence

Leader 0012's weakest completed match is seed 2207 (702.77 chips/hand, 1717 seconds). On hand 82, step 189, hero holds Kc3s on Kd9s5c4c and faces a 3,750 turn wager after a prior flop raise/call sequence. The `complete_range_search` log reports equity .5981 and ranks `raise_to 17500` at EV 3,610.18 versus call EV 2,975.9, with called equity .5962. The next observation shows Kh2c and a 20,000-chip loss. Hero was ahead at commitment; this is not evidence that the shove was negative EV. It is a useful concrete test of how a highly selected turn response should be valued.

Parent 0010 seed 2207 hand 22 loses 6,562 after J8 bets 4,687 on Td4h9dJcTs and folds to 14,530. Hand 82 loses 3,908 after calling flop and turn value bets, then folding to a river bet. These sequences motivate testing whether policy evidence and predicted replies remain coherent across actions sharing one cached estimate. They do not establish negative expectation from individual outcomes. The leader also spends 1,352 seconds by step 189 and 1,717 seconds to finish the same match; the added inference must remain deadline-aware.

## Change from parent

Retain 0010's standard-library evaluator, preflop baseline, particle support before the river, exact full-support river ranks, legal action menu, staged sizing, and one-step re-solving. For each hypothetical opponent hand, build a small quadrature distribution for a single per-street equity latent. Condition that distribution on every public action on the street, then predict fold/call/raise replies from the posterior latent distribution. River equity is exact; pre-river estimates and thresholds remain approximate. Opponent equity calculation uses only its hypothetical holes and public cards; hero runouts exclude both private hands. A likelihood floor preserves support.

Astra can adjust bounded latent-noise and likelihood-trust parameters after aggregate evidence. Observed low-probability responses, rejected/stale actions, and calculation failures trigger an audit with the exact public state; recommendations are revalidated and recomputed before action. Reviews include observed and expected response counts for the same immediate postflop-raise opportunities, descriptive response variance and accumulated Brier error. Settlement examples include hero seat explicitly. Every action is checked through `ctx.progress()`. Model output is optional and parsed defensively. Calls use a 210-second timeout, exceeding the observed 146-second maximum, with a 330-second wall reserve and at most four calls. Search checks the wall budget and returns before the grace period.

## Confirmation / refutation

Confirm if the joint likelihood predicts same-street replies better than the parent, avoids costly overcommitments in response-selected spots, and improves net chips across complete independent matches without giving up too much value. Refute if the documented caching behavior is insufficient to explain the trace, the nine-node latent approximation is overconfident, the new posterior merely overfolds, or CPU/model latency prevents full completion. Replays confirm mechanism and legality only; report uncertainty across match seeds, not individual hands.

## Verification

Initial preflight passed with arbitrary stand-in output: 81 actions, no exception, zero answered System-2 calls. This checks contract robustness, not match strength. A local replay of the leader's seed-2207 hand 82 selected `call` (estimated EV 1,449) instead of `raise_to 17500` (leader estimate 3,610). The replay took 0.79 seconds and emitted legal candidates. The parent 0010 river decision in seed 2207 hand 22 was legal and solved in 2.39 seconds; hand 82 river replay selected a fold. A numerical mechanism check showed a two-node joint likelihood of .48416 versus .246016 for the square of its .496 marginal likelihood, demonstrating that shared latent uncertainty is not equivalent to independent evidence. Timeout injection left policy unchanged. Garbage, JSON arrays, nonfinite numbers, booleans in numeric fields, and malformed actions were handled without exception.

Two dev matches completed all 100 hands: seed 901 scored +689.49 chips/hand in 270.9 seconds (244 actions), and seed 902 scored +688.51 in 247.7 seconds (228 actions). Each used two successful Astra calls with zero illegal actions or errors. Their mean is +689.00, but two match blocks do not establish reliable uncertainty or superiority over the leader. There were 58 and 59 checked immediate responses respectively, none below the .06 surprise threshold. These runs demonstrate legal complete execution and applied model recommendations; they do not establish calibrated response probabilities.

Both dev traces exposed weak review evidence. In seed 901 Astra changed noise/trust after a few losing marginal hands. In seed 902 Astra incorrectly called three ordinary +50 blind wins an inconsistent settlement pattern. The final revision adds hero_seat to settlement summaries, explains blind-win semantics, supplies aggregate predicted-versus-observed response counts, and instructs Astra to retain parameters absent concrete mismatch. The action solver is unchanged, but altered advice may change later play; both dev scores precede this payload fix. The two allowed dev evaluations are exhausted. No full training evaluation or held-out results were accessed.

Final preflight after the payload correction passed: 70 actions, no exception, harness_return within the short dry-run budget. It delivered no System-2 answer, so final real-model review behavior remains untested.
