parent: 0000-tier1_astra_naive

## Hypothesis and archive evidence

The leader's principal failure is decision throughput. Its four training episodes completed only 2, 3, 8 and 2 hands of 100; all ended at the 1800-second deadline. In trace_seed1103, the fifth action is `raise_to 400` with Ad Ah, leaving a flop of 7s 9h 4d and a 600-chip call. The episode ends before another action, at −5.5 chips/hand, after only two settlements. The 3301 score is mostly one early +20000-chip hand, not evidence of a stable full-match edge.

One Astra call per action cannot finish a 100-hand match at the measured roughly 57-second median. A code-owned range and bet-sizing policy, with asynchronous Astra audits and explicit failure checks, should complete matches and accumulate a repeatable edge against the frozen policy. The additional strategic hypothesis is that the opponent's random-hand equity model calls too much against a strong hero range: response-conditioned value sizing should outperform unconditional made-hand rules.

## Final implementation

- Exact standard-library 5–7 card evaluator; uniform-card equity Monte Carlo; no game or opponent imports, seeds, hidden state, or other files are read during play.
- An opponent distribution over legal hole combinations, updated by public action likelihoods from the documented policy. Likelihoods are softened, with a nonzero floor; they are hypotheses rather than asserted implementation facts. Preflop starts with every available combo, then uses a weighted particle approximation after board exposure.
- Public street history is replayed to recover call price, own raise counts, and action context. Hero equity is computed against that conditioned range, not merely against random cards. This avoids valuing board-only pairs as private strength.
- Search over legal integer current-street totals, including overbets, estimates opponent calls and hero equity conditional on those calls. Future-street realization and sampling uncertainty are discounted. This is an approximate terminal-value search, not a full game solver; reraises and future betting are important remaining limitations.
- Astra receives a compact observation, ranked action EVs, recent publicly revealed results, statistics, and explicit failure evidence. It returns a JSON action, reusable posture, and short lesson. Ordinary policy execution never waits for a model request. Audits run via `ctx.spawn`; only their reusable mode/lesson survives once the underlying decision has passed. Stale action suggestions are never posted.
- At most seven audits, normally an initial one and two spaced high-pot reviews; exceptions, a stalled/rejected step, repeated reraises, or a large revealed showdown trigger further reviews. Calls use 125-second timeouts, above the 69-second archive maximum, and catch all model exceptions. No new audit starts in the final 150 seconds.
- Every action is validated against legal bounds/candidates. Every resulting step checks `ctx.progress()` and action errors, and local computation failures use a legal passive fallback plus an audit. Progress without score change is normal inside a hand and is not treated as a failure.

## Verification and development evidence

The exact evaluator passed 10,000 independent random 5/6/7-card comparisons against a separate five-card combination enumerator. The final asynchronous version passed required preflight (`ok: true`, 141 actions, no exception, ended at the short preflight deadline). Preflight did not answer any System-2 calls, so it verifies the fallback path, not real-model behavior.

One permitted dev run on seed 901 used the preceding synchronous-audit revision. Its trace reached 94/100 completed hands and a provisional +461.88 chips/hand; it also showed long pauses around audits despite returned model latencies of 63.45 and 42.31 seconds. This motivated the final asynchronous architecture. A second permitted development check was launched on seed 902 for the final revision. These are development checks, not the training evaluation; provisional scores are not completed-match performance claims.

## Confirmation / refutation

Confirm the throughput hypothesis if all 100 hands complete within budget without illegal actions or uncaught errors. Confirm the strategic hypothesis only if completed-match scores improve across independent seeds, including weak seeds. Refute it if completion improves but scores are systematically negative, or if conditional value bets regularly encounter stronger reraises, particle support misses important holdings, or model recovery fails to affect policy. Compare independent match blocks; do not treat correlated hands as independent evidence. Neither one profitable development match nor a preflight score establishes an edge.
