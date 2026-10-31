# Leaderboard

| # | candidate | parent | mean score | seeds median / min | episodes | stale ticks | S1 calls | S2 calls | failed S1/S2 | Jev→act | Astra→act | Jev p50/max s | Astra p50/max s | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0000 | 0000-tier1_astra_naive | – | 0.000 | 0.000 / 0.000 | 4 | – | 0 | 66 | 0 / 0 | 0% | 100% | – | 52.0 / 113 | ok |
| 0001 | 0001 | 0000 | 1.000 | 1.000 / 1.000 | 4 | – | 0 | 0 | 0 / 0 | 0% | 0% | – | – | ok |
| 0002 | 0002 | 0000 | – | – | – | – | – | – | – | – | – | – | – | not evaluated |
| 0003 | 0003 | 0000 | – | – | – | – | – | – | – | – | – | – | – | not evaluated |

`seeds median / min`: the median and the worst per-seed score — a mean over a few seeds is noisy and can be carried by one lucky seed; an improvement that lifts the weak seeds is more likely to hold on unseen seeds. `Jev→act` / `Astra→act`: fraction of posted actions preceded by a fresh System-1 / System-2 answer since the previous action (reported, not used for selection). Latency columns are the median per-episode p50 and the worst observed call: design timeouts and pipelining from these measurements, not from assumptions.

Best so far: `0001` (mean score 1.000)
Only candidates whose status is `ok` can be selected: a fatal, partial or erroneous evaluation cannot win, whatever its score. A used-up call budget or the deadline is not an error. `parent` is the candidate a proposal says it was derived from (`parent: NNNN` in proposal.md).

Episodes (score, how it ended, wall time; ✗ = error):
- `0000-tier1_astra_naive`: seed 0: 0.00 deadline 1800s · seed 2: 0.00 deadline 1800s · seed 4: 0.00 deadline 1800s · seed 6: 0.00 deadline 1800s
- `0001`: seed 0: 1.00 game_done 0s · seed 2: 1.00 game_done 0s · seed 4: 1.00 game_done 0s · seed 6: 1.00 game_done 0s

System-2 latency ≈ 50.5 s + 25.5 ms × output tokens (fit over 62 archived calls, r = 0.56; slowest 113 s): long answers are slow — size timeouts from this, not from short baseline answers.

Score composition (mean per episode, from the adaptor's score_breakdown):
- `0000-tier1_astra_naive`: initial_score 0.00, lost 0.00, maximum_score 3.00, native_score 0.00, normalized_native_score 0.00, steps 15.50, won 0.00
- `0001`: initial_score 0.00, lost 0.00, maximum_score 3.00, native_score 3.00, normalized_native_score 1.00, steps 4.00, won 1.00
