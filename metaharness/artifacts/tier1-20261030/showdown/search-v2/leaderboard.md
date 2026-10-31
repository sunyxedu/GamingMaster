# Leaderboard

| # | candidate | parent | mean score | seeds median / min | episodes | stale ticks | S1 calls | S2 calls | failed S1/S2 | Jev→act | Astra→act | Jev p50/max s | Astra p50/max s | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0000 | 0000-tier1_astra_naive | – | -0.034 | -0.023 / -0.094 | 10 | – | 0 | 137 | 0 / 0 | 0% | 100% | – | 47.6 / 124 | ok |
| 0001 | 0001 | 0000 | -0.025 | -0.011 / -0.129 | 10 | – | 0 | 103 | 0 / 0 | 0% | 99% | – | 42.1 / 99 | ok |
| 0002 | 0002 | – | – | – | – | – | – | – | – | – | – | – | – | not evaluated |
| 0003 | 0003 | – | – | – | – | – | – | – | – | – | – | – | – | not evaluated |

`seeds median / min`: the median and the worst per-seed score — a mean over a few seeds is noisy and can be carried by one lucky seed; an improvement that lifts the weak seeds is more likely to hold on unseen seeds. `(= all seeds)`: every seed scored the same, which is what a scenario-independent script does; expect it to break on scenarios it was not written for. `Jev→act` / `Astra→act`: fraction of posted actions preceded by a fresh System-1 / System-2 answer since the previous action (reported, not used for selection). Latency columns are the median per-episode p50 and the worst observed call: design timeouts and pipelining from these measurements, not from assumptions.

Best so far: `0001` (mean score -0.025)
Only candidates whose status is `ok` can be selected: a fatal, partial or erroneous evaluation cannot win, whatever its score. A used-up call budget or the deadline is not an error. `parent` is the candidate a proposal says it was derived from (`parent: NNNN` in proposal.md).

Episodes (score, how it ended, wall time; ✗ = error):
- `0000-tier1_astra_naive`: seed 104729: -0.02 deadline 1800s · seed 130363: -0.06 deadline 1800s · seed 155921: -0.00 deadline 1800s · seed 181081: -0.01 deadline 1800s · seed 206369: 0.04 deadline 1800s · seed 232003: -0.04 deadline 1800s · seed 257053: -0.01 deadline 1800s · seed 282617: -0.08 deadline 1800s · seed 308081: -0.07 deadline 1800s · seed 333667: -0.09 deadline 1800s
- `0001`: seed 104729: -0.02 deadline 1800s · seed 130363: 0.02 deadline 1800s · seed 155921: -0.00 deadline 1800s · seed 181081: -0.01 deadline 1800s · seed 206369: -0.03 deadline 1800s · seed 232003: -0.02 deadline 1800s · seed 257053: -0.07 deadline 1800s · seed 282617: -0.13 deadline 1800s · seed 308081: 0.02 deadline 1800s · seed 333667: -0.01 deadline 1800s

System-2 latency ≈ 37.3 s + 32.1 ms × output tokens (fit over 220 archived calls, r = 0.53; slowest 124 s): long answers are slow — size timeouts from this, not from short baseline answers.

Score composition (mean per episode, from the adaptor's score_breakdown):
- `0000-tier1_astra_naive`: bot_search_visits 2454800.00, decisions 12.70, hp_margin -0.14, opponent_hp_fraction 0.68, opponent_survivors 4.80, outcome 0.00, score -0.03, self_hp_fraction 0.54, self_survivors 4.20, turns 11.90
- `0001`: bot_search_visits 1773700.00, decisions 9.40, hp_margin -0.10, opponent_hp_fraction 0.71, opponent_survivors 5.00, outcome 0.00, score -0.03, self_hp_fraction 0.61, self_survivors 4.40, turns 8.70
