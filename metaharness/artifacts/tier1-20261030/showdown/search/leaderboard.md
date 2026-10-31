# Leaderboard

| # | candidate | parent | mean score | seeds median / min | episodes | stale ticks | S1 calls | S2 calls | failed S1/S2 | Jev→act | Astra→act | Jev p50/max s | Astra p50/max s | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0000 | 0000-tier1_astra_naive | – | -0.687 | -1.004 / -1.185 | 10 | – | 0 | 311 | 0 / 0 | 0% | 100% | – | 46.5 / 104 | ok |
| 0001 | 0001 | – | – | – | – | – | – | – | – | – | – | – | – | not evaluated |
| 0002 | 0002 | – | – | – | – | – | – | – | – | – | – | – | – | not evaluated |
| 0003 | 0003 | – | – | – | – | – | – | – | – | – | – | – | – | not evaluated |

`seeds median / min`: the median and the worst per-seed score — a mean over a few seeds is noisy and can be carried by one lucky seed; an improvement that lifts the weak seeds is more likely to hold on unseen seeds. `(= all seeds)`: every seed scored the same, which is what a scenario-independent script does; expect it to break on scenarios it was not written for. `Jev→act` / `Astra→act`: fraction of posted actions preceded by a fresh System-1 / System-2 answer since the previous action (reported, not used for selection). Latency columns are the median per-episode p50 and the worst observed call: design timeouts and pipelining from these measurements, not from assumptions.

Best so far: `0000-tier1_astra_naive` (mean score -0.687)
Only candidates whose status is `ok` can be selected: a fatal, partial or erroneous evaluation cannot win, whatever its score. A used-up call budget or the deadline is not an error. `parent` is the candidate a proposal says it was derived from (`parent: NNNN` in proposal.md).

Episodes (score, how it ended, wall time; ✗ = error):
- `0000-tier1_astra_naive`: seed 104729: -0.10 deadline 1800s · seed 130363: -1.16 game_done 1544s · seed 155921: 0.01 deadline 1800s · seed 181081: 0.01 deadline 1800s · seed 206369: -1.00 game_done 1620s · seed 232003: -1.08 game_done 1290s · seed 257053: -0.06 harness_return 1733s · seed 282617: -1.13 game_done 889s · seed 308081: -1.17 game_done 787s · seed 333667: -1.18 game_done 1212s

System-2 latency ≈ 41.1 s + 26.0 ms × output tokens (fit over 308 archived calls, r = 0.41; slowest 104 s): long answers are slow — size timeouts from this, not from short baseline answers.

Score composition (mean per episode, from the adaptor's score_breakdown):
- `0000-tier1_astra_naive`: bot_deadline_hits 0.00, bot_decisions 28.00, bot_max_seconds 0.99, bot_search_visits 8909700.00, bot_seconds 26.02, decisions 30.80, engine_seconds 26.67, hp_margin -0.35, opponent_hp_fraction 0.45, opponent_survivors 4.00, outcome -0.60, score -0.69, self_hp_fraction 0.11, self_survivors 1.10, turns 25.90
