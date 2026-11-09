# Leaderboard

| # | candidate | parent | mean score | episodes | stale ticks | S1 calls | S2 calls | failed S1/S2 | Jev→act | Astra→act | Jev p50/max s | Astra p50/max s | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0000 | 0000-tier1_astra_naive | – | -0.944 | 6 | – | 0 | 47 | 0 / 3 (0 / 3 timed out) | 0% | 100% | – | 38.2 / 142 | ok |
| 0001 | 0001 | 0000 | – | – | – | – | – | – | – | – | – | – | not evaluated |
| 0002 | 0002 | – | 1.100 | 6 | – | 0 | 8 | 0 / 0 | 0% | 0% | – | 5.7 / 6 | ok |
| 0003 | 0003 | – | – | – | – | – | – | – | – | – | – | – | not evaluated |

`Jev→act` / `Astra→act`: fraction of posted actions preceded by a fresh System-1 / System-2 answer since the previous action (reported, not used for selection). Latency columns are the median per-episode p50 and the worst observed call: design timeouts and pipelining from these measurements, not from assumptions.

Best so far: `0002` (mean score 1.100)
Only candidates whose status is `ok` can be selected: a fatal, partial or erroneous evaluation cannot win, whatever its score. A used-up call budget or the deadline is not an error. `parent` is the candidate a proposal says it was derived from (`parent: NNNN` in proposal.md).

Episodes (score, how it ended, wall time; ✗ = error):
- `0000-tier1_astra_naive`: seed 1101: -0.67 deadline 1800s · seed 1102: -1.00 game_done 1590s · seed 1103: -1.00 game_done 1800s · seed 1104: -1.00 game_done 1264s · seed 1105: -1.00 game_done 938s · seed 1106: -1.00 game_done 1298s
- `0002`: seed 1101: 1.10 game_done 92s · seed 1102: 1.10 game_done 84s · seed 1103: 1.10 game_done 82s · seed 1104: 1.10 game_done 155s · seed 1105: 1.10 game_done 84s · seed 1106: 1.10 game_done 83s

System-2 latency ≈ 39.6 s + 46.2 ms × output tokens (fit over 44 archived calls, r = 0.13; slowest 142 s): long answers are slow — size timeouts from this, not from short baseline answers.

Score composition (mean per episode, from the adaptor's score_breakdown):
- `0000-tier1_astra_naive`: completed_duels 2.83, draws 0.00, forfeits 2.83, losses 0.00, lp_component 0.00, native_completed 0.00, outcome_component -0.94, score -0.94, technical_error 0.00, truncations 0.00, win_rate 0.00, wins 0.00
- `0002`: completed_duels 3.00, draws 0.00, forfeits 0.00, losses 0.00, lp_component 0.10, native_completed 3.00, outcome_component 1.00, score 1.10, technical_error 0.00, truncations 0.00, win_rate 1.00, wins 3.00
