# Leaderboard

| # | candidate | mean score | episodes | stale ticks | S1 calls | S2 calls | failed S1/S2 | Jev→act | Astra→act | Jev p50/max s | Astra p50/max s | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0000 | 0000-tier2_jev_only | 108.333 | 3 | 57% | 593 | 0 | 0 / 0 | 100% | 0% | 0.1 / 0 | – | ok |
| 0001 | 0001-tier2_astra_only | 86.667 | 3 | 100% | 0 | 5 | 0 / 0 | 0% | 100% | – | 26.2 / 38 | ok |
| 0002 | 0002-tier2_idle | 0.000 | 3 | 100% | 0 | 0 | 0 / 0 | – | – | – | – | ok |
| 0003 | 0003 | – | – | – | – | – | – | – | – | – | – | not evaluated |
| 0004 | 0004 | – | – | – | – | – | – | – | – | – | – | not evaluated |
| 0005 | 0005 | – | – | – | – | – | – | – | – | – | – | not evaluated |

`Jev→act` / `Astra→act`: fraction of posted actions preceded by a fresh System-1 / System-2 answer since the previous action (reported, not used for selection). Latency columns are the median per-episode p50 and the worst observed call: design timeouts and pipelining from these measurements, not from assumptions.

Best so far: `0000-tier2_jev_only` (mean score 108.333)
Only candidates whose status is `ok` can be selected: a fatal, partial or erroneous evaluation cannot win, whatever its score. A used-up call budget or the deadline is not an error.

Episodes (score, how it ended, wall time; ✗ = error):
- `0000-tier2_jev_only`: seed 0: 65.00 game_done 31s · seed 1: 105.00 game_done 28s · seed 2: 155.00 game_done 34s
- `0001-tier2_astra_only`: seed 0: 0.00 game_done 46s · seed 1: 180.00 game_done 41s · seed 2: 80.00 game_done 46s
- `0002-tier2_idle`: seed 0: 0.00 game_done 46s · seed 1: 0.00 game_done 46s · seed 2: 0.00 game_done 46s

Score composition (mean per episode, from the adaptor's score_breakdown):
- `0000-tier2_jev_only`: atari_score 108.33, invaders_killed 11.67, lives_remaining 0.00, survival_ticks 461.33, ufo_hits 0.00, waves_cleared 0.00
- `0001-tier2_astra_only`: atari_score 86.67, invaders_killed 10.33, lives_remaining 0.00, survival_ticks 661.33, ufo_hits 0.00, waves_cleared 0.00
- `0002-tier2_idle`: atari_score 0.00, invaders_killed 6.00, lives_remaining 0.00, survival_ticks 693.00, ufo_hits 0.00, waves_cleared 0.00
