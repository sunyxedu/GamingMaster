# Leaderboard

| # | candidate | mean score | episodes | stale ticks | S1 calls | S2 calls | failed S1/S2 | Jev→act | Astra→act | Jev p50/max s | Astra p50/max s | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0000 | 0000-tier2_jev_only | 6.000 | 6 | 90% | 2194 | 0 | 0 / 0 | 100% | 0% | 0.2 / 1 | – | ok |
| 0001 | 0001-tier2_astra_only | 1.000 | 6 | 100% | 0 | 19 | 0 / 1 | 0% | 100% | – | 68.5 / 87 | ok |
| 0002 | 0002-tier2_idle | 0.000 | 6 | 100% | 0 | 0 | 0 / 0 | – | – | – | – | ok |
| 0003 | 0003 | 12.000 | 6 | 87% | 1592 | 0 | 0 / 0 | 78% | 0% | 0.2 / 0 | – | ok |
| 0004 | 0004 | 5.167 | 6 | 0% | 3706 | 12 | 0 / 4 (0 / 4 timed out) | 8% | 0% | 0.2 / 0 | 50.3 / 77 | ok |
| 0005 | 0005 | 2.000 | 6 | 90% | 4852 | 6 | 0 / 0 | 99% | 0% | 0.2 / 0 | 90.5 / 119 | ok |
| 0006 | 0006 | 7.667 | 6 | 82% | 911 | 7 | 0 / 0 | 58% | 0% | 0.1 / 0 | 17.0 / 17 | ok |
| 0007 | 0007 | 1.333 | 6 | 0% | 4225 | 6 | 0 / 4 (0 / 4 timed out) | 9% | 0% | 0.2 / 0 | 31.7 / 35 | ok |
| 0008 | 0008 | 14.500 | 6 | 0% | 731 | 12 | 0 / 0 | 8% | 0% | 0.2 / 0 | – | ok |

`Jev→act` / `Astra→act`: fraction of posted actions preceded by a fresh System-1 / System-2 answer since the previous action (reported, not used for selection). Latency columns are the median per-episode p50 and the worst observed call: design timeouts and pipelining from these measurements, not from assumptions.

Best so far: `0008` (mean score 14.500)
Only candidates whose status is `ok` can be selected: a fatal, partial or erroneous evaluation cannot win, whatever its score. A used-up call budget or the deadline is not an error.

Episodes (score, how it ended, wall time; ✗ = error):
- `0000-tier2_jev_only`: seed 101: 6.00 game_done 70s · seed 103: 5.00 game_done 50s · seed 107: 6.00 game_done 52s · seed 109: 6.00 game_done 53s · seed 113: 7.00 game_done 60s · seed 127: 6.00 game_done 67s
- `0001-tier2_astra_only`: seed 101: 1.00 game_done 145s · seed 103: 1.00 game_done 143s · seed 107: 2.00 game_done 137s · seed 109: 0.00 game_done 171s · seed 113: 1.00 game_done 154s · seed 127: 1.00 game_done 170s
- `0002-tier2_idle`: seed 101: 0.00 game_done 169s · seed 103: 0.00 game_done 169s · seed 107: 0.00 game_done 170s · seed 109: 0.00 game_done 171s · seed 113: 0.00 game_done 170s · seed 127: 0.00 game_done 170s
- `0003` (refine of 0000-tier2_jev_only): seed 101: 10.00 game_done 45s · seed 103: 14.00 game_done 25s · seed 107: 10.00 game_done 21s · seed 109: 15.00 game_done 50s · seed 113: 10.00 game_done 53s · seed 127: 13.00 game_done 67s
- `0004` (rewrite of 0002-tier2_idle): seed 101: 7.00 game_done 145s · seed 103: 9.00 game_done 87s · seed 107: 1.00 game_done 158s · seed 109: 10.00 game_done 63s · seed 113: 3.00 game_done 142s · seed 127: 1.00 game_done 145s
- `0005` (crossover of 0000-tier2_jev_only, 0002-tier2_idle): seed 101: 3.00 game_done 111s · seed 103: 2.00 game_done 144s · seed 107: 2.00 game_done 160s · seed 109: 1.00 game_done 151s · seed 113: 3.00 game_done 110s · seed 127: 1.00 game_done 156s
- `0006` (rewrite of 0001-tier2_astra_only): seed 101: 9.00 game_done 22s · seed 103: 6.00 game_done 25s · seed 107: 8.00 game_done 34s · seed 109: 9.00 game_done 25s · seed 113: 6.00 game_done 19s · seed 127: 8.00 game_done 20s
- `0007` (crossover of 0003, 0005): seed 101: 0.00 game_done 169s · seed 103: 2.00 game_done 112s · seed 107: 2.00 game_done 112s · seed 109: 0.00 game_done 171s · seed 113: 0.00 game_done 170s · seed 127: 4.00 game_done 76s
- `0008` (explore of 0004): seed 101: 16.00 game_done 26s · seed 103: 12.00 game_done 26s · seed 107: 10.00 game_done 16s · seed 109: 15.00 game_done 19s · seed 113: 18.00 game_done 29s · seed 127: 16.00 game_done 35s

System-2 latency ≈ 46.4 s + 11.2 ms × output tokens (fit over 25 archived calls, r = 0.14; slowest 119 s): long answers are slow — size timeouts from this, not from short baseline answers.

Score composition (mean per episode, from the adaptor's score_breakdown):
- `0000-tier2_jev_only`: ale_reward 6.00, deaths 4.00, frogs_home 0.00, level 1.00, reserve_frogs 0.00, score 6.00, score_delta 0.00, time_bonus 48.00
- `0001-tier2_astra_only`: ale_reward 1.00, deaths 4.00, frogs_home 0.00, level 1.00, reserve_frogs 0.00, score 1.00, score_delta 0.00, time_bonus 434.67
- `0002-tier2_idle`: ale_reward 0.00, deaths 4.00, frogs_home 0.00, level 1.00, reserve_frogs 0.00, score 0.00, score_delta 0.00, time_bonus 510.00
- `0003`: ale_reward 12.00, deaths 4.00, frogs_home 0.00, level 1.00, reserve_frogs 0.00, score 12.00, score_delta 0.00, time_bonus 357.67
- `0004`: ale_reward 5.17, deaths 4.00, frogs_home 0.00, level 1.00, reserve_frogs 0.00, score 5.17, score_delta 0.00, time_bonus 124.00
- `0005`: ale_reward 2.00, deaths 4.00, frogs_home 0.00, level 1.00, reserve_frogs 0.00, score 2.00, score_delta 0.00, time_bonus 429.33
- `0006`: ale_reward 7.67, deaths 4.00, frogs_home 0.00, level 1.00, reserve_frogs 0.00, score 7.67, score_delta 0.00, time_bonus 54.67
- `0007`: ale_reward 1.33, deaths 4.00, frogs_home 0.00, level 1.00, reserve_frogs 0.00, score 1.33, score_delta 0.00, time_bonus 510.00
- `0008`: ale_reward 14.50, deaths 4.00, frogs_home 0.00, level 1.00, reserve_frogs 0.00, score 14.50, score_delta 0.00, time_bonus 54.00
