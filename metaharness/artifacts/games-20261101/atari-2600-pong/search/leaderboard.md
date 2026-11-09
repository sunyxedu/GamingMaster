# Leaderboard

| # | candidate | mean score | episodes | stale ticks | S1 calls | S2 calls | failed S1/S2 | Jev→act | Astra→act | Jev p50/max s | Astra p50/max s | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0000 | 0000-tier2_jev_only | -20.333 | 3 | 89% | 1315 | 0 | 0 / 0 | 100% | 0% | 0.1 / 0 | – | ok |
| 0001 | 0001-tier2_astra_only | -21.000 | 3 | 100% | 0 | 7 | 0 / 1 | 0% | 100% | – | 37.0 / 40 | ok |
| 0002 | 0002-tier2_idle | -21.000 | 3 | 100% | 0 | 0 | 0 / 0 | – | – | – | – | ok |
| 0003 | 0003 | 8.667 | 3 | 0% | 6502 | 0 | 2 / 0 (2 / 0 timed out) | 6% | 0% | 0.2 / 1 | – | ok |
| 0004 | 0004 | 3.667 | 3 | 0% | 88 | 0 | 0 / 0 | 0% | 0% | 0.2 / 0 | – | ok |
| 0005 | 0005 | 7.333 | 3 | 0% | 2005 | 4 | 0 / 0 | 1% | 0% | 0.2 / 0 | 31.8 / 35 | ok |
| 0006 | 0006 | 20.333 | 3 | 0% | 2670 | 9 | 1 / 3 (1 / 3 timed out) | 3% | 0% | 0.2 / 1 | 29.8 / 42 | ok |
| 0007 | 0007 | 20.000 | 3 | 0% | 1083 | 6 | 1 / 3 (1 / 2 timed out) | 1% | 0% | 0.2 / 1 | 53.6 / 57 | ok |
| 0008 | 0008 | 7.667 | 3 | 0% | 4492 | 6 | 3 / 2 (3 / 1 timed out) | 3% | 0% | 0.2 / 1 | 49.1 / 53 | ok |
| 0009 | 0009 | 20.333 | 3 | 0% | 2825 | 9 | 3 / 5 (3 / 5 timed out) | 3% | 0% | 0.2 / 1 | 47.1 / 53 | ok |
| 0010 | 0010 | 20.333 | 3 | 0% | 3273 | 9 | 4 / 4 (4 / 3 timed out) | 3% | 0% | 0.2 / 0 | 40.0 / 49 | ok |
| 0011 | 0011 | 19.667 | 3 | 0% | 2768 | 9 | 4 / 5 (4 / 4 timed out) | 3% | 0% | 0.2 / 1 | 51.3 / 56 | ok |

`Jev→act` / `Astra→act`: fraction of posted actions preceded by a fresh System-1 / System-2 answer since the previous action (reported, not used for selection). Latency columns are the median per-episode p50 and the worst observed call: design timeouts and pipelining from these measurements, not from assumptions.

Best so far: `0006` (mean score 20.333)
Only candidates whose status is `ok` can be selected: a fatal, partial or erroneous evaluation cannot win, whatever its score. A used-up call budget or the deadline is not an error.

Episodes (score, how it ended, wall time; ✗ = error):
- `0000-tier2_jev_only`: seed 0: -20.00 game_done 70s · seed 1: -20.00 game_done 66s · seed 2: -21.00 game_done 63s
- `0001-tier2_astra_only`: seed 0: -21.00 game_done 51s · seed 1: -21.00 game_done 51s · seed 2: -21.00 game_done 51s
- `0002-tier2_idle`: seed 0: -21.00 game_done 51s · seed 1: -21.00 game_done 51s · seed 2: -21.00 game_done 51s
- `0003` (refine of 0000-tier2_jev_only): seed 0: 7.00 deadline 900s · seed 1: 0.00 deadline 900s · seed 2: 19.00 game_done 153s
- `0004` (rewrite of 0002-tier2_idle): seed 0: 1.00 deadline 900s · seed 1: 0.00 deadline 900s · seed 2: 10.00 deadline 900s
- `0005` (crossover of 0000-tier2_jev_only, 0002-tier2_idle): seed 0: 0.00 deadline 900s · seed 1: 2.00 deadline 900s · seed 2: 20.00 game_done 657s
- `0006` (rewrite of 0003): seed 0: 20.00 game_done 580s · seed 1: 20.00 deadline 900s · seed 2: 21.00 game_done 265s
- `0007` (crossover of 0003, 0005): seed 0: 18.00 deadline 900s · seed 1: 21.00 game_done 505s · seed 2: 21.00 game_done 498s
- `0008` (explore of 0005): seed 0: 1.00 deadline 900s · seed 1: 2.00 deadline 900s · seed 2: 20.00 game_done 463s
- `0009` (crossover of 0006, 0008): seed 0: 20.00 game_done 580s · seed 1: 20.00 deadline 900s · seed 2: 21.00 game_done 325s
- `0010` (explore of 0008): seed 0: 20.00 deadline 900s · seed 1: 20.00 deadline 900s · seed 2: 21.00 game_done 269s
- `0011` (refine of 0006): seed 0: 17.00 game_done 807s · seed 1: 21.00 game_done 613s · seed 2: 21.00 game_done 353s

System-2 latency ≈ 29.1 s + 20.2 ms × output tokens (fit over 33 archived calls, r = 0.33; slowest 57 s): long answers are slow — size timeouts from this, not from short baseline answers.

Score composition (mean per episode, from the adaptor's score_breakdown):
- `0000-tier2_jev_only`: loss 1.00, opponent_points 21.00, own_points 0.67, return -20.33, win 0.00
- `0001-tier2_astra_only`: loss 1.00, opponent_points 21.00, own_points 0.00, return -21.00, win 0.00
- `0002-tier2_idle`: loss 1.00, opponent_points 21.00, own_points 0.00, return -21.00, win 0.00
- `0003`: loss 0.00, opponent_points 1.00, own_points 9.67, return 8.67, win 0.33
- `0004`: loss 0.00, opponent_points 0.00, own_points 3.67, return 3.67, win 0.00
- `0005`: loss 0.00, opponent_points 0.33, own_points 7.67, return 7.33, win 0.33
- `0006`: loss 0.00, opponent_points 0.33, own_points 20.67, return 20.33, win 0.67
- `0007`: loss 0.00, opponent_points 0.33, own_points 20.33, return 20.00, win 0.67
- `0008`: loss 0.00, opponent_points 0.67, own_points 8.33, return 7.67, win 0.33
- `0009`: loss 0.00, opponent_points 0.33, own_points 20.67, return 20.33, win 0.67
- `0010`: loss 0.00, opponent_points 0.00, own_points 20.33, return 20.33, win 0.33
- `0011`: loss 0.00, opponent_points 1.33, own_points 21.00, return 19.67, win 1.00

Validation seeds (finalists only; the final pick ranks training + validation together):
- `0006`: validation mean score 17.833 over 6 episodes
- `0009`: validation mean score 19.000 over 6 episodes
- `0010`: validation mean score 14.000 over 6 episodes

Final pick: `0009` (training 20.333333333333332, validation 19.0).

Held-out evaluations (not used for selection):
- `0009` / heldout (Atari 2600 Pong (ALE Robot Pong)): mean score 14.500 over 4 episodes

Ablation: `0009` without runtime System-1 / System-2 on its training seeds: 20.667 (with them: 20.333).
