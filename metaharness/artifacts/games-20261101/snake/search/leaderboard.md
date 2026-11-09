# Leaderboard

| # | candidate | mean score | episodes | stale ticks | S1 calls | S2 calls | failed S1/S2 | Jev→act | Astra→act | Jev p50/max s | Astra p50/max s | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0000 | 0000-tier2_jev_only | 0.000 | 6 | 42% | 37 | 0 | 0 / 0 | 100% | 0% | 0.2 / 0 | – | ok |
| 0001 | 0001-tier2_astra_only | 0.000 | 6 | 100% | 0 | 6 | 0 / 0 | – | – | – | – | ok |
| 0002 | 0002-tier2_idle | 0.000 | 6 | 100% | 0 | 0 | 0 / 0 | – | – | – | – | ok |
| 0003 | 0003 | 1.333 | 6 | 41% | 1287 | 0 | 0 / 0 | 100% | 0% | 0.1 / 1 | – | ok |
| 0004 | 0004 | – | 0 | – | 0 | 0 | 0 / 0 | – | – | – | – | ✗ fatal: proposer failed: ModelTimeout: timeout: claude -p exceeded 3600s (session 0db1563f-8059-4543-ab3d-34fdd849110e) |
| 0005 | 0005 | 6.000 | 6 | 12% | 40 | 0 | 0 / 0 | 14% | 0% | 0.2 / 0 | – | ok |
| 0006 | 0006 | 50.500 | 6 | 0% | 437 | 9 | 0 / 1 | 11% | 0% | 0.2 / 0 | 41.8 / 46 | ok |
| 0007 | 0007 | 38.333 | 6 | 0% | 115 | 0 | 0 / 0 | 4% | 0% | 0.2 / 0 | – | ok |
| 0008 | 0008 | 121.167 | 6 | 0% | 0 | 0 | 0 / 0 | 0% | 0% | – | – | ok |
| 0009 | 0009 | 124.333 | 6 | 0% | 0 | 0 | 0 / 0 | 0% | 0% | – | – | ok |
| 0010 | 0010 | 123.167 | 6 | 0% | 0 | 0 | 0 / 0 | 0% | 0% | – | – | ok |
| 0011 | 0011 | 121.500 | 6 | 0% | 0 | 0 | 0 / 0 | 0% | 0% | – | – | ok |
| 0012 | 0012 | 123.500 | 6 | 0% | 0 | 0 | 0 / 0 | 0% | 0% | – | – | ok |
| 0013 | 0013 | 125.000 | 6 | 0% | 0 | 0 | 0 / 0 | 0% | 0% | – | – | ok |
| 0014 | 0014 | 122.833 | 6 | 0% | 49 | 6 | 0 / 1 (0 / 1 timed out) | 0% | 0% | 0.2 / 0 | 35.9 / 51 | ok |
| 0015 | 0015 | 124.667 | 6 | 0% | 0 | 0 | 0 / 0 | 0% | 0% | – | – | ok |
| 0016 | 0016 | 92.500 | 6 | 0% | 0 | 0 | 0 / 0 | 0% | 0% | – | – | ok |
| 0017 | 0017 | 112.833 | 6 | 0% | 0 | 0 | 0 / 0 | 0% | 0% | – | – | ok |
| 0018 | 0018 | 124.167 | 6 | 0% | 104 | 0 | 1 / 0 (1 / 0 timed out) | 0% | 0% | 0.2 / 0 | – | ok |
| 0019 | 0019 | 122.833 | 6 | 0% | 0 | 0 | 0 / 0 | 0% | 0% | – | – | ok |
| 0020 | 0020 | 141.000 | 6 | 0% | 0 | 0 | 0 / 0 | 0% | 0% | – | – | ok |

`Jev→act` / `Astra→act`: fraction of posted actions preceded by a fresh System-1 / System-2 answer since the previous action (reported, not used for selection). Latency columns are the median per-episode p50 and the worst observed call: design timeouts and pipelining from these measurements, not from assumptions.

Best so far: `0020` (mean score 141.000)
Only candidates whose status is `ok` can be selected: a fatal, partial or erroneous evaluation cannot win, whatever its score. A used-up call budget or the deadline is not an error.

Episodes (score, how it ended, wall time; ✗ = error):
- `0000-tier2_jev_only`: seed 0: 0.00 game_done 1s · seed 1: 0.00 game_done 1s · seed 2: 0.00 game_done 1s · seed 3: 0.00 game_done 1s · seed 4: 0.00 game_done 1s · seed 5: 0.00 game_done 1s
- `0001-tier2_astra_only`: seed 0: 0.00 game_done 1s · seed 1: 0.00 game_done 1s · seed 2: 0.00 game_done 1s · seed 3: 0.00 game_done 1s · seed 4: 0.00 game_done 1s · seed 5: 0.00 game_done 1s
- `0002-tier2_idle`: seed 0: 0.00 game_done 1s · seed 1: 0.00 game_done 1s · seed 2: 0.00 game_done 1s · seed 3: 0.00 game_done 1s · seed 4: 0.00 game_done 1s · seed 5: 0.00 game_done 1s
- `0003` (refine of 0000-tier2_jev_only): seed 0: 1.00 game_done 47s · seed 1: 3.00 game_done 32s · seed 2: 0.00 game_done 32s · seed 3: 0.00 game_done 29s · seed 4: 3.00 game_done 47s · seed 5: 1.00 game_done 30s
- `0005` (crossover of 0000-tier2_jev_only, 0002-tier2_idle): seed 0: 3.00 game_done 3s · seed 1: 4.00 game_done 4s · seed 2: 9.00 game_done 8s · seed 3: 6.00 game_done 6s · seed 4: 8.00 game_done 8s · seed 5: 6.00 game_done 5s
- `0006` (rewrite of 0001-tier2_astra_only): seed 0: 44.00 game_done 65s · seed 1: 55.00 game_done 66s · seed 2: 55.00 game_done 80s · seed 3: 47.00 game_done 49s · seed 4: 47.00 game_done 60s · seed 5: 55.00 game_done 83s
- `0007` (crossover of 0005, 0003): seed 0: 35.00 game_done 51s · seed 1: 18.00 game_done 22s · seed 2: 55.00 game_done 79s · seed 3: 51.00 game_done 68s · seed 4: 44.00 game_done 69s · seed 5: 27.00 game_done 35s
- `0008` (explore of 0005): seed 0: 126.00 game_done 400s · seed 1: 118.00 game_done 400s · seed 2: 119.00 game_done 400s · seed 3: 123.00 game_done 400s · seed 4: 123.00 game_done 400s · seed 5: 118.00 game_done 400s
- `0009` (crossover of 0008, 0007): seed 0: 128.00 game_done 400s · seed 1: 120.00 game_done 400s · seed 2: 129.00 game_done 400s · seed 3: 126.00 game_done 400s · seed 4: 124.00 game_done 400s · seed 5: 119.00 game_done 400s
- `0010` (explore of 0008): seed 0: 120.00 game_done 400s · seed 1: 131.00 game_done 400s · seed 2: 118.00 game_done 400s · seed 3: 124.00 game_done 400s · seed 4: 120.00 game_done 400s · seed 5: 126.00 game_done 400s
- `0011` (refine of 0008): seed 0: 127.00 game_done 394s · seed 1: 118.00 game_done 398s · seed 2: 120.00 game_done 395s · seed 3: 123.00 game_done 393s · seed 4: 123.00 game_done 400s · seed 5: 118.00 game_done 400s
- `0012` (explore of 0006): seed 0: 121.00 game_done 400s · seed 1: 126.00 game_done 400s · seed 2: 124.00 game_done 400s · seed 3: 122.00 game_done 400s · seed 4: 124.00 game_done 400s · seed 5: 124.00 game_done 400s
- `0013` (refine of 0009): seed 0: 128.00 game_done 400s · seed 1: 120.00 game_done 400s · seed 2: 129.00 game_done 400s · seed 3: 126.00 game_done 400s · seed 4: 128.00 game_done 400s · seed 5: 119.00 game_done 400s
- `0014` (rewrite of 0009): seed 0: 127.00 game_done 400s · seed 1: 117.00 game_done 400s · seed 2: 122.00 game_done 400s · seed 3: 116.00 game_done 400s · seed 4: 126.00 game_done 400s · seed 5: 129.00 game_done 400s
- `0015` (refine of 0013): seed 0: 128.00 game_done 400s · seed 1: 120.00 game_done 400s · seed 2: 129.00 game_done 400s · seed 3: 126.00 game_done 400s · seed 4: 128.00 game_done 400s · seed 5: 117.00 game_done 400s
- `0016` (rewrite of 0010): seed 0: 55.00 game_done 94s · seed 1: 90.00 game_done 231s · seed 2: 131.00 game_done 397s · seed 3: 128.00 game_done 387s · seed 4: 47.00 game_done 77s · seed 5: 104.00 game_done 239s
- `0017` (crossover of 0013, 0009): seed 0: 117.00 game_done 363s · seed 1: 99.00 game_done 250s · seed 2: 116.00 game_done 345s · seed 3: 101.00 game_done 279s · seed 4: 120.00 game_done 392s · seed 5: 124.00 game_done 400s
- `0018` (rewrite of 0014): seed 0: 121.00 game_done 400s · seed 1: 124.00 game_done 400s · seed 2: 124.00 game_done 400s · seed 3: 124.00 game_done 400s · seed 4: 126.00 game_done 400s · seed 5: 126.00 game_done 400s
- `0019` (crossover of 0013, 0015): seed 0: 125.00 game_done 400s · seed 1: 124.00 game_done 400s · seed 2: 118.00 game_done 400s · seed 3: 121.00 game_done 400s · seed 4: 122.00 game_done 400s · seed 5: 127.00 game_done 400s
- `0020` (explore of 0016): seed 0: 141.00 game_done 345s · seed 1: 141.00 game_done 330s · seed 2: 141.00 game_done 360s · seed 3: 141.00 game_done 360s · seed 4: 141.00 game_done 385s · seed 5: 141.00 game_done 322s

System-2 latency ≈ 39.3 s + -7.5 ms × output tokens (fit over 7 archived calls, r = -0.11; slowest 51 s): long answers are slow — size timeouts from this, not from short baseline answers.

Score composition (mean per episode, from the adaptor's score_breakdown):
- `0000-tier2_jev_only`: filled_board 0.00, final_length 3.00, food_count 0.00, score_fraction 0.00, self_collision 0.00, ticks_survived 8.83, time_limit 0.00, wall_collision 1.00
- `0001-tier2_astra_only`: filled_board 0.00, final_length 3.00, food_count 0.00, score_fraction 0.00, self_collision 0.00, ticks_survived 6.50, time_limit 0.00, wall_collision 1.00
- `0002-tier2_idle`: filled_board 0.00, final_length 3.00, food_count 0.00, score_fraction 0.00, self_collision 0.00, ticks_survived 6.50, time_limit 0.00, wall_collision 1.00
- `0003`: filled_board 0.00, final_length 4.33, food_count 1.33, score_fraction 0.01, self_collision 0.00, ticks_survived 361.50, time_limit 0.00, wall_collision 1.00
- `0005`: filled_board 0.00, final_length 9.00, food_count 6.00, score_fraction 0.04, self_collision 0.00, ticks_survived 54.00, time_limit 0.00, wall_collision 1.00
- `0006`: filled_board 0.00, final_length 53.50, food_count 50.50, score_fraction 0.36, self_collision 0.83, ticks_survived 670.17, time_limit 0.00, wall_collision 0.17
- `0007`: filled_board 0.00, final_length 41.33, food_count 38.33, score_fraction 0.27, self_collision 0.83, ticks_survived 538.33, time_limit 0.00, wall_collision 0.17
- `0008`: filled_board 0.00, final_length 124.17, food_count 121.17, score_fraction 0.86, self_collision 0.00, ticks_survived 4000.00, time_limit 1.00, wall_collision 0.00
- `0009`: filled_board 0.00, final_length 127.33, food_count 124.33, score_fraction 0.88, self_collision 0.00, ticks_survived 4000.00, time_limit 1.00, wall_collision 0.00
- `0010`: filled_board 0.00, final_length 126.17, food_count 123.17, score_fraction 0.87, self_collision 0.00, ticks_survived 4000.00, time_limit 1.00, wall_collision 0.00
- `0011`: filled_board 0.00, final_length 124.50, food_count 121.50, score_fraction 0.86, self_collision 0.67, ticks_survived 3966.67, time_limit 0.33, wall_collision 0.00
- `0012`: filled_board 0.00, final_length 126.50, food_count 123.50, score_fraction 0.88, self_collision 0.00, ticks_survived 4000.00, time_limit 1.00, wall_collision 0.00
- `0013`: filled_board 0.00, final_length 128.00, food_count 125.00, score_fraction 0.89, self_collision 0.00, ticks_survived 4000.00, time_limit 1.00, wall_collision 0.00
- `0014`: filled_board 0.00, final_length 125.83, food_count 122.83, score_fraction 0.87, self_collision 0.00, ticks_survived 4000.00, time_limit 1.00, wall_collision 0.00
- `0015`: filled_board 0.00, final_length 127.67, food_count 124.67, score_fraction 0.88, self_collision 0.00, ticks_survived 4000.00, time_limit 1.00, wall_collision 0.00
- `0016`: filled_board 0.00, final_length 95.50, food_count 92.50, score_fraction 0.66, self_collision 0.50, ticks_survived 2373.50, time_limit 0.00, wall_collision 0.50
- `0017`: filled_board 0.00, final_length 115.83, food_count 112.83, score_fraction 0.80, self_collision 0.67, ticks_survived 3381.50, time_limit 0.17, wall_collision 0.17
- `0018`: filled_board 0.00, final_length 127.17, food_count 124.17, score_fraction 0.88, self_collision 0.00, ticks_survived 4000.00, time_limit 1.00, wall_collision 0.00
- `0019`: filled_board 0.00, final_length 125.83, food_count 122.83, score_fraction 0.87, self_collision 0.00, ticks_survived 4000.00, time_limit 1.00, wall_collision 0.00
- `0020`: filled_board 1.00, final_length 144.00, food_count 141.00, score_fraction 1.00, self_collision 0.00, ticks_survived 3500.67, time_limit 0.00, wall_collision 0.00

Validation seeds (finalists only; the final pick ranks training + validation together):
- `0013`: validation mean score 127.500 over 2 episodes
- `0015`: validation mean score 128.000 over 2 episodes
- `0020`: validation mean score 141.000 over 2 episodes

Final pick: `0020` (training 141.0, validation 141.0).

Held-out evaluations (not used for selection):
- `0020` / heldout (Snake (local classic)): mean score 140.286 over 7 episodes

Ablation: `0020` without runtime System-1 / System-2 on its training seeds: 141.000 (with them: 141.000).
