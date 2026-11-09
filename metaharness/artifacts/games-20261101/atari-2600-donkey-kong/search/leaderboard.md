# Leaderboard

| # | candidate | mean score | episodes | stale ticks | S1 calls | S2 calls | failed S1/S2 | Jev→act | Astra→act | Jev p50/max s | Astra p50/max s | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0000 | 0000-tier2_jev_only | 0.000 | 6 | 58% | 1907 | 0 | 0 / 0 | 100% | 0% | 0.2 / 0 | – | ok |
| 0001 | 0001-tier2_astra_only | 0.000 | 6 | 100% | 0 | 18 | 0 / 0 | 0% | 100% | – | 45.4 / 177 | ok |
| 0002 | 0002-tier2_idle | 0.000 | 6 | 100% | 0 | 0 | 0 / 0 | – | – | – | – | ok |
| 0003 | 0003 | 0.000 | 6 | 54% | 24000 | 0 | 8 / 0 (8 / 0 timed out) | 100% | 0% | 0.1 / 0 | – | ok |
| 0004 | 0004 | 0.000 | 6 | 0% | 18730 | 95 | 27 / 11 (27 / 8 timed out) | 23% | 0% | 0.2 / 1 | 48.6 / 96 | ok |
| 0005 | 0005 | – | 0 | – | 0 | 0 | 0 / 0 | – | – | – | – | ✗ fatal: ImportError: harness /home/hemlix/GamingMaster-mh-tier2/runs/atari-2600-donkey-kong/search/candidates/0005/harnes |
| 0006 | 0006 | 0.000 | 6 | 0% | 494 | 7 | 0 / 4 (0 / 4 timed out) | 1% | 0% | 0.2 / 0 | 66.8 / 82 | ok |
| 0007 | 0007 | 100.000 | 6 | 0% | 0 | 0 | 0 / 0 | 0% | 0% | – | – | ok |
| 0008 | 0008 | 0.000 | 6 | 0% | 412 | 6 | 0 / 0 | 13% | 0% | 0.2 / 0 | – | ok |
| 0009 | 0009 | 250.000 | 6 | 0% | 0 | 0 | 0 / 0 | 0% | 0% | – | – | ok |
| 0010 | 0010 | 133.333 | 6 | 0% | 324 | 8 | 1 / 1 (1 / 0 timed out) | 10% | 0% | 0.2 / 0 | 25.3 / 25 | ok |
| 0011 | 0011 | – | 0 | – | 0 | 0 | 0 / 0 | – | – | – | – | ✗ fatal: proposer produced no harness.py |
| 0012 | 0012 | 316.667 | 6 | 0% | 0 | 0 | 0 / 0 | 0% | 0% | – | – | ok |
| 0013 | 0013 | 266.667 | 6 | 0% | 0 | 0 | 0 / 0 | 0% | 0% | – | – | ok |
| 0014 | 0014 | 0.000 | 6 | 0% | 0 | 0 | 0 / 0 | 0% | 0% | – | – | ok |
| 0015 | 0015 | 333.333 | 6 | 0% | 0 | 0 | 0 / 0 | 0% | 0% | – | – | ok |
| 0016 | 0016 | 300.000 | 6 | 0% | 48 | 0 | 0 / 0 | 1% | 0% | 0.2 / 0 | – | ok |
| 0017 | 0017 | 4000.000 | 6 | 0% | 0 | 0 | 0 / 0 | 0% | 0% | – | – | ok |
| 0018 | 0018 | 4133.333 | 6 | 0% | 250 | 0 | 3 / 0 (3 / 0 timed out) | 1% | 0% | 0.2 / 1 | – | ok |
| 0019 | 0019 | 233.333 | 6 | 0% | 0 | 0 | 0 / 0 | 0% | 0% | – | – | ok |
| 0020 | 0020 | 2250.000 | 6 | 0% | 0 | 0 | 0 / 0 | 0% | 0% | – | – | ok |
| 0021 | 0021 | 2833.333 | 6 | 0% | 1 | 3 | 0 / 0 | 0% | 0% | 0.1 / 0 | 9.2 / 16 | ok |
| 0022 | 0022 | 216.667 | 6 | 0% | 0 | 0 | 0 / 0 | 0% | 0% | – | – | ok |
| 0023 | 0023 | 1516.667 | 6 | 0% | 104 | 1 | 0 / 0 | 1% | 0% | 0.2 / 0 | 17.1 / 17 | ok |
| 0024 | 0024 | 2200.000 | 6 | 0% | 0 | 0 | 0 / 0 | 0% | 0% | – | – | ok |
| 0025 | 0025 | 4133.333 | 6 | 0% | 250 | 0 | 0 / 0 | 1% | 0% | 0.2 / 1 | – | ok |
| 0026 | 0026 | 0.000 | 6 | 0% | 18 | 6 | 0 / 0 | 1% | 0% | 0.2 / 0 | 11.9 / 17 | ok |

`Jev→act` / `Astra→act`: fraction of posted actions preceded by a fresh System-1 / System-2 answer since the previous action (reported, not used for selection). Latency columns are the median per-episode p50 and the worst observed call: design timeouts and pipelining from these measurements, not from assumptions.

Best so far: `0018` (mean score 4133.333)
Only candidates whose status is `ok` can be selected: a fatal, partial or erroneous evaluation cannot win, whatever its score. A used-up call budget or the deadline is not an error.

Episodes (score, how it ended, wall time; ✗ = error):
- `0000-tier2_jev_only`: seed 0: 0.00 game_done 47s · seed 1: 0.00 game_done 59s · seed 2: 0.00 game_done 42s · seed 3: 0.00 game_done 50s · seed 4: 0.00 game_done 53s · seed 5: 0.00 game_done 50s
- `0001-tier2_astra_only`: seed 0: 0.00 game_done 96s · seed 1: 0.00 game_done 83s · seed 2: 0.00 game_done 133s · seed 3: 0.00 game_done 226s · seed 4: 0.00 game_done 76s · seed 5: 0.00 game_done 92s
- `0002-tier2_idle`: seed 0: 0.00 deadline 900s · seed 1: 0.00 deadline 900s · seed 2: 0.00 deadline 900s · seed 3: 0.00 deadline 900s · seed 4: 0.00 deadline 900s · seed 5: 0.00 deadline 900s
- `0003` (refine of 0000-tier2_jev_only): seed 0: 0.00 call_budget 582s · seed 1: 0.00 call_budget 586s · seed 2: 0.00 call_budget 598s · seed 3: 0.00 call_budget 572s · seed 4: 0.00 call_budget 579s · seed 5: 0.00 call_budget 581s
- `0004` (rewrite of 0002-tier2_idle): seed 0: 0.00 deadline 900s · seed 1: 0.00 deadline 900s · seed 2: 0.00 deadline 900s · seed 3: 0.00 deadline 900s · seed 4: 0.00 deadline 900s · seed 5: 0.00 deadline 900s
- `0006` (rewrite of 0001-tier2_astra_only): seed 0: 0.00 deadline 900s · seed 1: 0.00 deadline 900s · seed 2: 0.00 deadline 900s · seed 3: 0.00 deadline 900s · seed 4: 0.00 deadline 900s · seed 5: 0.00 deadline 900s
- `0007` (crossover of 0004, 0003): seed 0: 100.00 harness_return 60s · seed 1: 100.00 harness_return 60s · seed 2: 100.00 harness_return 60s · seed 3: 100.00 harness_return 60s · seed 4: 100.00 harness_return 60s · seed 5: 100.00 harness_return 60s
- `0008` (explore of 0004): seed 0: 0.00 harness_return 36s · seed 1: 0.00 harness_return 25s · seed 2: 0.00 harness_return 36s · seed 3: 0.00 harness_return 36s · seed 4: 0.00 harness_return 45s · seed 5: 0.00 harness_return 36s
- `0009` (crossover of 0007, 0008): seed 0: 300.00 game_done 44s · seed 1: 100.00 harness_return 31s · seed 2: 300.00 game_done 40s · seed 3: 200.00 harness_return 32s · seed 4: 300.00 game_done 44s · seed 5: 300.00 game_done 44s
- `0010` (explore of 0008): seed 0: 100.00 harness_return 29s · seed 1: 200.00 harness_return 39s · seed 2: 200.00 harness_return 39s · seed 3: 100.00 harness_return 29s · seed 4: 200.00 harness_return 48s · seed 5: 0.00 harness_return 29s
- `0012` (explore of 0004): seed 0: 300.00 game_done 87s · seed 1: 300.00 game_done 33s · seed 2: 600.00 game_done 44s · seed 3: 200.00 game_done 87s · seed 4: 300.00 game_done 37s · seed 5: 200.00 deadline 900s
- `0013` (refine of 0009): seed 0: 300.00 game_done 44s · seed 1: 300.00 game_done 44s · seed 2: 200.00 harness_return 30s · seed 3: 200.00 harness_return 30s · seed 4: 300.00 game_done 44s · seed 5: 300.00 game_done 44s
- `0014` (rewrite of 0009): seed 0: 0.00 harness_return 24s · seed 1: 0.00 harness_return 35s · seed 2: 0.00 harness_return 24s · seed 3: 0.00 harness_return 24s · seed 4: 0.00 harness_return 24s · seed 5: 0.00 harness_return 24s
- `0015` (refine of 0012): seed 0: 300.00 game_done 87s · seed 1: 300.00 game_done 33s · seed 2: 600.00 game_done 44s · seed 3: 200.00 game_done 87s · seed 4: 300.00 game_done 37s · seed 5: 300.00 game_done 36s
- `0016` (rewrite of 0006): seed 0: 300.00 game_done 41s · seed 1: 300.00 harness_return 26s · seed 2: 300.00 game_done 40s · seed 3: 300.00 game_done 40s · seed 4: 300.00 game_done 37s · seed 5: 300.00 game_done 40s
- `0017` (crossover of 0012, 0002-tier2_idle): seed 0: 4000.00 harness_return 216s · seed 1: 4100.00 game_done 225s · seed 2: 4000.00 harness_return 216s · seed 3: 3900.00 harness_return 216s · seed 4: 4000.00 harness_return 216s · seed 5: 4000.00 harness_return 216s
- `0018` (rewrite of 0015): seed 0: 4100.00 harness_return 216s · seed 1: 4200.00 game_done 228s · seed 2: 4100.00 harness_return 216s · seed 3: 4000.00 harness_return 216s · seed 4: 4200.00 harness_return 216s · seed 5: 4200.00 harness_return 216s
- `0019` (crossover of 0017, 0014): seed 0: 200.00 harness_return 30s · seed 1: 200.00 harness_return 30s · seed 2: 300.00 game_done 34s · seed 3: 300.00 game_done 36s · seed 4: 200.00 harness_return 29s · seed 5: 200.00 harness_return 29s
- `0020` (explore of 0016): seed 0: 2300.00 game_done 129s · seed 1: 2300.00 game_done 134s · seed 2: 2200.00 game_done 135s · seed 3: 2300.00 game_done 129s · seed 4: 2300.00 game_done 130s · seed 5: 2100.00 game_done 130s
- `0021` (crossover of 0018, 0019): seed 0: 2300.00 game_done 133s · seed 1: 4100.00 game_done 230s · seed 2: 2100.00 game_done 130s · seed 3: 2200.00 game_done 135s · seed 4: 4100.00 game_done 229s · seed 5: 2200.00 game_done 140s
- `0022` (explore of 0019): seed 0: 200.00 game_done 38s · seed 1: 300.00 game_done 33s · seed 2: 200.00 game_done 38s · seed 3: 200.00 game_done 38s · seed 4: 200.00 game_done 39s · seed 5: 200.00 game_done 38s
- `0023` (refine of 0018): seed 0: 2100.00 harness_return 120s · seed 1: 200.00 game_done 44s · seed 2: 2100.00 harness_return 120s · seed 3: 2200.00 harness_return 120s · seed 4: 300.00 game_done 34s · seed 5: 2200.00 harness_return 120s
- `0024` (explore of 0013): seed 0: 2100.00 game_done 134s · seed 1: 2200.00 game_done 127s · seed 2: 2300.00 game_done 130s · seed 3: 2200.00 game_done 132s · seed 4: 2200.00 game_done 132s · seed 5: 2200.00 game_done 131s
- `0025` (refine of 0018): seed 0: 4100.00 harness_return 216s · seed 1: 4200.00 game_done 226s · seed 2: 4100.00 harness_return 216s · seed 3: 4000.00 harness_return 216s · seed 4: 4200.00 harness_return 216s · seed 5: 4200.00 harness_return 216s
- `0026` (rewrite of 0007): seed 0: 0.00 game_done 35s · seed 1: 0.00 game_done 35s · seed 2: 0.00 game_done 35s · seed 3: 0.00 game_done 35s · seed 4: 0.00 game_done 35s · seed 5: 0.00 game_done 35s

System-2 latency ≈ 41.8 s + 25.0 ms × output tokens (fit over 104 archived calls, r = 0.21; slowest 177 s): long answers are slow — size timeouts from this, not from short baseline answers.

Score composition (mean per episode, from the adaptor's score_breakdown):
- `0000-tier2_jev_only`: ale_reward 0.00, current_bonus 4400.00, lives_remaining 0.00, rivets_removed 0.00, screens_completed 0.00, terminated 1.00, truncated 0.00
- `0001-tier2_astra_only`: ale_reward 0.00, current_bonus 4400.00, lives_remaining 0.00, rivets_removed 0.00, screens_completed 0.00, terminated 1.00, truncated 0.00
- `0002-tier2_idle`: ale_reward 0.00, current_bonus 0.00, lives_remaining 2.00, rivets_removed 0.00, screens_completed 0.00, terminated 0.00, truncated 0.00
- `0003`: ale_reward 0.00, current_bonus 4350.00, lives_remaining 0.00, rivets_removed 0.00, screens_completed 0.00, terminated 0.00, truncated 0.00
- `0004`: ale_reward 0.00, current_bonus 4200.00, lives_remaining 1.00, rivets_removed 0.00, screens_completed 0.00, terminated 0.00, truncated 0.00
- `0006`: ale_reward 0.00, current_bonus 4233.33, lives_remaining 0.00, rivets_removed 0.00, screens_completed 0.00, terminated 0.00, truncated 0.00
- `0007`: ale_reward 100.00, current_bonus 4433.33, lives_remaining 0.00, rivets_removed 0.00, screens_completed 0.00, terminated 0.00, truncated 0.00
- `0008`: ale_reward 0.00, current_bonus 4433.33, lives_remaining 0.00, rivets_removed 0.00, screens_completed 0.00, terminated 0.00, truncated 0.00
- `0009`: ale_reward 250.00, current_bonus 4433.33, lives_remaining 0.00, rivets_removed 0.00, screens_completed 0.00, terminated 0.67, truncated 0.00
- `0010`: ale_reward 133.33, current_bonus 4516.67, lives_remaining 0.00, rivets_removed 0.00, screens_completed 0.00, terminated 0.00, truncated 0.00
- `0012`: ale_reward 316.67, current_bonus 4516.67, lives_remaining 0.00, rivets_removed 0.00, screens_completed 0.00, terminated 0.83, truncated 0.00
- `0013`: ale_reward 266.67, current_bonus 4366.67, lives_remaining 0.00, rivets_removed 0.00, screens_completed 0.00, terminated 0.67, truncated 0.00
- `0014`: ale_reward 0.00, current_bonus 4500.00, lives_remaining 0.00, rivets_removed 0.00, screens_completed 0.00, terminated 0.00, truncated 0.00
- `0015`: ale_reward 333.33, current_bonus 4566.67, lives_remaining 0.00, rivets_removed 0.00, screens_completed 0.00, terminated 1.00, truncated 0.00
- `0016`: ale_reward 300.00, current_bonus 4400.00, lives_remaining 0.00, rivets_removed 0.00, screens_completed 0.00, terminated 0.83, truncated 0.00
- `0017`: ale_reward 4000.00, current_bonus 766.67, lives_remaining 0.00, rivets_removed 0.00, screens_completed 0.00, terminated 0.17, truncated 0.00
- `0018`: ale_reward 4133.33, current_bonus 750.00, lives_remaining 0.00, rivets_removed 0.00, screens_completed 0.00, terminated 0.17, truncated 0.00
- `0019`: ale_reward 233.33, current_bonus 4483.33, lives_remaining 0.00, rivets_removed 0.00, screens_completed 0.00, terminated 0.33, truncated 0.00
- `0020`: ale_reward 2250.00, current_bonus 4516.67, lives_remaining 0.00, rivets_removed 0.00, screens_completed 0.00, terminated 1.00, truncated 0.00
- `0021`: ale_reward 2833.33, current_bonus 4416.67, lives_remaining 0.00, rivets_removed 0.00, screens_completed 0.00, terminated 1.00, truncated 0.00
- `0022`: ale_reward 216.67, current_bonus 4416.67, lives_remaining 0.00, rivets_removed 0.00, screens_completed 0.00, terminated 1.00, truncated 0.00
- `0023`: ale_reward 1516.67, current_bonus 1483.33, lives_remaining 0.00, rivets_removed 0.00, screens_completed 0.00, terminated 0.33, truncated 0.00
- `0024`: ale_reward 2200.00, current_bonus 4533.33, lives_remaining 0.00, rivets_removed 0.00, screens_completed 0.00, terminated 1.00, truncated 0.00
- `0025`: ale_reward 4133.33, current_bonus 766.67, lives_remaining 0.00, rivets_removed 0.00, screens_completed 0.00, terminated 0.17, truncated 0.00
- `0026`: ale_reward 0.00, current_bonus 4483.33, lives_remaining 0.00, rivets_removed 0.00, screens_completed 0.00, terminated 1.00, truncated 0.00

Validation seeds (finalists only; the final pick ranks training + validation together):
- `0017`: validation mean score 4050.000 over 2 episodes
- `0018`: validation mean score 4150.000 over 2 episodes
- `0025`: validation mean score 4150.000 over 2 episodes

Final pick: `0018` (training 4133.333333333333, validation 4150.0).

Held-out evaluations (not used for selection):
- `0018` / heldout (Atari 2600 Donkey Kong): mean score 4075.000 over 8 episodes

Ablation: `0018` with every runtime model call failing, on its training seeds: 4133.333 (with them: 4133.333).
