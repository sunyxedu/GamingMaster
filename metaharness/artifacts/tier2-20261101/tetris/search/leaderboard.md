# Leaderboard

| # | candidate | mean score | episodes | stale ticks | S1 calls | S2 calls | failed S1/S2 | Jev→act | Astra→act | Jev p50/max s | Astra p50/max s | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0000 | 0000-tier2_jev_only | 202.000 | 6 | 73% | 29 | 0 | 0 / 0 | 100% | 0% | 0.2 / 0 | – | ok |
| 0001 | 0001-tier2_astra_only | 113.333 | 6 | 100% | 0 | 12 | 0 / 0 | 0% | 100% | – | 44.9 / 70 | ok |
| 0002 | 0002-tier2_idle | 0.000 | 6 | 100% | 0 | 0 | 0 / 0 | – | – | – | – | ok |
| 0003 | 0003 | 335.000 | 6 | 49% | 31 | 0 | 0 / 0 | 25% | 0% | 0.2 / 0 | – | ok |
| 0004 | 0004 | 337.333 | 6 | 41% | 44 | 6 | 1 / 0 (1 / 0 timed out) | 20% | 0% | 0.2 / 0 | – | ok |
| 0005 | 0005 | – | 0 | – | 0 | 0 | 0 / 0 | – | – | – | – | ✗ fatal: proposer produced no harness.py |
| 0006 | 0006 | 49615.000 | 6 | 42% | 1200 | 6 | 101 / 3 (101 / 2 timed out) | 17% | 0% | 0.2 / 0 | 66.6 / 68 | ok |
| 0007 | 0007 | 50094.000 | 6 | 30% | 1125 | 6 | 26 / 2 (26 / 2 timed out) | 16% | 0% | 0.2 / 0 | 36.1 / 46 | ok |
| 0008 | 0008 | 671.667 | 6 | 26% | 41 | 0 | 8 / 0 (8 / 0 timed out) | 8% | 0% | 0.2 / 0 | – | ok |
| 0009 | 0009 | 31117.000 | 6 | 10% | 869 | 0 | 0 / 0 | 16% | 0% | 0.2 / 0 | – | ok |
| 0010 | 0010 | 49159.333 | 6 | 0% | 1194 | 6 | 0 / 3 (0 / 3 timed out) | 12% | 0% | 0.2 / 0 | 40.7 / 53 | ok |
| 0011 | 0011 | 48097.000 | 6 | 0% | 1196 | 6 | 5 / 4 (5 / 4 timed out) | 12% | 0% | 0.2 / 0 | 52.2 / 60 | ok |
| 0012 | 0012 | 47711.667 | 6 | 29% | 1165 | 6 | 4 / 5 (4 / 5 timed out) | 18% | 0% | 0.2 / 0 | 61.4 / 61 | ok |
| 0013 | 0013 | 48354.667 | 6 | 30% | 1172 | 6 | 3 / 6 (3 / 6 timed out) | 18% | 0% | 0.2 / 0 | – | ok |
| 0014 | 0014 | 60758.333 | 6 | 0% | 404 | 6 | 0 / 0 | 6% | 0% | 0.3 / 0 | – | ok |
| 0015 | 0015 | 59819.667 | 6 | 0% | 393 | 7 | 3 / 0 (3 / 0 timed out) | 6% | 0% | 0.3 / 1 | 28.3 / 41 | ok |
| 0016 | 0016 | – | 0 | – | 0 | 0 | 0 / 0 | – | – | – | – | ✗ fatal: duplicate of candidates/0014 (the same code, comments aside): not evaluated |
| 0017 | 0017 | 63216.667 | 6 | 0% | 141 | 7 | 5 / 0 (5 / 0 timed out) | 2% | 0% | 0.3 / 1 | 34.9 / 35 | ok |
| 0018 | 0018 | 52982.000 | 6 | 0% | 493 | 7 | 5 / 0 (5 / 0 timed out) | 8% | 0% | 0.3 / 1 | 41.9 / 42 | ok |
| 0019 | 0019 | 56215.667 | 6 | 0% | 167 | 10 | 2 / 4 (2 / 3 timed out) | 1% | 0% | 0.2 / 1 | 51.5 / 63 | ok |
| 0020 | 0020 | 61632.000 | 6 | 0% | 146 | 6 | 0 / 0 | 2% | 0% | 0.3 / 1 | – | ok |
| 0021 | 0021 | 59684.000 | 6 | 0% | 128 | 10 | 1 / 5 (1 / 5 timed out) | 1% | 0% | 0.3 / 1 | – | ✗ 1 episode error(s), first: 1 ctx.spawn task(s) still running 10s after run_episode returned (model calls are cancelled o |
| 0022 | 0022 | 62594.333 | 6 | 3% | 0 | 6 | 0 / 0 | 0% | 0% | – | – | ✗ 5 episode error(s), first: 1 ctx.spawn task(s) still running 10s after run_episode returned (model calls are cancelled o |
| 0023 | 0023 | 62916.333 | 6 | 0% | 91 | 6 | 1 / 0 (1 / 0 timed out) | 1% | 0% | 0.3 / 1 | – | ✗ 1 episode error(s), first: 1 ctx.spawn task(s) still running 10s after run_episode returned (model calls are cancelled o |

`Jev→act` / `Astra→act`: fraction of posted actions preceded by a fresh System-1 / System-2 answer since the previous action (reported, not used for selection). Latency columns are the median per-episode p50 and the worst observed call: design timeouts and pipelining from these measurements, not from assumptions.

Best so far: `0017` (mean score 63216.667)
Only candidates whose status is `ok` can be selected: a fatal, partial or erroneous evaluation cannot win, whatever its score. A used-up call budget or the deadline is not an error.

Episodes (score, how it ended, wall time; ✗ = error):
- `0000-tier2_jev_only`: seed 0: 202.00 game_done 1s · seed 1: 196.00 game_done 1s · seed 2: 210.00 game_done 1s · seed 3: 190.00 game_done 1s · seed 4: 202.00 game_done 1s · seed 5: 212.00 game_done 1s
- `0001-tier2_astra_only`: seed 0: 68.00 game_done 70s · seed 1: 146.00 game_done 27s · seed 2: 114.00 game_done 50s · seed 3: 112.00 game_done 41s · seed 4: 86.00 game_done 60s · seed 5: 154.00 game_done 30s
- `0002-tier2_idle`: seed 0: 0.00 game_done 107s · seed 1: 0.00 game_done 103s · seed 2: 0.00 game_done 111s · seed 3: 0.00 game_done 101s · seed 4: 0.00 game_done 107s · seed 5: 0.00 game_done 112s
- `0003` (refine of 0000-tier2_jev_only): seed 0: 340.00 game_done 2s · seed 1: 336.00 game_done 2s · seed 2: 368.00 game_done 2s · seed 3: 306.00 game_done 2s · seed 4: 346.00 game_done 2s · seed 5: 314.00 game_done 2s
- `0004` (rewrite of 0001-tier2_astra_only): seed 0: 286.00 game_done 2s · seed 1: 474.00 game_done 3s · seed 2: 292.00 game_done 3s · seed 3: 296.00 game_done 2s · seed 4: 338.00 game_done 3s · seed 5: 338.00 game_done 3s
- `0006` (rewrite of 0002-tier2_idle): seed 0: 49056.00 game_done 95s · seed 1: 48578.00 game_done 97s · seed 2: 50652.00 game_done 95s · seed 3: 50242.00 game_done 93s · seed 4: 50128.00 game_done 91s · seed 5: 49034.00 game_done 94s
- `0007` (crossover of 0004, 0003): seed 0: 53628.00 game_done 80s · seed 1: 51632.00 game_done 83s · seed 2: 50536.00 game_done 84s · seed 3: 48974.00 game_done 84s · seed 4: 47386.00 game_done 76s · seed 5: 48408.00 game_done 82s
- `0008` (explore of 0004): seed 0: 574.00 game_done 4s · seed 1: 644.00 game_done 4s · seed 2: 588.00 game_done 4s · seed 3: 1038.00 game_done 4s · seed 4: 610.00 game_done 4s · seed 5: 576.00 game_done 4s
- `0009` (crossover of 0007, 0008): seed 0: 53584.00 game_done 69s · seed 1: 49036.00 game_done 63s · seed 2: 50354.00 game_done 66s · seed 3: 5502.00 game_done 21s · seed 4: 13186.00 game_done 34s · seed 5: 15040.00 game_done 39s
- `0010` (explore of 0008): seed 0: 50514.00 game_done 87s · seed 1: 45276.00 game_done 85s · seed 2: 50244.00 game_done 81s · seed 3: 50416.00 game_done 84s · seed 4: 50086.00 game_done 82s · seed 5: 48420.00 game_done 86s
- `0011` (refine of 0007): seed 0: 50044.00 game_done 85s · seed 1: 47232.00 game_done 85s · seed 2: 49330.00 game_done 85s · seed 3: 43600.00 game_done 84s · seed 4: 49168.00 game_done 80s · seed 5: 49208.00 game_done 81s
- `0012` (explore of 0006): seed 0: 48208.00 game_done 77s · seed 1: 47494.00 game_done 76s · seed 2: 47024.00 game_done 77s · seed 3: 49780.00 game_done 78s · seed 4: 48506.00 game_done 78s · seed 5: 45258.00 game_done 76s
- `0013` (refine of 0007): seed 0: 46566.00 game_done 78s · seed 1: 48196.00 game_done 82s · seed 2: 47600.00 game_done 75s · seed 3: 51980.00 game_done 84s · seed 4: 49136.00 game_done 76s · seed 5: 46650.00 game_done 78s
- `0014` (rewrite of 0009): seed 0: 62292.00 game_done 62s · seed 1: 63076.00 game_done 63s · seed 2: 58220.00 game_done 63s · seed 3: 59064.00 game_done 62s · seed 4: 64956.00 game_done 63s · seed 5: 56942.00 game_done 62s
- `0015` (refine of 0014): seed 0: 61984.00 game_done 45s · seed 1: 58982.00 game_done 45s · seed 2: 57110.00 game_done 45s · seed 3: 60928.00 game_done 46s · seed 4: 66594.00 game_done 47s · seed 5: 53320.00 game_done 47s
- `0017` (crossover of 0014, 0006): seed 0: 66410.00 game_done 46s · seed 1: 65108.00 game_done 44s · seed 2: 62868.00 game_done 44s · seed 3: 70450.00 game_done 46s · seed 4: 57630.00 game_done 43s · seed 5: 56834.00 game_done 46s
- `0018` (rewrite of 0014): seed 0: 27344.00 game_done 31s · seed 1: 64718.00 game_done 45s · seed 2: 55404.00 game_done 45s · seed 3: 65912.00 game_done 43s · seed 4: 68876.00 game_done 45s · seed 5: 35638.00 game_done 41s
- `0019` (crossover of 0017, 0014): seed 0: 62120.00 game_done 114s · seed 1: 69782.00 game_done 87s · seed 2: 54504.00 game_done 61s · seed 3: 67058.00 game_done 83s · seed 4: 19520.00 game_done 57s · seed 5: 64310.00 game_done 108s
- `0020` (explore of 0015): seed 0: 59064.00 game_done 46s · seed 1: 66208.00 game_done 45s · seed 2: 61144.00 game_done 45s · seed 3: 63126.00 game_done 45s · seed 4: 58334.00 game_done 45s · seed 5: 61916.00 game_done 45s
- `0021` (crossover of 0017, 0019): seed 0: 68814.00 game_done 94s · seed 1: 52414.00 game_done 75s · seed 2: 62954.00 game_done 68s · seed 3: 54134.00 game_done 77s · seed 4: 62298.00 game_done 67s ✗ 1 ctx.spawn task(s) still running 10s after run_episode returned (model calls are cancelled or end … · seed 5: 57490.00 game_done 78s
- `0022` (explore of 0019): seed 0: 55578.00 game_done 67s ✗ 1 ctx.spawn task(s) still running 10s after run_episode returned (model calls are cancelled or end … · seed 1: 70896.00 game_done 67s ✗ 1 ctx.spawn task(s) still running 10s after run_episode returned (model calls are cancelled or end … · seed 2: 69114.00 game_done 67s ✗ 1 ctx.spawn task(s) still running 10s after run_episode returned (model calls are cancelled or end … · seed 3: 60892.00 game_done 67s ✗ 1 ctx.spawn task(s) still running 10s after run_episode returned (model calls are cancelled or end … · seed 4: 62896.00 game_done 67s ✗ 1 ctx.spawn task(s) still running 10s after run_episode returned (model calls are cancelled or end … · seed 5: 56190.00 game_done 58s
- `0023` (refine of 0017): seed 0: 63294.00 game_done 55s · seed 1: 61072.00 game_done 44s · seed 2: 62864.00 game_done 67s ✗ 1 ctx.spawn task(s) still running 10s after run_episode returned (model calls are cancelled or end … · seed 3: 71414.00 game_done 66s · seed 4: 58728.00 game_done 44s · seed 5: 60126.00 game_done 67s

System-2 latency ≈ 37.3 s + 121.1 ms × output tokens (fit over 25 archived calls, r = 0.36; slowest 70 s): long answers are slow — size timeouts from this, not from short baseline answers.

Score composition (mean per episode, from the adaptor's score_breakdown):
- `0000-tier2_jev_only`: aggregate_height 67.67, back_to_back_bonus 0.00, bumpiness 39.33, combo_points 0.00, drop_points 202.00, holes 23.67, illegal_actions 0.00, level 1.00, line_clear_points 0.00, lines 0.00, pieces 11.00, score 202.00, survival_s 0.71, top_out 1.00, top_out_height 19.67
- `0001-tier2_astra_only`: aggregate_height 67.67, back_to_back_bonus 0.00, bumpiness 39.33, combo_points 0.00, drop_points 113.33, holes 23.67, illegal_actions 0.00, level 1.00, line_clear_points 0.00, lines 0.00, pieces 11.00, score 113.33, survival_s 46.24, top_out 1.00, top_out_height 19.67
- `0002-tier2_idle`: aggregate_height 67.67, back_to_back_bonus 0.00, bumpiness 39.33, combo_points 0.00, drop_points 0.00, holes 23.67, illegal_actions 0.00, level 1.00, line_clear_points 0.00, lines 0.00, pieces 11.00, score 0.00, survival_s 106.50, top_out 1.00, top_out_height 19.67
- `0003`: aggregate_height 87.83, back_to_back_bonus 0.00, bumpiness 35.83, combo_points 0.00, drop_points 335.00, holes 27.83, illegal_actions 0.00, level 1.00, line_clear_points 0.00, lines 0.00, pieces 15.00, score 335.00, survival_s 1.58, top_out 1.00, top_out_height 19.67
- `0004`: aggregate_height 83.17, back_to_back_bonus 0.00, bumpiness 37.50, combo_points 0.00, drop_points 320.67, holes 26.83, illegal_actions 0.83, level 1.00, line_clear_points 16.67, lines 0.17, pieces 14.50, score 337.33, survival_s 2.43, top_out 1.00, top_out_height 19.33
- `0006`: aggregate_height 13.33, back_to_back_bonus 0.00, bumpiness 6.00, combo_points 4316.67, drop_points 6665.00, holes 0.00, illegal_actions 0.00, level 8.00, line_clear_points 38633.33, lines 78.67, pieces 200.00, score 49615.00, survival_s 94.02, top_out 0.00, top_out_height 5.50
- `0007`: aggregate_height 17.17, back_to_back_bonus 0.00, bumpiness 9.00, combo_points 4358.33, drop_points 6602.33, holes 0.50, illegal_actions 0.00, level 8.00, line_clear_points 39133.33, lines 78.33, pieces 200.00, score 50094.00, survival_s 81.37, top_out 0.00, top_out_height 6.17
- `0008`: aggregate_height 99.50, back_to_back_bonus 0.00, bumpiness 35.17, combo_points 0.00, drop_points 505.00, holes 26.83, illegal_actions 2.17, level 1.00, line_clear_points 166.67, lines 1.00, pieces 20.67, score 671.67, survival_s 3.68, top_out 1.00, top_out_height 19.50
- `0009`: aggregate_height 59.00, back_to_back_bonus 0.00, bumpiness 35.17, combo_points 2391.67, drop_points 4808.67, holes 4.33, illegal_actions 0.00, level 5.67, line_clear_points 23916.67, lines 54.00, pieces 148.67, score 31117.00, survival_s 48.53, top_out 0.50, top_out_height 12.50
- `0010`: aggregate_height 15.17, back_to_back_bonus 0.00, bumpiness 6.50, combo_points 5041.67, drop_points 6534.33, holes 0.17, illegal_actions 0.00, level 8.00, line_clear_points 37583.33, lines 78.50, pieces 200.00, score 49159.33, survival_s 84.27, top_out 0.00, top_out_height 6.17
- `0011`: aggregate_height 21.83, back_to_back_bonus 0.00, bumpiness 7.50, combo_points 3883.33, drop_points 6613.67, holes 0.17, illegal_actions 0.00, level 8.00, line_clear_points 37600.00, lines 77.83, pieces 200.00, score 48097.00, survival_s 83.40, top_out 0.00, top_out_height 6.00
- `0012`: aggregate_height 18.83, back_to_back_bonus 0.00, bumpiness 5.17, combo_points 3375.00, drop_points 6686.67, holes 0.50, illegal_actions 0.00, level 8.00, line_clear_points 37650.00, lines 78.17, pieces 200.00, score 47711.67, survival_s 77.03, top_out 0.00, top_out_height 5.33
- `0013`: aggregate_height 18.50, back_to_back_bonus 0.00, bumpiness 9.33, combo_points 2725.00, drop_points 6663.00, holes 0.17, illegal_actions 0.00, level 8.00, line_clear_points 38966.67, lines 78.17, pieces 200.00, score 48354.67, survival_s 78.93, top_out 0.00, top_out_height 5.67
- `0014`: aggregate_height 25.83, back_to_back_bonus 200.00, bumpiness 9.00, combo_points 4491.67, drop_points 6666.67, holes 0.83, illegal_actions 0.00, level 8.00, line_clear_points 49400.00, lines 77.50, pieces 200.00, score 60758.33, survival_s 45.54, top_out 0.00, top_out_height 7.17
- `0015`: aggregate_height 17.00, back_to_back_bonus 0.00, bumpiness 5.67, combo_points 4641.67, drop_points 6711.33, holes 0.33, illegal_actions 0.00, level 8.00, line_clear_points 48466.67, lines 78.33, pieces 200.00, score 59819.67, survival_s 45.75, top_out 0.00, top_out_height 6.17
- `0017`: aggregate_height 18.83, back_to_back_bonus 333.33, bumpiness 6.33, combo_points 4966.67, drop_points 6716.67, holes 0.50, illegal_actions 0.00, level 8.00, line_clear_points 51200.00, lines 78.17, pieces 200.00, score 63216.67, survival_s 44.66, top_out 0.00, top_out_height 6.17
- `0018`: aggregate_height 95.50, back_to_back_bonus 0.00, bumpiness 16.67, combo_points 7241.67, drop_points 5490.33, holes 9.17, illegal_actions 0.00, level 7.33, line_clear_points 40250.00, lines 67.17, pieces 189.50, score 52982.00, survival_s 41.10, top_out 0.33, top_out_height 14.50
- `0019`: aggregate_height 40.33, back_to_back_bonus 333.33, bumpiness 13.17, combo_points 5483.33, drop_points 5732.33, holes 2.00, illegal_actions 0.00, level 7.50, line_clear_points 44666.67, lines 71.17, pieces 187.50, score 56215.67, survival_s 84.78, top_out 0.17, top_out_height 11.50
- `0020`: aggregate_height 20.50, back_to_back_bonus 33.33, bumpiness 6.67, combo_points 4983.33, drop_points 6632.00, holes 0.50, illegal_actions 0.00, level 8.00, line_clear_points 49983.33, lines 78.00, pieces 200.00, score 61632.00, survival_s 44.88, top_out 0.00, top_out_height 7.33
- `0021`: aggregate_height 50.33, back_to_back_bonus 0.00, bumpiness 9.67, combo_points 5500.00, drop_points 6284.00, holes 3.67, illegal_actions 0.00, level 7.83, line_clear_points 47900.00, lines 75.33, pieces 200.00, score 59684.00, survival_s 72.07, top_out 0.00, top_out_height 10.00
- `0022`: aggregate_height 16.67, back_to_back_bonus 0.00, bumpiness 7.50, combo_points 5775.00, drop_points 6569.33, holes 0.00, illegal_actions 5.67, level 8.00, line_clear_points 50250.00, lines 78.33, pieces 200.00, score 62594.33, survival_s 49.34, top_out 0.00, top_out_height 7.33
- `0023`: aggregate_height 20.83, back_to_back_bonus 133.33, bumpiness 6.83, combo_points 5183.33, drop_points 6666.33, holes 0.83, illegal_actions 0.00, level 8.00, line_clear_points 50933.33, lines 78.00, pieces 200.00, score 62916.33, survival_s 44.62, top_out 0.00, top_out_height 6.67

Validation seeds (finalists only; the final pick ranks training + validation together):
- `0014`: validation mean score 61969.000 over 6 episodes
- `0017`: validation mean score 66181.333 over 6 episodes
- `0020`: validation mean score 66966.333 over 6 episodes

Final pick: `0017` (training 63216.666666666664, validation 66181.33333333333).

Held-out evaluations (not used for selection):
- `0017` / heldout (Tetris (Guideline-compatible marathon)): mean score 62259.143 over 7 episodes

Ablation: `0017` without runtime System-1 / System-2 on its training seeds: 61753.333 (with them: 63216.667).
