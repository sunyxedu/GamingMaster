# Leaderboard

| # | candidate | mean score | episodes | stale ticks | S1 calls | S2 calls | failed S1/S2 | Jev→act | Astra→act | Jev p50/max s | Astra p50/max s | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0000 | 0000-tier2_jev_only | 0.000 | 6 | 80% | 66 | 0 | 0 / 0 | 100% | 0% | 0.2 / 0 | – | ok |
| 0001 | 0001-tier2_astra_only | 0.000 | 6 | 100% | 0 | 6 | 0 / 0 | – | – | – | – | ok |
| 0002 | 0002-tier2_idle | 0.000 | 6 | 100% | 0 | 0 | 0 / 0 | – | – | – | – | ok |
| 0003 | 0003 | 0.000 | 6 | 95% | 13 | 0 | 0 / 0 | 53% | 0% | 1.0 / 2 | – | ok |
| 0004 | 0004 | 81.833 | 6 | 0% | 1197 | 6 | 1172 / 3 (1172 / 3 timed out) | 0% | 0% | 0.4 / 0 | 40.8 / 41 | ok |
| 0005 | 0005 | 716.000 | 6 | 0% | 11074 | 12 | 0 / 2 (0 / 2 timed out) | 6% | 0% | 0.3 / 2 | 58.6 / 105 | ok |
| 0006 | 0006 | 716.000 | 6 | 0% | 178 | 0 | 0 / 0 | 0% | 0% | 0.2 / 1 | – | ok |
| 0007 | 0007 | 632.667 | 6 | 1% | 8919 | 6 | 5 / 1 (5 / 1 timed out) | 6% | 0% | 0.2 / 1 | 71.2 / 81 | ok |
| 0008 | 0008 | 716.000 | 6 | 0% | 0 | 0 | 0 / 0 | 0% | 0% | – | – | ok |

`Jev→act` / `Astra→act`: fraction of posted actions preceded by a fresh System-1 / System-2 answer since the previous action (reported, not used for selection). Latency columns are the median per-episode p50 and the worst observed call: design timeouts and pipelining from these measurements, not from assumptions.

Best so far: `0006` (mean score 716.000)
Only candidates whose status is `ok` can be selected: a fatal, partial or erroneous evaluation cannot win, whatever its score. A used-up call budget or the deadline is not an error.

Episodes (score, how it ended, wall time; ✗ = error):
- `0000-tier2_jev_only`: seed 0: 0.00 game_done 2s · seed 1: 0.00 game_done 2s · seed 2: 0.00 game_done 2s · seed 3: 0.00 game_done 2s · seed 4: 0.00 game_done 2s · seed 5: 0.00 game_done 2s
- `0001-tier2_astra_only`: seed 0: 0.00 game_done 1s · seed 1: 0.00 game_done 1s · seed 2: 0.00 game_done 1s · seed 3: 0.00 game_done 1s · seed 4: 0.00 game_done 1s · seed 5: 0.00 game_done 1s
- `0002-tier2_idle`: seed 0: 0.00 game_done 1s · seed 1: 0.00 game_done 1s · seed 2: 0.00 game_done 1s · seed 3: 0.00 game_done 1s · seed 4: 0.00 game_done 1s · seed 5: 0.00 game_done 1s
- `0003` (refine of 0000-tier2_jev_only): seed 0: 0.00 game_done 2s · seed 1: 0.00 game_done 2s · seed 2: 0.00 game_done 2s · seed 3: 0.00 game_done 2s · seed 4: 0.00 game_done 2s · seed 5: 0.00 game_done 3s
- `0004` (rewrite of 0002-tier2_idle): seed 0: 215.00 game_done 272s · seed 1: 20.00 game_done 27s · seed 2: 85.00 game_done 108s · seed 3: 43.00 game_done 56s · seed 4: 109.00 game_done 138s · seed 5: 19.00 game_done 26s
- `0005` (crossover of 0000-tier2_jev_only, 0002-tier2_idle): seed 0: 716.00 deadline 900s · seed 1: 716.00 deadline 900s · seed 2: 716.00 deadline 900s · seed 3: 716.00 deadline 900s · seed 4: 716.00 deadline 900s · seed 5: 716.00 deadline 900s
- `0006` (rewrite of 0001-tier2_astra_only): seed 0: 716.00 deadline 900s · seed 1: 716.00 deadline 900s · seed 2: 716.00 deadline 900s · seed 3: 716.00 deadline 900s · seed 4: 716.00 deadline 900s · seed 5: 716.00 deadline 900s
- `0007` (crossover of 0005, 0004): seed 0: 716.00 deadline 900s · seed 1: 716.00 deadline 900s · seed 2: 716.00 deadline 900s · seed 3: 716.00 deadline 900s · seed 4: 216.00 game_done 273s · seed 5: 716.00 deadline 900s
- `0008` (explore of 0005): seed 0: 716.00 deadline 900s · seed 1: 716.00 deadline 900s · seed 2: 716.00 deadline 900s · seed 3: 716.00 deadline 900s · seed 4: 716.00 deadline 900s · seed 5: 716.00 deadline 900s

System-2 latency ≈ 60.2 s + -38.9 ms × output tokens (fit over 16 archived calls, r = -0.14; slowest 105 s): long answers are slow — size timeouts from this, not from short baseline answers.

Score composition (mean per episode, from the adaptor's score_breakdown):
- `0000-tier2_jev_only`: alive_frames 50.00, death_penalty 1.00, flap_count 42.67, max_score_seen 0.00, pipe_pairs_passed 0.00, pipe_reward 0.00, primary_score_pipe_pairs 0.00, shaped_reward -8.00, survival_shaping_reward 2.92
- `0001-tier2_astra_only`: alive_frames 31.00, death_penalty 1.00, flap_count 0.00, max_score_seen 0.00, pipe_pairs_passed 0.00, pipe_reward 0.00, primary_score_pipe_pairs 0.00, shaped_reward 2.00, survival_shaping_reward 3.00
- `0002-tier2_idle`: alive_frames 31.00, death_penalty 1.00, flap_count 0.00, max_score_seen 0.00, pipe_pairs_passed 0.00, pipe_reward 0.00, primary_score_pipe_pairs 0.00, shaped_reward 2.00, survival_shaping_reward 3.00
- `0003`: alive_frames 46.83, death_penalty 1.00, flap_count 1.33, max_score_seen 0.00, pipe_pairs_passed 0.00, pipe_reward 0.00, primary_score_pipe_pairs 0.00, shaped_reward 3.58, survival_shaping_reward 4.58
- `0004`: alive_frames 3130.17, death_penalty 1.00, flap_count 170.00, max_score_seen 81.83, pipe_pairs_passed 81.83, pipe_reward 81.83, primary_score_pipe_pairs 81.83, shaped_reward 385.57, survival_shaping_reward 304.73
- `0005`: alive_frames 26999.00, death_penalty 0.00, flap_count 1457.83, max_score_seen 716.00, pipe_pairs_passed 716.00, pipe_reward 716.00, primary_score_pipe_pairs 716.00, shaped_reward 3344.30, survival_shaping_reward 2628.30
- `0006`: alive_frames 26999.00, death_penalty 0.00, flap_count 1458.17, max_score_seen 716.00, pipe_pairs_passed 716.00, pipe_reward 716.00, primary_score_pipe_pairs 716.00, shaped_reward 3344.30, survival_shaping_reward 2628.30
- `0007`: alive_frames 23863.50, death_penalty 0.17, flap_count 1366.17, max_score_seen 632.67, pipe_pairs_passed 632.67, pipe_reward 632.67, primary_score_pipe_pairs 632.67, shaped_reward 2955.57, survival_shaping_reward 2323.07
- `0008`: alive_frames 26999.00, death_penalty 0.00, flap_count 1459.00, max_score_seen 716.00, pipe_pairs_passed 716.00, pipe_reward 716.00, primary_score_pipe_pairs 716.00, shaped_reward 3344.30, survival_shaping_reward 2628.30

Validation seeds (finalists only; the final pick ranks training + validation together):
- `0005`: validation mean score 716.000 over 2 episodes
- `0006`: validation mean score 716.000 over 2 episodes
- `0008`: validation mean score 716.000 over 2 episodes

Final pick: `0006` (training 716.0, validation 716.0).

Held-out evaluations (not used for selection):
- `0006` / heldout (Flappy Bird (flappy-bird-gymnasium)): mean score 716.000 over 8 episodes

Ablation: `0006` without runtime System-1 / System-2 on its training seeds: 716.000 (with them: 716.000).
