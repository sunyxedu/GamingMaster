# Leaderboard

| # | candidate | mean score | episodes | stale ticks | S1 calls | S2 calls | failed S1/S2 | Jev→act | Astra→act | Jev p50/max s | Astra p50/max s | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0000 | 0000-tier2_jev_only | 0.000 | 6 | 93% | 330 | 0 | 0 / 0 | 100% | 0% | 0.2 / 0 | – | ok |
| 0001 | 0001-tier2_astra_only | 0.000 | 6 | 100% | 0 | 6 | 0 / 0 | – | – | – | – | ok |
| 0002 | 0002-tier2_idle | 0.000 | 6 | 100% | 0 | 0 | 0 / 0 | – | – | – | – | ok |
| 0003 | 0003 | 984018.333 | 6 | 93% | 303 | 0 | 0 / 0 | 100% | 0% | 0.2 / 1 | – | ok |
| 0004 | 0004 | 873744.500 | 6 | 96% | 173 | 6 | 0 / 0 | 89% | 0% | 0.2 / 2 | – | ok |
| 0005 | 0005 | 955261.167 | 6 | 35% | 88 | 0 | 1 / 0 (1 / 0 timed out) | 3% | 0% | 0.2 / 1 | – | ok |
| 0006 | 0006 | 1000000.000 | 6 | 16% | 77 | 0 | 0 / 0 | 2% | 0% | 0.1 / 0 | – | ok |
| 0007 | 0007 | 998898.667 | 6 | 19% | 86 | 0 | 0 / 0 | 2% | 0% | 0.2 / 0 | – | ok |
| 0008 | 0008 | 1000000.000 | 6 | 1% | 76 | 0 | 0 / 0 | 2% | 0% | 0.2 / 0 | – | ok |

`Jev→act` / `Astra→act`: fraction of posted actions preceded by a fresh System-1 / System-2 answer since the previous action (reported, not used for selection). Latency columns are the median per-episode p50 and the worst observed call: design timeouts and pipelining from these measurements, not from assumptions.

Best so far: `0008` (mean score 1000000.000)
Only candidates whose status is `ok` can be selected: a fatal, partial or erroneous evaluation cannot win, whatever its score. A used-up call budget or the deadline is not an error.

Episodes (score, how it ended, wall time; ✗ = error):
- `0000-tier2_jev_only`: seed 100: 0.00 game_done 14s · seed 101: 0.00 game_done 11s · seed 102: 0.00 game_done 12s · seed 103: 0.00 game_done 13s · seed 104: 0.00 game_done 11s · seed 105: 0.00 game_done 13s
- `0001-tier2_astra_only`: seed 100: 0.00 game_done 14s · seed 101: 0.00 game_done 11s · seed 102: 0.00 game_done 12s · seed 103: 0.00 game_done 13s · seed 104: 0.00 game_done 11s · seed 105: 0.00 game_done 13s
- `0002-tier2_idle`: seed 100: 0.00 game_done 14s · seed 101: 0.00 game_done 11s · seed 102: 0.00 game_done 12s · seed 103: 0.00 game_done 13s · seed 104: 0.00 game_done 10s · seed 105: 0.00 game_done 13s
- `0003` (refine of 0000-tier2_jev_only): seed 100: 1000000.00 game_done 14s · seed 101: 1000000.00 game_done 10s · seed 102: 1000000.00 game_done 12s · seed 103: 904110.00 game_done 13s · seed 104: 1000000.00 game_done 10s · seed 105: 1000000.00 game_done 13s
- `0004` (rewrite of 0002-tier2_idle): seed 100: 1000000.00 game_done 14s · seed 101: 1000000.00 game_done 10s · seed 102: 771591.00 game_done 12s · seed 103: 833181.00 game_done 13s · seed 104: 800029.00 game_done 10s · seed 105: 837666.00 game_done 13s
- `0005` (crossover of 0000-tier2_jev_only, 0002-tier2_idle): seed 100: 1000000.00 game_done 14s · seed 101: 1000000.00 game_done 10s · seed 102: 1000000.00 game_done 12s · seed 103: 955635.00 game_done 13s · seed 104: 775932.00 game_done 10s · seed 105: 1000000.00 game_done 13s
- `0006` (rewrite of 0003): seed 100: 1000000.00 game_done 14s · seed 101: 1000000.00 game_done 10s · seed 102: 1000000.00 game_done 12s · seed 103: 1000000.00 game_done 13s · seed 104: 1000000.00 game_done 10s · seed 105: 1000000.00 game_done 13s
- `0007` (crossover of 0003, 0005): seed 100: 1000000.00 game_done 14s · seed 101: 1000000.00 game_done 10s · seed 102: 1000000.00 game_done 12s · seed 103: 993392.00 game_done 13s · seed 104: 1000000.00 game_done 10s · seed 105: 1000000.00 game_done 13s
- `0008` (explore of 0005): seed 100: 1000000.00 game_done 14s · seed 101: 1000000.00 game_done 10s · seed 102: 1000000.00 game_done 12s · seed 103: 1000000.00 game_done 13s · seed 104: 1000000.00 game_done 10s · seed 105: 1000000.00 game_done 13s

Score composition (mean per episode, from the adaptor's score_breakdown):
- `0000-tier2_jev_only`: accuracy 0.00, combo_reliability 0.00, hold_break_rate 0.00, miss_rate 1.00, native_score_v1 0.00, progress 1.00
- `0001-tier2_astra_only`: accuracy 0.00, combo_reliability 0.00, hold_break_rate 0.00, miss_rate 1.00, native_score_v1 0.00, progress 1.00
- `0002-tier2_idle`: accuracy 0.00, combo_reliability 0.00, hold_break_rate 0.00, miss_rate 1.00, native_score_v1 0.00, progress 1.00
- `0003`: accuracy 0.99, combo_reliability 0.99, hold_break_rate 0.00, miss_rate 0.00, native_score_v1 984018.33, progress 1.00
- `0004`: accuracy 0.96, combo_reliability 0.78, hold_break_rate 0.00, miss_rate 0.04, native_score_v1 873744.50, progress 1.00
- `0005`: accuracy 0.98, combo_reliability 0.98, hold_break_rate 0.00, miss_rate 0.02, native_score_v1 955261.17, progress 1.00
- `0006`: accuracy 1.00, combo_reliability 1.00, hold_break_rate 0.00, miss_rate 0.00, native_score_v1 1000000.00, progress 1.00
- `0007`: accuracy 1.00, combo_reliability 1.00, hold_break_rate 0.00, miss_rate 0.00, native_score_v1 998898.67, progress 1.00
- `0008`: accuracy 1.00, combo_reliability 1.00, hold_break_rate 0.00, miss_rate 0.00, native_score_v1 1000000.00, progress 1.00

Validation seeds (finalists only; the final pick ranks training + validation together):
- `0006`: validation mean score 1000000.000 over 2 episodes
- `0007`: validation mean score 1000000.000 over 2 episodes
- `0008`: validation mean score 1000000.000 over 2 episodes

Final pick: `0008` (training 1000000.0, validation 1000000.0).

Held-out evaluations (not used for selection):
- `0008` / heldout (osu!mania (stable native ScoreV1 simulator)): mean score 1000000.000 over 8 episodes

Ablation: `0008` with every runtime model call failing, on its training seeds: 0.000 (with them: 1000000.000).
