# Leaderboard

| # | candidate | mean score | episodes | stale ticks | S1 calls | S2 calls | failed S1/S2 | Jev→act | Astra→act | Jev p50/max s | Astra p50/max s | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0000 | 0000-tier2_jev_only | 0.167 | 6 | 78% | 235 | 0 | 0 / 0 | 100% | 0% | 0.2 / 0 | – | ok |
| 0001 | 0001-tier2_astra_only | 0.000 | 6 | 100% | 0 | 6 | 0 / 0 | – | – | – | – | ok |
| 0002 | 0002-tier2_idle | 0.000 | 6 | 100% | 0 | 0 | 0 / 0 | – | – | – | – | ok |
| 0003 | 0003 | 1.000 | 6 | 3% | 6 | 0 | 0 / 0 | 1% | 0% | 0.2 / 0 | – | ok |
| 0004 | 0004 | 1.000 | 6 | 0% | 6 | 0 | 0 / 0 | 1% | 0% | 0.2 / 0 | – | ok |
| 0005 | 0005 | 1.000 | 6 | 3% | 6 | 0 | 0 / 0 | 1% | 0% | 0.2 / 0 | – | ok |

`Jev→act` / `Astra→act`: fraction of posted actions preceded by a fresh System-1 / System-2 answer since the previous action (reported, not used for selection). Latency columns are the median per-episode p50 and the worst observed call: design timeouts and pipelining from these measurements, not from assumptions.

Best so far: `0004` (mean score 1.000)
Only candidates whose status is `ok` can be selected: a fatal, partial or erroneous evaluation cannot win, whatever its score. A used-up call budget or the deadline is not an error.

Episodes (score, how it ended, wall time; ✗ = error):
- `0000-tier2_jev_only`: seed 100: 0.12 game_done 7s · seed 101: 0.24 game_done 7s · seed 102: 0.07 game_done 7s · seed 103: 0.16 game_done 6s · seed 104: 0.22 game_done 6s · seed 105: 0.18 game_done 9s
- `0001-tier2_astra_only`: seed 100: 0.00 game_done 7s · seed 101: 0.00 game_done 7s · seed 102: 0.00 game_done 7s · seed 103: 0.00 game_done 6s · seed 104: 0.00 game_done 6s · seed 105: 0.00 game_done 8s
- `0002-tier2_idle`: seed 100: 0.00 game_done 7s · seed 101: 0.00 game_done 7s · seed 102: 0.00 game_done 7s · seed 103: 0.00 game_done 6s · seed 104: 0.00 game_done 6s · seed 105: 0.00 game_done 8s
- `0003` (refine of 0000-tier2_jev_only): seed 100: 1.00 game_done 7s · seed 101: 1.00 game_done 7s · seed 102: 1.00 game_done 7s · seed 103: 1.00 game_done 6s · seed 104: 1.00 game_done 6s · seed 105: 1.00 game_done 8s
- `0004` (rewrite of 0002-tier2_idle): seed 100: 1.00 game_done 7s · seed 101: 1.00 game_done 7s · seed 102: 1.00 game_done 7s · seed 103: 1.00 game_done 6s · seed 104: 1.00 game_done 6s · seed 105: 1.00 game_done 8s
- `0005` (crossover of 0000-tier2_jev_only, 0002-tier2_idle): seed 100: 1.00 game_done 7s · seed 101: 1.00 game_done 7s · seed 102: 1.00 game_done 7s · seed 103: 1.00 game_done 6s · seed 104: 1.00 game_done 6s · seed 105: 1.00 game_done 8s

Score composition (mean per episode, from the adaptor's score_breakdown):
- `0000-tier2_jev_only`: broken_holds 2.17, completion 1.00, judged_heads 33.33, max_combo 3.67, misses 22.00, rm_proxy_2025 1116.42, rm_proxy_normalized 0.14, score 0.17, total_heads 33.33, weighted_accuracy 0.17
- `0001-tier2_astra_only`: broken_holds 2.17, completion 1.00, judged_heads 33.33, max_combo 0.00, misses 33.33, rm_proxy_2025 0.00, rm_proxy_normalized 0.00, score 0.00, total_heads 33.33, weighted_accuracy 0.00
- `0002-tier2_idle`: broken_holds 2.17, completion 1.00, judged_heads 33.33, max_combo 0.00, misses 33.33, rm_proxy_2025 0.00, rm_proxy_normalized 0.00, score 0.00, total_heads 33.33, weighted_accuracy 0.00
- `0003`: broken_holds 0.00, completion 1.00, judged_heads 33.33, max_combo 33.33, misses 0.00, rm_proxy_2025 8406.90, rm_proxy_normalized 1.00, score 1.00, total_heads 33.33, weighted_accuracy 1.00
- `0004`: broken_holds 0.00, completion 1.00, judged_heads 33.33, max_combo 33.33, misses 0.00, rm_proxy_2025 8406.90, rm_proxy_normalized 1.00, score 1.00, total_heads 33.33, weighted_accuracy 1.00
- `0005`: broken_holds 0.00, completion 1.00, judged_heads 33.33, max_combo 33.33, misses 0.00, rm_proxy_2025 8406.90, rm_proxy_normalized 1.00, score 1.00, total_heads 33.33, weighted_accuracy 1.00

Validation seeds (finalists only; the final pick ranks training + validation together):
- `0003`: validation mean score 1.000 over 2 episodes
- `0004`: validation mean score 1.000 over 2 episodes
- `0005`: validation mean score 1.000 over 2 episodes

Final pick: `0004` (training 1.0, validation 1.0).

Held-out evaluations (not used for selection):
- `0004` / heldout (Malody Key (offline native .mc emulator)): mean score 1.000 over 8 episodes

Ablation: `0004` without runtime System-1 / System-2 on its training seeds: 1.000 (with them: 1.000).
