# Leaderboard

| # | candidate | mean score | episodes | stale ticks | S1 calls | S2 calls | failed S1/S2 | Jev→act | Astra→act | Jev p50/max s | Astra p50/max s | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0000 | 0000-tier2_jev_only | 7564.804 | 12 | 90% | 263 | 0 | 0 / 0 | 100% | 0% | 0.2 / 0 | – | ok |
| 0001 | 0001-tier2_astra_only | 0.000 | 12 | 100% | 0 | 12 | 0 / 0 | – | – | – | – | ok |
| 0002 | 0002-tier2_idle | 0.000 | 12 | 100% | 0 | 0 | 0 / 0 | – | – | – | – | ok |
| 0003 | 0003 | 889547.082 | 12 | 93% | 329 | 0 | 0 / 0 | 55% | 0% | 0.4 / 2 | – | ok |
| 0004 | 0004 | 1000000.000 | 12 | 0% | 0 | 0 | 0 / 0 | 0% | 0% | – | – | ok |
| 0005 | 0005 | 1000000.000 | 12 | 0% | 12 | 0 | 0 / 0 | 0% | 0% | 0.2 / 0 | – | ok |

`Jev→act` / `Astra→act`: fraction of posted actions preceded by a fresh System-1 / System-2 answer since the previous action (reported, not used for selection). Latency columns are the median per-episode p50 and the worst observed call: design timeouts and pipelining from these measurements, not from assumptions.

Best so far: `0004` (mean score 1000000.000)
Only candidates whose status is `ok` can be selected: a fatal, partial or erroneous evaluation cannot win, whatever its score. A used-up call budget or the deadline is not an error.

Episodes (score, how it ended, wall time; ✗ = error):
- `0000-tier2_jev_only`: seed 1001: 3378.38 game_done 4s · seed 1002: 0.00 game_done 3s · seed 1003: 0.00 game_done 1s · seed 1004: 5937.50 game_done 3s · seed 1005: 12971.70 game_done 9s · seed 1006: 26903.55 game_done 4s · seed 1007: 0.00 game_done 3s · seed 1008: 0.00 game_done 1s · seed 1009: 13590.03 game_done 3s · seed 1010: 6082.73 game_done 8s · seed 1011: 18987.34 game_done 3s · seed 1012: 2926.42 game_done 4s
- `0001-tier2_astra_only`: seed 1001: 0.00 game_done 3s · seed 1002: 0.00 game_done 3s · seed 1003: 0.00 game_done 1s · seed 1004: 0.00 game_done 2s · seed 1005: 0.00 game_done 8s · seed 1006: 0.00 game_done 3s · seed 1007: 0.00 game_done 3s · seed 1008: 0.00 game_done 1s · seed 1009: 0.00 game_done 3s · seed 1010: 0.00 game_done 7s · seed 1011: 0.00 game_done 3s · seed 1012: 0.00 game_done 4s
- `0002-tier2_idle`: seed 1001: 0.00 game_done 3s · seed 1002: 0.00 game_done 3s · seed 1003: 0.00 game_done 1s · seed 1004: 0.00 game_done 2s · seed 1005: 0.00 game_done 8s · seed 1006: 0.00 game_done 3s · seed 1007: 0.00 game_done 3s · seed 1008: 0.00 game_done 1s · seed 1009: 0.00 game_done 2s · seed 1010: 0.00 game_done 7s · seed 1011: 0.00 game_done 3s · seed 1012: 0.00 game_done 3s
- `0003` (refine of 0000-tier2_jev_only): seed 1001: 1000000.00 game_done 12s · seed 1002: 1000000.00 game_done 13s · seed 1003: 193891.10 game_done 4s · seed 1004: 877500.00 game_done 11s · seed 1005: 1000000.00 game_done 17s · seed 1006: 842131.98 game_done 11s · seed 1007: 1000000.00 game_done 16s · seed 1008: 1000000.00 game_done 8s · seed 1009: 761041.90 game_done 14s · seed 1010: 1000000.00 game_done 15s · seed 1011: 1000000.00 game_done 12s · seed 1012: 1000000.00 game_done 15s
- `0004` (rewrite of 0002-tier2_idle): seed 1001: 1000000.00 game_done 12s · seed 1002: 1000000.00 game_done 13s · seed 1003: 1000000.00 game_done 9s · seed 1004: 1000000.00 game_done 11s · seed 1005: 1000000.00 game_done 17s · seed 1006: 1000000.00 game_done 10s · seed 1007: 1000000.00 game_done 16s · seed 1008: 1000000.00 game_done 7s · seed 1009: 1000000.00 game_done 14s · seed 1010: 1000000.00 game_done 15s · seed 1011: 1000000.00 game_done 11s · seed 1012: 1000000.00 game_done 14s
- `0005` (crossover of 0000-tier2_jev_only, 0002-tier2_idle): seed 1001: 1000000.00 game_done 12s · seed 1002: 1000000.00 game_done 13s · seed 1003: 1000000.00 game_done 9s · seed 1004: 1000000.00 game_done 11s · seed 1005: 1000000.00 game_done 17s · seed 1006: 1000000.00 game_done 10s · seed 1007: 1000000.00 game_done 16s · seed 1008: 1000000.00 game_done 7s · seed 1009: 1000000.00 game_done 14s · seed 1010: 1000000.00 game_done 15s · seed 1011: 1000000.00 game_done 11s · seed 1012: 1000000.00 game_done 14s

Score composition (mean per episode, from the adaptor's score_breakdown):
- `0000-tier2_jev_only`: accuracy_fraction 0.00, combo_fraction 0.01, health_fraction 0.00, normalized_score_fraction 0.01, survival 0.00
- `0001-tier2_astra_only`: accuracy_fraction 0.00, combo_fraction 0.00, health_fraction 0.00, normalized_score_fraction 0.00, survival 0.00
- `0002-tier2_idle`: accuracy_fraction 0.00, combo_fraction 0.00, health_fraction 0.00, normalized_score_fraction 0.00, survival 0.00
- `0003`: accuracy_fraction 0.96, combo_fraction 0.87, health_fraction 0.91, normalized_score_fraction 0.89, survival 0.92
- `0004`: accuracy_fraction 1.00, combo_fraction 1.00, health_fraction 1.00, normalized_score_fraction 1.00, survival 1.00
- `0005`: accuracy_fraction 1.00, combo_fraction 1.00, health_fraction 1.00, normalized_score_fraction 1.00, survival 1.00

Validation seeds (finalists only; the final pick ranks training + validation together):
- `0003`: validation mean score 770685.133 over 4 episodes
- `0004`: validation mean score 1000000.000 over 4 episodes
- `0005`: validation mean score 1000000.000 over 4 episodes

Final pick: `0004` (training 1000000.0, validation 1000000.0).

Held-out evaluations (not used for selection):
- `0004` / heldout (Quaver Keys (local .qua simulator)): mean score 1000000.000 over 16 episodes

Ablation: `0004` with every runtime model call failing, on its training seeds: 1000000.000 (with them: 1000000.000).
