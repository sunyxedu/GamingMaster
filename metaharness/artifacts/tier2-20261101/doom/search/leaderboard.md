# Leaderboard

| # | candidate | mean score | episodes | stale ticks | S1 calls | S2 calls | failed S1/S2 | Jev→act | Astra→act | Jev p50/max s | Astra p50/max s | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0000 | 0000-tier2_jev_only | 4.439 | 12 | 83% | 24599 | 0 | 1 / 0 (1 / 0 timed out) | 100% | 0% | 0.2 / 2 | – | ok |
| 0001 | 0001-tier2_astra_only | 4.310 | 12 | 100% | 0 | 88 | 0 / 11 (0 / 2 timed out) | 0% | 100% | – | 74.0 / 213 | ok |
| 0002 | 0002-tier2_idle | 4.250 | 12 | 100% | 0 | 0 | 0 / 0 | – | – | – | – | ok |
| 0003 | 0003 | 4.791 | 12 | 97% | 5321 | 0 | 8 / 0 (8 / 0 timed out) | 100% | 0% | 0.8 / 5 | – | ok |
| 0004 | 0004 | 3.445 | 12 | 1% | 2402 | 21 | 18 / 0 (18 / 0 timed out) | 2% | 0% | 0.8 / 2 | 31.4 / 127 | ok |
| 0005 | 0005 | 4.272 | 12 | 1% | 625 | 9 | 6 / 0 (6 / 0 timed out) | 3% | 0% | 0.9 / 2 | 44.0 / 73 | ok |
| 0006 | 0006 | 6.957 | 12 | 1% | 2090 | 15 | 0 / 0 | 3% | 0% | 0.2 / 2 | 46.0 / 103 | ok |
| 0007 | 0007 | 5.555 | 12 | 90% | 913 | 13 | 0 / 0 | 46% | 0% | 0.2 / 1 | 67.6 / 76 | ok |
| 0008 | 0008 | 9.382 | 12 | 1% | 2758 | 20 | 2 / 3 (2 / 3 timed out) | 2% | 0% | 0.2 / 1 | 97.3 / 167 | ok |
| 0009 | 0009 | 6.680 | 12 | 1% | 2275 | 16 | 6 / 0 (6 / 0 timed out) | 2% | 0% | 0.2 / 0 | 73.8 / 131 | ok |
| 0010 | 0010 | 5.837 | 12 | 1% | 3524 | 14 | 5 / 0 (5 / 0 timed out) | 3% | 0% | 0.2 / 1 | 142.9 / 147 | ok |
| 0011 | 0011 | 10.958 | 12 | 0% | 3059 | 17 | 6 / 3 (6 / 3 timed out) | 2% | 0% | 0.2 / 1 | 107.4 / 162 | ok |
| 0012 | 0012 | – | 0 | – | 0 | 0 | 0 / 0 | – | – | – | – | ✗ fatal: proposer produced no harness.py |
| 0013 | 0013 | 8.177 | 12 | 1% | 2147 | 17 | 30 / 0 (30 / 0 timed out) | 2% | 0% | 0.2 / 1 | 80.4 / 157 | ok |
| 0014 | 0014 | 6.235 | 12 | 0% | 2147 | 15 | 58 / 1 (58 / 1 timed out) | 2% | 0% | 0.2 / 0 | 88.7 / 167 | ok |
| 0015 | 0015 | 9.324 | 12 | 1% | 3248 | 17 | 0 / 5 (0 / 4 timed out) | 2% | 0% | 0.2 / 1 | 112.9 / 137 | ok |
| 0016 | 0016 | 7.920 | 12 | 1% | 2397 | 19 | 0 / 3 (0 / 2 timed out) | 2% | 0% | 0.2 / 1 | 79.8 / 116 | ok |
| 0017 | 0017 | 7.104 | 12 | 0% | 1311 | 14 | 0 / 1 (0 / 1 timed out) | 2% | 0% | 0.2 / 1 | 121.7 / 127 | ok |
| 0018 | 0018 | 4.602 | 12 | 0% | 651 | 12 | 0 / 0 | 3% | 0% | 0.2 / 0 | 26.7 / 61 | ok |
| 0019 | 0019 | 5.709 | 12 | 0% | 665 | 14 | 0 / 0 | 3% | 0% | 0.2 / 0 | 79.0 / 95 | ok |
| 0020 | 0020 | 9.715 | 12 | 1% | 3709 | 18 | 0 / 0 | 2% | 0% | 0.2 / 1 | 84.8 / 101 | ok |

`Jev→act` / `Astra→act`: fraction of posted actions preceded by a fresh System-1 / System-2 answer since the previous action (reported, not used for selection). Latency columns are the median per-episode p50 and the worst observed call: design timeouts and pipelining from these measurements, not from assumptions.

Best so far: `0011` (mean score 10.958)
Only candidates whose status is `ok` can be selected: a fatal, partial or erroneous evaluation cannot win, whatever its score. A used-up call budget or the deadline is not an error.

Episodes (score, how it ended, wall time; ✗ = error):
- `0000-tier2_jev_only`: seed 1001: 1.53 game_done 12s · seed 1002: 4.32 call_budget 649s · seed 1003: 5.37 game_done 19s · seed 1004: 4.32 call_budget 654s · seed 1005: 1.68 game_done 9s · seed 1006: 4.32 call_budget 659s · seed 1007: 6.03 game_done 15s · seed 1008: 4.32 call_budget 655s · seed 1011: 8.23 game_done 23s · seed 1012: 4.32 call_budget 651s · seed 1013: 4.53 game_done 17s · seed 1014: 4.32 call_budget 651s
- `0001-tier2_astra_only`: seed 1001: 2.12 game_done 145s · seed 1002: 6.08 deadline 900s · seed 1003: 4.70 game_done 140s · seed 1004: 1.91 game_done 655s · seed 1005: 2.29 game_done 71s · seed 1006: 5.96 deadline 900s · seed 1007: 7.68 game_done 104s · seed 1008: 4.25 deadline 900s · seed 1011: 6.81 game_done 802s · seed 1012: 1.49 game_done 150s · seed 1013: 3.11 game_done 113s · seed 1014: 5.33 deadline 900s
- `0002-tier2_idle`: seed 1001: 4.25 deadline 900s · seed 1002: 4.25 deadline 900s · seed 1003: 4.25 deadline 900s · seed 1004: 4.25 deadline 900s · seed 1005: 4.25 deadline 900s · seed 1006: 4.25 deadline 900s · seed 1007: 4.25 deadline 900s · seed 1008: 4.25 deadline 900s · seed 1011: 4.25 deadline 900s · seed 1012: 4.25 deadline 900s · seed 1013: 4.25 deadline 900s · seed 1014: 4.25 deadline 900s
- `0003` (refine of 0000-tier2_jev_only): seed 1001: 3.20 game_done 21s · seed 1002: 6.64 deadline 900s · seed 1003: 4.93 game_done 30s · seed 1004: 4.25 deadline 900s · seed 1005: 3.95 game_done 23s · seed 1006: 6.06 deadline 900s · seed 1007: 4.65 game_done 39s · seed 1008: 2.44 game_done 58s · seed 1011: 6.35 game_done 57s · seed 1012: 4.25 deadline 900s · seed 1013: 4.15 game_done 25s · seed 1014: 6.63 deadline 900s
- `0004` (rewrite of 0001-tier2_astra_only): seed 1001: 1.68 game_done 24s · seed 1002: 6.91 deadline 900s · seed 1003: 2.41 game_done 5s · seed 1004: 2.01 game_done 25s · seed 1005: 3.39 game_done 36s · seed 1006: 7.74 deadline 900s · seed 1007: 1.51 game_done 6s · seed 1008: 5.61 game_done 99s · seed 1011: 1.58 game_done 4s · seed 1012: 2.45 game_done 34s · seed 1013: 2.04 game_done 22s · seed 1014: 4.01 game_done 881s
- `0005` (crossover of 0000-tier2_jev_only, 0002-tier2_idle): seed 1001: 1.55 game_done 12s · seed 1002: 2.15 game_done 18s · seed 1003: 4.19 game_done 18s · seed 1004: 5.50 game_done 75s · seed 1005: 3.35 game_done 23s · seed 1006: 4.19 game_done 24s · seed 1007: 4.62 game_done 28s · seed 1008: 6.90 game_done 179s · seed 1011: 3.64 game_done 18s · seed 1012: 5.32 game_done 31s · seed 1013: 4.36 game_done 30s · seed 1014: 5.49 game_done 139s
- `0006` (rewrite of 0001-tier2_astra_only): seed 1001: 4.01 game_done 34s · seed 1002: 8.73 game_done 70s · seed 1003: 7.80 game_done 20s · seed 1004: 10.97 deadline 900s · seed 1005: 5.36 game_done 53s · seed 1006: 4.13 game_done 328s · seed 1007: 6.28 game_done 28s · seed 1008: 3.01 game_done 30s · seed 1011: 12.95 game_done 296s · seed 1012: 6.50 game_done 77s · seed 1013: 7.39 game_done 81s · seed 1014: 6.34 game_done 43s
- `0007` (crossover of 0003, 0005): seed 1001: 3.60 game_done 15s · seed 1002: 4.23 game_done 23s · seed 1003: 6.91 game_done 32s · seed 1004: 2.59 game_done 28s · seed 1005: 6.54 game_done 48s · seed 1006: 10.77 game_done 160s · seed 1007: 3.98 game_done 21s · seed 1008: 4.92 game_done 36s · seed 1011: 5.38 game_done 17s · seed 1012: 8.17 game_done 103s · seed 1013: 6.11 game_done 32s · seed 1014: 3.46 game_done 22s
- `0008` (explore of 0004): seed 1001: 7.93 game_done 64s · seed 1002: 11.12 game_done 188s · seed 1003: 10.26 game_done 118s · seed 1004: 11.76 deadline 900s · seed 1005: 11.19 game_done 138s · seed 1006: 9.51 game_done 107s · seed 1007: 11.12 deadline 900s · seed 1008: 11.94 deadline 900s · seed 1011: 8.53 game_done 38s · seed 1012: 4.66 game_done 208s · seed 1013: 5.48 game_done 22s · seed 1014: 9.09 game_done 171s
- `0009` (crossover of 0008, 0007): seed 1001: 4.70 game_done 44s · seed 1002: 7.00 game_done 45s · seed 1003: 4.30 game_done 17s · seed 1004: 6.49 deadline 900s · seed 1005: 3.67 game_done 22s · seed 1006: 11.52 game_done 97s · seed 1007: 6.23 game_done 52s · seed 1008: 8.62 game_done 161s · seed 1011: 5.13 game_done 21s · seed 1012: 8.77 game_done 770s · seed 1013: 4.40 game_done 31s · seed 1014: 9.33 deadline 900s
- `0010` (explore of 0008): seed 1001: 8.38 deadline 900s · seed 1002: 3.37 game_done 24s · seed 1003: 4.50 game_done 26s · seed 1004: 4.74 game_done 45s · seed 1005: 3.99 game_done 34s · seed 1006: 6.79 game_done 121s · seed 1007: 3.64 game_done 18s · seed 1008: 8.75 deadline 900s · seed 1011: 8.09 deadline 900s · seed 1012: 3.92 game_done 39s · seed 1013: 7.68 game_done 86s · seed 1014: 6.19 game_done 60s
- `0011` (refine of 0008): seed 1001: 13.50 game_done 104s · seed 1002: 14.87 deadline 900s · seed 1003: 14.45 game_done 423s · seed 1004: 12.97 deadline 900s · seed 1005: 5.29 game_done 28s · seed 1006: 9.15 game_done 101s · seed 1007: 9.82 game_done 28s · seed 1008: 18.43 deadline 900s · seed 1011: 6.33 game_done 21s · seed 1012: 3.21 game_done 78s · seed 1013: 5.96 game_done 49s · seed 1014: 17.51 game_done 606s
- `0013` (refine of 0011): seed 1001: 6.45 game_done 52s · seed 1002: 5.55 game_done 42s · seed 1003: 5.57 game_done 19s · seed 1004: 9.88 game_done 528s · seed 1005: 6.66 game_done 73s · seed 1006: 8.01 game_done 84s · seed 1007: 15.23 game_done 840s · seed 1008: 12.03 deadline 900s · seed 1011: 8.24 game_done 23s · seed 1012: 8.06 game_done 283s · seed 1013: 4.12 game_done 23s · seed 1014: 8.32 game_done 104s
- `0014` (rewrite of 0009): seed 1001: 5.38 game_done 31s · seed 1002: 5.67 game_done 80s · seed 1003: 9.51 game_done 40s · seed 1004: 6.56 deadline 900s · seed 1005: 3.19 game_done 29s · seed 1006: 6.40 game_done 61s · seed 1007: 6.79 game_done 30s · seed 1008: 3.56 game_done 248s · seed 1011: 8.09 game_done 37s · seed 1012: 5.46 deadline 900s · seed 1013: 5.56 game_done 24s · seed 1014: 8.64 game_done 94s
- `0015` (refine of 0011): seed 1001: 9.26 game_done 131s · seed 1002: 7.67 game_done 82s · seed 1003: 4.26 game_done 24s · seed 1004: 11.72 deadline 900s · seed 1005: 8.27 game_done 89s · seed 1006: 17.47 game_done 750s · seed 1007: 8.22 game_done 104s · seed 1008: 8.97 game_done 821s · seed 1011: 5.83 game_done 33s · seed 1012: 4.75 game_done 263s · seed 1013: 10.11 game_done 127s · seed 1014: 15.36 deadline 900s
- `0016` (rewrite of 0006): seed 1001: 4.50 game_done 81s · seed 1002: 5.17 game_done 56s · seed 1003: 10.28 deadline 900s · seed 1004: 6.55 game_done 48s · seed 1005: 6.86 game_done 65s · seed 1006: 14.60 deadline 900s · seed 1007: 6.76 game_done 58s · seed 1008: 8.36 game_done 248s · seed 1011: 8.80 game_done 19s · seed 1012: 7.77 game_done 463s · seed 1013: 6.48 game_done 53s · seed 1014: 8.91 game_done 57s
- `0017` (crossover of 0011, 0003): seed 1001: 7.42 game_done 58s · seed 1002: 5.79 game_done 57s · seed 1003: 10.35 game_done 881s · seed 1004: 4.41 game_done 28s · seed 1005: 6.80 game_done 69s · seed 1006: 8.84 game_done 101s · seed 1007: 4.11 game_done 10s · seed 1008: 6.86 game_done 203s · seed 1011: 12.28 game_done 39s · seed 1012: 4.01 game_done 91s · seed 1013: 8.70 game_done 94s · seed 1014: 5.69 game_done 64s
- `0018` (rewrite of 0015): seed 1001: 4.29 game_done 43s · seed 1002: 4.78 game_done 47s · seed 1003: 4.82 game_done 54s · seed 1004: 2.98 game_done 69s · seed 1005: 5.54 game_done 52s · seed 1006: 4.31 game_done 55s · seed 1007: 8.00 game_done 102s · seed 1008: 2.68 game_done 43s · seed 1011: 6.81 game_done 125s · seed 1012: 2.51 game_done 44s · seed 1013: 4.57 game_done 27s · seed 1014: 3.92 game_done 28s
- `0019` (crossover of 0011, 0015): seed 1001: 11.09 game_done 126s · seed 1002: 6.99 game_done 74s · seed 1003: 6.88 game_done 28s · seed 1004: 1.63 game_done 31s · seed 1005: 4.72 game_done 24s · seed 1006: 8.67 game_done 159s · seed 1007: 3.59 game_done 21s · seed 1008: 1.99 game_done 42s · seed 1011: 4.59 game_done 19s · seed 1012: 3.99 game_done 94s · seed 1013: 6.32 game_done 43s · seed 1014: 8.05 game_done 84s
- `0020` (explore of 0016): seed 1001: 6.02 game_done 32s · seed 1002: 11.84 game_done 73s · seed 1003: 5.10 game_done 31s · seed 1004: 8.69 game_done 825s · seed 1005: 5.78 game_done 75s · seed 1006: 15.84 deadline 900s · seed 1007: 8.96 deadline 900s · seed 1008: 8.98 game_done 388s · seed 1011: 20.49 deadline 900s · seed 1012: 4.94 game_done 94s · seed 1013: 5.94 game_done 25s · seed 1014: 14.00 deadline 900s

System-2 latency ≈ 64.5 s + 7.6 ms × output tokens (fit over 173 archived calls, r = 0.07; slowest 213 s): long answers are slow — size timeouts from this, not from short baseline answers.

Score composition (mean per episode, from the adaptor's score_breakdown):
- `0000-tier2_jev_only`: armor 0.38, exit_success 0.00, health 1.64, item_progress 0.00, kill_progress 0.07, normalized_progress 1.00, remaining_ammo 0.52, secret_progress 0.00, time_tiebreak 0.91, total 4.44, vizdoom_reward 0.00
- `0001-tier2_astra_only`: armor 0.39, exit_success 0.00, health 1.08, item_progress 0.06, kill_progress 0.07, normalized_progress 1.52, remaining_ammo 0.45, secret_progress 0.00, time_tiebreak 0.87, total 4.31, vizdoom_reward 0.00
- `0002-tier2_idle`: armor 0.00, exit_success 0.00, health 3.00, item_progress 0.00, kill_progress 0.00, normalized_progress 0.00, remaining_ammo 0.50, secret_progress 0.00, time_tiebreak 0.75, total 4.25, vizdoom_reward 0.00
- `0003`: armor 0.39, exit_success 0.00, health 1.40, item_progress 0.11, kill_progress 0.06, normalized_progress 1.57, remaining_ammo 0.54, secret_progress 0.00, time_tiebreak 0.89, total 4.79, vizdoom_reward 0.00
- `0004`: armor 0.14, exit_success 0.00, health 0.74, item_progress 0.12, kill_progress 0.02, normalized_progress 1.16, remaining_ammo 0.48, secret_progress 0.00, time_tiebreak 0.93, total 3.44, vizdoom_reward 0.00
- `0005`: armor 0.62, exit_success 0.00, health 0.28, item_progress 0.07, kill_progress 0.09, normalized_progress 1.84, remaining_ammo 0.54, secret_progress 0.00, time_tiebreak 0.99, total 4.27, vizdoom_reward 0.00
- `0006`: armor 0.44, exit_success 0.00, health 0.31, item_progress 0.24, kill_progress 0.21, normalized_progress 4.80, remaining_ammo 0.44, secret_progress 0.03, time_tiebreak 0.95, total 6.96, vizdoom_reward 0.00
- `0007`: armor 0.55, exit_success 0.00, health 0.34, item_progress 0.13, kill_progress 0.14, normalized_progress 3.09, remaining_ammo 0.59, secret_progress 0.03, time_tiebreak 0.99, total 5.55, vizdoom_reward 0.00
- `0008`: armor 0.60, exit_success 0.00, health 0.80, item_progress 0.34, kill_progress 0.28, normalized_progress 6.39, remaining_ammo 0.68, secret_progress 0.02, time_tiebreak 0.91, total 9.38, vizdoom_reward 0.00
- `0009`: armor 0.55, exit_success 0.00, health 0.59, item_progress 0.21, kill_progress 0.18, normalized_progress 4.05, remaining_ammo 0.56, secret_progress 0.03, time_tiebreak 0.93, total 6.68, vizdoom_reward 0.00
- `0010`: armor 0.46, exit_success 0.00, health 0.43, item_progress 0.20, kill_progress 0.15, normalized_progress 3.66, remaining_ammo 0.35, secret_progress 0.05, time_tiebreak 0.93, total 5.84, vizdoom_reward 0.00
- `0011`: armor 0.64, exit_success 0.00, health 0.81, item_progress 0.36, kill_progress 0.37, normalized_progress 7.85, remaining_ammo 0.76, secret_progress 0.03, time_tiebreak 0.90, total 10.96, vizdoom_reward 0.00
- `0013`: armor 0.63, exit_success 0.00, health 0.48, item_progress 0.30, kill_progress 0.23, normalized_progress 5.48, remaining_ammo 0.66, secret_progress 0.05, time_tiebreak 0.93, total 8.18, vizdoom_reward 0.00
- `0014`: armor 0.38, exit_success 0.00, health 0.66, item_progress 0.19, kill_progress 0.18, normalized_progress 3.84, remaining_ammo 0.42, secret_progress 0.00, time_tiebreak 0.94, total 6.23, vizdoom_reward 0.00
- `0015`: armor 0.52, exit_success 0.00, health 0.51, item_progress 0.34, kill_progress 0.29, normalized_progress 6.69, remaining_ammo 0.70, secret_progress 0.03, time_tiebreak 0.90, total 9.32, vizdoom_reward 0.00
- `0016`: armor 0.61, exit_success 0.00, health 0.54, item_progress 0.26, kill_progress 0.23, normalized_progress 5.24, remaining_ammo 0.59, secret_progress 0.05, time_tiebreak 0.93, total 7.92, vizdoom_reward 0.00
- `0017`: armor 0.35, exit_success 0.00, health 0.41, item_progress 0.27, kill_progress 0.21, normalized_progress 4.80, remaining_ammo 0.58, secret_progress 0.00, time_tiebreak 0.96, total 7.10, vizdoom_reward 0.00
- `0018`: armor 0.32, exit_success 0.00, health 0.22, item_progress 0.16, kill_progress 0.10, normalized_progress 2.51, remaining_ammo 0.57, secret_progress 0.00, time_tiebreak 0.98, total 4.60, vizdoom_reward 0.00
- `0019`: armor 0.38, exit_success 0.00, health 0.18, item_progress 0.21, kill_progress 0.15, normalized_progress 3.58, remaining_ammo 0.60, secret_progress 0.00, time_tiebreak 0.98, total 5.71, vizdoom_reward 0.00
- `0020`: armor 0.49, exit_success 0.00, health 0.73, item_progress 0.38, kill_progress 0.29, normalized_progress 6.89, remaining_ammo 0.73, secret_progress 0.03, time_tiebreak 0.88, total 9.71, vizdoom_reward 0.00

Validation seeds (finalists only; the final pick ranks training + validation together):
- `0008`: validation mean score 6.503 over 4 episodes
- `0011`: validation mean score 10.962 over 4 episodes
- `0020`: validation mean score 11.320 over 4 episodes

Final pick: `0011` (training 10.958099867724867, validation 10.961875).

Held-out evaluations (not used for selection):
- `0011` / heldout (Doom (ViZDoom ASYNC_PLAYER)): mean score 6.572 over 16 episodes

Ablation: `0011` without runtime System-1 / System-2 on its training seeds: 11.618 (with them: 10.958).
