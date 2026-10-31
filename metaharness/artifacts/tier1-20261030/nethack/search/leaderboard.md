# Leaderboard

| # | candidate | parent | mean score | seeds median / min | episodes | stale ticks | S1 calls | S2 calls | failed S1/S2 | Jev→act | Astra→act | Jev p50/max s | Astra p50/max s | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0000 | 0000-tier1_astra_naive | – | 0.000 | 0.000 / 0.000 (= all seeds) | 8 | – | 0 | 136 | 0 / 0 | 0% | 100% | – | 57.4 / 205 | ok |
| 0001 | 0001 | 0000 | 38.375 | 24.000 / 0.000 | 8 | – | 0 | 49 | 0 / 0 | 0% | 1% | – | 54.8 / 97 | ok |
| 0002 | 0002 | – | 25.625 | 12.000 / 0.000 | 8 | – | 0 | 27 | 0 / 0 | 0% | 1% | – | 58.0 / 135 | ok |
| 0003 | 0003 | – | 47.875 | 39.000 / 0.000 | 8 | – | 0 | 55 | 0 / 0 | 0% | 1% | – | 51.6 / 140 | ok |
| 0004 | 0004 | – | 8.625 | 2.000 / 0.000 | 8 | – | 0 | 34 | 0 / 0 | 0% | 1% | – | 65.7 / 161 | ok |
| 0005 | 0005 | 0002 | 48.375 | 17.000 / 0.000 | 8 | – | 0 | 7 | 0 / 0 | 0% | 0% | – | 53.0 / 63 | ok |
| 0006 | 0006 | 0005 | 71.250 | 62.000 / 0.000 | 8 | – | 0 | 9 | 0 / 0 | 0% | 0% | – | 72.6 / 81 | ok |
| 0007 | 0007 | 0006 | 98.375 | 123.000 / 0.000 | 8 | – | 0 | 0 | 0 / 0 | 0% | 0% | – | – | ok |
| 0008 | 0008 | 0006 | 58.750 | 62.000 / 0.000 | 8 | – | 0 | 4 | 0 / 0 | 0% | 0% | – | – | ok |
| 0009 | 0009 | 0006 | 91.375 | 100.000 / 0.000 | 8 | – | 0 | 16 | 0 / 0 | 0% | 0% | – | 60.6 / 106 | ok |
| 0010 | 0010 | 0009 | 70.625 | 70.000 / 0.000 | 8 | – | 0 | 13 | 0 / 0 | 0% | 0% | – | 36.8 / 87 | ok |
| 0011 | 0011 | 0007 | 84.500 | 82.000 / 0.000 | 8 | – | 0 | 0 | 0 / 0 | 0% | 0% | – | – | ok |
| 0012 | 0012 | 0007 | 120.000 | 177.000 / 0.000 | 8 | – | 0 | 0 | 0 / 0 | 0% | 0% | – | – | ok |
| 0013 | 0013 | 0012 | 120.000 | 177.000 / 0.000 | 8 | – | 0 | 0 | 0 / 0 | 0% | 0% | – | – | ok |
| 0014 | 0014 | 0011 | 84.500 | 82.000 / 0.000 | 8 | – | 0 | 0 | 0 / 0 | 0% | 0% | – | – | ok |
| 0015 | 0015 | 0012 | 120.000 | 177.000 / 0.000 | 8 | – | 0 | 0 | 0 / 0 | 0% | 0% | – | – | ok |
| 0016 | 0016 | 0013 | 123.000 | 177.000 / 0.000 | 8 | – | 0 | 0 | 0 / 0 | 0% | 0% | – | – | ok |
| 0017 | 0017 | 0012 | 99.125 | 123.000 / 2.000 | 8 | – | 0 | 0 | 0 / 0 | 0% | 0% | – | – | ok |
| 0018 | 0018 | 0011 | 120.000 | 177.000 / 0.000 | 8 | – | 0 | 0 | 0 / 0 | 0% | 0% | – | – | ok |

`seeds median / min`: the median and the worst per-seed score — a mean over a few seeds is noisy and can be carried by one lucky seed; an improvement that lifts the weak seeds is more likely to hold on unseen seeds. `(= all seeds)`: every seed scored the same, which is what a scenario-independent script does; expect it to break on scenarios it was not written for. `Jev→act` / `Astra→act`: fraction of posted actions preceded by a fresh System-1 / System-2 answer since the previous action (reported, not used for selection). Latency columns are the median per-episode p50 and the worst observed call: design timeouts and pipelining from these measurements, not from assumptions.

Best so far: `0016` (mean score 123.000)
Only candidates whose status is `ok` can be selected: a fatal, partial or erroneous evaluation cannot win, whatever its score. A used-up call budget or the deadline is not an error. `parent` is the candidate a proposal says it was derived from (`parent: NNNN` in proposal.md).

Validation (top candidates re-evaluated on unseen seeds; the final selection is the best train+validation mean among them):
- `0012`: validation 77.875 over 8 episodes; train+validation 98.938
- `0013`: validation 77.875 over 8 episodes; train+validation 98.938
- `0015`: validation 77.875 over 8 episodes; train+validation 98.938
- `0016`: validation 77.875 over 8 episodes; train+validation 100.438
- `0018`: validation 77.875 over 8 episodes; train+validation 98.938

Episodes (score, how it ended, wall time; ✗ = error):
- `0000-tier1_astra_naive`: seed 1101: 0.00 deadline 1800s · seed 1102: 0.00 deadline 1800s · seed 1103: 0.00 deadline 1800s · seed 1104: 0.00 deadline 1800s · seed 1105: 0.00 deadline 1800s · seed 1106: 0.00 deadline 1800s · seed 1107: 0.00 deadline 1800s · seed 1108: 0.00 deadline 1800s
- `0001`: seed 1101: 4.00 deadline 1800s · seed 1102: 116.00 deadline 1800s · seed 1103: 0.00 game_done 1746s · seed 1104: 8.00 deadline 1800s · seed 1105: 81.00 deadline 1800s · seed 1106: 0.00 game_done 1765s · seed 1107: 74.00 deadline 1800s · seed 1108: 24.00 deadline 1800s
- `0002`: seed 1101: 4.00 deadline 1800s · seed 1102: 120.00 deadline 1800s · seed 1103: 50.00 deadline 1800s · seed 1104: 0.00 game_done 1689s · seed 1105: 17.00 deadline 1800s · seed 1106: 0.00 game_done 1366s · seed 1107: 2.00 deadline 1800s · seed 1108: 12.00 harness_return 1761s
- `0003`: seed 1101: 13.00 deadline 1800s · seed 1102: 88.00 deadline 1800s · seed 1103: 155.00 deadline 1800s · seed 1104: 39.00 deadline 1800s · seed 1105: 23.00 harness_return 1775s · seed 1106: 57.00 deadline 1800s · seed 1107: 0.00 game_done 1751s · seed 1108: 8.00 deadline 1800s
- `0004`: seed 1101: 0.00 game_done 1779s · seed 1102: 0.00 deadline 1800s · seed 1103: 54.00 deadline 1800s · seed 1104: 0.00 game_done 1022s · seed 1105: 9.00 deadline 1800s · seed 1106: 0.00 deadline 1800s · seed 1107: 2.00 deadline 1800s · seed 1108: 4.00 deadline 1800s
- `0005`: seed 1101: 4.00 harness_return 163s · seed 1102: 58.00 harness_return 1002s · seed 1103: 123.00 harness_return 7s · seed 1104: 0.00 game_done 7s · seed 1105: 17.00 harness_return 123s · seed 1106: 0.00 harness_return 332s · seed 1107: 177.00 harness_return 329s · seed 1108: 8.00 harness_return 302s
- `0006`: seed 1101: 188.00 deadline 1800s · seed 1102: 62.00 harness_return 1279s · seed 1103: 123.00 harness_return 9s · seed 1104: 0.00 game_done 9s · seed 1105: 18.00 harness_return 6s · seed 1106: 117.00 harness_return 1423s · seed 1107: 62.00 harness_return 1s · seed 1108: 0.00 deadline 1800s
- `0007`: seed 1101: 82.00 harness_return 18s · seed 1102: 202.00 harness_return 57s · seed 1103: 123.00 harness_return 13s · seed 1104: 0.00 game_done 17s · seed 1105: 21.00 harness_return 50s · seed 1106: 178.00 harness_return 38s · seed 1107: 177.00 harness_return 28s · seed 1108: 4.00 harness_return 14s
- `0008`: seed 1101: 140.00 harness_return 59s · seed 1102: 127.00 harness_return 37s · seed 1103: 123.00 harness_return 13s · seed 1104: 0.00 game_done 12s · seed 1105: 18.00 harness_return 18s · seed 1106: 0.00 game_done 44s · seed 1107: 62.00 harness_return 10s · seed 1108: 0.00 game_done 26s
- `0009`: seed 1101: 224.00 harness_return 1074s · seed 1102: 62.00 harness_return 385s · seed 1103: 123.00 harness_return 8s · seed 1104: 0.00 game_done 8s · seed 1105: 29.00 harness_return 1425s · seed 1106: 100.00 harness_return 971s · seed 1107: 177.00 harness_return 723s · seed 1108: 16.00 harness_return 225s
- `0010`: seed 1101: 70.00 harness_return 214s · seed 1102: 62.00 harness_return 124s · seed 1103: 123.00 harness_return 8s · seed 1104: 0.00 game_done 8s · seed 1105: 29.00 deadline 1800s · seed 1106: 100.00 harness_return 197s · seed 1107: 177.00 harness_return 1065s · seed 1108: 4.00 harness_return 351s
- `0011`: seed 1101: 82.00 harness_return 19s · seed 1102: 202.00 harness_return 59s · seed 1103: 123.00 harness_return 13s · seed 1104: 0.00 game_done 16s · seed 1105: 21.00 harness_return 49s · seed 1106: 178.00 harness_return 36s · seed 1107: 66.00 harness_return 19s · seed 1108: 4.00 harness_return 21s
- `0012`: seed 1101: 255.00 harness_return 24s · seed 1102: 202.00 harness_return 61s · seed 1103: 123.00 harness_return 13s · seed 1104: 0.00 game_done 19s · seed 1105: 21.00 harness_return 56s · seed 1106: 178.00 harness_return 41s · seed 1107: 177.00 harness_return 25s · seed 1108: 4.00 harness_return 16s
- `0013`: seed 1101: 255.00 harness_return 23s · seed 1102: 202.00 harness_return 53s · seed 1103: 123.00 harness_return 14s · seed 1104: 0.00 game_done 17s · seed 1105: 21.00 harness_return 47s · seed 1106: 178.00 harness_return 35s · seed 1107: 177.00 harness_return 26s · seed 1108: 4.00 harness_return 7s
- `0014`: seed 1101: 82.00 harness_return 19s · seed 1102: 202.00 harness_return 52s · seed 1103: 123.00 harness_return 13s · seed 1104: 0.00 game_done 16s · seed 1105: 21.00 harness_return 45s · seed 1106: 178.00 harness_return 34s · seed 1107: 66.00 harness_return 17s · seed 1108: 4.00 harness_return 18s
- `0015`: seed 1101: 255.00 harness_return 23s · seed 1102: 202.00 harness_return 54s · seed 1103: 123.00 harness_return 14s · seed 1104: 0.00 game_done 18s · seed 1105: 21.00 harness_return 51s · seed 1106: 178.00 harness_return 36s · seed 1107: 177.00 harness_return 28s · seed 1108: 4.00 harness_return 10s
- `0016`: seed 1101: 255.00 harness_return 31s · seed 1102: 202.00 harness_return 90s · seed 1103: 123.00 harness_return 17s · seed 1104: 0.00 game_done 20s · seed 1105: 21.00 harness_return 117s · seed 1106: 178.00 harness_return 62s · seed 1107: 177.00 harness_return 53s · seed 1108: 28.00 harness_return 55s
- `0017`: seed 1101: 255.00 harness_return 43s · seed 1102: 202.00 harness_return 99s · seed 1103: 123.00 harness_return 25s · seed 1104: 8.00 harness_return 25s · seed 1105: 21.00 harness_return 86s · seed 1106: 178.00 harness_return 62s · seed 1107: 2.00 harness_return 14s · seed 1108: 4.00 harness_return 26s
- `0018`: seed 1101: 255.00 harness_return 26s · seed 1102: 202.00 harness_return 54s · seed 1103: 123.00 harness_return 13s · seed 1104: 0.00 game_done 18s · seed 1105: 21.00 harness_return 49s · seed 1106: 178.00 harness_return 39s · seed 1107: 177.00 harness_return 28s · seed 1108: 4.00 harness_return 5s

System-2 latency ≈ 39.3 s + 31.3 ms × output tokens (fit over 312 archived calls, r = 0.77; slowest 205 s): long answers are slow — size timeouts from this, not from short baseline answers.

Score composition (mean per episode, from the adaptor's score_breakdown):
- `0000-tier1_astra_naive`: ascended 0.00, maximum_dungeon_depth 1.00, scenario_success 0.00, terminal_score 0.00
- `0001`: ascended 0.00, maximum_dungeon_depth 1.38, scenario_success 0.00, terminal_score 38.38
- `0002`: ascended 0.00, maximum_dungeon_depth 1.50, scenario_success 0.12, terminal_score 25.62
- `0003`: ascended 0.00, maximum_dungeon_depth 1.50, scenario_success 0.00, terminal_score 47.88
- `0004`: ascended 0.00, maximum_dungeon_depth 1.12, scenario_success 0.12, terminal_score 8.62
- `0005`: ascended 0.00, maximum_dungeon_depth 1.75, scenario_success 0.12, terminal_score 48.38
- `0006`: ascended 0.00, maximum_dungeon_depth 2.00, scenario_success 0.12, terminal_score 71.25
- `0007`: ascended 0.00, maximum_dungeon_depth 2.50, scenario_success 0.12, terminal_score 98.38
- `0008`: ascended 0.00, maximum_dungeon_depth 2.12, scenario_success 0.12, terminal_score 58.75
- `0009`: ascended 0.00, maximum_dungeon_depth 2.25, scenario_success 0.12, terminal_score 91.38
- `0010`: ascended 0.00, maximum_dungeon_depth 2.12, scenario_success 0.12, terminal_score 70.62
- `0011`: ascended 0.00, maximum_dungeon_depth 2.25, scenario_success 0.12, terminal_score 84.50
- `0012`: ascended 0.00, maximum_dungeon_depth 2.62, scenario_success 0.12, terminal_score 120.00
- `0013`: ascended 0.00, maximum_dungeon_depth 2.62, scenario_success 0.12, terminal_score 120.00
- `0014`: ascended 0.00, maximum_dungeon_depth 2.25, scenario_success 0.12, terminal_score 84.50
- `0015`: ascended 0.00, maximum_dungeon_depth 2.62, scenario_success 0.12, terminal_score 120.00
- `0016`: ascended 0.00, maximum_dungeon_depth 2.62, scenario_success 0.12, terminal_score 123.00
- `0017`: ascended 0.00, maximum_dungeon_depth 2.25, scenario_success 0.00, terminal_score 99.12
- `0018`: ascended 0.00, maximum_dungeon_depth 2.62, scenario_success 0.12, terminal_score 120.00

Held-out evaluations (not used for selection):
- `0016` / heldout (NetHack (NetHack Learning Environment)): mean score 114.000 over 8 episodes
