# Leaderboard

| # | candidate | parent | mean score | seeds median / min | episodes | stale ticks | S1 calls | S2 calls | failed S1/S2 | Jev→act | Astra→act | Jev p50/max s | Astra p50/max s | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0000 | 0000-tier1_astra_naive | – | 0.010 | 0.010 / 0.010 (= all seeds) | 4 | – | 0 | 79 | 0 / 0 | 0% | 100% | – | 58.2 / 182 | ok |
| 0001 | 0001 | 0000 | 0.335 | 0.090 / 0.060 | 4 | – | 0 | 7 | 0 / 3 (0 / 3 timed out) | 0% | 1% | – | 108.1 / 211 | ok |
| 0002 | 0002 | – | 0.568 | 1.100 / 0.030 | 4 | – | 0 | 6 | 0 / 0 | 0% | 1% | – | 68.1 / 68 | ok |
| 0003 | 0003 | 0000 | 0.033 | 0.030 / 0.030 | 4 | – | 0 | 4 | 0 / 0 | 0% | 1% | – | 91.3 / 151 | ok |
| 0004 | 0004 | 0002 | 0.312 | 0.060 / 0.040 | 4 | – | 0 | 14 | 0 / 0 | 0% | 3% | – | 64.3 / 138 | ok |
| 0005 | 0005 | 0002 | 0.562 | 1.100 / 0.020 | 4 | – | 0 | 9 | 0 / 0 | 0% | 2% | – | 57.2 / 74 | ok |
| 0006 | 0006 | 0002 | 0.315 | 0.090 / 0.030 | 4 | – | 0 | 7 | 0 / 0 | 0% | 1% | – | 68.6 / 76 | ok |
| 0007 | 0007 | 0002 | 0.570 | 1.100 / 0.040 | 4 | – | 0 | 20 | 0 / 0 | 0% | 4% | – | 55.3 / 92 | ok |
| 0008 | 0008 | 0006 | 0.565 | 1.100 / 0.030 | 4 | – | 0 | 11 | 0 / 0 | 0% | 3% | – | 53.6 / 87 | ok |
| 0009 | 0009 | 0005 | 0.583 | 1.100 / 0.060 | 4 | – | 0 | 15 | 0 / 0 | 0% | 3% | – | 53.8 / 61 | ok |
| 0010 | 0010 | 0009 | 0.575 | 1.100 / 0.040 | 4 | – | 0 | 20 | 0 / 0 | 0% | 5% | – | 50.0 / 134 | ok |
| 0011 | 0011 | 0005 | 0.320 | 0.070 / 0.050 | 4 | – | 0 | 17 | 0 / 0 | 0% | 4% | – | 58.3 / 77 | ok |
| 0012 | 0012 | 0002 | 0.318 | 0.070 / 0.040 | 4 | – | 0 | 27 | 0 / 0 | 0% | 8% | – | 59.0 / 95 | ok |
| 0013 | 0013 | 0009 | 0.840 | 1.100 / 0.060 | 4 | – | 0 | 9 | 0 / 0 | 0% | 1% | – | 53.9 / 80 | ok |
| 0014 | 0014 | 0005 | 0.312 | 0.080 / 0.030 | 4 | – | 0 | 16 | 0 / 0 | 0% | 4% | – | 59.4 / 65 | ok |
| 0015 | 0015 | 0002 | 0.318 | 0.070 / 0.040 | 4 | – | 0 | 10 | 0 / 0 | 0% | 1% | – | 62.5 / 67 | ok |
| 0016 | 0016 | 0013 | 0.595 | 1.100 / 0.090 | 4 | – | 0 | 16 | 0 / 0 | 0% | 3% | – | 47.3 / 68 | ok |
| 0017 | 0017 | 0014 | 0.573 | 1.100 / 0.030 | 4 | – | 0 | 19 | 0 / 0 | 0% | 4% | – | 49.2 / 82 | ok |
| 0018 | 0018 | 0017 | 0.568 | 1.100 / 0.030 | 4 | – | 0 | 19 | 0 / 0 | 0% | 5% | – | 36.8 / 55 | ok |

`seeds median / min`: the median and the worst per-seed score — a mean over a few seeds is noisy and can be carried by one lucky seed; an improvement that lifts the weak seeds is more likely to hold on unseen seeds. `(= all seeds)`: every seed scored the same, which is what a scenario-independent script does; expect it to break on scenarios it was not written for. `Jev→act` / `Astra→act`: fraction of posted actions preceded by a fresh System-1 / System-2 answer since the previous action (reported, not used for selection). Latency columns are the median per-episode p50 and the worst observed call: design timeouts and pipelining from these measurements, not from assumptions.

Best so far: `0009` (mean score 0.583)
Only candidates whose status is `ok` can be selected: a fatal, partial or erroneous evaluation cannot win, whatever its score. A used-up call budget or the deadline is not an error. `parent` is the candidate a proposal says it was derived from (`parent: NNNN` in proposal.md).

Validation (top candidates re-evaluated on unseen seeds; the final selection is the best train+validation mean among them):
- `0009`: validation 0.585 over 4 episodes; train+validation 0.584
- `0013`: validation 0.323 over 4 episodes; train+validation 0.581
- `0016`: validation 0.325 over 4 episodes; train+validation 0.460

Episodes (score, how it ended, wall time; ✗ = error):
- `0000-tier1_astra_naive`: seed 101: 0.01 deadline 1800s · seed 102: 0.01 deadline 1800s · seed 103: 0.01 deadline 1800s · seed 104: 0.01 deadline 1801s
- `0001`: seed 101: 0.09 game_done 1731s · seed 102: 1.10 game_done 1543s · seed 103: 0.06 game_done 1684s · seed 104: 0.09 game_done 1619s
- `0002`: seed 101: 1.10 game_done 288s · seed 102: 0.03 game_done 333s · seed 103: 0.04 game_done 420s · seed 104: 1.10 game_done 748s
- `0003`: seed 101: 0.03 game_done 329s · seed 102: 0.03 game_done 502s · seed 103: 0.04 game_done 256s · seed 104: 0.03 game_done 422s
- `0004`: seed 101: 1.10 game_done 942s · seed 102: 0.04 game_done 1186s · seed 103: 0.05 game_done 1288s · seed 104: 0.06 game_done 842s
- `0005`: seed 101: 1.10 game_done 328s · seed 102: 0.03 game_done 427s · seed 103: 0.02 game_done 998s · seed 104: 1.10 game_done 956s
- `0006`: seed 101: 1.10 game_done 567s · seed 102: 0.03 game_done 599s · seed 103: 0.04 game_done 762s · seed 104: 0.09 game_done 985s
- `0007`: seed 101: 0.04 game_done 502s · seed 102: 1.10 game_done 891s · seed 103: 1.10 game_done 1124s · seed 104: 0.04 game_done 334s
- `0008`: seed 101: 1.10 game_done 293s · seed 102: 0.03 game_done 279s · seed 103: 0.03 game_done 559s · seed 104: 1.10 game_done 641s
- `0009`: seed 101: 1.10 game_done 339s · seed 102: 0.06 game_done 266s · seed 103: 0.07 game_done 751s · seed 104: 1.10 game_done 1041s
- `0010`: seed 101: 1.10 game_done 1052s · seed 102: 0.04 game_done 1319s · seed 103: 0.06 game_done 1402s · seed 104: 1.10 game_done 1293s
- `0011`: seed 101: 1.10 game_done 966s · seed 102: 0.05 game_done 789s · seed 103: 0.07 game_done 538s · seed 104: 0.06 game_done 1206s
- `0012`: seed 101: 1.10 game_done 1013s · seed 102: 0.06 game_done 452s · seed 103: 0.04 game_done 853s · seed 104: 0.07 game_done 1013s
- `0013`: seed 101: 1.10 game_done 756s · seed 102: 1.10 game_done 638s · seed 103: 0.06 game_done 851s · seed 104: 1.10 game_done 508s
- `0014`: seed 101: 1.10 game_done 869s · seed 102: 0.04 game_done 442s · seed 103: 0.03 game_done 524s · seed 104: 0.08 game_done 1300s
- `0015`: seed 101: 1.10 game_done 649s · seed 102: 0.04 game_done 436s · seed 103: 0.07 game_done 1085s · seed 104: 0.06 game_done 619s
- `0016`: seed 101: 1.10 game_done 676s · seed 102: 1.10 game_done 829s · seed 103: 0.09 game_done 1165s · seed 104: 0.09 game_done 932s
- `0017`: seed 101: 0.06 game_done 959s · seed 102: 1.10 game_done 1274s · seed 103: 0.03 game_done 689s · seed 104: 1.10 game_done 857s
- `0018`: seed 101: 1.10 game_done 669s · seed 102: 0.04 game_done 589s · seed 103: 0.03 game_done 641s · seed 104: 1.10 game_done 1094s

System-2 latency ≈ 41.7 s + 28.1 ms × output tokens (fit over 263 archived calls, r = 0.79; slowest 211 s): long answers are slow — size timeouts from this, not from short baseline answers.

Score composition (mean per episode, from the adaptor's score_breakdown):
- `0000-tier1_astra_naive`: action_count 4.00, public_vp 1.00, true_vp 1.00, truncated 0.00, turns 0.00, win 0.00
- `0001`: action_count 333.00, public_vp 8.00, true_vp 8.50, truncated 0.00, turns 79.00, win 0.25
- `0002`: action_count 272.50, public_vp 6.25, true_vp 7.00, truncated 0.00, turns 70.00, win 0.50
- `0003`: action_count 306.25, public_vp 3.00, true_vp 3.25, truncated 0.00, turns 76.75, win 0.00
- `0004`: action_count 307.00, public_vp 6.25, true_vp 6.50, truncated 0.00, turns 77.25, win 0.25
- `0005`: action_count 262.25, public_vp 5.50, true_vp 6.25, truncated 0.00, turns 68.00, win 0.50
- `0006`: action_count 289.50, public_vp 5.75, true_vp 6.50, truncated 0.00, turns 73.50, win 0.25
- `0007`: action_count 315.50, public_vp 5.50, true_vp 7.00, truncated 0.00, turns 79.50, win 0.50
- `0008`: action_count 260.25, public_vp 5.75, true_vp 6.50, truncated 0.00, turns 67.25, win 0.50
- `0009`: action_count 293.25, public_vp 6.75, true_vp 8.25, truncated 0.00, turns 72.00, win 0.50
- `0010`: action_count 330.50, public_vp 6.75, true_vp 7.50, truncated 0.00, turns 83.25, win 0.50
- `0011`: action_count 334.25, public_vp 6.75, true_vp 7.00, truncated 0.00, turns 86.25, win 0.25
- `0012`: action_count 288.25, public_vp 5.75, true_vp 6.75, truncated 0.00, turns 75.75, win 0.25
- `0013`: action_count 324.25, public_vp 7.75, true_vp 9.00, truncated 0.00, turns 82.25, win 0.75
- `0014`: action_count 295.00, public_vp 5.50, true_vp 6.25, truncated 0.00, turns 76.75, win 0.25
- `0015`: action_count 317.00, public_vp 6.00, true_vp 6.75, truncated 0.00, turns 80.25, win 0.25
- `0016`: action_count 402.50, public_vp 9.00, true_vp 9.75, truncated 0.00, turns 102.25, win 0.50
- `0017`: action_count 316.25, public_vp 6.75, true_vp 7.50, truncated 0.00, turns 82.25, win 0.50
- `0018`: action_count 303.00, public_vp 6.25, true_vp 6.75, truncated 0.00, turns 78.25, win 0.50

Held-out evaluations (not used for selection):
- `0009` / heldout (Settlers of Catan (Catanatron AlphaBeta depth 2)): mean score 1.100 over 4 episodes
