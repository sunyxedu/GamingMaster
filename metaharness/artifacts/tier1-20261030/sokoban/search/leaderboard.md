# Leaderboard

| # | candidate | parent | mean score | seeds median / min | episodes | stale ticks | S1 calls | S2 calls | failed S1/S2 | Jev→act | Astra→act | Jev p50/max s | Astra p50/max s | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0000 | 0000-tier1_astra_naive | – | 1167.266 | 197.220 / -0.570 | 10 | – | 0 | 193 | 0 / 0 | 0% | 100% | – | 50.1 / 178 | ok |
| 0001 | 0001 | 0000 | 10393.351 | 10393.140 / 10389.550 | 10 | – | 0 | 0 | 0 / 0 | 0% | 0% | – | – | ok |
| 0002 | 0002 | 0000 | 6304.215 | 10392.520 / 97.350 | 10 | – | 0 | 121 | 0 / 3 (0 / 3 timed out) | 0% | 34% | – | 48.8 / 227 | ok |
| 0003 | 0003 | 0000 | 7299.769 | 10392.630 / -21.500 | 10 | – | 0 | 59 | 0 / 9 (0 / 9 timed out) | 0% | 1% | – | 116.0 / 170 | ok |
| 0004 | 0004 | 0001 | 10393.351 | 10393.140 / 10389.550 | 10 | – | 0 | 0 | 0 / 0 | 0% | 0% | – | – | ok |
| 0005 | 0005 | – | 10393.360 | 10393.140 / 10389.560 | 10 | – | 0 | 0 | 0 / 0 | 0% | 0% | – | – | ok |
| 0006 | 0006 | 0002 | 6304.182 | 10392.520 / -6.420 | 10 | – | 0 | 108 | 0 / 7 (0 / 7 timed out) | 0% | 26% | – | 56.5 / 218 | ok |
| 0007 | 0007 | – | – | – | – | – | – | – | – | – | – | – | – | not evaluated |
| 0008 | 0008 | – | – | – | – | – | – | – | – | – | – | – | – | not evaluated |
| 0009 | 0009 | – | – | – | – | – | – | – | – | – | – | – | – | not evaluated |

`seeds median / min`: the median and the worst per-seed score — a mean over a few seeds is noisy and can be carried by one lucky seed; an improvement that lifts the weak seeds is more likely to hold on unseen seeds. `(= all seeds)`: every seed scored the same, which is what a scenario-independent script does; expect it to break on scenarios it was not written for. `Jev→act` / `Astra→act`: fraction of posted actions preceded by a fresh System-1 / System-2 answer since the previous action (reported, not used for selection). Latency columns are the median per-episode p50 and the worst observed call: design timeouts and pipelining from these measurements, not from assumptions.

Best so far: `0005` (mean score 10393.360)
Only candidates whose status is `ok` can be selected: a fatal, partial or erroneous evaluation cannot win, whatever its score. A used-up call budget or the deadline is not an error. `parent` is the candidate a proposal says it was derived from (`parent: NNNN` in proposal.md).

Validation (top candidates re-evaluated on unseen seeds; the final selection is the best train+validation mean among them):
- `0001`: validation 10393.361 over 10 episodes; train+validation 10393.356
- `0004`: validation 10393.361 over 10 episodes; train+validation 10393.356
- `0005`: validation 10393.390 over 10 episodes; train+validation 10393.375

Episodes (score, how it ended, wall time; ✗ = error):
- `0000-tier1_astra_naive`: seed 100: 10395.18 game_done 1531s · seed 101: 0.00 deadline 1800s · seed 102: -0.57 deadline 1800s · seed 103: 0.00 deadline 1800s · seed 104: 197.29 deadline 1800s · seed 105: 294.25 deadline 1800s · seed 106: 196.25 deadline 1800s · seed 107: 97.30 deadline 1800s · seed 108: 197.22 deadline 1800s · seed 109: 295.74 deadline 1800s
- `0001`: seed 100: 10395.25 game_done 1s · seed 101: 10392.56 game_done 1s · seed 102: 10393.14 game_done 1s · seed 103: 10392.14 game_done 1s · seed 104: 10395.25 game_done 0s · seed 105: 10392.69 game_done 0s · seed 106: 10393.12 game_done 1s · seed 107: 10389.55 game_done 5s · seed 108: 10395.08 game_done 1s · seed 109: 10394.73 game_done 1s
- `0002`: seed 100: 97.35 deadline 1800s · seed 101: 10392.52 game_done 1277s · seed 102: 10393.14 game_done 853s · seed 103: 10392.06 game_done 929s · seed 104: 10395.19 game_done 1701s · seed 105: 195.85 deadline 1800s · seed 106: 195.67 deadline 1800s · seed 107: 192.62 deadline 1800s · seed 108: 10394.06 game_done 1508s · seed 109: 10393.69 game_done 1491s
- `0003`: seed 100: 10395.15 game_done 664s · seed 101: 74.50 game_done 1799s · seed 102: 10393.12 game_done 1461s · seed 103: -21.50 game_done 1799s · seed 104: 10395.19 game_done 501s · seed 105: 10392.63 game_done 354s · seed 106: 192.51 deadline 1800s · seed 107: 10389.40 game_done 320s · seed 108: 10395.07 game_done 166s · seed 109: 10391.62 game_done 220s
- `0004`: seed 100: 10395.25 game_done 1s · seed 101: 10392.56 game_done 0s · seed 102: 10393.14 game_done 1s · seed 103: 10392.14 game_done 1s · seed 104: 10395.25 game_done 0s · seed 105: 10392.69 game_done 1s · seed 106: 10393.12 game_done 1s · seed 107: 10389.55 game_done 4s · seed 108: 10395.08 game_done 0s · seed 109: 10394.73 game_done 0s
- `0005`: seed 100: 10395.25 game_done 0s · seed 101: 10392.58 game_done 0s · seed 102: 10393.14 game_done 0s · seed 103: 10392.14 game_done 0s · seed 104: 10395.25 game_done 0s · seed 105: 10392.71 game_done 0s · seed 106: 10393.12 game_done 0s · seed 107: 10389.56 game_done 1s · seed 108: 10395.11 game_done 0s · seed 109: 10394.74 game_done 0s
- `0006`: seed 100: 10395.15 game_done 819s · seed 101: 10392.52 game_done 1155s · seed 102: -6.42 deadline 1800s · seed 103: 10392.08 game_done 1543s · seed 104: 10395.19 game_done 1123s · seed 105: 293.61 deadline 1800s · seed 106: 99.44 deadline 1800s · seed 107: 290.48 deadline 1800s · seed 108: 10395.05 game_done 896s · seed 109: 10394.72 game_done 573s

System-2 latency ≈ 48.0 s + 15.9 ms × output tokens (fit over 444 archived calls, r = 0.74; slowest 227 s): long answers are slow — size timeouts from this, not from short baseline answers.

Score composition (mean per episode, from the adaptor's score_breakdown):
- `0000-tier1_astra_naive`: action_cap_terminated 0.00, boxes_on_goals 1.70, pushes 5.10, solved 0.10, total_actions 18.40, walk_steps 13.30
- `0001`: action_cap_terminated 0.00, boxes_on_goals 4.00, pushes 12.60, solved 1.00, total_actions 34.90, walk_steps 22.30
- `0002`: action_cap_terminated 0.00, boxes_on_goals 3.10, pushes 10.90, solved 0.60, total_actions 33.50, walk_steps 22.60
- `0003`: action_cap_terminated 0.20, boxes_on_goals 3.10, pushes 11.70, solved 0.70, total_actions 438.10, walk_steps 426.40
- `0004`: action_cap_terminated 0.00, boxes_on_goals 4.00, pushes 12.60, solved 1.00, total_actions 34.90, walk_steps 22.30
- `0005`: action_cap_terminated 0.00, boxes_on_goals 4.00, pushes 12.60, solved 1.00, total_actions 34.00, walk_steps 21.40
- `0006`: action_cap_terminated 0.00, boxes_on_goals 3.10, pushes 10.90, solved 0.60, total_actions 36.80, walk_steps 25.90

Held-out evaluations (not used for selection):
- `0005` / heldout (Sokoban (Boxoban ASCII)): mean score 10392.443 over 10 episodes
