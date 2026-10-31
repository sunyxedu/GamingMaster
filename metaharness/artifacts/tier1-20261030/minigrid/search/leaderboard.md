# Leaderboard

| # | candidate | parent | mean score | seeds median / min | episodes | stale ticks | S1 calls | S2 calls | failed S1/S2 | Jev→act | Astra→act | Jev p50/max s | Astra p50/max s | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0000 | 0000-tier1_astra_naive | – | 0.658 | 0.930 / 0.000 | 8 | – | 0 | 71 | 0 / 0 | 0% | 100% | – | 51.1 / 102 | ok |
| 0001 | 0001 | 0000 | 0.836 | 0.916 / 0.438 | 8 | – | 0 | 31 | 0 / 0 | 0% | 23% | – | 70.0 / 127 | ok |
| 0002 | 0002 | – | 0.918 | 0.963 / 0.817 | 8 | – | 0 | 2 | 0 / 0 | 0% | 3% | – | 73.6 / 74 | ok |
| 0003 | 0003 | 0000 | 0.863 | 0.944 / 0.620 | 8 | – | 0 | 26 | 0 / 0 | 0% | 23% | – | 68.4 / 129 | ok |
| 0004 | 0004 | 0002 | 0.907 | 0.963 / 0.730 | 8 | – | 0 | 0 | 0 / 0 | 0% | 0% | – | – | ok |
| 0005 | 0005 | 0003 | 0.911 | 0.963 / 0.775 | 8 | – | 0 | 0 | 0 / 0 | 0% | 0% | – | – | ok |
| 0006 | 0006 | 0003 | 0.907 | 0.963 / 0.730 | 8 | – | 0 | 0 | 0 / 0 | 0% | 0% | – | – | ok |
| 0007 | 0007 | 0002 | 0.918 | 0.963 / 0.817 | 8 | – | 0 | 0 | 0 / 0 | 0% | 0% | – | – | ok |
| 0008 | 0008 | 0005 | 0.910 | 0.953 / 0.775 | 8 | – | 0 | 6 | 0 / 0 | 0% | 4% | – | 84.3 / 135 | ok |
| 0009 | 0009 | 0005 | 0.803 | 0.963 / 0.000 | 8 | – | 0 | 0 | 0 / 0 | 0% | 0% | – | – | ok |
| 0010 | 0010 | 0007 | 0.918 | 0.963 / 0.817 | 8 | – | 0 | 0 | 0 / 0 | 0% | 0% | – | – | ok |
| 0011 | 0011 | 0009 | 0.905 | 0.963 / 0.803 | 8 | – | 0 | 0 | 0 / 0 | 0% | 0% | – | – | ok |
| 0012 | 0012 | 0005 | 0.918 | 0.963 / 0.817 | 8 | – | 0 | 0 | 0 / 0 | 0% | 0% | – | – | ok |
| 0013 | 0013 | – | – | – | – | – | – | – | – | – | – | – | – | not evaluated |
| 0014 | 0014 | – | – | – | – | – | – | – | – | – | – | – | – | not evaluated |
| 0015 | 0015 | – | – | – | – | – | – | – | – | – | – | – | – | not evaluated |

`seeds median / min`: the median and the worst per-seed score — a mean over a few seeds is noisy and can be carried by one lucky seed; an improvement that lifts the weak seeds is more likely to hold on unseen seeds. `(= all seeds)`: every seed scored the same, which is what a scenario-independent script does; expect it to break on scenarios it was not written for. `Jev→act` / `Astra→act`: fraction of posted actions preceded by a fresh System-1 / System-2 answer since the previous action (reported, not used for selection). Latency columns are the median per-episode p50 and the worst observed call: design timeouts and pipelining from these measurements, not from assumptions.

Best so far: `0002` (mean score 0.918)
Only candidates whose status is `ok` can be selected: a fatal, partial or erroneous evaluation cannot win, whatever its score. A used-up call budget or the deadline is not an error. `parent` is the candidate a proposal says it was derived from (`parent: NNNN` in proposal.md).

Validation (top candidates re-evaluated on unseen seeds; the final selection is the best train+validation mean among them):
- `0002`: validation 0.941 over 8 episodes; train+validation 0.929
- `0007`: validation 0.941 over 8 episodes; train+validation 0.929
- `0010`: validation 0.941 over 8 episodes; train+validation 0.929
- `0012`: validation 0.941 over 8 episodes; train+validation 0.929

Episodes (score, how it ended, wall time; ✗ = error):
- `0000-tier1_astra_naive`: seed 5: 0.95 game_done 1219s · seed 1: 0.00 deadline 1800s · seed 3: 0.80 game_done 642s · seed 6: 0.00 deadline 1800s · seed 14: 0.93 game_done 285s · seed 2: 0.68 game_done 1503s · seed 8: 0.95 game_done 1561s · seed 7: 0.96 game_done 1518s
- `0001`: seed 5: 0.96 game_done 849s · seed 1: 0.93 game_done 1358s · seed 3: 0.82 game_done 689s · seed 6: 0.78 game_done 786s · seed 14: 0.92 game_done 69s · seed 2: 0.44 game_done 921s · seed 8: 0.89 game_done 381s · seed 7: 0.96 game_done 1135s
- `0002`: seed 5: 0.97 game_done 0s · seed 1: 0.97 game_done 0s · seed 3: 0.82 game_done 123s · seed 6: 0.90 game_done 0s · seed 14: 0.93 game_done 0s · seed 2: 0.82 game_done 0s · seed 8: 0.96 game_done 0s · seed 7: 0.97 game_done 0s
- `0003`: seed 5: 0.97 game_done 105s · seed 1: 0.96 game_done 474s · seed 3: 0.82 game_done 376s · seed 6: 0.72 game_done 893s · seed 14: 0.93 game_done 44s · seed 2: 0.62 game_done 1203s · seed 8: 0.94 game_done 117s · seed 7: 0.94 game_done 1267s
- `0004`: seed 5: 0.97 game_done 0s · seed 1: 0.97 game_done 0s · seed 3: 0.73 game_done 0s · seed 6: 0.90 game_done 0s · seed 14: 0.93 game_done 0s · seed 2: 0.82 game_done 0s · seed 8: 0.96 game_done 0s · seed 7: 0.97 game_done 0s
- `0005`: seed 5: 0.97 game_done 0s · seed 1: 0.96 game_done 0s · seed 3: 0.78 game_done 0s · seed 6: 0.90 game_done 0s · seed 14: 0.93 game_done 0s · seed 2: 0.82 game_done 0s · seed 8: 0.96 game_done 0s · seed 7: 0.96 game_done 0s
- `0006`: seed 5: 0.97 game_done 0s · seed 1: 0.97 game_done 0s · seed 3: 0.73 game_done 0s · seed 6: 0.90 game_done 0s · seed 14: 0.93 game_done 0s · seed 2: 0.82 game_done 0s · seed 8: 0.96 game_done 0s · seed 7: 0.97 game_done 0s
- `0007`: seed 5: 0.97 game_done 0s · seed 1: 0.97 game_done 0s · seed 3: 0.82 game_done 0s · seed 6: 0.90 game_done 0s · seed 14: 0.93 game_done 0s · seed 2: 0.82 game_done 0s · seed 8: 0.96 game_done 0s · seed 7: 0.97 game_done 0s
- `0008`: seed 5: 0.97 game_done 0s · seed 1: 0.95 game_done 180s · seed 3: 0.78 game_done 180s · seed 6: 0.90 game_done 121s · seed 14: 0.93 game_done 0s · seed 2: 0.82 game_done 180s · seed 8: 0.96 game_done 0s · seed 7: 0.97 game_done 188s
- `0009`: seed 5: 0.97 game_done 0s · seed 1: 0.97 game_done 0s · seed 3: 0.82 game_done 0s · seed 6: 0.80 game_done 0s · seed 14: 0.93 game_done 0s · seed 2: 0.00 game_done 0s · seed 8: 0.96 game_done 0s · seed 7: 0.96 game_done 0s
- `0010`: seed 5: 0.97 game_done 0s · seed 1: 0.97 game_done 0s · seed 3: 0.82 game_done 0s · seed 6: 0.90 game_done 0s · seed 14: 0.93 game_done 0s · seed 2: 0.82 game_done 0s · seed 8: 0.96 game_done 0s · seed 7: 0.97 game_done 0s
- `0011`: seed 5: 0.97 game_done 0s · seed 1: 0.97 game_done 0s · seed 3: 0.82 game_done 0s · seed 6: 0.80 game_done 0s · seed 14: 0.93 game_done 0s · seed 2: 0.82 game_done 0s · seed 8: 0.96 game_done 0s · seed 7: 0.96 game_done 0s
- `0012`: seed 5: 0.97 game_done 0s · seed 1: 0.97 game_done 0s · seed 3: 0.82 game_done 0s · seed 6: 0.90 game_done 0s · seed 14: 0.93 game_done 0s · seed 2: 0.82 game_done 0s · seed 8: 0.96 game_done 0s · seed 7: 0.97 game_done 0s

System-2 latency ≈ 44.8 s + 28.2 ms × output tokens (fit over 131 archived calls, r = 0.84; slowest 135 s): long answers are slow — size timeouts from this, not from short baseline answers.

Score composition (mean per episode, from the adaptor's score_breakdown):
- `0000-tier1_astra_naive`: max_steps 155.00, raw_return 0.66, remaining_fraction 0.90, remaining_steps 146.38, score 0.66, steps_used 8.62, success 0.75, terminated 0.75, truncated 0.00
- `0001`: max_steps 155.00, raw_return 0.84, remaining_fraction 0.82, remaining_steps 138.00, score 0.84, steps_used 17.00, success 1.00, terminated 1.00, truncated 0.00
- `0002`: max_steps 155.00, raw_return 0.92, remaining_fraction 0.91, remaining_steps 146.50, score 0.92, steps_used 8.50, success 1.00, terminated 1.00, truncated 0.00
- `0003`: max_steps 155.00, raw_return 0.86, remaining_fraction 0.85, remaining_steps 140.88, score 0.86, steps_used 14.12, success 1.00, terminated 1.00, truncated 0.00
- `0004`: max_steps 155.00, raw_return 0.91, remaining_fraction 0.90, remaining_steps 146.00, score 0.91, steps_used 9.00, success 1.00, terminated 1.00, truncated 0.00
- `0005`: max_steps 155.00, raw_return 0.91, remaining_fraction 0.90, remaining_steps 145.75, score 0.91, steps_used 9.25, success 1.00, terminated 1.00, truncated 0.00
- `0006`: max_steps 155.00, raw_return 0.91, remaining_fraction 0.90, remaining_steps 146.00, score 0.91, steps_used 9.00, success 1.00, terminated 1.00, truncated 0.00
- `0007`: max_steps 155.00, raw_return 0.92, remaining_fraction 0.91, remaining_steps 146.50, score 0.92, steps_used 8.50, success 1.00, terminated 1.00, truncated 0.00
- `0008`: max_steps 155.00, raw_return 0.91, remaining_fraction 0.90, remaining_steps 145.50, score 0.91, steps_used 9.50, success 1.00, terminated 1.00, truncated 0.00
- `0009`: max_steps 155.00, raw_return 0.80, remaining_fraction 0.79, remaining_steps 139.00, score 0.80, steps_used 16.00, success 0.88, terminated 0.88, truncated 0.12
- `0010`: max_steps 155.00, raw_return 0.92, remaining_fraction 0.91, remaining_steps 146.50, score 0.92, steps_used 8.50, success 1.00, terminated 1.00, truncated 0.00
- `0011`: max_steps 155.00, raw_return 0.91, remaining_fraction 0.89, remaining_steps 145.38, score 0.91, steps_used 9.62, success 1.00, terminated 1.00, truncated 0.00
- `0012`: max_steps 155.00, raw_return 0.92, remaining_fraction 0.91, remaining_steps 146.50, score 0.92, steps_used 8.50, success 1.00, terminated 1.00, truncated 0.00

Held-out evaluations (not used for selection):
- `0002` / heldout (MiniGrid / BabyAI): mean score 0.907 over 8 episodes
