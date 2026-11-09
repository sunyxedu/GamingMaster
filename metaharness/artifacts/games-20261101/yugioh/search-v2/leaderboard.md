# Leaderboard

| # | candidate | parent | mean score | seeds median / min | episodes | stale ticks | S1 calls | S2 calls | failed S1/S2 | Jev→act | Astra→act | Jev p50/max s | Astra p50/max s | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0000 | 0000-tier1_astra_naive | – | -0.889 | -1.000 / -1.000 | 6 | – | 0 | 196 | 0 / 0 | 0% | 100% | – | 12.9 / 155 | ok |
| 0001 | 0001 | 0000 | -0.333 | -0.333 / -1.000 | 6 | – | 0 | 130 | 0 / 3 (0 / 3 timed out) | 0% | 0% | – | 43.8 / 87 | ok |
| 0002 | 0002 | – | -0.488 | -0.333 / -0.667 | 6 | – | 0 | 21 | 0 / 4 (0 / 4 timed out) | 0% | 0% | – | 42.8 / 62 | ok |
| 0003 | 0003 | – | 1.096 | 1.100 / 1.091 | 6 | – | 0 | 9 | 0 / 0 | 0% | 0% | – | 32.5 / 40 | ok |
| 0004 | 0004 | 0003 | 1.096 | 1.100 / 1.091 | 6 | – | 0 | 6 | 0 / 0 | 0% | 0% | – | – | ok |
| 0005 | 0005 | – | 1.096 | 1.100 / 1.091 | 6 | – | 0 | 9 | 0 / 1 (0 / 1 timed out) | 0% | 0% | – | 49.1 / 49 | ok |
| 0006 | 0006 | 0003 | 1.096 | 1.100 / 1.091 | 6 | – | 0 | 6 | 0 / 0 | 0% | 0% | – | – | ok |

`seeds median / min`: the median and the worst per-seed score — a mean over a few seeds is noisy and can be carried by one lucky seed; an improvement that lifts the weak seeds is more likely to hold on unseen seeds. `Jev→act` / `Astra→act`: fraction of posted actions preceded by a fresh System-1 / System-2 answer since the previous action (reported, not used for selection). Latency columns are the median per-episode p50 and the worst observed call: design timeouts and pipelining from these measurements, not from assumptions.

Best so far: `0004` (mean score 1.096)
Only candidates whose status is `ok` can be selected: a fatal, partial or erroneous evaluation cannot win, whatever its score. A used-up call budget or the deadline is not an error. `parent` is the candidate a proposal says it was derived from (`parent: NNNN` in proposal.md).

Validation (top candidates re-evaluated on unseen seeds; the final selection is the best train+validation mean among them):
- `0003`: validation 0.978 over 6 episodes; train+validation 1.037
- `0004`: validation 1.094 over 6 episodes; train+validation 1.095
- `0006`: validation 0.978 over 6 episodes; train+validation 1.037

Episodes (score, how it ended, wall time; ✗ = error):
- `0000-tier1_astra_naive`: seed 1101: -1.00 game_done 952s · seed 1102: -1.00 game_done 994s · seed 1103: -1.00 game_done 1139s · seed 1104: -0.67 harness_return 736s · seed 1105: -0.67 harness_return 644s · seed 1106: -1.00 game_done 1568s
- `0001`: seed 1101: -0.33 deadline 1800s · seed 1102: -0.33 deadline 1800s · seed 1103: -1.00 game_done 1777s · seed 1104: -0.33 deadline 1800s · seed 1105: 0.00 deadline 1800s · seed 1106: 0.00 deadline 1800s
- `0002`: seed 1101: -0.66 game_done 1800s · seed 1102: -0.30 game_done 761s · seed 1103: -0.33 game_done 870s · seed 1104: -0.67 game_done 771s · seed 1105: -0.65 game_done 1274s · seed 1106: -0.32 game_done 1193s
- `0003`: seed 1101: 1.09 game_done 78s · seed 1102: 1.10 game_done 79s · seed 1103: 1.10 game_done 76s · seed 1104: 1.09 game_done 77s · seed 1105: 1.09 game_done 218s · seed 1106: 1.10 game_done 73s
- `0004`: seed 1101: 1.09 game_done 73s · seed 1102: 1.10 game_done 72s · seed 1103: 1.10 game_done 72s · seed 1104: 1.09 game_done 72s · seed 1105: 1.09 game_done 72s · seed 1106: 1.10 game_done 72s
- `0005`: seed 1101: 1.09 game_done 246s · seed 1102: 1.10 game_done 76s · seed 1103: 1.10 game_done 74s · seed 1104: 1.09 game_done 74s · seed 1105: 1.09 game_done 74s · seed 1106: 1.10 game_done 73s
- `0006`: seed 1101: 1.09 game_done 76s · seed 1102: 1.10 game_done 77s · seed 1103: 1.10 game_done 73s · seed 1104: 1.09 game_done 75s · seed 1105: 1.09 game_done 73s · seed 1106: 1.10 game_done 72s

System-2 latency ≈ 22.4 s + 53.8 ms × output tokens (fit over 341 archived calls, r = 0.34; slowest 155 s): long answers are slow — size timeouts from this, not from short baseline answers.

Score composition (mean per episode, from the adaptor's score_breakdown):
- `0000-tier1_astra_naive`: completed_duels 2.67, draws 0.00, forfeits 2.67, losses 0.00, lp_component 0.00, native_completed 0.00, outcome_component -0.89, score -0.89, technical_error 0.00, truncations 0.00, win_rate 0.00, wins 0.00
- `0001`: completed_duels 1.33, draws 0.00, forfeits 1.00, losses 0.00, lp_component 0.00, native_completed 0.00, outcome_component -0.33, score -0.33, technical_error 0.00, truncations 0.33, win_rate 0.00, wins 0.00
- `0002`: completed_duels 3.00, draws 0.00, forfeits 1.67, losses 0.00, lp_component 0.01, native_completed 0.17, outcome_component -0.50, score -0.49, technical_error 0.00, truncations 1.17, win_rate 0.06, wins 0.17
- `0003`: completed_duels 3.00, draws 0.00, forfeits 0.00, losses 0.00, lp_component 0.10, native_completed 3.00, outcome_component 1.00, score 1.10, technical_error 0.00, truncations 0.00, win_rate 1.00, wins 3.00
- `0004`: completed_duels 3.00, draws 0.00, forfeits 0.00, losses 0.00, lp_component 0.10, native_completed 3.00, outcome_component 1.00, score 1.10, technical_error 0.00, truncations 0.00, win_rate 1.00, wins 3.00
- `0005`: completed_duels 3.00, draws 0.00, forfeits 0.00, losses 0.00, lp_component 0.10, native_completed 3.00, outcome_component 1.00, score 1.10, technical_error 0.00, truncations 0.00, win_rate 1.00, wins 3.00
- `0006`: completed_duels 3.00, draws 0.00, forfeits 0.00, losses 0.00, lp_component 0.10, native_completed 3.00, outcome_component 1.00, score 1.10, technical_error 0.00, truncations 0.00, win_rate 1.00, wins 3.00

Held-out evaluations (not used for selection):
- `0004` / heldout (Yu-Gi-Oh! EDOPro — offline three-duel benchmark): mean score 1.094 over 6 episodes
