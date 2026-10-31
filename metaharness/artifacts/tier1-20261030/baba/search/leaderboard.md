# Leaderboard

| # | candidate | parent | mean score | seeds median / min | episodes | stale ticks | S1 calls | S2 calls | failed S1/S2 | Jev→act | Astra→act | Jev p50/max s | Astra p50/max s | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0000 | 0000-tier1_astra_naive | – | 0.750 | 1.000 / 0.000 | 8 | – | 0 | 122 | 0 / 0 | 0% | 100% | – | 43.9 / 208 | ok |
| 0001 | 0001 | 0000 | 0.750 | 1.000 / 0.000 | 8 | – | 0 | 24 | 0 / 2 (0 / 2 timed out) | 0% | 20% | – | 77.7 / 224 | ok |
| 0002 | 0002 | 0000 | 1.000 | 1.000 / 1.000 (= all seeds) | 8 | – | 0 | 10 | 0 / 0 | 0% | 12% | – | 96.9 / 193 | ok |
| 0003 | 0003 | 0000 | 0.875 | 1.000 / 0.000 | 8 | – | 0 | 16 | 0 / 0 | 0% | 21% | – | 67.6 / 144 | ok |
| 0004 | 0004 | – | – | – | – | – | – | – | – | – | – | – | – | not evaluated |
| 0005 | 0005 | – | – | – | – | – | – | – | – | – | – | – | – | not evaluated |
| 0006 | 0006 | – | – | – | – | – | – | – | – | – | – | – | – | not evaluated |

`seeds median / min`: the median and the worst per-seed score — a mean over a few seeds is noisy and can be carried by one lucky seed; an improvement that lifts the weak seeds is more likely to hold on unseen seeds. `(= all seeds)`: every seed scored the same, which is what a scenario-independent script does; expect it to break on scenarios it was not written for. `Jev→act` / `Astra→act`: fraction of posted actions preceded by a fresh System-1 / System-2 answer since the previous action (reported, not used for selection). Latency columns are the median per-episode p50 and the worst observed call: design timeouts and pipelining from these measurements, not from assumptions.

Best so far: `0003` (mean score 0.875)
Only candidates whose status is `ok` can be selected: a fatal, partial or erroneous evaluation cannot win, whatever its score. A used-up call budget or the deadline is not an error. `parent` is the candidate a proposal says it was derived from (`parent: NNNN` in proposal.md).

Validation (top candidates re-evaluated on unseen seeds; the final selection is the best train+validation mean among them):
- `0000-tier1_astra_naive`: validation 0.500 over 8 episodes; train+validation 0.625
- `0001`: validation 0.125 over 8 episodes; train+validation 0.438
- `0002`: validation 0.375 over 8 episodes; train+validation 0.688
- `0003`: validation 0.750 over 8 episodes; train+validation 0.812

Episodes (score, how it ended, wall time; ✗ = error):
- `0000-tier1_astra_naive`: seed 100: 1.00 game_done 134s · seed 101: 1.00 game_done 81s · seed 102: 1.00 game_done 52s · seed 103: 0.00 deadline 1800s · seed 104: 1.00 game_done 772s · seed 105: 1.00 game_done 1073s · seed 106: 1.00 game_done 646s · seed 107: 0.00 deadline 1800s
- `0001`: seed 100: 0.00 deadline 1800s · seed 101: 1.00 game_done 697s · seed 102: 0.00 deadline 1800s · seed 103: 1.00 game_done 1699s · seed 104: 1.00 game_done 201s · seed 105: 1.00 game_done 325s · seed 106: 1.00 game_done 652s · seed 107: 1.00 game_done 915s
- `0002`: seed 100: 1.00 game_done 0s · seed 101: 1.00 game_done 0s · seed 102: 1.00 game_done 0s · seed 103: 1.00 game_done 238s · seed 104: 1.00 game_done 422s · seed 105: 1.00 game_done 1640s · seed 106: 1.00 game_done 478s · seed 107: 1.00 game_done 992s
- `0003`: seed 100: 1.00 game_done 393s · seed 101: 1.00 game_done 1571s · seed 102: 1.00 game_done 68s · seed 103: 0.00 deadline 1800s · seed 104: 1.00 game_done 121s · seed 105: 1.00 game_done 1529s · seed 106: 1.00 game_done 1659s · seed 107: 1.00 game_done 1200s

System-2 latency ≈ 45.2 s + 18.1 ms × output tokens (fit over 165 archived calls, r = 0.81; slowest 224 s): long answers are slow — size timeouts from this, not from short baseline answers.

Score composition (mean per episode, from the adaptor's score_breakdown):
- `0000-tier1_astra_naive`: puzzles_attempted 1.00, puzzles_solved 0.75, solved_fraction 0.75
- `0001`: puzzles_attempted 1.00, puzzles_solved 0.75, solved_fraction 0.75
- `0002`: puzzles_attempted 1.00, puzzles_solved 1.00, solved_fraction 1.00
- `0003`: puzzles_attempted 1.00, puzzles_solved 0.88, solved_fraction 0.88

Held-out evaluations (not used for selection):
- `0003` / heldout (Baba Is AI): mean score 1.000 over 8 episodes
