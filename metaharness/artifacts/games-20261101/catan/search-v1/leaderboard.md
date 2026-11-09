# Leaderboard

| # | candidate | parent | mean score | seeds median / min | episodes | stale ticks | S1 calls | S2 calls | failed S1/S2 | Jev→act | Astra→act | Jev p50/max s | Astra p50/max s | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0000 | 0000-tier1_astra_naive | – | 1.875 | 2.000 / 1.000 | 8 | – | 0 | 217 | 0 / 0 | 0% | 100% | – | 48.6 / 263 | ok |
| 0001 | 0001 | 0000 | 6.500 | 10.000 / 0.000 | 8 | – | 0 | 83 | 0 / 7 (0 / 7 timed out) | 0% | 9% | – | 58.3 / 90 | ok |
| 0002 | 0002 | 0000 | 4.625 | 4.000 / 3.000 | 8 | – | 0 | 110 | 0 / 0 | 0% | 22% | – | 50.4 / 101 | ok |
| 0003 | 0003 | 0000 | 9.500 | 10.000 / 5.000 | 8 | – | 0 | 0 | 0 / 0 | 0% | 0% | – | – | ok |
| 0004 | 0004 | 0003 | 10.000 | 10.000 / 10.000 | 8 | – | 0 | 0 | 0 / 0 | 0% | 0% | – | – | ok |
| 0005 | 0005 | 0003 | 10.000 | 10.000 / 10.000 | 8 | – | 0 | 0 | 0 / 0 | 0% | 0% | – | – | ok |
| 0006 | 0006 | 0002 | 9.625 | 10.000 / 7.000 | 8 | – | 0 | 0 | 0 / 0 | 0% | 0% | – | – | ok |
| 0007 | 0007 | 0004 | 10.000 | 10.000 / 10.000 | 8 | – | 0 | 0 | 0 / 0 | 0% | 0% | – | – | ok |
| 0008 | 0008 | 0006 | 10.000 | 10.000 / 10.000 | 8 | – | 0 | 0 | 0 / 0 | 0% | 0% | – | – | ok |
| 0009 | 0009 | 0006 | 10.000 | 10.000 / 10.000 | 8 | – | 0 | 0 | 0 / 0 | 0% | 0% | – | – | ok |
| 0010 | 0010 | – | – | – | – | – | – | – | – | – | – | – | – | not evaluated |
| 0011 | 0011 | – | – | – | – | – | – | – | – | – | – | – | – | not evaluated |
| 0012 | 0012 | – | – | – | – | – | – | – | – | – | – | – | – | not evaluated |

`seeds median / min`: the median and the worst per-seed score — a mean over a few seeds is noisy and can be carried by one lucky seed; an improvement that lifts the weak seeds is more likely to hold on unseen seeds. `Jev→act` / `Astra→act`: fraction of posted actions preceded by a fresh System-1 / System-2 answer since the previous action (reported, not used for selection). Latency columns are the median per-episode p50 and the worst observed call: design timeouts and pipelining from these measurements, not from assumptions.

Best so far: `0009` (mean score 10.000)
Only candidates whose status is `ok` can be selected: a fatal, partial or erroneous evaluation cannot win, whatever its score. A used-up call budget or the deadline is not an error. `parent` is the candidate a proposal says it was derived from (`parent: NNNN` in proposal.md).

Validation (top candidates re-evaluated on unseen seeds; the final selection is the best train+validation mean among them):
- `0004`: validation 10.000 over 8 episodes; train+validation 10.000
- `0008`: validation 10.000 over 8 episodes; train+validation 10.000
- `0009`: validation 10.375 over 8 episodes; train+validation 10.188

Episodes (score, how it ended, wall time; ✗ = error):
- `0000-tier1_astra_naive`: seed 101: 2.00 deadline 1800s · seed 102: 2.00 deadline 1800s · seed 103: 2.00 deadline 1800s · seed 104: 2.00 harness_return 1672s · seed 105: 2.00 deadline 1800s · seed 106: 1.00 deadline 1800s · seed 107: 2.00 deadline 1800s · seed 108: 2.00 deadline 1800s
- `0001`: seed 101: 3.00 deadline 1801s · seed 102: 10.00 game_done 1719s · seed 103: 10.00 game_done 1738s · seed 104: 10.00 game_done 1717s · seed 105: 7.00 game_done 1547s · seed 106: 10.00 game_done 1731s · seed 107: 0.00 deadline 1800s · seed 108: 2.00 deadline 1800s
- `0002`: seed 101: 4.00 deadline 1800s · seed 102: 6.00 deadline 1800s · seed 103: 3.00 deadline 1800s · seed 104: 4.00 deadline 1800s · seed 105: 8.00 deadline 1800s · seed 106: 4.00 deadline 1800s · seed 107: 4.00 deadline 1800s · seed 108: 4.00 deadline 1800s
- `0003`: seed 101: 10.00 game_done 2s · seed 102: 10.00 game_done 1s · seed 103: 5.00 game_done 2s · seed 104: 10.00 game_done 2s · seed 105: 11.00 game_done 2s · seed 106: 10.00 game_done 1s · seed 107: 10.00 game_done 2s · seed 108: 10.00 game_done 1s
- `0004`: seed 101: 10.00 game_done 2s · seed 102: 10.00 game_done 2s · seed 103: 10.00 game_done 2s · seed 104: 10.00 game_done 2s · seed 105: 10.00 game_done 0s · seed 106: 10.00 game_done 2s · seed 107: 10.00 game_done 1s · seed 108: 10.00 game_done 1s
- `0005`: seed 101: 10.00 game_done 11s · seed 102: 10.00 game_done 11s · seed 103: 10.00 game_done 10s · seed 104: 10.00 game_done 13s · seed 105: 10.00 game_done 8s · seed 106: 10.00 game_done 9s · seed 107: 10.00 game_done 8s · seed 108: 10.00 game_done 5s
- `0006`: seed 101: 10.00 game_done 1s · seed 102: 10.00 game_done 2s · seed 103: 10.00 game_done 1s · seed 104: 10.00 game_done 1s · seed 105: 10.00 game_done 2s · seed 106: 10.00 game_done 2s · seed 107: 10.00 game_done 1s · seed 108: 7.00 game_done 1s
- `0007`: seed 101: 10.00 game_done 2s · seed 102: 10.00 game_done 2s · seed 103: 10.00 game_done 2s · seed 104: 10.00 game_done 2s · seed 105: 10.00 game_done 1s · seed 106: 10.00 game_done 1s · seed 107: 10.00 game_done 1s · seed 108: 10.00 game_done 2s
- `0008`: seed 101: 10.00 game_done 2s · seed 102: 10.00 game_done 5s · seed 103: 10.00 game_done 3s · seed 104: 10.00 game_done 3s · seed 105: 10.00 game_done 2s · seed 106: 10.00 game_done 2s · seed 107: 10.00 game_done 1s · seed 108: 10.00 game_done 1s
- `0009`: seed 101: 10.00 game_done 7s · seed 102: 10.00 game_done 4s · seed 103: 10.00 game_done 4s · seed 104: 10.00 game_done 4s · seed 105: 10.00 game_done 5s · seed 106: 10.00 game_done 3s · seed 107: 10.00 game_done 3s · seed 108: 10.00 game_done 2s

System-2 latency ≈ 44.0 s + 27.0 ms × output tokens (fit over 385 archived calls, r = 0.66; slowest 263 s): long answers are slow — size timeouts from this, not from short baseline answers.

Score composition (mean per episode, from the adaptor's score_breakdown):
- `0000-tier1_astra_naive`: actions 38.50, public_victory_points 1.88, true_victory_points 1.88, turns 14.50, win 0.00
- `0001`: actions 356.25, public_victory_points 5.75, true_victory_points 6.50, turns 117.00, win 0.50
- `0002`: actions 207.38, public_victory_points 4.50, true_victory_points 4.62, turns 70.00, win 0.00
- `0003`: actions 463.75, public_victory_points 8.25, true_victory_points 9.50, turns 152.38, win 0.88
- `0004`: actions 257.25, public_victory_points 9.88, true_victory_points 10.00, turns 85.50, win 1.00
- `0005`: actions 381.00, public_victory_points 9.25, true_victory_points 10.00, turns 122.50, win 1.00
- `0006`: actions 327.38, public_victory_points 8.62, true_victory_points 9.62, turns 110.88, win 0.88
- `0007`: actions 318.50, public_victory_points 10.00, true_victory_points 10.00, turns 106.00, win 1.00
- `0008`: actions 233.88, public_victory_points 9.50, true_victory_points 10.00, turns 78.00, win 1.00
- `0009`: actions 212.25, public_victory_points 9.38, true_victory_points 10.00, turns 72.00, win 1.00

Held-out evaluations (not used for selection):
- `0009` / heldout (Settlers of Catan (Catanatron)): mean score 10.125 over 8 episodes
