# Leaderboard

| # | candidate | parent | mean score | seeds median / min | episodes | stale ticks | S1 calls | S2 calls | failed S1/S2 | Jev→act | Astra→act | Jev p50/max s | Astra p50/max s | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0000 | 0000-tier1_astra_naive | – | 0.000 | 0.000 / 0.000 (= all seeds) | 9 | – | 0 | 125 | 0 / 0 | 0% | 100% | – | 51.5 / 132 | ok |
| 0001 | 0001 | 0000 | 1.000 | 1.000 / 1.000 (= all seeds) | 9 | – | 0 | 51 | 0 / 0 | 0% | 100% | – | 60.8 / 107 | ok |
| 0002 | 0002 | – | 1.000 | 1.000 / 1.000 (= all seeds) | 9 | – | 0 | 10 | 0 / 0 | 0% | 20% | – | 63.0 / 90 | ok |
| 0003 | 0003 | 0000 | 0.667 | 1.000 / 0.000 | 9 | – | 0 | 53 | 0 / 0 | 0% | 100% | – | 56.0 / 118 | ok |
| 0004 | 0004 | – | – | – | – | – | – | – | – | – | – | – | – | not evaluated |
| 0005 | 0005 | – | – | – | – | – | – | – | – | – | – | – | – | not evaluated |
| 0006 | 0006 | – | – | – | – | – | – | – | – | – | – | – | – | not evaluated |

`seeds median / min`: the median and the worst per-seed score — a mean over a few seeds is noisy and can be carried by one lucky seed; an improvement that lifts the weak seeds is more likely to hold on unseen seeds. `(= all seeds)`: every seed scored the same, which is what a scenario-independent script does; expect it to break on scenarios it was not written for. `Jev→act` / `Astra→act`: fraction of posted actions preceded by a fresh System-1 / System-2 answer since the previous action (reported, not used for selection). Latency columns are the median per-episode p50 and the worst observed call: design timeouts and pipelining from these measurements, not from assumptions.

Best so far: `0001` (mean score 1.000)
Only candidates whose status is `ok` can be selected: a fatal, partial or erroneous evaluation cannot win, whatever its score. A used-up call budget or the deadline is not an error. `parent` is the candidate a proposal says it was derived from (`parent: NNNN` in proposal.md).

Validation (top candidates re-evaluated on unseen seeds; the final selection is the best train+validation mean among them):
- `0001`: validation 1.000 over 9 episodes; train+validation 1.000
- `0002`: validation 1.000 over 9 episodes; train+validation 1.000
- `0003`: validation 0.889 over 9 episodes; train+validation 0.778

Episodes (score, how it ended, wall time; ✗ = error):
- `0000-tier1_astra_naive`: seed 100: 0.00 deadline 1800s · seed 101: 0.00 deadline 1800s · seed 102: 0.00 deadline 1800s · seed 103: 0.00 deadline 1800s · seed 104: 0.00 deadline 1800s · seed 105: 0.00 deadline 1800s · seed 106: 0.00 deadline 1800s · seed 107: 0.00 deadline 1800s · seed 108: 0.00 deadline 1800s
- `0001`: seed 100: 1.00 game_done 923s · seed 101: 1.00 game_done 1619s · seed 102: 1.00 game_done 370s · seed 103: 1.00 game_done 772s · seed 104: 1.00 game_done 1239s · seed 105: 1.00 game_done 253s · seed 106: 1.00 game_done 947s · seed 107: 1.00 game_done 984s · seed 108: 1.00 game_done 283s
- `0002`: seed 100: 1.00 game_done 182s · seed 101: 1.00 game_done 698s · seed 102: 1.00 game_done 403s · seed 103: 1.00 game_done 192s · seed 104: 1.00 game_done 132s · seed 105: 1.00 game_done 123s · seed 106: 1.00 game_done 196s · seed 107: 1.00 game_done 167s · seed 108: 1.00 game_done 280s
- `0003`: seed 100: 0.00 deadline 1800s · seed 101: 1.00 game_done 535s · seed 102: 0.00 deadline 1800s · seed 103: 1.00 game_done 285s · seed 104: 0.00 deadline 1800s · seed 105: 1.00 game_done 636s · seed 106: 1.00 game_done 1766s · seed 107: 1.00 game_done 1215s · seed 108: 1.00 game_done 286s

System-2 latency ≈ 52.9 s + 12.2 ms × output tokens (fit over 227 archived calls, r = 0.26; slowest 132 s): long answers are slow — size timeouts from this, not from short baseline answers.

Score composition (mean per episode, from the adaptor's score_breakdown):
- `0000-tier1_astra_naive`: native_max_score 1.00, native_score 0.00, normalized_score 0.00
- `0001`: native_max_score 1.00, native_score 1.00, normalized_score 1.00
- `0002`: native_max_score 1.00, native_score 1.00, normalized_score 1.00
- `0003`: native_max_score 1.00, native_score 0.67, normalized_score 0.67

Held-out evaluations (not used for selection):
- `0001` / heldout (TextWorld generic Glulx quests (admissible commands)): mean score 1.000 over 9 episodes
