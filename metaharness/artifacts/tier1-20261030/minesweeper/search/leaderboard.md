# Leaderboard

| # | candidate | parent | mean score | seeds median / min | episodes | stale ticks | S1 calls | S2 calls | failed S1/S2 | Jev→act | Astra→act | Jev p50/max s | Astra p50/max s | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0000 | 0000-tier1_astra_naive | – | 0.350 | 0.338 / 0.000 | 12 | – | 0 | 159 | 0 / 0 | 0% | 100% | – | 55.4 / 141 | ok |
| 0001 | 0001 | 0000 | 0.767 | 1.000 / 0.042 | 12 | – | 0 | 0 | 0 / 0 | 0% | 0% | – | – | ok |
| 0002 | 0002 | 0000 | 0.604 | 0.941 / 0.003 | 12 | – | 0 | 0 | 0 / 0 | 0% | 0% | – | – | ok |
| 0003 | 0003 | 0001 | 0.715 | 0.941 / 0.019 | 12 | – | 0 | 0 | 0 / 0 | 0% | 0% | – | – | ok |
| 0004 | 0004 | 0001 | 0.492 | 0.898 / 0.005 | 12 | – | 0 | 0 | 0 / 0 | 0% | 0% | – | – | ok |
| 0005 | 0005 | 0006 | 0.915 | 1.000 / 0.010 | 12 | – | 0 | 0 | 0 / 0 | 0% | 0% | – | – | ok |
| 0006 | 0006 | 0003 | 0.917 | 1.000 / 0.009 | 12 | – | 0 | 0 | 0 / 0 | 0% | 0% | – | – | ok |
| 0007 | 0007 | 0006 | 0.918 | 1.000 / 0.023 | 12 | – | 0 | 0 | 0 / 0 | 0% | 0% | – | – | ok |
| 0008 | 0008 | 0005 | 0.999 | 1.000 / 0.988 | 12 | – | 0 | 0 | 0 / 0 | 0% | 0% | – | – | ok |
| 0009 | 0009 | 0005 | 0.915 | 1.000 / 0.005 | 12 | – | 0 | 0 | 0 / 0 | 0% | 0% | – | – | ok |
| 0010 | 0010 | 0008 | 0.999 | 1.000 / 0.992 | 12 | – | 0 | 0 | 0 / 0 | 0% | 0% | – | – | ok |
| 0011 | 0011 | 0008 | 0.999 | 1.000 / 0.992 | 12 | – | 0 | 0 | 0 / 0 | 0% | 0% | – | – | ok |
| 0012 | 0012 | 0008 | 0.999 | 1.000 / 0.988 | 12 | – | 0 | 0 | 0 / 0 | 0% | 0% | – | – | ok |
| 0013 | 0013 | – | – | – | – | – | – | – | – | – | – | – | – | not evaluated |
| 0014 | 0014 | – | – | – | – | – | – | – | – | – | – | – | – | not evaluated |
| 0015 | 0015 | – | – | – | – | – | – | – | – | – | – | – | – | not evaluated |

`seeds median / min`: the median and the worst per-seed score — a mean over a few seeds is noisy and can be carried by one lucky seed; an improvement that lifts the weak seeds is more likely to hold on unseen seeds. `(= all seeds)`: every seed scored the same, which is what a scenario-independent script does; expect it to break on scenarios it was not written for. `Jev→act` / `Astra→act`: fraction of posted actions preceded by a fresh System-1 / System-2 answer since the previous action (reported, not used for selection). Latency columns are the median per-episode p50 and the worst observed call: design timeouts and pipelining from these measurements, not from assumptions.

Best so far: `0011` (mean score 0.999)
Only candidates whose status is `ok` can be selected: a fatal, partial or erroneous evaluation cannot win, whatever its score. A used-up call budget or the deadline is not an error. `parent` is the candidate a proposal says it was derived from (`parent: NNNN` in proposal.md).

Validation (top candidates re-evaluated on unseen seeds; the final selection is the best train+validation mean among them):
- `0010`: validation 0.670 over 12 episodes; train+validation 0.835
- `0011`: validation 0.671 over 12 episodes; train+validation 0.835
- `0012`: validation 0.672 over 12 episodes; train+validation 0.835

Episodes (score, how it ended, wall time; ✗ = error):
- `0000-tier1_astra_naive`: seed 600: 0.00 deadline 1800s · seed 601: 0.00 game_done 737s · seed 602: 0.09 game_done 933s · seed 603: 0.68 deadline 1800s · seed 604: 0.43 game_done 1092s · seed 605: 0.07 game_done 454s · seed 606: 0.99 game_done 1469s · seed 607: 0.00 game_done 110s · seed 608: 0.02 game_done 219s · seed 609: 0.68 deadline 1800s · seed 610: 0.34 deadline 1800s · seed 611: 0.89 game_done 1236s
- `0001`: seed 600: 1.00 game_done 0s · seed 601: 0.04 game_done 1s · seed 602: 0.99 game_done 5s · seed 603: 1.00 game_done 0s · seed 604: 0.14 game_done 0s · seed 605: 1.00 game_done 0s · seed 606: 1.00 game_done 0s · seed 607: 1.00 game_done 1s · seed 608: 0.06 game_done 1s · seed 609: 1.00 game_done 0s · seed 610: 1.00 game_done 2s · seed 611: 0.98 game_done 0s
- `0002`: seed 600: 1.00 game_done 0s · seed 601: 1.00 game_done 2s · seed 602: 0.00 game_done 0s · seed 603: 0.94 deadline 1800s · seed 604: 0.88 deadline 1800s · seed 605: 1.00 game_done 0s · seed 606: 1.00 game_done 0s · seed 607: 0.00 game_done 0s · seed 608: 0.34 deadline 1800s · seed 609: 1.00 game_done 0s · seed 610: 0.07 game_done 0s · seed 611: 0.01 game_done 0s
- `0003`: seed 600: 1.00 game_done 0s · seed 601: 0.02 game_done 0s · seed 602: 0.93 game_done 6s · seed 603: 0.94 game_done 0s · seed 604: 0.04 game_done 0s · seed 605: 1.00 game_done 0s · seed 606: 1.00 game_done 0s · seed 607: 0.90 game_done 1s · seed 608: 0.04 game_done 0s · seed 609: 1.00 game_done 0s · seed 610: 0.74 game_done 1s · seed 611: 0.98 game_done 0s
- `0004`: seed 600: 1.00 game_done 0s · seed 601: 0.00 game_done 0s · seed 602: 0.01 game_done 0s · seed 603: 0.94 game_done 0s · seed 604: 0.02 game_done 0s · seed 605: 0.01 game_done 0s · seed 606: 1.00 game_done 0s · seed 607: 0.90 game_done 0s · seed 608: 0.03 game_done 0s · seed 609: 1.00 game_done 0s · seed 610: 0.01 game_done 0s · seed 611: 0.99 game_done 0s
- `0005`: seed 600: 1.00 game_done 0s · seed 601: 1.00 game_done 1s · seed 602: 0.99 game_done 3s · seed 603: 1.00 game_done 0s · seed 604: 1.00 game_done 1s · seed 605: 1.00 game_done 0s · seed 606: 1.00 game_done 0s · seed 607: 1.00 game_done 4s · seed 608: 1.00 game_done 4s · seed 609: 1.00 game_done 1s · seed 610: 0.01 game_done 1s · seed 611: 0.98 game_done 0s
- `0006`: seed 600: 1.00 game_done 0s · seed 601: 1.00 game_done 1s · seed 602: 0.99 game_done 1s · seed 603: 1.00 game_done 0s · seed 604: 1.00 game_done 1s · seed 605: 1.00 game_done 0s · seed 606: 1.00 game_done 0s · seed 607: 0.01 game_done 0s · seed 608: 1.00 game_done 1s · seed 609: 1.00 game_done 0s · seed 610: 1.00 game_done 0s · seed 611: 1.00 game_done 0s
- `0007`: seed 600: 1.00 game_done 0s · seed 601: 1.00 game_done 1s · seed 602: 0.99 game_done 2s · seed 603: 1.00 game_done 0s · seed 604: 1.00 game_done 1s · seed 605: 1.00 game_done 0s · seed 606: 1.00 game_done 0s · seed 607: 0.02 game_done 3s · seed 608: 1.00 game_done 3s · seed 609: 1.00 game_done 0s · seed 610: 1.00 game_done 1s · seed 611: 1.00 game_done 0s
- `0008`: seed 600: 1.00 game_done 0s · seed 601: 1.00 game_done 1s · seed 602: 1.00 game_done 10s · seed 603: 1.00 game_done 0s · seed 604: 1.00 game_done 7s · seed 605: 1.00 game_done 26s · seed 606: 1.00 game_done 0s · seed 607: 1.00 game_done 103s · seed 608: 1.00 game_done 82s · seed 609: 1.00 game_done 0s · seed 610: 1.00 game_done 45s · seed 611: 0.99 game_done 1s
- `0009`: seed 600: 1.00 game_done 0s · seed 601: 1.00 game_done 0s · seed 602: 1.00 game_done 11s · seed 603: 1.00 game_done 0s · seed 604: 1.00 game_done 31s · seed 605: 1.00 game_done 35s · seed 606: 1.00 game_done 0s · seed 607: 0.00 game_done 35s · seed 608: 1.00 game_done 65s · seed 609: 1.00 game_done 0s · seed 610: 1.00 game_done 27s · seed 611: 0.98 game_done 1s
- `0010`: seed 600: 1.00 game_done 0s · seed 601: 1.00 game_done 1s · seed 602: 0.99 game_done 4s · seed 603: 1.00 game_done 0s · seed 604: 1.00 game_done 7s · seed 605: 1.00 game_done 28s · seed 606: 1.00 game_done 0s · seed 607: 1.00 game_done 101s · seed 608: 1.00 game_done 83s · seed 609: 1.00 game_done 0s · seed 610: 1.00 game_done 46s · seed 611: 1.00 game_done 0s
- `0011`: seed 600: 1.00 game_done 0s · seed 601: 1.00 game_done 1s · seed 602: 0.99 game_done 4s · seed 603: 1.00 game_done 0s · seed 604: 1.00 game_done 1s · seed 605: 1.00 game_done 29s · seed 606: 1.00 game_done 0s · seed 607: 1.00 game_done 154s · seed 608: 1.00 game_done 84s · seed 609: 1.00 game_done 0s · seed 610: 1.00 game_done 46s · seed 611: 1.00 game_done 0s
- `0012`: seed 600: 1.00 game_done 0s · seed 601: 1.00 game_done 3s · seed 602: 1.00 game_done 14s · seed 603: 1.00 game_done 0s · seed 604: 1.00 game_done 10s · seed 605: 1.00 game_done 35s · seed 606: 1.00 game_done 0s · seed 607: 1.00 game_done 102s · seed 608: 1.00 game_done 89s · seed 609: 1.00 game_done 0s · seed 610: 1.00 game_done 46s · seed 611: 0.99 game_done 1s

System-2 latency ≈ 37.4 s + 33.4 ms × output tokens (fit over 155 archived calls, r = 0.68; slowest 141 s): long answers are slow — size timeouts from this, not from short baseline answers.

Score composition (mean per episode, from the adaptor's score_breakdown):
- `0000-tier1_astra_naive`: progress 0.35, safe_revealed 40.58, safe_total 175.00, won 0.00
- `0001`: progress 0.77, safe_revealed 113.50, safe_total 175.00, won 0.58
- `0002`: progress 0.60, safe_revealed 79.92, safe_total 175.00, won 0.42
- `0003`: progress 0.71, safe_revealed 102.25, safe_total 175.00, won 0.33
- `0004`: progress 0.49, safe_revealed 53.33, safe_total 175.00, won 0.25
- `0005`: progress 0.91, safe_revealed 158.42, safe_total 175.00, won 0.67
- `0006`: progress 0.92, safe_revealed 156.92, safe_total 175.00, won 0.75
- `0007`: progress 0.92, safe_revealed 157.08, safe_total 175.00, won 0.75
- `0008`: progress 1.00, safe_revealed 174.75, safe_total 175.00, won 0.75
- `0009`: progress 0.91, safe_revealed 156.75, safe_total 175.00, won 0.67
- `0010`: progress 1.00, safe_revealed 174.67, safe_total 175.00, won 0.83
- `0011`: progress 1.00, safe_revealed 174.67, safe_total 175.00, won 0.83
- `0012`: progress 1.00, safe_revealed 174.75, safe_total 175.00, won 0.75

Held-out evaluations (not used for selection):
- `0011` / heldout (Minesweeper (offline deterministic)): mean score 0.837 over 12 episodes
