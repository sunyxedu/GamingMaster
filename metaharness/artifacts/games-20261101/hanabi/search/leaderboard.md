# Leaderboard

| # | candidate | parent | mean score | episodes | stale ticks | S1 calls | S2 calls | failed S1/S2 | Jev→act | Astra→act | Jev p50/max s | Astra p50/max s | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0000 | 0000-tier1_astra_naive | – | 10.250 | 4 | – | 0 | 146 | 0 / 3 (0 / 3 timed out) | 0% | 100% | – | 34.9 / 155 | ok |
| 0001 | 0001 | 0000 | 21.250 | 4 | – | 0 | 4 | 0 / 0 | 0% | 0% | – | – | ok |
| 0002 | 0002 | – | 16.000 | 4 | – | 0 | 16 | 0 / 0 | 0% | 7% | – | 32.5 / 74 | ok |
| 0003 | 0003 | – | 21.500 | 4 | – | 0 | 5 | 0 / 1 (0 / 1 timed out) | 0% | 0% | – | 56.9 / 57 | ok |
| 0004 | 0004 | 0003 | 21.000 | 4 | – | 0 | 8 | 0 / 0 | 0% | 2% | – | 24.4 / 61 | ok |
| 0005 | 0005 | 0001 | 19.750 | 4 | – | 0 | 4 | 0 / 0 | 0% | 0% | – | – | ok |
| 0006 | 0006 | 0001 | 20.250 | 4 | – | 0 | 6 | 0 / 0 | 0% | 1% | – | 20.2 / 22 | ok |
| 0007 | 0007 | 0003 | 22.250 | 4 | – | 0 | 4 | 0 / 0 | 0% | 1% | – | 13.1 / 18 | ok |
| 0008 | 0008 | 0001 | 22.750 | 4 | – | 0 | 4 | 0 / 0 | 0% | 0% | – | – | ok |
| 0009 | 0009 | 0006 | 21.500 | 4 | – | 0 | 4 | 0 / 0 | 0% | 0% | – | – | ok |
| 0010 | 0010 | 0008 | 22.750 | 4 | – | 0 | 3 | 0 / 0 | 0% | 0% | – | 25.4 / 25 | ok |
| 0011 | 0011 | 0008 | 6.500 | 4 | – | 0 | 8 | 0 / 0 | 0% | 2% | – | 15.7 / 19 | ok |
| 0012 | 0012 | 0003 | 21.750 | 4 | – | 0 | 5 | 0 / 0 | 0% | 0% | – | 3.6 / 4 | ok |
| 0013 | 0013 | 0010 | 23.250 | 4 | – | 0 | 4 | 0 / 0 | 0% | 1% | – | 58.3 / 62 | ok |
| 0014 | 0014 | 0010 | 23.000 | 4 | – | 0 | 3 | 0 / 0 | 0% | 1% | – | 48.1 / 52 | ok |
| 0015 | 0015 | 0010 | 20.750 | 4 | – | 0 | 2 | 0 / 0 | 0% | 0% | – | – | ok |
| 0016 | 0016 | 0013 | 23.250 | 4 | – | 0 | 4 | 0 / 0 | 0% | 1% | – | 18.6 / 23 | ok |
| 0017 | 0017 | 0014 | 22.750 | 4 | – | 0 | 5 | 0 / 0 | 0% | 2% | – | 60.1 / 77 | ok |
| 0018 | 0018 | 0012 | 21.000 | 4 | – | 0 | 4 | 0 / 0 | 0% | 0% | – | – | ok |

`Jev→act` / `Astra→act`: fraction of posted actions preceded by a fresh System-1 / System-2 answer since the previous action (reported, not used for selection). Latency columns are the median per-episode p50 and the worst observed call: design timeouts and pipelining from these measurements, not from assumptions.

Best so far: `0013` (mean score 23.250)
Only candidates whose status is `ok` can be selected: a fatal, partial or erroneous evaluation cannot win, whatever its score. A used-up call budget or the deadline is not an error. `parent` is the candidate a proposal says it was derived from (`parent: NNNN` in proposal.md).

Validation (top candidates re-evaluated on unseen seeds; the final selection is the best train+validation mean among them):
- `0013`: validation 23.000 over 3 episodes; train+validation 23.143
- `0014`: validation 23.333 over 3 episodes; train+validation 23.143
- `0016`: validation 22.000 over 3 episodes; train+validation 22.714

Episodes (score, how it ended, wall time; ✗ = error):
- `0000-tier1_astra_naive`: seed 11: 11.00 harness_return 1592s · seed 12: 9.00 deadline 1800s · seed 13: 12.00 deadline 1800s · seed 14: 9.00 deadline 1800s
- `0001`: seed 11: 22.00 game_done 122s · seed 12: 22.00 game_done 122s · seed 13: 23.00 game_done 122s · seed 14: 18.00 game_done 122s
- `0002`: seed 11: 3.00 deadline 1800s · seed 12: 20.00 game_done 1008s · seed 13: 24.00 game_done 276s · seed 14: 17.00 game_done 1710s
- `0003`: seed 11: 22.00 game_done 91s · seed 12: 21.00 game_done 91s · seed 13: 23.00 game_done 90s · seed 14: 20.00 game_done 147s
- `0004`: seed 11: 21.00 game_done 81s · seed 12: 23.00 game_done 80s · seed 13: 22.00 game_done 111s · seed 14: 18.00 game_done 64s
- `0005`: seed 11: 21.00 game_done 125s · seed 12: 23.00 game_done 125s · seed 13: 21.00 game_done 125s · seed 14: 14.00 game_done 125s
- `0006`: seed 11: 20.00 game_done 101s · seed 12: 18.00 game_done 82s · seed 13: 24.00 game_done 82s · seed 14: 19.00 game_done 140s
- `0007`: seed 11: 22.00 game_done 405s · seed 12: 21.00 game_done 406s · seed 13: 25.00 game_done 402s · seed 14: 21.00 game_done 406s
- `0008`: seed 11: 23.00 game_done 122s · seed 12: 23.00 game_done 123s · seed 13: 24.00 game_done 123s · seed 14: 21.00 game_done 123s
- `0009`: seed 11: 22.00 game_done 83s · seed 12: 22.00 game_done 84s · seed 13: 23.00 game_done 84s · seed 14: 19.00 game_done 84s
- `0010`: seed 11: 24.00 game_done 391s · seed 12: 23.00 game_done 491s · seed 13: 24.00 game_done 426s · seed 14: 20.00 game_done 535s
- `0011`: seed 11: 10.00 game_done 59s · seed 12: 6.00 game_done 66s · seed 13: 4.00 game_done 157s · seed 14: 6.00 game_done 45s
- `0012`: seed 11: 21.00 game_done 72s · seed 12: 21.00 game_done 72s · seed 13: 25.00 game_done 88s · seed 14: 20.00 game_done 71s
- `0013`: seed 11: 24.00 game_done 401s · seed 12: 23.00 game_done 449s · seed 13: 25.00 game_done 522s · seed 14: 21.00 game_done 594s
- `0014`: seed 11: 24.00 game_done 496s · seed 12: 23.00 game_done 543s · seed 13: 25.00 game_done 534s · seed 14: 20.00 game_done 561s
- `0015`: seed 11: 20.00 game_done 398s · seed 12: 21.00 game_done 449s · seed 13: 22.00 game_done 560s · seed 14: 20.00 game_done 560s
- `0016`: seed 11: 24.00 game_done 386s · seed 12: 23.00 game_done 530s · seed 13: 25.00 game_done 541s · seed 14: 21.00 game_done 648s
- `0017`: seed 11: 23.00 game_done 450s · seed 12: 23.00 game_done 408s · seed 13: 24.00 game_done 525s · seed 14: 21.00 game_done 537s
- `0018`: seed 11: 21.00 game_done 80s · seed 12: 21.00 game_done 85s · seed 13: 21.00 game_done 85s · seed 14: 21.00 game_done 84s

System-2 latency ≈ 39.5 s + -4.4 ms × output tokens (fit over 191 archived calls, r = -0.03; slowest 155 s): long answers are slow — size timeouts from this, not from short baseline answers.

Score composition (mean per episode, from the adaptor's score_breakdown):
- `0000-tier1_astra_naive`: benchmark_score 10.25, fireworks_sum 10.25, information_tokens 3.00, life_tokens 3.00
- `0001`: benchmark_score 21.25, fireworks_sum 21.25, information_tokens 6.00, life_tokens 2.75
- `0002`: benchmark_score 16.00, fireworks_sum 16.00, information_tokens 7.25, life_tokens 3.00
- `0003`: benchmark_score 21.50, fireworks_sum 21.50, information_tokens 1.50, life_tokens 3.00
- `0004`: benchmark_score 21.00, fireworks_sum 21.00, information_tokens 2.50, life_tokens 2.75
- `0005`: benchmark_score 19.75, fireworks_sum 19.75, information_tokens 4.50, life_tokens 2.75
- `0006`: benchmark_score 20.25, fireworks_sum 20.25, information_tokens 7.00, life_tokens 2.25
- `0007`: benchmark_score 22.25, fireworks_sum 22.25, information_tokens 1.75, life_tokens 2.75
- `0008`: benchmark_score 22.75, fireworks_sum 22.75, information_tokens 6.00, life_tokens 3.00
- `0009`: benchmark_score 21.50, fireworks_sum 21.50, information_tokens 7.50, life_tokens 2.75
- `0010`: benchmark_score 22.75, fireworks_sum 22.75, information_tokens 6.25, life_tokens 2.75
- `0011`: benchmark_score 6.50, fireworks_sum 6.50, information_tokens 7.50, life_tokens 3.00
- `0012`: benchmark_score 21.75, fireworks_sum 21.75, information_tokens 2.50, life_tokens 2.75
- `0013`: benchmark_score 23.25, fireworks_sum 23.25, information_tokens 6.00, life_tokens 3.00
- `0014`: benchmark_score 23.00, fireworks_sum 23.00, information_tokens 1.50, life_tokens 2.75
- `0015`: benchmark_score 20.75, fireworks_sum 20.75, information_tokens 5.00, life_tokens 2.75
- `0016`: benchmark_score 23.25, fireworks_sum 23.25, information_tokens 5.75, life_tokens 3.00
- `0017`: benchmark_score 22.75, fireworks_sum 22.75, information_tokens 5.00, life_tokens 3.00
- `0018`: benchmark_score 21.00, fireworks_sum 21.00, information_tokens 4.25, life_tokens 2.75

Held-out evaluations (not used for selection):
- `0013` / heldout (Hanabi (offline standard)): mean score 23.500 over 4 episodes
