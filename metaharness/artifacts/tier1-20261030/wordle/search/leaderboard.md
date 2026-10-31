# Leaderboard

| # | candidate | parent | mean score | seeds median / min | episodes | stale ticks | S1 calls | S2 calls | failed S1/S2 | Jev→act | Astra→act | Jev p50/max s | Astra p50/max s | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0000 | 0000-tier1_astra_naive | – | 0.000 | 0.000 / 0.000 (= all seeds) | 12 | – | 0 | 274 | 0 / 0 | 0% | 100% | – | 54.0 / 176 | ok |
| 0001 | 0001 | 0000 | 0.278 | 0.333 / 0.000 | 12 | – | 0 | 47 | 0 / 1 (0 / 1 timed out) | 0% | 79% | – | 68.8 / 136 | ok |
| 0002 | 0002 | 0000 | 0.569 | 0.667 / 0.333 | 12 | – | 0 | 0 | 0 / 0 | 0% | 0% | – | – | ok |
| 0003 | 0003 | – | 0.583 | 0.667 / 0.500 | 12 | – | 0 | 0 | 0 / 0 | 0% | 0% | – | – | ok |
| 0004 | 0004 | 0003 | 0.611 | 0.667 / 0.500 | 12 | – | 0 | 0 | 0 / 0 | 0% | 0% | – | – | ok |
| 0005 | 0005 | 0003 | 0.556 | 0.500 / 0.500 | 12 | – | 0 | 0 | 0 / 0 | 0% | 0% | – | – | ok |
| 0006 | 0006 | 0002 | 0.583 | 0.667 / 0.500 | 12 | – | 0 | 0 | 0 / 0 | 0% | 0% | – | – | ok |
| 0007 | 0007 | 0004 | 0.597 | 0.667 / 0.500 | 12 | – | 0 | 0 | 0 / 0 | 0% | 0% | – | – | ok |
| 0008 | 0008 | 0006 | 0.583 | 0.667 / 0.333 | 12 | – | 0 | 0 | 0 / 0 | 0% | 0% | – | – | ok |
| 0009 | 0009 | 0006 | 0.583 | 0.667 / 0.500 | 12 | – | 0 | 0 | 0 / 0 | 0% | 0% | – | – | ok |
| 0010 | 0010 | 0004 | 0.611 | 0.667 / 0.500 | 12 | – | 0 | 0 | 0 / 0 | 0% | 0% | – | – | ok |
| 0011 | 0011 | 0009 | 0.569 | 0.667 / 0.333 | 12 | – | 0 | 0 | 0 / 0 | 0% | 0% | – | – | ok |
| 0012 | 0012 | 0004 | 0.625 | 0.667 / 0.500 | 12 | – | 0 | 0 | 0 / 0 | 0% | 0% | – | – | ok |
| 0013 | 0013 | 0012 | 0.597 | 0.667 / 0.500 | 12 | – | 0 | 0 | 0 / 0 | 0% | 0% | – | – | ok |
| 0014 | 0014 | 0008 | 0.583 | 0.667 / 0.500 | 12 | – | 0 | 0 | 0 / 0 | 0% | 0% | – | – | ok |
| 0015 | 0015 | 0009 | 0.583 | 0.667 / 0.500 | 12 | – | 0 | 0 | 0 / 0 | 0% | 0% | – | – | ok |
| 0016 | 0016 | 0015 | 0.611 | 0.667 / 0.500 | 12 | – | 0 | 0 | 0 / 0 | 0% | 0% | – | – | ok |
| 0017 | 0017 | 0015 | 0.583 | 0.667 / 0.500 | 12 | – | 0 | 0 | 0 / 0 | 0% | 0% | – | – | ok |
| 0018 | 0018 | 0015 | 0.583 | 0.667 / 0.500 | 12 | – | 0 | 0 | 0 / 0 | 0% | 0% | – | – | ok |

`seeds median / min`: the median and the worst per-seed score — a mean over a few seeds is noisy and can be carried by one lucky seed; an improvement that lifts the weak seeds is more likely to hold on unseen seeds. `(= all seeds)`: every seed scored the same, which is what a scenario-independent script does; expect it to break on scenarios it was not written for. `Jev→act` / `Astra→act`: fraction of posted actions preceded by a fresh System-1 / System-2 answer since the previous action (reported, not used for selection). Latency columns are the median per-episode p50 and the worst observed call: design timeouts and pipelining from these measurements, not from assumptions.

Best so far: `0016` (mean score 0.611)
Only candidates whose status is `ok` can be selected: a fatal, partial or erroneous evaluation cannot win, whatever its score. A used-up call budget or the deadline is not an error. `parent` is the candidate a proposal says it was derived from (`parent: NNNN` in proposal.md).

Validation (top candidates re-evaluated on unseen seeds; the final selection is the best train+validation mean among them):
- `0004`: validation 0.583 over 12 episodes; train+validation 0.597
- `0010`: validation 0.569 over 12 episodes; train+validation 0.590
- `0012`: validation 0.569 over 12 episodes; train+validation 0.597
- `0016`: validation 0.597 over 12 episodes; train+validation 0.604

Episodes (score, how it ended, wall time; ✗ = error):
- `0000-tier1_astra_naive`: seed 0: 0.00 deadline 1800s · seed 1: 0.00 deadline 1800s · seed 2: 0.00 deadline 1800s · seed 3: 0.00 deadline 1800s · seed 4: 0.00 deadline 1800s · seed 5: 0.00 deadline 1800s · seed 7: 0.00 deadline 1800s · seed 8: 0.00 deadline 1800s · seed 11: 0.00 deadline 1800s · seed 13: 0.00 deadline 1800s · seed 17: 0.00 deadline 1800s · seed 43: 0.00 deadline 1800s
- `0001`: seed 0: 0.00 harness_return 1261s · seed 1: 0.33 game_done 1176s · seed 2: 0.17 game_done 1682s · seed 3: 0.17 game_done 1743s · seed 4: 0.33 game_done 400s · seed 5: 0.83 game_done 443s · seed 7: 0.33 game_done 460s · seed 8: 0.00 game_done 594s · seed 11: 0.17 game_done 550s · seed 13: 0.17 game_done 654s · seed 17: 0.50 game_done 487s · seed 43: 0.33 game_done 310s
- `0002`: seed 0: 0.67 game_done 190s · seed 1: 0.67 game_done 192s · seed 2: 0.50 game_done 189s · seed 3: 0.50 game_done 188s · seed 4: 0.50 game_done 190s · seed 5: 0.67 game_done 194s · seed 7: 0.67 game_done 192s · seed 8: 0.50 game_done 181s · seed 11: 0.67 game_done 178s · seed 13: 0.67 game_done 189s · seed 17: 0.50 game_done 187s · seed 43: 0.33 game_done 181s
- `0003`: seed 0: 0.50 game_done 6s · seed 1: 0.67 game_done 8s · seed 2: 0.67 game_done 2s · seed 3: 0.50 game_done 3s · seed 4: 0.67 game_done 0s · seed 5: 0.50 game_done 1s · seed 7: 0.67 game_done 3s · seed 8: 0.50 game_done 3s · seed 11: 0.67 game_done 1s · seed 13: 0.50 game_done 1s · seed 17: 0.50 game_done 0s · seed 43: 0.67 game_done 0s
- `0004`: seed 0: 0.50 game_done 81s · seed 1: 0.67 game_done 80s · seed 2: 0.67 game_done 12s · seed 3: 0.50 game_done 35s · seed 4: 0.67 game_done 4s · seed 5: 0.50 game_done 9s · seed 7: 0.67 game_done 25s · seed 8: 0.50 game_done 47s · seed 11: 0.67 game_done 8s · seed 13: 0.67 game_done 11s · seed 17: 0.67 game_done 4s · seed 43: 0.67 game_done 4s
- `0005`: seed 0: 0.50 game_done 240s · seed 1: 0.50 game_done 245s · seed 2: 0.50 game_done 257s · seed 3: 0.67 game_done 254s · seed 4: 0.67 game_done 237s · seed 5: 0.50 game_done 227s · seed 7: 0.67 game_done 235s · seed 8: 0.50 game_done 232s · seed 11: 0.67 game_done 236s · seed 13: 0.50 game_done 233s · seed 17: 0.50 game_done 226s · seed 43: 0.50 game_done 224s
- `0006`: seed 0: 0.50 game_done 63s · seed 1: 0.67 game_done 51s · seed 2: 0.50 game_done 62s · seed 3: 0.50 game_done 61s · seed 4: 0.67 game_done 7s · seed 5: 0.67 game_done 4s · seed 7: 0.67 game_done 13s · seed 8: 0.67 game_done 43s · seed 11: 0.50 game_done 23s · seed 13: 0.67 game_done 7s · seed 17: 0.50 game_done 62s · seed 43: 0.50 game_done 21s
- `0007`: seed 0: 0.50 game_done 20s · seed 1: 0.67 game_done 24s · seed 2: 0.67 game_done 81s · seed 3: 0.67 game_done 27s · seed 4: 0.50 game_done 37s · seed 5: 0.67 game_done 2s · seed 7: 0.50 game_done 82s · seed 8: 0.50 game_done 83s · seed 11: 0.50 game_done 25s · seed 13: 0.67 game_done 10s · seed 17: 0.67 game_done 29s · seed 43: 0.67 game_done 6s
- `0008`: seed 0: 0.67 game_done 70s · seed 1: 0.67 game_done 12s · seed 2: 0.50 game_done 73s · seed 3: 0.50 game_done 76s · seed 4: 0.67 game_done 21s · seed 5: 0.33 game_done 93s · seed 7: 0.67 game_done 18s · seed 8: 0.67 game_done 29s · seed 11: 0.67 game_done 52s · seed 13: 0.50 game_done 77s · seed 17: 0.67 game_done 16s · seed 43: 0.50 game_done 52s
- `0009`: seed 0: 0.50 game_done 151s · seed 1: 0.67 game_done 46s · seed 2: 0.67 game_done 150s · seed 3: 0.50 game_done 43s · seed 4: 0.67 game_done 4s · seed 5: 0.67 game_done 3s · seed 7: 0.50 game_done 10s · seed 8: 0.67 game_done 24s · seed 11: 0.50 game_done 15s · seed 13: 0.67 game_done 7s · seed 17: 0.50 game_done 32s · seed 43: 0.50 game_done 11s
- `0010`: seed 0: 0.50 game_done 80s · seed 1: 0.67 game_done 80s · seed 2: 0.67 game_done 11s · seed 3: 0.50 game_done 37s · seed 4: 0.67 game_done 4s · seed 5: 0.50 game_done 10s · seed 7: 0.67 game_done 21s · seed 8: 0.50 game_done 44s · seed 11: 0.67 game_done 5s · seed 13: 0.67 game_done 9s · seed 17: 0.67 game_done 9s · seed 43: 0.67 game_done 2s
- `0011`: seed 0: 0.33 game_done 74s · seed 1: 0.67 game_done 39s · seed 2: 0.67 game_done 61s · seed 3: 0.50 game_done 41s · seed 4: 0.67 game_done 3s · seed 5: 0.67 game_done 3s · seed 7: 0.50 game_done 13s · seed 8: 0.67 game_done 22s · seed 11: 0.50 game_done 15s · seed 13: 0.67 game_done 6s · seed 17: 0.50 game_done 9s · seed 43: 0.50 game_done 7s
- `0012`: seed 0: 0.50 game_done 121s · seed 1: 0.67 game_done 120s · seed 2: 0.67 game_done 9s · seed 3: 0.50 game_done 40s · seed 4: 0.67 game_done 4s · seed 5: 0.50 game_done 10s · seed 7: 0.67 game_done 35s · seed 8: 0.67 game_done 63s · seed 11: 0.67 game_done 9s · seed 13: 0.67 game_done 10s · seed 17: 0.67 game_done 11s · seed 43: 0.67 game_done 5s
- `0013`: seed 0: 0.50 game_done 109s · seed 1: 0.50 game_done 109s · seed 2: 0.67 game_done 13s · seed 3: 0.50 game_done 65s · seed 4: 0.67 game_done 4s · seed 5: 0.50 game_done 12s · seed 7: 0.67 game_done 45s · seed 8: 0.50 game_done 55s · seed 11: 0.67 game_done 10s · seed 13: 0.67 game_done 13s · seed 17: 0.67 game_done 11s · seed 43: 0.67 game_done 5s
- `0014`: seed 0: 0.50 game_done 51s · seed 1: 0.67 game_done 31s · seed 2: 0.50 game_done 32s · seed 3: 0.50 game_done 52s · seed 4: 0.50 game_done 31s · seed 5: 0.67 game_done 51s · seed 7: 0.50 game_done 31s · seed 8: 0.67 game_done 53s · seed 11: 0.50 game_done 33s · seed 13: 0.67 game_done 43s · seed 17: 0.67 game_done 31s · seed 43: 0.67 game_done 28s
- `0015`: seed 0: 0.50 game_done 150s · seed 1: 0.67 game_done 47s · seed 2: 0.67 game_done 151s · seed 3: 0.50 game_done 45s · seed 4: 0.67 game_done 4s · seed 5: 0.67 game_done 2s · seed 7: 0.50 game_done 13s · seed 8: 0.67 game_done 22s · seed 11: 0.50 game_done 16s · seed 13: 0.67 game_done 6s · seed 17: 0.50 game_done 25s · seed 43: 0.50 game_done 11s
- `0016`: seed 0: 0.50 game_done 150s · seed 1: 0.67 game_done 151s · seed 2: 0.67 game_done 14s · seed 3: 0.50 game_done 30s · seed 4: 0.67 game_done 6s · seed 5: 0.50 game_done 11s · seed 7: 0.67 game_done 25s · seed 8: 0.50 game_done 39s · seed 11: 0.67 game_done 10s · seed 13: 0.67 game_done 9s · seed 17: 0.67 game_done 9s · seed 43: 0.67 game_done 6s
- `0017`: seed 0: 0.50 game_done 13s · seed 1: 0.50 game_done 9s · seed 2: 0.50 game_done 152s · seed 3: 0.50 game_done 48s · seed 4: 0.67 game_done 3s · seed 5: 0.67 game_done 37s · seed 7: 0.67 game_done 9s · seed 8: 0.67 game_done 7s · seed 11: 0.67 game_done 19s · seed 13: 0.50 game_done 126s · seed 17: 0.67 game_done 18s · seed 43: 0.50 game_done 58s
- `0018`: seed 0: 0.50 game_done 150s · seed 1: 0.67 game_done 47s · seed 2: 0.67 game_done 152s · seed 3: 0.50 game_done 44s · seed 4: 0.67 game_done 3s · seed 5: 0.67 game_done 3s · seed 7: 0.50 game_done 10s · seed 8: 0.67 game_done 20s · seed 11: 0.50 game_done 17s · seed 13: 0.67 game_done 7s · seed 17: 0.50 game_done 25s · seed 43: 0.50 game_done 12s

System-2 latency ≈ 53.4 s + 17.6 ms × output tokens (fit over 308 archived calls, r = 0.52; slowest 176 s): long answers are slow — size timeouts from this, not from short baseline answers.

Score composition (mean per episode, from the adaptor's score_breakdown):
- `0000-tier1_astra_naive`: guesses_used 0.00, max_guesses 6.00, score 0.00, solved 0.00
- `0001`: guesses_used 4.83, max_guesses 6.00, score 0.28, solved 0.83
- `0002`: guesses_used 3.58, max_guesses 6.00, score 0.57, solved 1.00
- `0003`: guesses_used 3.50, max_guesses 6.00, score 0.58, solved 1.00
- `0004`: guesses_used 3.33, max_guesses 6.00, score 0.61, solved 1.00
- `0005`: guesses_used 3.67, max_guesses 6.00, score 0.56, solved 1.00
- `0006`: guesses_used 3.50, max_guesses 6.00, score 0.58, solved 1.00
- `0007`: guesses_used 3.42, max_guesses 6.00, score 0.60, solved 1.00
- `0008`: guesses_used 3.50, max_guesses 6.00, score 0.58, solved 1.00
- `0009`: guesses_used 3.50, max_guesses 6.00, score 0.58, solved 1.00
- `0010`: guesses_used 3.33, max_guesses 6.00, score 0.61, solved 1.00
- `0011`: guesses_used 3.58, max_guesses 6.00, score 0.57, solved 1.00
- `0012`: guesses_used 3.25, max_guesses 6.00, score 0.62, solved 1.00
- `0013`: guesses_used 3.42, max_guesses 6.00, score 0.60, solved 1.00
- `0014`: guesses_used 3.50, max_guesses 6.00, score 0.58, solved 1.00
- `0015`: guesses_used 3.50, max_guesses 6.00, score 0.58, solved 1.00
- `0016`: guesses_used 3.33, max_guesses 6.00, score 0.61, solved 1.00
- `0017`: guesses_used 3.50, max_guesses 6.00, score 0.58, solved 1.00
- `0018`: guesses_used 3.50, max_guesses 6.00, score 0.58, solved 1.00

Held-out evaluations (not used for selection):
- `0016` / heldout (Wordle): mean score 0.569 over 12 episodes
