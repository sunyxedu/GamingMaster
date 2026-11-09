# Leaderboard

| # | candidate | parent | mean score | seeds median / min | episodes | stale ticks | S1 calls | S2 calls | failed S1/S2 | Jev→act | Astra→act | Jev p50/max s | Astra p50/max s | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0001 | 0001-previous_winner | – | 4073.000 | 4092.000 / 4007.000 | 4 | – | 0 | 1 | 0 / 0 | 0% | 0% | – | – | ok |
| 0002 | 0002-tier1_astra_naive | – | 0.000 | 0.000 / 0.000 (= all seeds) | 4 | – | 0 | 17 | 0 / 0 | 0% | 100% | – | 65.5 / 156 | ok |
| 0003 | 0003 | 0001 | 4009.222 | 4092.000 / 3751.890 | 4 | – | 0 | 1 | 0 / 0 | 0% | 0% | – | 139.3 / 139 | ok |
| 0004 | 0004 | 0001 | 3896.250 | 3941.000 / 3839.000 | 4 | – | 0 | 2 | 0 / 0 | 0% | 0% | – | – | ok |
| 0005 | 0005 | 0001 | 4073.000 | 4092.000 / 4007.000 | 4 | – | 0 | 0 | 0 / 0 | 0% | 0% | – | – | ok |
| 0006 | 0006 | 0005 | 4073.000 | 4092.000 / 4007.000 | 4 | – | 0 | 0 | 0 / 0 | 0% | 0% | – | – | ok |
| 0007 | 0007 | 0003 | 2841.368 | 2891.000 / 2713.630 | 4 | – | 0 | 4 | 0 / 1 (0 / 1 timed out) | 0% | 1% | – | 151.9 / 159 | ok |
| 0008 | 0008 | 0005 | 4073.000 | 4092.000 / 4007.000 | 4 | – | 0 | 0 | 0 / 0 | 0% | 0% | – | – | ok |
| 0009 | 0009 | 0008 | 4073.000 | 4092.000 / 4007.000 | 4 | – | 0 | 0 | 0 / 0 | 0% | 0% | – | – | ok |
| 0010 | 0010 | 0008 | 4073.000 | 4092.000 / 4007.000 | 4 | – | 0 | 0 | 0 / 0 | 0% | 0% | – | – | ok |
| 0011 | 0011 | 0008 | 4073.000 | 4092.000 / 4007.000 | 4 | – | 0 | 0 | 0 / 0 | 0% | 0% | – | – | ok |
| 0012 | 0012 | – | – | – | – | – | – | – | – | – | – | – | – | not evaluated |
| 0013 | 0013 | – | – | – | – | – | – | – | – | – | – | – | – | not evaluated |
| 0014 | 0014 | – | – | – | – | – | – | – | – | – | – | – | – | not evaluated |
| 0015 | 0015 | 0005 | 4073.000 | 4092.000 / 4007.000 | 4 | – | 0 | 0 | 0 / 0 | 0% | 0% | – | – | ok |
| 0016 | 0016 | 0005 | 4073.000 | 4092.000 / 4007.000 | 4 | – | 0 | 0 | 0 / 0 | 0% | 0% | – | – | ok |
| 0017 | 0017 | 0011 | 4073.000 | 4092.000 / 4007.000 | 4 | – | 0 | 0 | 0 / 0 | 0% | 0% | – | – | ok |

`seeds median / min`: the median and the worst per-seed score — a mean over a few seeds is noisy and can be carried by one lucky seed; an improvement that lifts the weak seeds is more likely to hold on unseen seeds. `(= all seeds)`: every seed scored the same, which is what a scenario-independent script does; expect it to break on scenarios it was not written for. `Jev→act` / `Astra→act`: fraction of posted actions preceded by a fresh System-1 / System-2 answer since the previous action (reported, not used for selection). Latency columns are the median per-episode p50 and the worst observed call: design timeouts and pipelining from these measurements, not from assumptions.

Best so far: `0011` (mean score 4073.000)
Only candidates whose status is `ok` can be selected: a fatal, partial or erroneous evaluation cannot win, whatever its score. A used-up call budget or the deadline is not an error. `parent` is the candidate a proposal says it was derived from (`parent: NNNN` in proposal.md).

Validation (top candidates re-evaluated on unseen seeds; the final selection is the best train+validation mean among them):
- `0001-previous_winner`: validation 3081.213 over 4 episodes; train+validation 3577.106
- `0005`: validation 3224.765 over 4 episodes; train+validation 3648.883
- `0006`: validation 3224.765 over 4 episodes; train+validation 3648.883
- `0008`: validation 3228.278 over 4 episodes; train+validation 3650.639
- `0009`: validation 3390.095 over 4 episodes; train+validation 3731.548
- `0010`: validation 3331.733 over 4 episodes; train+validation 3702.366
- `0011`: validation 3606.670 over 4 episodes; train+validation 3839.835
- `0015`: validation 3558.262 over 4 episodes; train+validation 3815.631
- `0016`: validation 3224.765 over 4 episodes; train+validation 3648.883
- `0017`: validation 3597.903 over 4 episodes; train+validation 3835.451

Episodes (score, how it ended, wall time; ✗ = error):
- `0001-previous_winner`: seed 101: 4030.00 game_done 220s · seed 202: 4007.00 game_done 219s · seed 303: 4163.00 game_done 221s · seed 404: 4092.00 game_done 219s
- `0002-tier1_astra_naive`: seed 101: 0.00 deadline 1800s · seed 202: 0.00 deadline 1800s · seed 303: 0.00 deadline 1800s · seed 404: 0.00 deadline 1800s
- `0003`: seed 101: 4030.00 game_done 220s · seed 202: 3751.89 game_done 358s · seed 303: 4163.00 game_done 219s · seed 404: 4092.00 game_done 221s
- `0004`: seed 101: 3840.00 game_done 221s · seed 202: 3839.00 game_done 223s · seed 303: 3941.00 game_done 221s · seed 404: 3965.00 game_done 222s
- `0005`: seed 101: 4030.00 game_done 224s · seed 202: 4007.00 game_done 222s · seed 303: 4163.00 game_done 224s · seed 404: 4092.00 game_done 222s
- `0006`: seed 101: 4030.00 game_done 223s · seed 202: 4007.00 game_done 223s · seed 303: 4163.00 game_done 222s · seed 404: 4092.00 game_done 224s
- `0007`: seed 101: 2713.63 game_done 521s · seed 202: 2891.00 game_done 221s · seed 303: 2777.84 game_done 651s · seed 404: 2983.00 game_done 382s
- `0008`: seed 101: 4030.00 game_done 224s · seed 202: 4007.00 game_done 224s · seed 303: 4163.00 game_done 223s · seed 404: 4092.00 game_done 223s
- `0009`: seed 101: 4030.00 game_done 222s · seed 202: 4007.00 game_done 223s · seed 303: 4163.00 game_done 221s · seed 404: 4092.00 game_done 223s
- `0010`: seed 101: 4030.00 game_done 219s · seed 202: 4007.00 game_done 222s · seed 303: 4163.00 game_done 221s · seed 404: 4092.00 game_done 220s
- `0011`: seed 101: 4030.00 game_done 224s · seed 202: 4007.00 game_done 225s · seed 303: 4163.00 game_done 224s · seed 404: 4092.00 game_done 223s
- `0015`: seed 101: 4030.00 game_done 226s · seed 202: 4007.00 game_done 227s · seed 303: 4163.00 game_done 227s · seed 404: 4092.00 game_done 227s
- `0016`: seed 101: 4030.00 game_done 223s · seed 202: 4007.00 game_done 224s · seed 303: 4163.00 game_done 226s · seed 404: 4092.00 game_done 225s
- `0017`: seed 101: 4030.00 game_done 224s · seed 202: 4007.00 game_done 224s · seed 303: 4163.00 game_done 224s · seed 404: 4092.00 game_done 226s

System-2 latency ≈ 55.4 s + 21.8 ms × output tokens (fit over 17 archived calls, r = 0.84; slowest 159 s): long answers are slow — size timeouts from this, not from short baseline answers.

Score composition (mean per episode, from the adaptor's score_breakdown):
- `0001-previous_winner`: automation-science-pack 1908.00, bootstrap 20.00, chemical-science-pack 0.00, logistic-science-pack 2145.00, production-science-pack 0.00, space-science-pack 0.00, utility-science-pack 0.00
- `0002-tier1_astra_naive`: automation-science-pack 0.00, bootstrap 0.00, chemical-science-pack 0.00, logistic-science-pack 0.00, production-science-pack 0.00, space-science-pack 0.00, utility-science-pack 0.00
- `0003`: automation-science-pack 1894.25, bootstrap 19.97, chemical-science-pack 0.00, logistic-science-pack 2095.00, production-science-pack 0.00, space-science-pack 0.00, utility-science-pack 0.00
- `0004`: automation-science-pack 1846.25, bootstrap 20.00, chemical-science-pack 0.00, logistic-science-pack 2030.00, production-science-pack 0.00, space-science-pack 0.00, utility-science-pack 0.00
- `0005`: automation-science-pack 1908.00, bootstrap 20.00, chemical-science-pack 0.00, logistic-science-pack 2145.00, production-science-pack 0.00, space-science-pack 0.00, utility-science-pack 0.00
- `0006`: automation-science-pack 1908.00, bootstrap 20.00, chemical-science-pack 0.00, logistic-science-pack 2145.00, production-science-pack 0.00, space-science-pack 0.00, utility-science-pack 0.00
- `0007`: automation-science-pack 1544.00, bootstrap 19.87, chemical-science-pack 0.00, logistic-science-pack 1277.50, production-science-pack 0.00, space-science-pack 0.00, utility-science-pack 0.00
- `0008`: automation-science-pack 1908.00, bootstrap 20.00, chemical-science-pack 0.00, logistic-science-pack 2145.00, production-science-pack 0.00, space-science-pack 0.00, utility-science-pack 0.00
- `0009`: automation-science-pack 1908.00, bootstrap 20.00, chemical-science-pack 0.00, logistic-science-pack 2145.00, production-science-pack 0.00, space-science-pack 0.00, utility-science-pack 0.00
- `0010`: automation-science-pack 1908.00, bootstrap 20.00, chemical-science-pack 0.00, logistic-science-pack 2145.00, production-science-pack 0.00, space-science-pack 0.00, utility-science-pack 0.00
- `0011`: automation-science-pack 1908.00, bootstrap 20.00, chemical-science-pack 0.00, logistic-science-pack 2145.00, production-science-pack 0.00, space-science-pack 0.00, utility-science-pack 0.00
- `0015`: automation-science-pack 1908.00, bootstrap 20.00, chemical-science-pack 0.00, logistic-science-pack 2145.00, production-science-pack 0.00, space-science-pack 0.00, utility-science-pack 0.00
- `0016`: automation-science-pack 1908.00, bootstrap 20.00, chemical-science-pack 0.00, logistic-science-pack 2145.00, production-science-pack 0.00, space-science-pack 0.00, utility-science-pack 0.00
- `0017`: automation-science-pack 1908.00, bootstrap 20.00, chemical-science-pack 0.00, logistic-science-pack 2145.00, production-science-pack 0.00, space-science-pack 0.00, utility-science-pack 0.00

Held-out evaluations (not used for selection):
- `0011` / heldout (Factorio): mean score 3342.580 over 4 episodes
