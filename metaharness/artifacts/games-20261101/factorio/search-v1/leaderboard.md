# Leaderboard

| # | candidate | parent | mean score | episodes | stale ticks | S1 calls | S2 calls | failed S1/S2 | Jev→act | Astra→act | Jev p50/max s | Astra p50/max s | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0000 | 0000-tier1_astra_naive | – | 0.000 | 4 | – | 0 | 51 | 0 / 0 | 0% | 100% | – | 45.2 / 133 | ok |
| 0001 | 0001 | 0000 | 425.090 | 4 | – | 0 | 20 | 0 / 6 (0 / 6 timed out) | 0% | 2% | – | 129.1 / 179 | ok |
| 0002 | 0002 | – | 2262.310 | 4 | – | 0 | 4 | 0 / 0 | 0% | 0% | – | 65.0 / 65 | ok |
| 0003 | 0003 | – | 341.058 | 4 | – | 0 | 28 | 0 / 0 | 0% | 5% | – | 74.3 / 182 | ok |
| 0004 | 0004 | 0002 | 2577.477 | 4 | – | 0 | 4 | 0 / 0 | 0% | 0% | – | 13.0 / 13 | ok |
| 0005 | 0005 | 0002 | 3835.310 | 4 | – | 0 | 0 | 0 / 0 | 0% | 0% | – | – | ok |
| 0006 | 0006 | 0002 | 2427.810 | 4 | – | 0 | 6 | 0 / 0 | 0% | 0% | – | 38.4 / 60 | ok |
| 0007 | 0007 | 0005 | 4073.000 | 4 | – | 0 | 1 | 0 / 0 | 0% | 0% | – | 29.2 / 29 | ok |
| 0008 | 0008 | 0002 | 4029.500 | 4 | – | 0 | 0 | 0 / 0 | 0% | 0% | – | – | ok |
| 0009 | 0009 | 0005 | 2999.500 | 4 | – | 0 | 0 | 0 / 0 | 0% | 0% | – | – | ok |
| 0010 | 0010 | 0007 | 3251.000 | 4 | – | 0 | 4 | 0 / 0 | 0% | 0% | – | – | ok |
| 0011 | 0011 | 0007 | 4046.750 | 4 | – | 0 | 1 | 0 / 0 | 0% | 0% | – | – | ok |
| 0012 | 0012 | 0007 | 3966.750 | 4 | – | 0 | 0 | 0 / 0 | 0% | 0% | – | – | ok |
| 0013 | 0013 | 0007 | 3948.250 | 4 | – | 0 | 0 | 0 / 0 | 0% | 0% | – | – | ok |
| 0014 | 0014 | 0007 | 4071.250 | 4 | – | 0 | 1 | 0 / 0 | 0% | 0% | – | 53.3 / 53 | ok |
| 0015 | 0015 | 0007 | 4060.250 | 4 | – | 0 | 1 | 0 / 0 | 0% | 0% | – | 40.6 / 41 | ok |
| 0016 | 0016 | 0014 | 4071.250 | 4 | – | 0 | 1 | 0 / 0 | 0% | 0% | – | – | ok |
| 0017 | 0017 | 0015 | 3966.000 | 4 | – | 0 | 1 | 0 / 0 | 0% | 0% | – | – | ok |
| 0018 | 0018 | 0015 | 3483.000 | 4 | – | 0 | 0 | 0 / 0 | 0% | 0% | – | – | ok |

`Jev→act` / `Astra→act`: fraction of posted actions preceded by a fresh System-1 / System-2 answer since the previous action (reported, not used for selection). Latency columns are the median per-episode p50 and the worst observed call: design timeouts and pipelining from these measurements, not from assumptions.

Best so far: `0007` (mean score 4073.000)
Only candidates whose status is `ok` can be selected: a fatal, partial or erroneous evaluation cannot win, whatever its score. A used-up call budget or the deadline is not an error. `parent` is the candidate a proposal says it was derived from (`parent: NNNN` in proposal.md).

Validation (top candidates re-evaluated on unseen seeds; the final selection is the best train+validation mean among them):
- `0007`: validation 3081.213 over 4 episodes; train+validation 3577.106
- `0014`: validation 3081.713 over 4 episodes; train+validation 3576.481
- `0016`: validation 3081.713 over 4 episodes; train+validation 3576.481

Episodes (score, how it ended, wall time; ✗ = error):
- `0000-tier1_astra_naive`: seed 101: 0.00 deadline 1800s · seed 202: 0.00 deadline 1800s · seed 303: 0.00 deadline 1800s · seed 404: 0.00 deadline 1800s
- `0001`: seed 101: 10.31 game_done 1688s · seed 202: 1657.00 game_done 1696s · seed 303: 16.34 game_done 1682s · seed 404: 16.71 game_done 1684s
- `0002`: seed 101: 2264.06 game_done 218s · seed 202: 2232.06 game_done 217s · seed 303: 2278.06 game_done 219s · seed 404: 2275.06 game_done 219s
- `0003`: seed 101: 190.61 game_done 1768s · seed 202: 1136.69 game_done 1770s · seed 303: 18.62 game_done 1769s · seed 404: 18.31 game_done 1770s
- `0004`: seed 101: 2694.72 game_done 223s · seed 202: 2582.23 game_done 223s · seed 303: 2455.32 game_done 223s · seed 404: 2577.64 game_done 224s
- `0005`: seed 101: 3741.06 game_done 227s · seed 202: 3728.06 game_done 226s · seed 303: 3972.06 game_done 227s · seed 404: 3900.06 game_done 226s
- `0006`: seed 101: 2443.06 game_done 225s · seed 202: 2386.06 game_done 227s · seed 303: 2457.06 game_done 226s · seed 404: 2425.06 game_done 229s
- `0007`: seed 101: 4030.00 game_done 237s · seed 202: 4007.00 game_done 236s · seed 303: 4163.00 game_done 237s · seed 404: 4092.00 game_done 238s
- `0008`: seed 101: 4020.00 game_done 220s · seed 202: 3958.00 game_done 220s · seed 303: 4076.00 game_done 221s · seed 404: 4064.00 game_done 220s
- `0009`: seed 101: 2954.00 game_done 234s · seed 202: 2933.00 game_done 236s · seed 303: 3061.00 game_done 236s · seed 404: 3050.00 game_done 237s
- `0010`: seed 101: 3216.00 game_done 228s · seed 202: 3155.00 game_done 228s · seed 303: 3099.00 game_done 229s · seed 404: 3534.00 game_done 229s
- `0011`: seed 101: 3993.00 game_done 223s · seed 202: 4005.00 game_done 222s · seed 303: 4093.00 game_done 225s · seed 404: 4096.00 game_done 225s
- `0012`: seed 101: 3851.00 game_done 230s · seed 202: 3889.00 game_done 230s · seed 303: 4105.00 game_done 231s · seed 404: 4022.00 game_done 234s
- `0013`: seed 101: 3877.00 game_done 223s · seed 202: 3883.00 game_done 221s · seed 303: 4034.00 game_done 223s · seed 404: 3999.00 game_done 224s
- `0014`: seed 101: 4078.00 game_done 221s · seed 202: 4049.00 game_done 220s · seed 303: 4095.00 game_done 220s · seed 404: 4063.00 game_done 219s
- `0015`: seed 101: 4012.00 game_done 223s · seed 202: 4007.00 game_done 223s · seed 303: 4163.00 game_done 224s · seed 404: 4059.00 game_done 225s
- `0016`: seed 101: 4078.00 game_done 226s · seed 202: 4049.00 game_done 225s · seed 303: 4095.00 game_done 227s · seed 404: 4063.00 game_done 224s
- `0017`: seed 101: 3896.00 game_done 219s · seed 202: 3933.00 game_done 217s · seed 303: 4057.00 game_done 219s · seed 404: 3978.00 game_done 218s
- `0018`: seed 101: 3437.00 game_done 220s · seed 202: 3563.00 game_done 222s · seed 303: 3399.00 game_done 220s · seed 404: 3533.00 game_done 223s

System-2 latency ≈ 46.8 s + 9.7 ms × output tokens (fit over 90 archived calls, r = 0.61; slowest 182 s): long answers are slow — size timeouts from this, not from short baseline answers.

Score composition (mean per episode, from the adaptor's score_breakdown):
- `0000-tier1_astra_naive`: automation-science-pack 0.00, bootstrap 0.00, chemical-science-pack 0.00, logistic-science-pack 0.00, production-science-pack 0.00, space-science-pack 0.00, utility-science-pack 0.00
- `0001`: automation-science-pack 408.25, bootstrap 16.84, chemical-science-pack 0.00, logistic-science-pack 0.00, production-science-pack 0.00, space-science-pack 0.00, utility-science-pack 0.00
- `0002`: automation-science-pack 2243.25, bootstrap 19.06, chemical-science-pack 0.00, logistic-science-pack 0.00, production-science-pack 0.00, space-science-pack 0.00, utility-science-pack 0.00
- `0003`: automation-science-pack 322.50, bootstrap 18.56, chemical-science-pack 0.00, logistic-science-pack 0.00, production-science-pack 0.00, space-science-pack 0.00, utility-science-pack 0.00
- `0004`: automation-science-pack 2289.00, bootstrap 18.48, chemical-science-pack 0.00, logistic-science-pack 270.00, production-science-pack 0.00, space-science-pack 0.00, utility-science-pack 0.00
- `0005`: automation-science-pack 2016.25, bootstrap 19.06, chemical-science-pack 0.00, logistic-science-pack 1800.00, production-science-pack 0.00, space-science-pack 0.00, utility-science-pack 0.00
- `0006`: automation-science-pack 2408.75, bootstrap 19.06, chemical-science-pack 0.00, logistic-science-pack 0.00, production-science-pack 0.00, space-science-pack 0.00, utility-science-pack 0.00
- `0007`: automation-science-pack 1908.00, bootstrap 20.00, chemical-science-pack 0.00, logistic-science-pack 2145.00, production-science-pack 0.00, space-science-pack 0.00, utility-science-pack 0.00
- `0008`: automation-science-pack 1761.00, bootstrap 20.00, chemical-science-pack 0.00, logistic-science-pack 2248.50, production-science-pack 0.00, space-science-pack 0.00, utility-science-pack 0.00
- `0009`: automation-science-pack 1875.50, bootstrap 20.00, chemical-science-pack 0.00, logistic-science-pack 1104.00, production-science-pack 0.00, space-science-pack 0.00, utility-science-pack 0.00
- `0010`: automation-science-pack 1649.00, bootstrap 20.00, chemical-science-pack 0.00, logistic-science-pack 1582.00, production-science-pack 0.00, space-science-pack 0.00, utility-science-pack 0.00
- `0011`: automation-science-pack 1889.75, bootstrap 20.00, chemical-science-pack 0.00, logistic-science-pack 2137.00, production-science-pack 0.00, space-science-pack 0.00, utility-science-pack 0.00
- `0012`: automation-science-pack 1883.25, bootstrap 20.00, chemical-science-pack 0.00, logistic-science-pack 2063.50, production-science-pack 0.00, space-science-pack 0.00, utility-science-pack 0.00
- `0013`: automation-science-pack 1865.75, bootstrap 20.00, chemical-science-pack 0.00, logistic-science-pack 2062.50, production-science-pack 0.00, space-science-pack 0.00, utility-science-pack 0.00
- `0014`: automation-science-pack 1896.75, bootstrap 20.00, chemical-science-pack 0.00, logistic-science-pack 2154.50, production-science-pack 0.00, space-science-pack 0.00, utility-science-pack 0.00
- `0015`: automation-science-pack 1910.75, bootstrap 20.00, chemical-science-pack 0.00, logistic-science-pack 2129.50, production-science-pack 0.00, space-science-pack 0.00, utility-science-pack 0.00
- `0016`: automation-science-pack 1896.75, bootstrap 20.00, chemical-science-pack 0.00, logistic-science-pack 2154.50, production-science-pack 0.00, space-science-pack 0.00, utility-science-pack 0.00
- `0017`: automation-science-pack 1901.50, bootstrap 20.00, chemical-science-pack 0.00, logistic-science-pack 2044.50, production-science-pack 0.00, space-science-pack 0.00, utility-science-pack 0.00
- `0018`: automation-science-pack 1631.50, bootstrap 20.00, chemical-science-pack 0.00, logistic-science-pack 1831.50, production-science-pack 0.00, space-science-pack 0.00, utility-science-pack 0.00

Held-out evaluations (not used for selection):
- `0007` / heldout (Factorio): mean score 3041.213 over 4 episodes
