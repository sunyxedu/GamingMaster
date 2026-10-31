# Leaderboard

| # | candidate | parent | mean score | seeds median / min | episodes | stale ticks | S1 calls | S2 calls | failed S1/S2 | Jev→act | Astra→act | Jev p50/max s | Astra p50/max s | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0000 | 0000-tier1_astra_naive | – | 0.000 | 0.000 / 0.000 | 10 | – | 0 | 10 | 0 / 0 | – | – | – | 41.9 / 69 | ✗ 10 episode error(s), first: StopIteration |
| 0001 | 0001 | 0000 | -0.249 | -0.009 / -1.158 | 10 | – | 0 | 120 | 0 / 4 (0 / 4 timed out) | 0% | 47% | – | 55.2 / 80 | ok |
| 0002 | 0002 | 0000 | -0.770 | -1.039 / -1.200 | 10 | – | 0 | 66 | 0 / 1 (0 / 1 timed out) | 0% | 28% | – | 62.6 / 89 | ok |
| 0003 | 0003 | – | -0.766 | -1.034 / -1.138 | 10 | – | 0 | 149 | 0 / 0 | 0% | 45% | – | 60.2 / 102 | ok |
| 0004 | 0004 | 0001 | 0.635 | 1.049 / -1.107 | 10 | – | 0 | 118 | 0 / 3 (0 / 3 timed out) | 0% | 37% | – | 77.2 / 143 | ok |
| 0005 | 0005 | 0001 | -0.333 | -1.018 / -1.105 | 10 | – | 0 | 117 | 0 / 2 (0 / 2 timed out) | 0% | 52% | – | 55.2 / 83 | ok |
| 0006 | 0006 | 0005 | -0.237 | -0.065 / -1.158 | 10 | – | 0 | 86 | 0 / 4 (0 / 4 timed out) | 0% | 35% | – | 50.4 / 77 | ok |
| 0007 | 0007 | 0004 | 0.659 | 1.071 / -1.067 | 10 | – | 0 | 56 | 0 / 1 (0 / 1 timed out) | 0% | 16% | – | 60.6 / 141 | ok |
| 0008 | 0008 | 0006 | -0.117 | 0.000 / -1.083 | 10 | – | 0 | 21 | 0 / 0 | 0% | 16% | – | 52.4 / 76 | ok |
| 0009 | 0009 | 0004 | 0.451 | 1.073 / -1.082 | 10 | – | 0 | 42 | 0 / 1 (0 / 1 timed out) | 0% | 11% | – | 85.2 / 126 | ok |
| 0010 | 0010 | 0007 | 0.010 | 1.012 / -1.108 | 10 | – | 0 | 43 | 0 / 1 (0 / 1 timed out) | 0% | 13% | – | 75.2 / 142 | ok |
| 0011 | 0011 | 0009 | 0.010 | 1.037 / -1.125 | 10 | – | 0 | 35 | 0 / 2 (0 / 2 timed out) | 0% | 8% | – | 84.0 / 132 | ok |
| 0012 | 0012 | 0007 | 0.863 | 1.087 / -1.128 | 10 | – | 0 | 37 | 0 / 0 | 0% | 10% | – | 78.8 / 129 | ok |
| 0013 | 0013 | 0012 | 0.207 | 1.025 / -1.164 | 10 | – | 0 | 59 | 0 / 1 (0 / 1 timed out) | 0% | 17% | – | 51.0 / 126 | ok |
| 0014 | 0014 | 0011 | 0.854 | 1.052 / -1.067 | 10 | – | 0 | 46 | 0 / 1 (0 / 1 timed out) | 0% | 12% | – | 68.0 / 109 | ok |
| 0015 | 0015 | 0009 | 0.654 | 1.074 / -1.075 | 10 | – | 0 | 45 | 0 / 1 (0 / 1 timed out) | 0% | 11% | – | 77.2 / 133 | ok |
| 0016 | 0016 | 0012 | 0.215 | 1.025 / -1.141 | 10 | – | 0 | 14 | 0 / 0 | 0% | 4% | – | 66.4 / 102 | ok |
| 0017 | 0017 | 0012 | 0.009 | 1.012 / -1.117 | 10 | – | 0 | 44 | 0 / 0 | 0% | 14% | – | 88.3 / 140 | ok |
| 0018 | 0018 | 0015 | 0.461 | 1.118 / -1.075 | 10 | – | 0 | 27 | 0 / 0 | 0% | 6% | – | 97.1 / 126 | ok |

`seeds median / min`: the median and the worst per-seed score — a mean over a few seeds is noisy and can be carried by one lucky seed; an improvement that lifts the weak seeds is more likely to hold on unseen seeds. `Jev→act` / `Astra→act`: fraction of posted actions preceded by a fresh System-1 / System-2 answer since the previous action (reported, not used for selection). Latency columns are the median per-episode p50 and the worst observed call: design timeouts and pipelining from these measurements, not from assumptions.

Best so far: `0014` (mean score 0.854)
Only candidates whose status is `ok` can be selected: a fatal, partial or erroneous evaluation cannot win, whatever its score. A used-up call budget or the deadline is not an error. `parent` is the candidate a proposal says it was derived from (`parent: NNNN` in proposal.md).

Validation (top candidates re-evaluated on unseen seeds; the final selection is the best train+validation mean among them):
- `0007`: validation 0.446 over 10 episodes; train+validation 0.552
- `0012`: validation 0.653 over 10 episodes; train+validation 0.758
- `0014`: validation 0.847 over 10 episodes; train+validation 0.850

Episodes (score, how it ended, wall time; ✗ = error):
- `0000-tier1_astra_naive`: seed 104729: 0.00 harness_error 75s ✗ StopIteration · seed 130363: 0.00 harness_error 126s ✗ StopIteration · seed 155921: 0.00 harness_error 168s ✗ StopIteration · seed 181081: 0.00 harness_error 151s ✗ StopIteration · seed 206369: 0.00 harness_error 44s ✗ StopIteration · seed 232003: 0.00 harness_error 285s ✗ StopIteration · seed 257053: 0.00 harness_error 72s ✗ StopIteration · seed 282617: 0.00 harness_error 84s ✗ StopIteration · seed 308081: 0.00 harness_error 163s ✗ StopIteration · seed 333667: 0.00 harness_error 236s ✗ StopIteration
- `0001`: seed 104729: 1.00 game_done 846s · seed 130363: -0.04 deadline 1800s · seed 155921: 1.02 game_done 1736s · seed 181081: -1.08 game_done 1420s · seed 206369: 0.03 deadline 1800s · seed 232003: -1.10 game_done 1711s · seed 257053: -0.00 deadline 1800s · seed 282617: -1.15 game_done 510s · seed 308081: -0.01 deadline 1800s · seed 333667: -1.16 game_done 1494s
- `0002`: seed 104729: 1.02 game_done 1598s · seed 130363: -1.10 game_done 1227s · seed 155921: -1.04 game_done 1714s · seed 181081: -0.05 deadline 1800s · seed 206369: -1.03 game_done 1250s · seed 232003: -1.01 game_done 1749s · seed 257053: -1.16 game_done 1074s · seed 282617: -1.10 game_done 1499s · seed 308081: -1.04 game_done 1383s · seed 333667: -1.20 game_done 726s
- `0003`: seed 104729: -1.03 game_done 1709s · seed 130363: -1.12 game_done 1703s · seed 155921: -1.09 game_done 1727s · seed 181081: -1.01 game_done 1764s · seed 206369: 1.11 game_done 1652s · seed 232003: -1.13 game_done 1745s · seed 257053: -1.08 game_done 1735s · seed 282617: -1.14 game_done 1756s · seed 308081: -1.02 game_done 1689s · seed 333667: -0.13 deadline 1800s
- `0004`: seed 104729: 1.05 game_done 1700s · seed 130363: 1.03 game_done 1679s · seed 155921: 1.02 game_done 1649s · seed 181081: 1.11 game_done 1700s · seed 206369: 1.13 game_done 1272s · seed 232003: -1.07 game_done 1619s · seed 257053: 1.02 game_done 1701s · seed 282617: 1.11 game_done 1361s · seed 308081: 1.06 game_done 1648s · seed 333667: -1.11 game_done 1700s
- `0005`: seed 104729: 1.05 game_done 1715s · seed 130363: -1.09 game_done 1690s · seed 155921: 1.05 game_done 1719s · seed 181081: -1.06 game_done 1770s · seed 206369: 1.16 game_done 1586s · seed 232003: -0.12 deadline 1800s · seed 257053: -1.11 game_done 1732s · seed 282617: -1.09 game_done 1727s · seed 308081: -1.02 game_done 1373s · seed 333667: -1.10 game_done 1058s
- `0006`: seed 104729: 1.12 game_done 1779s · seed 130363: -1.12 game_done 1005s · seed 155921: 1.07 game_done 1369s · seed 181081: -1.10 game_done 972s · seed 206369: 0.07 deadline 1800s · seed 232003: -0.07 deadline 1800s · seed 257053: -1.08 game_done 1679s · seed 282617: -1.11 game_done 1503s · seed 308081: 1.01 game_done 1106s · seed 333667: -1.16 game_done 666s
- `0007`: seed 104729: 1.17 game_done 1701s · seed 130363: 1.10 game_done 1701s · seed 155921: 1.04 game_done 1652s · seed 181081: 1.07 game_done 1681s · seed 206369: 1.12 game_done 1701s · seed 232003: 1.07 game_done 1701s · seed 257053: -1.06 game_done 1701s · seed 282617: -1.07 game_done 1700s · seed 308081: 1.11 game_done 1666s · seed 333667: 1.03 game_done 1643s
- `0008`: seed 104729: 0.00 deadline 1800s · seed 130363: -1.06 game_done 1636s · seed 155921: 0.00 deadline 1800s · seed 181081: -1.08 game_done 1556s · seed 206369: 1.12 game_done 615s · seed 232003: -0.06 deadline 1800s · seed 257053: 0.00 deadline 1800s · seed 282617: -0.10 deadline 1800s · seed 308081: 0.00 deadline 1800s · seed 333667: -0.01 deadline 1800s
- `0009`: seed 104729: 1.13 game_done 1688s · seed 130363: 1.07 game_done 1688s · seed 155921: -1.01 game_done 1690s · seed 181081: -1.04 game_done 1690s · seed 206369: 1.06 game_done 1680s · seed 232003: 1.01 game_done 1386s · seed 257053: -1.08 game_done 1681s · seed 282617: 1.07 game_done 1681s · seed 308081: 1.13 game_done 1681s · seed 333667: 1.17 game_done 1627s
- `0010`: seed 104729: 1.08 game_done 1701s · seed 130363: -1.01 game_done 1702s · seed 155921: 1.04 game_done 1701s · seed 181081: -1.03 game_done 1702s · seed 206369: 1.13 game_done 1701s · seed 232003: 1.01 game_done 1701s · seed 257053: -1.03 game_done 1701s · seed 282617: 1.05 game_done 1701s · seed 308081: -1.04 game_done 1701s · seed 333667: -1.11 game_done 1664s
- `0011`: seed 104729: 1.13 game_done 1684s · seed 130363: -1.01 game_done 1684s · seed 155921: -1.02 game_done 1685s · seed 181081: -1.12 game_done 1681s · seed 206369: 1.07 game_done 1681s · seed 232003: 1.04 game_done 1684s · seed 257053: -1.07 game_done 1681s · seed 282617: 1.07 game_done 1683s · seed 308081: 1.12 game_done 1682s · seed 333667: -1.11 game_done 1682s
- `0012`: seed 104729: 1.17 game_done 1701s · seed 130363: 1.10 game_done 1702s · seed 155921: 1.04 game_done 1701s · seed 181081: 1.11 game_done 1701s · seed 206369: 1.13 game_done 1635s · seed 232003: 1.06 game_done 1701s · seed 257053: 1.09 game_done 1701s · seed 282617: 1.02 game_done 1701s · seed 308081: 1.04 game_done 1701s · seed 333667: -1.13 game_done 1701s
- `0013`: seed 104729: 1.15 game_done 1701s · seed 130363: 1.06 game_done 1703s · seed 155921: 1.01 game_done 1702s · seed 181081: 1.03 game_done 1702s · seed 206369: 1.13 game_done 1700s · seed 232003: -1.05 game_done 1701s · seed 257053: 1.04 game_done 1626s · seed 282617: -1.09 game_done 1701s · seed 308081: -1.05 game_done 1686s · seed 333667: -1.16 game_done 1700s
- `0014`: seed 104729: 1.13 game_done 1686s · seed 130363: 1.05 game_done 1696s · seed 155921: 1.02 game_done 1617s · seed 181081: 1.02 game_done 1694s · seed 206369: 1.05 game_done 1680s · seed 232003: 1.02 game_done 1683s · seed 257053: -1.07 game_done 1608s · seed 282617: 1.07 game_done 1681s · seed 308081: 1.14 game_done 552s · seed 333667: 1.10 game_done 1031s
- `0015`: seed 104729: 1.13 game_done 1686s · seed 130363: 1.12 game_done 1689s · seed 155921: 1.04 game_done 1688s · seed 181081: -1.06 game_done 1688s · seed 206369: 1.07 game_done 1682s · seed 232003: 1.01 game_done 1687s · seed 257053: -1.07 game_done 1683s · seed 282617: 1.07 game_done 1684s · seed 308081: 1.13 game_done 1682s · seed 333667: 1.10 game_done 1683s
- `0016`: seed 104729: 1.17 game_done 1700s · seed 130363: -1.12 game_done 1152s · seed 155921: 1.04 game_done 1438s · seed 181081: 1.11 game_done 1401s · seed 206369: 1.13 game_done 1701s · seed 232003: 1.01 game_done 1189s · seed 257053: -1.04 game_done 1701s · seed 282617: 1.03 game_done 1701s · seed 308081: -1.04 game_done 1701s · seed 333667: -1.14 game_done 1478s
- `0017`: seed 104729: 1.17 game_done 1701s · seed 130363: -1.07 game_done 1702s · seed 155921: 1.04 game_done 1701s · seed 181081: -1.01 game_done 1701s · seed 206369: 1.13 game_done 1701s · seed 232003: 1.01 game_done 1701s · seed 257053: 1.04 game_done 1701s · seed 282617: -1.02 game_done 1662s · seed 308081: -1.08 game_done 1700s · seed 333667: -1.12 game_done 1701s
- `0018`: seed 104729: 1.13 game_done 1691s · seed 130363: 1.05 game_done 1693s · seed 155921: 1.13 game_done 1694s · seed 181081: -1.03 game_done 1695s · seed 206369: 1.08 game_done 1682s · seed 232003: -1.04 game_done 1686s · seed 257053: -1.07 game_done 1683s · seed 282617: 1.12 game_done 1683s · seed 308081: 1.12 game_done 1682s · seed 333667: 1.12 game_done 1684s

System-2 latency ≈ 48.0 s + 20.4 ms × output tokens (fit over 1004 archived calls, r = 0.61; slowest 143 s): long answers are slow — size timeouts from this, not from short baseline answers.

Score composition (mean per episode, from the adaptor's score_breakdown):
- `0000-tier1_astra_naive`: decisions 0.00, hp_margin 0.00, opponent_hp_fraction 1.00, opponent_survivors 6.00, outcome 0.00, score 0.00, self_hp_fraction 1.00, self_survivors 6.00, turns 1.00
- `0001`: decisions 23.80, hp_margin -0.19, opponent_hp_fraction 0.39, opponent_survivors 2.90, outcome -0.20, score -0.25, self_hp_fraction 0.19, self_survivors 2.00, turns 20.50
- `0002`: decisions 23.10, hp_margin -0.28, opponent_hp_fraction 0.29, opponent_survivors 2.30, outcome -0.70, score -0.77, self_hp_fraction 0.01, self_survivors 0.50, turns 18.50
- `0003`: decisions 32.60, hp_margin -0.26, opponent_hp_fraction 0.32, opponent_survivors 2.60, outcome -0.70, score -0.77, self_hp_fraction 0.06, self_survivors 0.50, turns 28.00
- `0004`: decisions 29.80, hp_margin 0.14, opponent_hp_fraction 0.07, opponent_survivors 0.60, outcome 0.60, score 0.64, self_hp_fraction 0.21, self_survivors 2.40, turns 26.30
- `0005`: decisions 21.90, hp_margin -0.13, opponent_hp_fraction 0.25, opponent_survivors 2.10, outcome -0.30, score -0.33, self_hp_fraction 0.12, self_survivors 1.10, turns 17.80
- `0006`: decisions 22.90, hp_margin -0.15, opponent_hp_fraction 0.39, opponent_survivors 2.90, outcome -0.20, score -0.24, self_hp_fraction 0.24, self_survivors 1.70, turns 19.30
- `0007`: decisions 29.70, hp_margin 0.23, opponent_hp_fraction 0.05, opponent_survivors 0.50, outcome 0.60, score 0.66, self_hp_fraction 0.29, self_survivors 2.60, turns 26.00
- `0008`: decisions 8.50, hp_margin -0.07, opponent_hp_fraction 0.70, opponent_survivors 4.50, outcome -0.10, score -0.12, self_hp_fraction 0.63, self_survivors 4.00, turns 7.40
- `0009`: decisions 31.00, hp_margin 0.20, opponent_hp_fraction 0.05, opponent_survivors 0.70, outcome 0.40, score 0.45, self_hp_fraction 0.26, self_survivors 2.50, turns 27.40
- `0010`: decisions 25.60, hp_margin 0.04, opponent_hp_fraction 0.09, opponent_survivors 1.10, outcome 0.00, score 0.01, self_hp_fraction 0.13, self_survivors 1.50, turns 21.20
- `0011`: decisions 29.60, hp_margin 0.04, opponent_hp_fraction 0.14, opponent_survivors 1.20, outcome 0.00, score 0.01, self_hp_fraction 0.18, self_survivors 2.20, turns 26.00
- `0012`: decisions 27.40, hp_margin 0.25, opponent_hp_fraction 0.05, opponent_survivors 0.50, outcome 0.80, score 0.86, self_hp_fraction 0.30, self_survivors 3.20, turns 24.20
- `0013`: decisions 30.00, hp_margin 0.03, opponent_hp_fraction 0.14, opponent_survivors 1.20, outcome 0.20, score 0.21, self_hp_fraction 0.17, self_survivors 1.60, turns 25.30
- `0014`: decisions 32.90, hp_margin 0.22, opponent_hp_fraction 0.03, opponent_survivors 0.30, outcome 0.80, score 0.85, self_hp_fraction 0.24, self_survivors 3.30, turns 29.80
- `0015`: decisions 29.90, hp_margin 0.22, opponent_hp_fraction 0.05, opponent_survivors 0.50, outcome 0.60, score 0.65, self_hp_fraction 0.27, self_survivors 3.70, turns 27.10
- `0016`: decisions 22.50, hp_margin 0.06, opponent_hp_fraction 0.13, opponent_survivors 1.20, outcome 0.20, score 0.22, self_hp_fraction 0.20, self_survivors 2.00, turns 18.70
- `0017`: decisions 24.70, hp_margin 0.04, opponent_hp_fraction 0.12, opponent_survivors 1.20, outcome 0.00, score 0.01, self_hp_fraction 0.16, self_survivors 1.60, turns 20.70
- `0018`: decisions 30.60, hp_margin 0.24, opponent_hp_fraction 0.06, opponent_survivors 0.70, outcome 0.40, score 0.46, self_hp_fraction 0.30, self_survivors 3.10, turns 27.70

Held-out evaluations (not used for selection):
- `0014` / heldout (Pokémon Showdown — Gen 9 Random Battle): mean score 0.020 over 10 episodes
