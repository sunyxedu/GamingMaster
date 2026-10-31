# Leaderboard

| # | candidate | parent | mean score | seeds median / min | episodes | stale ticks | S1 calls | S2 calls | failed S1/S2 | Jev→act | Astra→act | Jev p50/max s | Astra p50/max s | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0000 | 0000-tier1_astra_naive | – | 0.007 | 0.000 / 0.000 | 8 | – | 0 | 217 | 0 / 0 | 0% | 100% | – | 52.6 / 121 | ok |
| 0001 | 0001 | 0000 | 0.067 | 0.050 / 0.000 | 8 | – | 0 | 68 | 0 / 1 (0 / 1 timed out) | 0% | 44% | – | 42.9 / 87 | ok |
| 0002 | 0002 | – | 0.088 | 0.098 / 0.000 | 8 | – | 0 | 87 | 0 / 2 (0 / 2 timed out) | 0% | 38% | – | 51.2 / 90 | ok |
| 0003 | 0003 | 0000 | 0.090 | 0.098 / 0.000 | 8 | – | 0 | 124 | 0 / 0 | 0% | 61% | – | 34.2 / 78 | ok |
| 0004 | 0004 | 0003 | 0.125 | 0.194 / 0.000 | 8 | – | 0 | 142 | 0 / 4 (0 / 4 timed out) | 0% | 72% | – | 43.8 / 163 | ok |
| 0005 | 0005 | 0003 | 0.149 | 0.098 / 0.000 | 8 | – | 0 | 85 | 0 / 2 (0 / 2 timed out) | 0% | 38% | – | 50.2 / 72 | ok |
| 0006 | 0006 | 0002 | 0.052 | 0.028 / 0.000 | 8 | – | 0 | 40 | 0 / 15 (0 / 15 timed out) | 0% | 24% | – | 120.9 / 179 | ok |
| 0007 | 0007 | 0005 | 0.046 | 0.050 / 0.000 | 8 | – | 0 | 91 | 0 / 20 (0 / 20 timed out) | 0% | 40% | – | 56.8 / 98 | ok |
| 0008 | 0008 | 0005 | 0.091 | 0.098 / 0.000 | 8 | – | 0 | 120 | 0 / 22 (0 / 22 timed out) | 0% | 45% | – | 55.6 / 97 | ok |
| 0009 | 0009 | 0002 | 0.051 | 0.000 / 0.000 | 8 | – | 0 | 64 | 0 / 26 (0 / 26 timed out) | 0% | 33% | – | 55.0 / 111 | ok |
| 0010 | 0010 | 0005 | 0.029 | 0.028 / 0.000 | 8 | – | 0 | 94 | 0 / 27 (0 / 27 timed out) | 0% | 50% | – | 38.2 / 106 | ok |
| 0011 | 0011 | 0005 | 0.198 | 0.196 / 0.000 | 8 | – | 0 | 159 | 0 / 22 (0 / 22 timed out) | 0% | 33% | – | 40.4 / 72 | ok |
| 0012 | 0012 | 0005 | 0.031 | 0.000 / 0.000 | 8 | – | 0 | 75 | 0 / 35 (0 / 35 timed out) | 0% | 37% | – | 58.8 / 82 | ok |
| 0013 | 0013 | 0011 | 0.024 | 0.000 / 0.000 | 8 | – | 0 | 88 | 0 / 35 (0 / 35 timed out) | 0% | 41% | – | 58.8 / 75 | ok |
| 0014 | 0014 | 0011 | 0.121 | 0.098 / 0.000 | 8 | – | 0 | 91 | 0 / 34 (0 / 34 timed out) | 0% | 29% | – | 56.5 / 74 | ok |
| 0015 | 0015 | 0005 | 0.085 | 0.098 / 0.000 | 8 | – | 0 | 90 | 0 / 30 (0 / 30 timed out) | 0% | 26% | – | 55.9 / 75 | ok |
| 0016 | 0016 | 0011 | 0.038 | 0.000 / 0.000 | 8 | – | 0 | 39 | 0 / 24 (0 / 24 timed out) | 0% | 29% | – | 52.4 / 61 | ok |
| 0017 | 0017 | 0011 | 0.032 | 0.028 / 0.000 | 8 | – | 0 | 66 | 0 / 17 (0 / 17 timed out) | 0% | 37% | – | 59.0 / 75 | ok |
| 0018 | 0018 | 0011 | 0.106 | 0.167 / 0.000 | 8 | – | 0 | 146 | 0 / 31 (0 / 31 timed out) | 0% | 39% | – | 44.5 / 70 | ok |

`seeds median / min`: the median and the worst per-seed score — a mean over a few seeds is noisy and can be carried by one lucky seed; an improvement that lifts the weak seeds is more likely to hold on unseen seeds. `(= all seeds)`: every seed scored the same, which is what a scenario-independent script does; expect it to break on scenarios it was not written for. `Jev→act` / `Astra→act`: fraction of posted actions preceded by a fresh System-1 / System-2 answer since the previous action (reported, not used for selection). Latency columns are the median per-episode p50 and the worst observed call: design timeouts and pipelining from these measurements, not from assumptions.

Best so far: `0005` (mean score 0.149)
Only candidates whose status is `ok` can be selected: a fatal, partial or erroneous evaluation cannot win, whatever its score. A used-up call budget or the deadline is not an error. `parent` is the candidate a proposal says it was derived from (`parent: NNNN` in proposal.md).

Validation (top candidates re-evaluated on unseen seeds; the final selection is the best train+validation mean among them):
- `0004`: validation 0.116 over 8 episodes; train+validation 0.120
- `0005`: validation 0.121 over 8 episodes; train+validation 0.135
- `0011`: validation 0.016 over 8 episodes; train+validation 0.107

Episodes (score, how it ended, wall time; ✗ = error):
- `0000-tier1_astra_naive`: seed 0: 0.00 deadline 1800s · seed 1: 0.03 deadline 1800s · seed 2: 0.00 deadline 1800s · seed 3: 0.00 deadline 1800s · seed 8: 0.00 deadline 1800s · seed 9: 0.03 deadline 1800s · seed 10: 0.00 deadline 1800s · seed 11: 0.00 deadline 1800s
- `0001`: seed 0: 0.00 deadline 1800s · seed 1: 0.03 deadline 1800s · seed 2: 0.05 deadline 1800s · seed 3: 0.00 harness_return 1005s · seed 8: 0.10 game_done 1144s · seed 9: 0.22 harness_return 1797s · seed 10: 0.14 deadline 1800s · seed 11: 0.00 harness_return 445s
- `0002`: seed 0: 0.20 game_done 952s · seed 1: 0.14 deadline 1800s · seed 2: 0.14 deadline 1800s · seed 3: 0.00 deadline 1800s · seed 8: 0.10 deadline 1800s · seed 9: 0.08 deadline 1800s · seed 10: 0.05 harness_return 1798s · seed 11: 0.00 harness_return 1798s
- `0003`: seed 0: 0.10 deadline 1800s · seed 1: 0.17 deadline 1800s · seed 2: 0.15 deadline 1800s · seed 3: 0.00 deadline 1800s · seed 8: 0.20 deadline 1800s · seed 9: 0.06 game_done 179s · seed 10: 0.05 deadline 1800s · seed 11: 0.00 deadline 1800s
- `0004`: seed 0: 0.20 deadline 1801s · seed 1: 0.19 harness_return 1798s · seed 2: 0.00 deadline 1800s · seed 3: 0.33 harness_return 1798s · seed 8: 0.20 harness_return 1798s · seed 9: 0.03 deadline 1800s · seed 10: 0.05 harness_return 1798s · seed 11: 0.00 deadline 1800s
- `0005`: seed 0: 0.10 deadline 1800s · seed 1: 0.03 deadline 1800s · seed 2: 0.00 deadline 1800s · seed 3: 0.23 harness_return 1797s · seed 8: 0.20 game_done 548s · seed 9: 0.08 game_done 200s · seed 10: 0.05 deadline 1800s · seed 11: 0.50 harness_return 1797s
- `0006`: seed 0: 0.00 deadline 1800s · seed 1: 0.03 deadline 1800s · seed 2: 0.00 harness_return 1798s · seed 3: 0.00 harness_return 833s · seed 8: 0.10 harness_return 791s · seed 9: 0.06 game_done 346s · seed 10: 0.00 harness_return 1798s · seed 11: 0.23 harness_return 1798s
- `0007`: seed 0: 0.00 deadline 1800s · seed 1: 0.06 game_done 814s · seed 2: 0.05 deadline 1800s · seed 3: 0.00 harness_return 1794s · seed 8: 0.10 deadline 1800s · seed 9: 0.17 deadline 1800s · seed 10: 0.00 deadline 1800s · seed 11: 0.00 harness_return 1794s
- `0008`: seed 0: 0.10 game_done 1760s · seed 1: 0.03 deadline 1800s · seed 2: 0.05 harness_return 1796s · seed 3: 0.17 deadline 1800s · seed 8: 0.00 harness_return 1796s · seed 9: 0.22 deadline 1800s · seed 10: 0.00 harness_return 1796s · seed 11: 0.17 harness_return 1796s
- `0009`: seed 0: 0.00 deadline 1800s · seed 1: 0.14 deadline 1800s · seed 2: 0.00 harness_return 895s · seed 3: 0.00 deadline 1800s · seed 8: 0.00 harness_return 360s · seed 9: 0.22 harness_return 1795s · seed 10: 0.05 harness_return 1795s · seed 11: 0.00 harness_return 402s
- `0010`: seed 0: 0.00 harness_return 1614s · seed 1: 0.03 deadline 1800s · seed 2: 0.00 harness_return 1799s · seed 3: 0.00 harness_return 1796s · seed 8: 0.10 harness_return 1314s · seed 9: 0.06 game_done 356s · seed 10: 0.05 harness_return 1756s · seed 11: 0.00 harness_return 1796s
- `0011`: seed 0: 0.20 harness_return 1367s · seed 1: 0.14 deadline 1800s · seed 2: 0.00 harness_return 1795s · seed 3: 0.50 deadline 1800s · seed 8: 0.20 harness_return 1795s · seed 9: 0.22 harness_return 1795s · seed 10: 0.00 harness_return 1792s · seed 11: 0.33 harness_return 1795s
- `0012`: seed 0: 0.00 harness_return 976s · seed 1: 0.14 harness_return 1796s · seed 2: 0.00 harness_return 901s · seed 3: 0.00 harness_return 331s · seed 8: 0.00 harness_return 1368s · seed 9: 0.11 deadline 1800s · seed 10: 0.00 harness_return 843s · seed 11: 0.00 harness_return 476s
- `0013`: seed 0: 0.00 harness_return 1795s · seed 1: 0.03 deadline 1800s · seed 2: 0.00 harness_return 1795s · seed 3: 0.00 harness_return 1307s · seed 8: 0.00 harness_return 1463s · seed 9: 0.17 harness_return 868s · seed 10: 0.00 harness_return 1795s · seed 11: 0.00 harness_return 1088s
- `0014`: seed 0: 0.10 harness_return 767s · seed 1: 0.17 deadline 1800s · seed 2: 0.00 harness_return 1006s · seed 3: 0.57 harness_return 1039s · seed 8: 0.00 deadline 1800s · seed 9: 0.14 harness_return 1717s · seed 10: 0.00 deadline 1800s · seed 11: 0.00 harness_return 1345s
- `0015`: seed 0: 0.00 deadline 1800s · seed 1: 0.17 harness_return 1795s · seed 2: 0.00 harness_return 232s · seed 3: 0.17 harness_return 1231s · seed 8: 0.10 deadline 1800s · seed 9: 0.25 game_done 871s · seed 10: 0.00 harness_return 1795s · seed 11: 0.00 harness_return 527s
- `0016`: seed 0: 0.00 harness_return 446s · seed 1: 0.06 deadline 1800s · seed 2: 0.00 harness_return 1487s · seed 3: 0.17 harness_return 1728s · seed 8: 0.00 harness_return 195s · seed 9: 0.08 game_done 1153s · seed 10: 0.00 harness_return 436s · seed 11: 0.00 harness_return 289s
- `0017`: seed 0: 0.00 deadline 1800s · seed 1: 0.03 deadline 1800s · seed 2: 0.00 deadline 1800s · seed 3: 0.00 deadline 1801s · seed 8: 0.10 harness_return 1795s · seed 9: 0.08 game_done 1503s · seed 10: 0.05 game_done 1569s · seed 11: 0.00 harness_return 1182s
- `0018`: seed 0: 0.10 deadline 1801s · seed 1: 0.22 harness_return 1792s · seed 2: 0.00 harness_return 1792s · seed 3: 0.00 deadline 1800s · seed 8: 0.20 game_done 1616s · seed 9: 0.17 game_done 1145s · seed 10: 0.00 harness_return 1792s · seed 11: 0.17 harness_return 1792s

System-2 latency ≈ 43.1 s + 9.7 ms × output tokens (fit over 1477 archived calls, r = 0.37; slowest 179 s): long answers are slow — size timeouts from this, not from short baseline answers.

Score composition (mean per episode, from the adaptor's score_breakdown):
- `0000-tier1_astra_naive`: initial_score 2.50, lost 0.00, maximum_score 210.25, native_score 2.50, normalized_native_score 0.01, steps 26.12, won 0.00
- `0001`: initial_score 2.50, lost 0.12, maximum_score 210.25, native_score 21.25, normalized_native_score 0.07, steps 16.12, won 0.00
- `0002`: initial_score 2.50, lost 0.12, maximum_score 210.25, native_score 21.25, normalized_native_score 0.09, steps 26.12, won 0.00
- `0003`: initial_score 2.50, lost 0.12, maximum_score 210.25, native_score 21.88, normalized_native_score 0.09, steps 24.12, won 0.00
- `0004`: initial_score 2.50, lost 0.00, maximum_score 210.25, native_score 16.25, normalized_native_score 0.12, steps 23.25, won 0.00
- `0005`: initial_score 2.50, lost 0.25, maximum_score 210.25, native_score 12.12, normalized_native_score 0.15, steps 26.00, won 0.00
- `0006`: initial_score 2.50, lost 0.12, maximum_score 210.25, native_score 5.25, normalized_native_score 0.05, steps 11.88, won 0.00
- `0007`: initial_score 2.50, lost 0.12, maximum_score 210.25, native_score 13.12, normalized_native_score 0.05, steps 20.50, won 0.00
- `0008`: initial_score 2.50, lost 0.12, maximum_score 210.25, native_score 15.62, normalized_native_score 0.09, steps 26.62, won 0.00
- `0009`: initial_score 2.50, lost 0.00, maximum_score 210.25, native_score 18.75, normalized_native_score 0.05, steps 13.38, won 0.00
- `0010`: initial_score 2.50, lost 0.12, maximum_score 210.25, native_score 6.88, normalized_native_score 0.03, steps 16.38, won 0.00
- `0011`: initial_score 2.50, lost 0.00, maximum_score 210.25, native_score 21.88, normalized_native_score 0.20, steps 43.88, won 0.00
- `0012`: initial_score 2.50, lost 0.00, maximum_score 210.25, native_score 11.25, normalized_native_score 0.03, steps 13.25, won 0.00
- `0013`: initial_score 2.50, lost 0.00, maximum_score 210.25, native_score 8.75, normalized_native_score 0.02, steps 16.00, won 0.00
- `0014`: initial_score 2.50, lost 0.00, maximum_score 210.25, native_score 16.50, normalized_native_score 0.12, steps 23.00, won 0.00
- `0015`: initial_score 2.50, lost 0.12, maximum_score 210.25, native_score 20.00, normalized_native_score 0.09, steps 28.00, won 0.00
- `0016`: initial_score 2.50, lost 0.12, maximum_score 210.25, native_score 6.88, normalized_native_score 0.04, steps 6.00, won 0.00
- `0017`: initial_score 2.50, lost 0.25, maximum_score 210.25, native_score 8.12, normalized_native_score 0.03, steps 15.00, won 0.00
- `0018`: initial_score 2.50, lost 0.25, maximum_score 210.25, native_score 20.00, normalized_native_score 0.11, steps 35.50, won 0.00

Held-out evaluations (not used for selection):
- `0005` / heldout (jericho-interactive-fiction): mean score 0.101 over 8 episodes
