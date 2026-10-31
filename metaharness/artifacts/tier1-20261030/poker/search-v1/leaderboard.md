# Leaderboard

| # | candidate | parent | mean score | episodes | stale ticks | S1 calls | S2 calls | failed S1/S2 | Jev→act | Astra→act | Jev p50/max s | Astra p50/max s | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0000 | 0000-tier1_astra_naive | – | 59.625 | 4 | – | 0 | 28 | 0 / 0 | 0% | 100% | – | 40.5 / 281 | ok |
| 0001 | 0001 | 0000 | -79.992 | 4 | – | 0 | 0 | 0 / 0 | 0% | 0% | – | – | ok |
| 0002 | 0002 | 0001 | 109.510 | 4 | – | 0 | 0 | 0 / 0 | 0% | 0% | – | – | ok |
| 0003 | 0003 | – | 548.688 | 4 | – | 0 | 0 | 0 / 0 | 0% | 0% | – | – | ok |
| 0004 | 0004 | 0003 | 517.750 | 4 | – | 0 | 0 | 0 / 0 | 0% | 0% | – | – | ok |
| 0005 | 0005 | 0002 | 33.023 | 4 | – | 0 | 0 | 0 / 0 | 0% | 0% | – | – | ok |
| 0006 | 0006 | 0002 | 14.027 | 4 | – | 0 | 0 | 0 / 0 | 0% | 0% | – | – | ok |
| 0007 | 0007 | 0003 | 268.688 | 4 | – | 0 | 0 | 0 / 0 | 0% | 0% | – | – | ok |
| 0008 | 0008 | 0006 | 2.240 | 4 | – | 0 | 0 | 0 / 0 | 0% | 0% | – | – | ok |
| 0009 | 0009 | 0004 | 681.062 | 4 | – | 0 | 0 | 0 / 0 | 0% | 0% | – | – | ✗ 1 episode error(s), first: game hung: a call into the game (tick/step/observe) did not return, so neither the harness no |
| 0010 | 0010 | 0003 | 659.438 | 4 | – | 0 | 0 | 0 / 0 | 0% | 0% | – | – | ok |
| 0011 | 0011 | 0003 | 410.000 | 4 | – | 0 | 0 | 0 / 0 | 0% | 0% | – | – | ✗ 1 episode error(s), first: game hung: a call into the game (tick/step/observe) did not return, so neither the harness no |
| 0012 | 0012 | 0009 | 548.842 | 4 | – | 0 | 0 | 0 / 0 | 0% | 0% | – | – | ok |
| 0013 | 0013 | 0012 | 788.312 | 4 | – | 0 | 0 | 0 / 0 | 0% | 0% | – | – | ok |
| 0014 | 0014 | 0012 | 799.362 | 4 | – | 0 | 0 | 0 / 0 | 0% | 0% | – | – | ok |
| 0015 | 0015 | 0007 | 539.312 | 4 | – | 0 | 0 | 0 / 0 | 0% | 0% | – | – | ok |
| 0016 | 0016 | 0014 | 642.643 | 4 | – | 0 | 0 | 0 / 0 | 0% | 0% | – | – | ok |
| 0017 | 0017 | 0015 | 475.562 | 4 | – | 0 | 0 | 0 / 0 | 0% | 0% | – | – | ok |
| 0018 | 0018 | 0013 | 529.375 | 4 | – | 0 | 0 | 0 / 0 | 0% | 0% | – | – | ok |

`Jev→act` / `Astra→act`: fraction of posted actions preceded by a fresh System-1 / System-2 answer since the previous action (reported, not used for selection). Latency columns are the median per-episode p50 and the worst observed call: design timeouts and pipelining from these measurements, not from assumptions.

Best so far: `0014` (mean score 799.362)
Only candidates whose status is `ok` can be selected: a fatal, partial or erroneous evaluation cannot win, whatever its score. A used-up call budget or the deadline is not an error. `parent` is the candidate a proposal says it was derived from (`parent: NNNN` in proposal.md).

Validation (top candidates re-evaluated on unseen seeds; the final selection is the best train+validation mean among them):
- `0010`: validation 399.438 over 4 episodes; train+validation 529.438
- `0013`: validation 788.375 over 4 episodes; train+validation 788.344
- `0014`: validation 829.375 over 4 episodes; train+validation 814.369

Episodes (score, how it ended, wall time; ✗ = error):
- `0000-tier1_astra_naive`: seed 1103: 200.50 deadline 1800s · seed 2207: -5.75 deadline 1800s · seed 3301: 31.25 deadline 1800s · seed 4409: 12.50 deadline 1800s
- `0001`: seed 1103: 8.53 game_done 609s · seed 2207: -559.76 game_done 563s · seed 3301: -90.78 game_done 577s · seed 4409: 322.04 game_done 599s
- `0002`: seed 1103: -51.75 game_done 652s · seed 2207: -27.82 game_done 633s · seed 3301: 198.76 game_done 640s · seed 4409: 318.85 game_done 633s
- `0003`: seed 1103: 836.00 game_done 1483s · seed 2207: 83.00 game_done 1430s · seed 3301: 819.00 game_done 1458s · seed 4409: 456.75 game_done 1468s
- `0004`: seed 1103: 1172.25 harness_return 1800s · seed 2207: 82.75 harness_return 1800s · seed 3301: 542.50 harness_return 1800s · seed 4409: 273.50 harness_return 1800s
- `0005`: seed 1103: -207.67 game_done 1284s · seed 2207: 195.19 game_done 912s · seed 3301: 55.15 game_done 1075s · seed 4409: 89.42 game_done 1120s
- `0006`: seed 1103: -16.15 game_done 1485s · seed 2207: -123.97 game_done 1109s · seed 3301: -77.11 game_done 1317s · seed 4409: 273.34 game_done 1360s
- `0007`: seed 1103: 240.00 harness_return 1800s · seed 2207: -158.50 deadline 1806s · seed 3301: 607.50 harness_return 1800s · seed 4409: 385.75 deadline 1804s
- `0008`: seed 1103: 112.60 game_done 1341s · seed 2207: -30.83 game_done 1239s · seed 3301: -294.60 game_done 1319s · seed 4409: 221.79 game_done 1328s
- `0009`: seed 1103: 1168.25 deadline 1803s · seed 2207: 486.25 game_error 1813s ✗ game hung: a call into the game (tick/step/observe) did not return, so neither the harness nor the … · seed 3301: 726.00 deadline 1808s · seed 4409: 343.75 harness_return 1800s
- `0010`: seed 1103: 945.50 harness_return 1755s · seed 2207: 276.00 harness_return 1751s · seed 3301: 787.50 harness_return 1742s · seed 4409: 628.75 harness_return 1743s
- `0011`: seed 1103: 644.00 harness_return 1800s · seed 2207: 46.75 game_error 1813s ✗ game hung: a call into the game (tick/step/observe) did not return, so neither the harness nor the … · seed 3301: 583.50 deadline 1806s · seed 4409: 365.75 deadline 1800s
- `0012`: seed 1103: 840.50 game_done 1703s · seed 2207: 269.75 game_done 1684s · seed 3301: 590.12 game_done 1694s · seed 4409: 495.00 game_done 1703s
- `0013`: seed 1103: 849.25 game_done 1595s · seed 2207: 546.25 game_done 1608s · seed 3301: 1195.75 game_done 1628s · seed 4409: 562.00 game_done 1596s
- `0014`: seed 1103: 881.25 game_done 1284s · seed 2207: 418.57 game_done 1208s · seed 3301: 1148.00 game_done 1281s · seed 4409: 749.63 game_done 1280s
- `0015`: seed 1103: 853.75 harness_return 1800s · seed 2207: 272.25 harness_return 1800s · seed 3301: 632.50 deadline 1804s · seed 4409: 398.75 deadline 1804s
- `0016`: seed 1103: 810.87 game_done 1741s · seed 2207: 497.92 game_done 1727s · seed 3301: 646.19 game_done 1734s · seed 4409: 615.59 harness_return 1725s
- `0017`: seed 1103: 848.50 deadline 1802s · seed 2207: 41.50 deadline 1800s · seed 3301: 630.50 deadline 1803s · seed 4409: 381.75 harness_return 1800s
- `0018`: seed 1103: 834.00 harness_return 1750s · seed 2207: 348.25 harness_return 1733s · seed 3301: 419.00 harness_return 1750s · seed 4409: 516.25 harness_return 1744s

System-2 latency ≈ 67.4 s + -218.4 ms × output tokens (fit over 24 archived calls, r = -0.16; slowest 281 s): long answers are slow — size timeouts from this, not from short baseline answers.

Score composition (mean per episode, from the adaptor's score_breakdown):
- `0000-tier1_astra_naive`: bb_per_100 59.62, bb_per_hand 0.60, chips_per_hand 59.62, complete 0.00, completed_hands 2.00, illegal_actions 0.00, required_hands 100.00, total_chips 5962.50
- `0001`: bb_per_100 -79.99, bb_per_hand -0.80, chips_per_hand -79.99, complete 1.00, completed_hands 100.00, illegal_actions 0.00, required_hands 100.00, total_chips -7999.25
- `0002`: bb_per_100 109.51, bb_per_hand 1.10, chips_per_hand 109.51, complete 1.00, completed_hands 100.00, illegal_actions 0.00, required_hands 100.00, total_chips 10951.00
- `0003`: bb_per_100 548.69, bb_per_hand 5.49, chips_per_hand 548.69, complete 1.00, completed_hands 100.00, illegal_actions 0.00, required_hands 100.00, total_chips 54868.75
- `0004`: bb_per_100 517.75, bb_per_hand 5.18, chips_per_hand 517.75, complete 0.00, completed_hands 76.75, illegal_actions 0.00, required_hands 100.00, total_chips 51775.00
- `0005`: bb_per_100 33.02, bb_per_hand 0.33, chips_per_hand 33.02, complete 1.00, completed_hands 100.00, illegal_actions 0.00, required_hands 100.00, total_chips 3302.25
- `0006`: bb_per_100 14.03, bb_per_hand 0.14, chips_per_hand 14.03, complete 1.00, completed_hands 100.00, illegal_actions 0.00, required_hands 100.00, total_chips 1402.75
- `0007`: bb_per_100 268.69, bb_per_hand 2.69, chips_per_hand 268.69, complete 0.00, completed_hands 43.25, illegal_actions 0.00, required_hands 100.00, total_chips 26868.75
- `0008`: bb_per_100 2.24, bb_per_hand 0.02, chips_per_hand 2.24, complete 1.00, completed_hands 100.00, illegal_actions 0.00, required_hands 100.00, total_chips 224.00
- `0009`: bb_per_100 746.00, bb_per_hand 7.46, chips_per_hand 746.00, complete 0.00, completed_hands 66.33, illegal_actions 0.00, required_hands 100.00, total_chips 74600.00
- `0010`: bb_per_100 659.44, bb_per_hand 6.59, chips_per_hand 659.44, complete 0.00, completed_hands 62.00, illegal_actions 0.00, required_hands 100.00, total_chips 65943.75
- `0011`: bb_per_100 531.08, bb_per_hand 5.31, chips_per_hand 531.08, complete 0.00, completed_hands 46.67, illegal_actions 0.00, required_hands 100.00, total_chips 53108.33
- `0012`: bb_per_100 548.84, bb_per_hand 5.49, chips_per_hand 548.84, complete 1.00, completed_hands 100.00, illegal_actions 0.00, required_hands 100.00, total_chips 54884.25
- `0013`: bb_per_100 788.31, bb_per_hand 7.88, chips_per_hand 788.31, complete 1.00, completed_hands 100.00, illegal_actions 0.00, required_hands 100.00, total_chips 78831.25
- `0014`: bb_per_100 799.36, bb_per_hand 7.99, chips_per_hand 799.36, complete 1.00, completed_hands 100.00, illegal_actions 0.00, required_hands 100.00, total_chips 79936.25
- `0015`: bb_per_100 539.31, bb_per_hand 5.39, chips_per_hand 539.31, complete 0.00, completed_hands 55.75, illegal_actions 0.00, required_hands 100.00, total_chips 53931.25
- `0016`: bb_per_100 642.64, bb_per_hand 6.43, chips_per_hand 642.64, complete 0.75, completed_hands 97.25, illegal_actions 0.00, required_hands 100.00, total_chips 64264.25
- `0017`: bb_per_100 475.56, bb_per_hand 4.76, chips_per_hand 475.56, complete 0.00, completed_hands 45.75, illegal_actions 0.00, required_hands 100.00, total_chips 47556.25
- `0018`: bb_per_100 529.38, bb_per_hand 5.29, chips_per_hand 529.38, complete 0.00, completed_hands 81.25, illegal_actions 0.00, required_hands 100.00, total_chips 52937.50

Held-out evaluations (not used for selection):
- `0014` / heldout (Heads-up no-limit Texas Hold'em — heuristic-v1): mean score 724.247 over 4 episodes
