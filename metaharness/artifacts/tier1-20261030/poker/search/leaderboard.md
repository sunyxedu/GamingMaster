# Leaderboard

| # | candidate | parent | mean score | seeds median / min | episodes | stale ticks | S1 calls | S2 calls | failed S1/S2 | Jev→act | Astra→act | Jev p50/max s | Astra p50/max s | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0000 | 0000-tier1_astra_naive | – | 47.333 | 10.830 / -5.500 | 4 | – | 0 | 46 | 0 / 0 | 0% | 100% | – | 56.1 / 69 | ok |
| 0001 | 0001 | 0000 | 484.933 | 644.960 / 80.950 | 4 | – | 0 | 8 | 0 / 1 (0 / 1 timed out) | 0% | 0% | – | 67.2 / 71 | ok |
| 0002 | 0002 | 0000 | -81.257 | -62.280 / -133.070 | 4 | – | 0 | 0 | 0 / 0 | 0% | 0% | – | – | ok |
| 0003 | 0003 | – | 390.478 | 551.990 / 0.000 | 4 | – | 0 | 10 | 0 / 0 | 0% | 2% | – | 57.6 / 80 | ok |
| 0004 | 0004 | – | 484.933 | 644.960 / 80.950 | 4 | – | 0 | 5 | 0 / 0 | 0% | 0% | – | 65.9 / 66 | ok |
| 0005 | 0005 | 0001 | 691.160 | 683.620 / 675.080 | 4 | – | 0 | 11 | 0 / 1 (0 / 1 timed out) | 0% | 1% | – | 93.4 / 121 | ok |
| 0006 | 0006 | 0002 | 82.885 | 92.740 / -227.830 | 4 | – | 0 | 11 | 0 / 0 | 0% | 1% | – | 56.6 / 89 | ok |
| 0007 | 0007 | – | -599.815 | -470.300 / -1130.570 | 4 | – | 0 | 0 | 0 / 0 | 0% | 0% | – | – | ok |
| 0008 | 0008 | 0006 | -24.838 | -21.200 / -46.700 | 4 | – | 0 | 8 | 0 / 0 | 0% | 1% | – | 60.1 / 73 | ok |
| 0009 | 0009 | 0006 | 519.400 | 567.090 / 425.450 | 4 | – | 0 | 12 | 0 / 0 | 0% | 1% | – | 37.1 / 48 | ok |
| 0010 | 0010 | 0005 | 708.430 | 738.250 / 662.690 | 4 | – | 0 | 13 | 0 / 0 | 0% | 1% | – | 75.9 / 146 | ok |
| 0011 | 0011 | 0009 | 390.645 | 461.460 / 176.530 | 4 | – | 0 | 14 | 0 / 0 | 0% | 1% | – | 55.1 / 116 | ok |
| 0012 | 0012 | 0009 | 1021.800 | 1301.670 / 702.770 | 4 | – | 0 | 8 | 0 / 0 | 0% | 1% | – | 82.4 / 93 | ok |
| 0013 | 0013 | 0012 | 930.690 | 1211.020 / 23.910 | 4 | – | 0 | 8 | 0 / 0 | 0% | 1% | – | 62.5 / 69 | ok |
| 0014 | 0014 | 0010 | 654.430 | 726.830 / 519.800 | 4 | – | 0 | 4 | 0 / 0 | 0% | 0% | – | 50.8 / 55 | ok |
| 0015 | 0015 | 0010 | 808.515 | 842.630 / 696.530 | 4 | – | 0 | 6 | 0 / 0 | 0% | 1% | – | 61.2 / 71 | ok |
| 0016 | 0016 | 0012 | 972.915 | 1320.210 / 189.900 | 4 | – | 0 | 8 | 0 / 0 | 0% | 1% | – | 66.5 / 70 | ok |
| 0017 | 0017 | 0015 | 563.640 | 642.940 / 452.410 | 4 | – | 0 | 6 | 0 / 0 | 0% | 1% | – | 59.7 / 120 | ok |
| 0018 | 0018 | 0015 | 600.827 | 714.150 / 245.510 | 4 | – | 0 | 7 | 0 / 0 | 0% | 1% | – | 63.4 / 80 | ok |

`seeds median / min`: the median and the worst per-seed score — a mean over a few seeds is noisy and can be carried by one lucky seed; an improvement that lifts the weak seeds is more likely to hold on unseen seeds. `(= all seeds)`: every seed scored the same, which is what a scenario-independent script does; expect it to break on scenarios it was not written for. `Jev→act` / `Astra→act`: fraction of posted actions preceded by a fresh System-1 / System-2 answer since the previous action (reported, not used for selection). Latency columns are the median per-episode p50 and the worst observed call: design timeouts and pipelining from these measurements, not from assumptions.

Best so far: `0013` (mean score 930.690)
Only candidates whose status is `ok` can be selected: a fatal, partial or erroneous evaluation cannot win, whatever its score. A used-up call budget or the deadline is not an error. `parent` is the candidate a proposal says it was derived from (`parent: NNNN` in proposal.md).

Validation (top candidates re-evaluated on unseen seeds; the final selection is the best train+validation mean among them):
- `0012`: validation 631.487 over 4 episodes; train+validation 826.644
- `0013`: validation 743.980 over 4 episodes; train+validation 837.335
- `0016`: validation 411.072 over 4 episodes; train+validation 691.994

Episodes (score, how it ended, wall time; ✗ = error):
- `0000-tier1_astra_naive`: seed 1103: -5.50 deadline 1800s · seed 2207: 4.50 deadline 1800s · seed 3301: 179.50 deadline 1800s · seed 4409: 10.83 deadline 1800s
- `0001`: seed 1103: 644.96 game_done 881s · seed 2207: 80.95 game_done 737s · seed 3301: 464.35 game_done 843s · seed 4409: 749.47 game_done 886s
- `0002`: seed 1103: -62.28 game_done 319s · seed 2207: -133.07 game_done 297s · seed 3301: -48.90 game_done 304s · seed 4409: -80.78 game_done 326s
- `0003`: seed 1103: 813.25 game_done 1494s · seed 2207: 196.67 deadline 1802s · seed 3301: 0.00 deadline 1801s · seed 4409: 551.99 deadline 1800s
- `0004`: seed 1103: 644.96 game_done 1002s · seed 2207: 80.95 game_done 860s · seed 3301: 464.35 game_done 969s · seed 4409: 749.47 game_done 1007s
- `0005`: seed 1103: 678.68 game_done 967s · seed 2207: 683.62 game_done 795s · seed 3301: 727.26 game_done 951s · seed 4409: 675.08 game_done 971s
- `0006`: seed 1103: 92.74 deadline 1800s · seed 2207: -227.83 game_done 1680s · seed 3301: 418.30 game_done 1793s · seed 4409: 48.33 game_done 1445s
- `0007`: seed 1103: -470.30 game_done 603s · seed 2207: -1130.57 game_done 572s · seed 3301: -289.94 game_done 576s · seed 4409: -508.45 game_done 602s
- `0008`: seed 1103: -46.70 game_done 688s · seed 2207: -3.90 game_done 507s · seed 3301: -21.20 game_done 426s · seed 4409: -27.55 game_done 494s
- `0009`: seed 1103: 452.91 game_done 672s · seed 2207: 567.09 game_done 458s · seed 3301: 632.15 game_done 494s · seed 4409: 425.45 game_done 679s
- `0010`: seed 1103: 662.69 game_done 1653s · seed 2207: 680.52 game_done 1482s · seed 3301: 738.25 game_done 1648s · seed 4409: 752.26 game_done 1661s
- `0011`: seed 1103: 461.46 harness_return 1793s · seed 2207: 176.53 deadline 1800s · seed 3301: 597.28 game_done 1425s · seed 4409: 327.31 game_done 1331s
- `0012`: seed 1103: 1358.26 game_done 969s · seed 2207: 702.77 game_done 1717s · seed 3301: 1301.67 deadline 1802s · seed 4409: 724.50 deadline 1801s
- `0013`: seed 1103: 1452.17 game_done 577s · seed 2207: 23.91 game_done 572s · seed 3301: 1211.02 game_done 639s · seed 4409: 1035.66 game_done 645s
- `0014`: seed 1103: 640.46 harness_return 1785s · seed 2207: 519.80 harness_return 1789s · seed 3301: 726.83 harness_return 1787s · seed 4409: 730.63 game_done 1438s
- `0015`: seed 1103: 696.53 game_done 1690s · seed 2207: 842.63 game_done 1550s · seed 3301: 917.87 game_done 1590s · seed 4409: 777.03 game_done 1438s
- `0016`: seed 1103: 1320.21 game_done 709s · seed 2207: 189.90 game_done 863s · seed 3301: 1333.58 game_done 792s · seed 4409: 1047.97 game_done 1082s
- `0017`: seed 1103: 692.86 deadline 1800s · seed 2207: 642.94 game_done 1768s · seed 3301: 466.35 deadline 1800s · seed 4409: 452.41 harness_return 1785s
- `0018`: seed 1103: 714.15 harness_return 1785s · seed 2207: 245.51 deadline 1800s · seed 3301: 853.25 game_done 1739s · seed 4409: 590.40 deadline 1800s

System-2 latency ≈ 52.3 s + 14.5 ms × output tokens (fit over 159 archived calls, r = 0.55; slowest 146 s): long answers are slow — size timeouts from this, not from short baseline answers.

Score composition (mean per episode, from the adaptor's score_breakdown):
- `0000-tier1_astra_naive`: bb_per_100 47.33, bb_per_hand 0.47, chips_per_hand 47.33, complete 0.00, completed_hands 3.75, illegal_actions 0.00, required_hands 100.00, total_chips 4733.25
- `0001`: bb_per_100 484.93, bb_per_hand 4.85, chips_per_hand 484.93, complete 1.00, completed_hands 100.00, illegal_actions 0.00, required_hands 100.00, total_chips 48493.25
- `0002`: bb_per_100 -81.26, bb_per_hand -0.81, chips_per_hand -81.26, complete 1.00, completed_hands 100.00, illegal_actions 0.00, required_hands 100.00, total_chips -8125.75
- `0003`: bb_per_100 390.48, bb_per_hand 3.90, chips_per_hand 390.48, complete 0.25, completed_hands 58.50, illegal_actions 0.00, required_hands 100.00, total_chips 39047.75
- `0004`: bb_per_100 484.93, bb_per_hand 4.85, chips_per_hand 484.93, complete 1.00, completed_hands 100.00, illegal_actions 0.00, required_hands 100.00, total_chips 48493.25
- `0005`: bb_per_100 691.16, bb_per_hand 6.91, chips_per_hand 691.16, complete 1.00, completed_hands 100.00, illegal_actions 0.00, required_hands 100.00, total_chips 69116.00
- `0006`: bb_per_100 82.88, bb_per_hand 0.83, chips_per_hand 82.88, complete 0.75, completed_hands 83.75, illegal_actions 0.00, required_hands 100.00, total_chips 8288.50
- `0007`: bb_per_100 -599.81, bb_per_hand -6.00, chips_per_hand -599.81, complete 1.00, completed_hands 100.00, illegal_actions 0.00, required_hands 100.00, total_chips -59981.50
- `0008`: bb_per_100 -24.84, bb_per_hand -0.25, chips_per_hand -24.84, complete 1.00, completed_hands 100.00, illegal_actions 0.00, required_hands 100.00, total_chips -2483.75
- `0009`: bb_per_100 519.40, bb_per_hand 5.19, chips_per_hand 519.40, complete 1.00, completed_hands 100.00, illegal_actions 0.00, required_hands 100.00, total_chips 51940.00
- `0010`: bb_per_100 708.43, bb_per_hand 7.08, chips_per_hand 708.43, complete 1.00, completed_hands 100.00, illegal_actions 0.00, required_hands 100.00, total_chips 70843.00
- `0011`: bb_per_100 390.64, bb_per_hand 3.91, chips_per_hand 390.64, complete 0.50, completed_hands 94.00, illegal_actions 0.00, required_hands 100.00, total_chips 39064.50
- `0012`: bb_per_100 1021.80, bb_per_hand 10.22, chips_per_hand 1021.80, complete 0.50, completed_hands 85.00, illegal_actions 0.00, required_hands 100.00, total_chips 102180.00
- `0013`: bb_per_100 930.69, bb_per_hand 9.31, chips_per_hand 930.69, complete 1.00, completed_hands 100.00, illegal_actions 0.00, required_hands 100.00, total_chips 93069.00
- `0014`: bb_per_100 654.43, bb_per_hand 6.54, chips_per_hand 654.43, complete 0.25, completed_hands 92.00, illegal_actions 0.00, required_hands 100.00, total_chips 65443.00
- `0015`: bb_per_100 808.51, bb_per_hand 8.09, chips_per_hand 808.51, complete 1.00, completed_hands 100.00, illegal_actions 0.00, required_hands 100.00, total_chips 80851.50
- `0016`: bb_per_100 972.91, bb_per_hand 9.73, chips_per_hand 972.91, complete 1.00, completed_hands 100.00, illegal_actions 0.00, required_hands 100.00, total_chips 97291.50
- `0017`: bb_per_100 563.64, bb_per_hand 5.64, chips_per_hand 563.64, complete 0.25, completed_hands 71.00, illegal_actions 0.00, required_hands 100.00, total_chips 56364.00
- `0018`: bb_per_100 600.83, bb_per_hand 6.01, chips_per_hand 600.83, complete 0.25, completed_hands 81.75, illegal_actions 0.00, required_hands 100.00, total_chips 60082.75

Held-out evaluations (not used for selection):
- `0013` / heldout (Heads-up no-limit Texas Hold'em — mc-pot-odds-v2): mean score 878.230 over 4 episodes
