# Leaderboard

| # | candidate | parent | mean score | seeds median / min | episodes | stale ticks | S1 calls | S2 calls | failed S1/S2 | Jev→act | Astra→act | Jev p50/max s | Astra p50/max s | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0000 | 0000-tier1_astra_naive | – | 211.500 | 240.000 / 24.000 | 8 | – | 0 | 239 | 0 / 0 | 0% | 100% | – | 53.2 / 112 | ok |
| 0001 | 0001 | 0000 | 2831.500 | 2872.000 / 2668.000 | 8 | – | 0 | 8 | 0 / 0 | 0% | 0% | – | 57.1 / 103 | ok |
| 0002 | 0002 | 0001 | 2877.000 | 2904.000 / 2744.000 | 8 | – | 0 | 39 | 0 / 0 | 0% | 2% | – | 53.5 / 65 | ok |
| 0003 | 0003 | – | 2549.500 | 2928.000 / 1384.000 | 8 | – | 0 | 35 | 0 / 0 | 0% | 2% | – | 49.7 / 108 | ok |
| 0004 | 0004 | 0002 | 2587.500 | 2868.000 / 1808.000 | 8 | – | 0 | 18 | 0 / 0 | 0% | 1% | – | 62.5 / 93 | ok |
| 0005 | 0005 | 0002 | 2140.000 | 2672.000 / 560.000 | 8 | – | 0 | 23 | 0 / 0 | 0% | 1% | – | 54.0 / 71 | ok |
| 0006 | 0006 | 0003 | 1757.750 | 2444.000 / 2.000 | 8 | – | 0 | 21 | 0 / 0 | 0% | 2% | – | 52.5 / 105 | ok |
| 0007 | 0007 | 0002 | 2859.000 | 2920.000 / 2684.000 | 8 | – | 0 | 19 | 0 / 0 | 0% | 1% | – | 69.5 / 91 | ok |
| 0008 | 0008 | – | – | – | 0 | – | 0 | 0 | 0 / 0 | – | – | – | – | ✗ fatal: Reconnecting... 1/8 ( |
| 0009 | 0009 | 0003 | 2976.000 | 2972.000 / 2928.000 | 8 | – | 0 | 8 | 0 / 0 | 0% | 0% | – | – | ok |
| 0010 | 0010 | 0009 | 2994.000 | 3032.000 / 2928.000 | 8 | – | 0 | 12 | 0 / 0 | 0% | 0% | – | 68.4 / 75 | ok |
| 0011 | 0011 | 0003 | 3047.500 | 3052.000 / 2972.000 | 8 | – | 0 | 9 | 0 / 0 | 0% | 0% | – | 83.5 / 83 | ok |
| 0012 | 0012 | 0009 | 2966.000 | 2972.000 / 2924.000 | 8 | – | 0 | 8 | 0 / 0 | 0% | 0% | – | – | ok |
| 0013 | 0013 | 0011 | 3067.500 | 3072.000 / 3056.000 | 8 | – | 0 | 15 | 0 / 0 | 0% | 0% | – | 67.1 / 76 | ok |
| 0014 | 0014 | 0009 | 2982.500 | 2976.000 / 2928.000 | 8 | – | 0 | 21 | 0 / 0 | 0% | 1% | – | 70.6 / 104 | ok |
| 0015 | 0015 | 0011 | 2786.500 | 2900.000 / 2320.000 | 8 | – | 0 | 3 | 0 / 0 | 0% | 0% | – | – | ok |
| 0016 | 0016 | 0013 | 3017.500 | 3040.000 / 2928.000 | 8 | – | 0 | 19 | 0 / 0 | 0% | 1% | – | 72.9 / 96 | ok |
| 0017 | 0017 | 0011 | 3030.500 | 3036.000 / 2940.000 | 8 | – | 0 | 8 | 0 / 0 | 0% | 0% | – | 56.7 / 59 | ok |
| 0018 | 0018 | 0010 | 2954.000 | 2944.000 / 2924.000 | 8 | – | 0 | 18 | 0 / 0 | 0% | 1% | – | 71.4 / 100 | ok |

`seeds median / min`: the median and the worst per-seed score — a mean over a few seeds is noisy and can be carried by one lucky seed; an improvement that lifts the weak seeds is more likely to hold on unseen seeds. `(= all seeds)`: every seed scored the same, which is what a scenario-independent script does; expect it to break on scenarios it was not written for. `Jev→act` / `Astra→act`: fraction of posted actions preceded by a fresh System-1 / System-2 answer since the previous action (reported, not used for selection). Latency columns are the median per-episode p50 and the worst observed call: design timeouts and pipelining from these measurements, not from assumptions.

Best so far: `0013` (mean score 3067.500)
Only candidates whose status is `ok` can be selected: a fatal, partial or erroneous evaluation cannot win, whatever its score. A used-up call budget or the deadline is not an error. `parent` is the candidate a proposal says it was derived from (`parent: NNNN` in proposal.md).

Validation (top candidates re-evaluated on unseen seeds; the final selection is the best train+validation mean among them):
- `0011`: validation 3006.500 over 8 episodes; train+validation 3027.000
- `0013`: validation 3016.500 over 8 episodes; train+validation 3042.000
- `0017`: validation 2999.500 over 8 episodes; train+validation 3015.000

Episodes (score, how it ended, wall time; ✗ = error):
- `0000-tier1_astra_naive`: seed 11: 236.00 deadline 1800s · seed 23: 264.00 deadline 1800s · seed 37: 220.00 deadline 1800s · seed 41: 244.00 deadline 1800s · seed 59: 252.00 deadline 1800s · seed 71: 240.00 deadline 1800s · seed 83: 24.00 deadline 1800s · seed 97: 212.00 deadline 1800s
- `0001`: seed 11: 2864.00 game_done 66s · seed 23: 2780.00 game_done 164s · seed 37: 2908.00 game_done 193s · seed 41: 2880.00 game_done 276s · seed 59: 2668.00 game_done 299s · seed 71: 2908.00 game_done 459s · seed 83: 2872.00 game_done 124s · seed 97: 2772.00 game_done 221s
- `0002`: seed 11: 2904.00 game_done 1645s · seed 23: 2900.00 game_done 1191s · seed 37: 2744.00 game_done 1296s · seed 41: 2972.00 game_done 1473s · seed 59: 2888.00 game_done 1040s · seed 71: 2768.00 game_done 1049s · seed 83: 2916.00 game_done 970s · seed 97: 2924.00 game_done 1216s
- `0003`: seed 11: 2928.00 game_done 1751s · seed 23: 2928.00 game_done 1618s · seed 37: 2928.00 game_done 1738s · seed 41: 2932.00 game_done 1495s · seed 59: 2912.00 game_done 1758s · seed 71: 2932.00 game_done 1531s · seed 83: 1452.00 deadline 1800s · seed 97: 1384.00 deadline 1800s
- `0004`: seed 11: 2896.00 game_done 806s · seed 23: 2868.00 game_done 395s · seed 37: 2416.00 game_done 332s · seed 41: 2928.00 game_done 629s · seed 59: 2208.00 deadline 1800s · seed 71: 1808.00 deadline 1800s · seed 83: 2656.00 game_done 1686s · seed 97: 2920.00 game_done 1447s
- `0005`: seed 11: 2672.00 game_done 1723s · seed 23: 2628.00 deadline 1800s · seed 37: 560.00 deadline 1800s · seed 41: 568.00 deadline 1800s · seed 59: 2708.00 game_done 1800s · seed 71: 2536.00 deadline 1800s · seed 83: 2716.00 game_done 1660s · seed 97: 2732.00 game_done 1483s
- `0006`: seed 11: 2.00 deadline 1800s · seed 23: 4.00 deadline 1800s · seed 37: 2408.00 deadline 1801s · seed 41: 500.00 deadline 1800s · seed 59: 2864.00 game_done 1520s · seed 71: 2928.00 game_done 1459s · seed 83: 2912.00 game_done 1728s · seed 97: 2444.00 deadline 1800s
- `0007`: seed 11: 2916.00 game_done 381s · seed 23: 2928.00 game_done 504s · seed 37: 2924.00 game_done 486s · seed 41: 2920.00 game_done 619s · seed 59: 2684.00 game_done 377s · seed 71: 2684.00 game_done 614s · seed 83: 2896.00 game_done 489s · seed 97: 2920.00 game_done 338s
- `0009`: seed 11: 2940.00 game_done 70s · seed 23: 3040.00 game_done 70s · seed 37: 2928.00 game_done 69s · seed 41: 3040.00 game_done 73s · seed 59: 2928.00 game_done 70s · seed 71: 3032.00 game_done 71s · seed 83: 2928.00 game_done 69s · seed 97: 2972.00 game_done 67s
- `0010`: seed 11: 3036.00 game_done 338s · seed 23: 3032.00 game_done 334s · seed 37: 2928.00 game_done 335s · seed 41: 3040.00 game_done 339s · seed 59: 2972.00 game_done 317s · seed 71: 3040.00 game_done 319s · seed 83: 2932.00 game_done 315s · seed 97: 2972.00 game_done 313s
- `0011`: seed 11: 3040.00 game_done 209s · seed 23: 3056.00 game_done 211s · seed 37: 3068.00 game_done 211s · seed 41: 3104.00 game_done 208s · seed 59: 2972.00 game_done 227s · seed 71: 3052.00 game_done 227s · seed 83: 3036.00 game_done 230s · seed 97: 3052.00 game_done 224s
- `0012`: seed 11: 2932.00 game_done 101s · seed 23: 2972.00 game_done 98s · seed 37: 2924.00 game_done 100s · seed 41: 2972.00 game_done 109s · seed 59: 2928.00 game_done 95s · seed 71: 3036.00 game_done 91s · seed 83: 2928.00 game_done 87s · seed 97: 3036.00 game_done 90s
- `0013`: seed 11: 3072.00 game_done 442s · seed 23: 3056.00 game_done 426s · seed 37: 3056.00 game_done 399s · seed 41: 3072.00 game_done 407s · seed 59: 3056.00 game_done 662s · seed 71: 3100.00 game_done 498s · seed 83: 3072.00 game_done 530s · seed 97: 3056.00 game_done 572s
- `0014`: seed 11: 2976.00 game_done 782s · seed 23: 2928.00 game_done 813s · seed 37: 2928.00 game_done 821s · seed 41: 3040.00 game_done 811s · seed 59: 2932.00 game_done 764s · seed 71: 3040.00 game_done 761s · seed 83: 3044.00 game_done 753s · seed 97: 2972.00 game_done 752s
- `0015`: seed 11: 2900.00 game_done 123s · seed 23: 2928.00 game_done 110s · seed 37: 2320.00 game_done 104s · seed 41: 2900.00 game_done 112s · seed 59: 2876.00 game_done 181s · seed 71: 2808.00 game_done 185s · seed 83: 2636.00 game_done 166s · seed 97: 2924.00 game_done 161s
- `0016`: seed 11: 2928.00 game_done 680s · seed 23: 3036.00 game_done 679s · seed 37: 2932.00 game_done 709s · seed 41: 3040.00 game_done 673s · seed 59: 3040.00 game_done 568s · seed 71: 3056.00 game_done 576s · seed 83: 3056.00 game_done 530s · seed 97: 3052.00 game_done 520s
- `0017`: seed 11: 3036.00 game_done 267s · seed 23: 3036.00 game_done 255s · seed 37: 3052.00 game_done 262s · seed 41: 3056.00 game_done 270s · seed 59: 2940.00 game_done 248s · seed 71: 3052.00 game_done 245s · seed 83: 3036.00 game_done 248s · seed 97: 3036.00 game_done 249s
- `0018`: seed 11: 2928.00 game_done 628s · seed 23: 2944.00 game_done 627s · seed 37: 2924.00 game_done 628s · seed 41: 2940.00 game_done 628s · seed 59: 2928.00 game_done 681s · seed 71: 3028.00 game_done 688s · seed 83: 2968.00 game_done 695s · seed 97: 2972.00 game_done 688s

System-2 latency ≈ 49.6 s + 20.4 ms × output tokens (fit over 433 archived calls, r = 0.51; slowest 112 s): long answers are slow — size timeouts from this, not from short baseline answers.

Score composition (mean per episode, from the adaptor's score_breakdown):
- `0000-tier1_astra_naive`: combined_score 211.50, max_tile 29.00, merge_score 182.50
- `0001`: combined_score 2831.50, max_tile 256.00, merge_score 2575.50
- `0002`: combined_score 2877.00, max_tile 256.00, merge_score 2621.00
- `0003`: combined_score 2549.50, max_tile 224.00, merge_score 2325.50
- `0004`: combined_score 2587.50, max_tile 208.00, merge_score 2379.50
- `0005`: combined_score 2140.00, max_tile 208.00, merge_score 1932.00
- `0006`: combined_score 1757.75, max_tile 168.75, merge_score 1589.00
- `0007`: combined_score 2859.00, max_tile 256.00, merge_score 2603.00
- `0009`: combined_score 2976.00, max_tile 256.00, merge_score 2720.00
- `0010`: combined_score 2994.00, max_tile 256.00, merge_score 2738.00
- `0011`: combined_score 3047.50, max_tile 256.00, merge_score 2791.50
- `0012`: combined_score 2966.00, max_tile 256.00, merge_score 2710.00
- `0013`: combined_score 3067.50, max_tile 256.00, merge_score 2811.50
- `0014`: combined_score 2982.50, max_tile 256.00, merge_score 2726.50
- `0015`: combined_score 2786.50, max_tile 240.00, merge_score 2546.50
- `0016`: combined_score 3017.50, max_tile 256.00, merge_score 2761.50
- `0017`: combined_score 3030.50, max_tile 256.00, merge_score 2774.50
- `0018`: combined_score 2954.00, max_tile 256.00, merge_score 2698.00

Held-out evaluations (not used for selection):
- `0013` / heldout (2048): mean score 3066.000 over 8 episodes
