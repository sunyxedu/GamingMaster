# Leaderboard

| # | candidate | parent | mean score | seeds median / min | episodes | stale ticks | S1 calls | S2 calls | failed S1/S2 | Jev→act | Astra→act | Jev p50/max s | Astra p50/max s | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0000 | 0000-tier1_astra_naive | – | -4.875 | -5.000 / -16.000 | 8 | – | 0 | 179 | 0 / 0 | 0% | 100% | – | 54.6 / 130 | ok |
| 0001 | 0001 | 0000 | 5.000 | 2.000 / 0.000 | 8 | – | 0 | 8 | 0 / 0 | 0% | 0% | – | – | ok |
| 0002 | 0002 | 0000 | -23.000 | -7.000 / -75.000 | 8 | – | 0 | 30 | 0 / 0 | 0% | 7% | – | 59.2 / 107 | ok |
| 0003 | 0003 | 0000 | -2.000 | -2.000 / -6.000 | 8 | – | 0 | 80 | 0 / 2 (0 / 2 timed out) | 0% | 91% | – | 58.1 / 114 | ok |
| 0004 | 0004 | 0001 | 4.625 | 4.000 / -8.000 | 8 | – | 0 | 17 | 0 / 0 | 0% | 2% | – | 67.0 / 95 | ok |
| 0005 | 0005 | 0004 | -25.750 | -10.000 / -73.000 | 8 | – | 0 | 24 | 0 / 4 (0 / 4 timed out) | 0% | 3% | – | 82.2 / 117 | ok |
| 0006 | 0006 | 0002 | -0.250 | -2.000 / -12.000 | 8 | – | 0 | 0 | 0 / 0 | 0% | 0% | – | – | ok |
| 0007 | 0007 | 0006 | -2.875 | -2.000 / -18.000 | 8 | – | 0 | 0 | 0 / 0 | 0% | 0% | – | – | ok |
| 0008 | 0008 | 0004 | 3.000 | 3.000 / -1.000 | 8 | – | 0 | 15 | 0 / 0 | 0% | 4% | – | 69.7 / 144 | ok |
| 0009 | 0009 | 0000 | -3.875 | 0.000 / -23.000 | 8 | – | 0 | 40 | 0 / 0 | 0% | 11% | – | 113.5 / 216 | ok |
| 0010 | 0010 | 0001 | 2.625 | 1.000 / -1.000 | 8 | – | 0 | 8 | 0 / 0 | 0% | 0% | – | – | ok |
| 0011 | 0011 | 0004 | 9.000 | 2.000 / -13.000 | 8 | – | 0 | 20 | 0 / 0 | 0% | 4% | – | 88.1 / 151 | ok |
| 0012 | 0012 | 0008 | -4.250 | -2.000 / -25.000 | 8 | – | 0 | 13 | 0 / 0 | 0% | 2% | – | 65.1 / 94 | ok |
| 0013 | 0013 | 0011 | 6.625 | 4.000 / -2.000 | 8 | – | 0 | 10 | 0 / 0 | 0% | 1% | – | 37.4 / 38 | ok |
| 0014 | 0014 | 0001 | 3.375 | 1.000 / -6.000 | 8 | – | 0 | 23 | 0 / 0 | 0% | 6% | – | 53.0 / 107 | ok |
| 0015 | 0015 | 0012 | 5.250 | 4.000 / -3.000 | 8 | – | 0 | 10 | 0 / 0 | 0% | 1% | – | 113.0 / 155 | ok |
| 0016 | 0016 | 0011 | 14.500 | 10.000 / 2.000 | 8 | – | 0 | 15 | 0 / 0 | 0% | 3% | – | 101.6 / 159 | ok |
| 0017 | 0017 | 0015 | 2.125 | 3.000 / -5.000 | 8 | – | 0 | 14 | 0 / 0 | 0% | 4% | – | 78.3 / 140 | ok |
| 0018 | 0018 | 0015 | 0.625 | 3.000 / -19.000 | 8 | – | 0 | 19 | 0 / 0 | 0% | 6% | – | 51.6 / 142 | ok |

`seeds median / min`: the median and the worst per-seed score — a mean over a few seeds is noisy and can be carried by one lucky seed; an improvement that lifts the weak seeds is more likely to hold on unseen seeds. `(= all seeds)`: every seed scored the same, which is what a scenario-independent script does; expect it to break on scenarios it was not written for. `Jev→act` / `Astra→act`: fraction of posted actions preceded by a fresh System-1 / System-2 answer since the previous action (reported, not used for selection). Latency columns are the median per-episode p50 and the worst observed call: design timeouts and pipelining from these measurements, not from assumptions.

Best so far: `0016` (mean score 14.500)
Only candidates whose status is `ok` can be selected: a fatal, partial or erroneous evaluation cannot win, whatever its score. A used-up call budget or the deadline is not an error. `parent` is the candidate a proposal says it was derived from (`parent: NNNN` in proposal.md).

Validation (top candidates re-evaluated on unseen seeds; the final selection is the best train+validation mean among them):
- `0011`: validation 10.625 over 8 episodes; train+validation 9.812
- `0013`: validation 7.125 over 8 episodes; train+validation 6.875
- `0016`: validation 11.000 over 8 episodes; train+validation 12.750

Episodes (score, how it ended, wall time; ✗ = error):
- `0000-tier1_astra_naive`: seed 1100: 6.00 game_done 1654s · seed 1101: -6.00 deadline 1800s · seed 1200: -6.00 deadline 1800s · seed 1201: -16.00 deadline 1800s · seed 1300: -8.00 deadline 1800s · seed 1301: 0.00 deadline 1800s · seed 1400: -4.00 deadline 1800s · seed 1401: -5.00 deadline 1800s
- `0001`: seed 1100: 2.00 game_done 201s · seed 1101: 1.00 game_done 201s · seed 1200: 3.00 game_done 201s · seed 1201: 2.00 game_done 201s · seed 1300: 24.00 game_done 201s · seed 1301: 8.00 game_done 201s · seed 1400: 0.00 game_done 201s · seed 1401: 0.00 game_done 201s
- `0002`: seed 1100: -4.00 game_done 1192s · seed 1101: 6.00 game_done 1253s · seed 1200: -14.00 game_done 1088s · seed 1201: -21.00 deadline 1800s · seed 1300: -75.00 game_done 606s · seed 1301: -64.00 game_done 856s · seed 1400: -7.00 game_done 547s · seed 1401: -5.00 game_done 318s
- `0003`: seed 1100: -3.00 deadline 1800s · seed 1101: -3.00 deadline 1800s · seed 1200: -1.00 deadline 1800s · seed 1201: 0.00 deadline 1800s · seed 1300: -2.00 deadline 1800s · seed 1301: -2.00 deadline 1800s · seed 1400: 1.00 deadline 1800s · seed 1401: -6.00 deadline 1800s
- `0004`: seed 1100: 2.00 game_done 201s · seed 1101: -8.00 game_done 201s · seed 1200: 4.00 game_done 331s · seed 1201: 6.00 game_done 507s · seed 1300: 14.00 game_done 615s · seed 1301: 12.00 game_done 401s · seed 1400: 3.00 game_done 239s · seed 1401: 4.00 game_done 386s
- `0005`: seed 1100: -2.00 game_done 797s · seed 1101: -2.00 game_done 921s · seed 1200: -73.00 game_done 1657s · seed 1201: -73.00 game_done 1191s · seed 1300: -10.00 game_done 280s · seed 1301: 4.00 game_done 201s · seed 1400: -25.00 game_done 598s · seed 1401: -25.00 game_done 1080s
- `0006`: seed 1100: -2.00 game_done 0s · seed 1101: 4.00 game_done 0s · seed 1200: -6.00 game_done 0s · seed 1201: -12.00 game_done 0s · seed 1300: -11.00 game_done 0s · seed 1301: 13.00 game_done 0s · seed 1400: 24.00 game_done 0s · seed 1401: -12.00 game_done 0s
- `0007`: seed 1100: -2.00 game_done 0s · seed 1101: 4.00 game_done 0s · seed 1200: -6.00 game_done 0s · seed 1201: -12.00 game_done 0s · seed 1300: -11.00 game_done 0s · seed 1301: 13.00 game_done 0s · seed 1400: 9.00 game_done 0s · seed 1401: -18.00 game_done 0s
- `0008`: seed 1100: 4.00 game_done 161s · seed 1101: 0.00 game_done 316s · seed 1200: 4.00 game_done 296s · seed 1201: 0.00 game_done 324s · seed 1300: 13.00 game_done 201s · seed 1301: -1.00 game_done 125s · seed 1400: 3.00 game_done 132s · seed 1401: 1.00 game_done 238s
- `0009`: seed 1100: 3.00 game_done 926s · seed 1101: -23.00 game_done 1050s · seed 1200: -6.00 game_done 997s · seed 1201: -18.00 game_done 911s · seed 1300: 15.00 game_done 847s · seed 1301: 9.00 game_done 1080s · seed 1400: 0.00 game_done 853s · seed 1401: -11.00 game_done 1006s
- `0010`: seed 1100: 3.00 game_done 1s · seed 1101: 0.00 game_done 1s · seed 1200: 4.00 game_done 1s · seed 1201: 0.00 game_done 1s · seed 1300: 13.00 game_done 1s · seed 1301: -1.00 game_done 1s · seed 1400: 1.00 game_done 1s · seed 1401: 1.00 game_done 1s
- `0011`: seed 1100: 1.00 game_done 351s · seed 1101: -3.00 game_done 332s · seed 1200: 11.00 game_done 422s · seed 1201: 1.00 game_done 407s · seed 1300: 41.00 game_done 733s · seed 1301: 32.00 game_done 626s · seed 1400: -13.00 game_done 385s · seed 1401: 2.00 game_done 518s
- `0012`: seed 1100: -3.00 game_done 201s · seed 1101: -2.00 game_done 201s · seed 1200: -11.00 game_done 184s · seed 1201: -18.00 game_done 201s · seed 1300: 13.00 game_done 200s · seed 1301: -25.00 game_done 449s · seed 1400: 3.00 game_done 201s · seed 1401: 9.00 game_done 163s
- `0013`: seed 1100: 3.00 game_done 97s · seed 1101: -2.00 game_done 143s · seed 1200: 4.00 game_done 71s · seed 1201: 6.00 game_done 140s · seed 1300: 22.00 game_done 142s · seed 1301: 12.00 game_done 68s · seed 1400: 4.00 game_done 69s · seed 1401: 4.00 game_done 67s
- `0014`: seed 1100: 2.00 game_done 710s · seed 1101: 1.00 game_done 321s · seed 1200: -2.00 game_done 483s · seed 1201: -6.00 game_done 402s · seed 1300: 13.00 game_done 519s · seed 1301: 19.00 game_done 702s · seed 1400: 0.00 game_done 401s · seed 1401: 0.00 game_done 347s
- `0015`: seed 1100: -3.00 game_done 320s · seed 1101: 1.00 game_done 257s · seed 1200: 4.00 game_done 321s · seed 1201: 11.00 game_done 321s · seed 1300: 17.00 game_done 320s · seed 1301: 8.00 game_done 320s · seed 1400: 3.00 game_done 329s · seed 1401: 1.00 game_done 320s
- `0016`: seed 1100: 2.00 game_done 536s · seed 1101: 10.00 game_done 503s · seed 1200: 26.00 game_done 658s · seed 1201: 8.00 game_done 486s · seed 1300: 35.00 game_done 543s · seed 1301: 28.00 game_done 294s · seed 1400: 4.00 game_done 373s · seed 1401: 3.00 game_done 275s
- `0017`: seed 1100: -3.00 game_done 321s · seed 1101: -3.00 game_done 330s · seed 1200: 4.00 game_done 229s · seed 1201: 11.00 game_done 321s · seed 1300: -5.00 game_done 199s · seed 1301: 9.00 game_done 413s · seed 1400: 3.00 game_done 327s · seed 1401: 1.00 game_done 328s
- `0018`: seed 1100: 4.00 game_done 468s · seed 1101: 0.00 game_done 333s · seed 1200: 4.00 game_done 331s · seed 1201: -19.00 game_done 364s · seed 1300: 11.00 game_done 186s · seed 1301: 2.00 game_done 302s · seed 1400: 3.00 game_done 404s · seed 1401: 0.00 game_done 494s

System-2 latency ≈ 49.2 s + 24.2 ms × output tokens (fit over 449 archived calls, r = 0.85; slowest 216 s): long answers are slow — size timeouts from this, not from short baseline answers.

Score composition (mean per episode, from the adaptor's score_breakdown):
- `0000-tier1_astra_naive`: agent_vp 14.75, fallback_count 0.00, illegal_action_count 0.12, opponent_vp 19.62, truncated 0.00, vp_margin -4.88, win_score 0.12
- `0001`: agent_vp 35.50, fallback_count 0.00, illegal_action_count 0.00, opponent_vp 30.50, truncated 0.00, vp_margin 5.00, win_score 0.88
- `0002`: agent_vp 23.38, fallback_count 0.00, illegal_action_count 0.00, opponent_vp 46.38, truncated 0.00, vp_margin -23.00, win_score 0.12
- `0003`: agent_vp 5.75, fallback_count 0.00, illegal_action_count 0.00, opponent_vp 7.75, truncated 0.00, vp_margin -2.00, win_score 0.00
- `0004`: agent_vp 36.62, fallback_count 0.00, illegal_action_count 0.00, opponent_vp 32.00, truncated 0.00, vp_margin 4.62, win_score 0.88
- `0005`: agent_vp 27.50, fallback_count 0.00, illegal_action_count 0.00, opponent_vp 53.25, truncated 0.00, vp_margin -25.75, win_score 0.12
- `0006`: agent_vp 38.25, fallback_count 0.00, illegal_action_count 0.00, opponent_vp 38.50, truncated 0.00, vp_margin -0.25, win_score 0.38
- `0007`: agent_vp 35.50, fallback_count 0.00, illegal_action_count 0.00, opponent_vp 38.38, truncated 0.00, vp_margin -2.88, win_score 0.38
- `0008`: agent_vp 37.75, fallback_count 0.00, illegal_action_count 0.00, opponent_vp 34.75, truncated 0.00, vp_margin 3.00, win_score 0.81
- `0009`: agent_vp 33.38, fallback_count 0.00, illegal_action_count 0.00, opponent_vp 37.25, truncated 0.00, vp_margin -3.88, win_score 0.44
- `0010`: agent_vp 36.88, fallback_count 0.00, illegal_action_count 0.00, opponent_vp 34.25, truncated 0.00, vp_margin 2.62, win_score 0.81
- `0011`: agent_vp 39.00, fallback_count 0.00, illegal_action_count 0.00, opponent_vp 30.00, truncated 0.00, vp_margin 9.00, win_score 0.75
- `0012`: agent_vp 33.12, fallback_count 0.00, illegal_action_count 0.00, opponent_vp 37.38, truncated 0.00, vp_margin -4.25, win_score 0.38
- `0013`: agent_vp 36.75, fallback_count 0.00, illegal_action_count 0.00, opponent_vp 30.12, truncated 0.00, vp_margin 6.62, win_score 0.88
- `0014`: agent_vp 33.50, fallback_count 0.00, illegal_action_count 0.00, opponent_vp 30.12, truncated 0.00, vp_margin 3.38, win_score 0.62
- `0015`: agent_vp 38.12, fallback_count 0.00, illegal_action_count 0.00, opponent_vp 32.88, truncated 0.00, vp_margin 5.25, win_score 0.88
- `0016`: agent_vp 38.00, fallback_count 0.00, illegal_action_count 0.00, opponent_vp 23.50, truncated 0.00, vp_margin 14.50, win_score 1.00
- `0017`: agent_vp 36.25, fallback_count 0.00, illegal_action_count 0.00, opponent_vp 34.12, truncated 0.00, vp_margin 2.12, win_score 0.62
- `0018`: agent_vp 34.62, fallback_count 0.00, illegal_action_count 0.00, opponent_vp 34.00, truncated 0.00, vp_margin 0.62, win_score 0.75

Held-out evaluations (not used for selection):
- `0016` / heldout (Dominion (offline pyminion)): mean score 1.625 over 8 episodes
