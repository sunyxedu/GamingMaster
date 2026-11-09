# Leaderboard

| # | candidate | parent | mean score | episodes | stale ticks | S1 calls | S2 calls | failed S1/S2 | Jev→act | Astra→act | Jev p50/max s | Astra p50/max s | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0000 | 0000-tier1_astra_naive | – | -4.500 | 6 | – | 0 | 141 | 0 / 1 (0 / 1 timed out) | 0% | 100% | – | 34.5 / 242 | ok |
| 0001 | 0001 | 0000 | 0.333 | 6 | – | 0 | 6 | 0 / 0 | 0% | 0% | – | – | ok |
| 0002 | 0002 | – | -2.167 | 6 | – | 0 | 28 | 0 / 0 | 0% | 12% | – | 19.9 / 81 | ok |
| 0003 | 0003 | – | 9.500 | 6 | – | 0 | 6 | 0 / 0 | 0% | 0% | – | – | ok |
| 0004 | 0004 | 0003 | 7.667 | 6 | – | 0 | 6 | 0 / 0 | 0% | 0% | – | – | ok |
| 0005 | 0005 | 0001 | 5.167 | 6 | – | 0 | 6 | 0 / 0 | 0% | 0% | – | – | ok |
| 0006 | 0006 | 0003 | 7.667 | 6 | – | 0 | 9 | 0 / 0 | 0% | 1% | – | 12.7 / 27 | ok |
| 0007 | 0007 | 0006 | 5.500 | 6 | – | 0 | 8 | 0 / 0 | 0% | 0% | – | 14.0 / 17 | ok |
| 0008 | 0008 | 0005 | 13.333 | 6 | – | 0 | 6 | 0 / 0 | 0% | 0% | – | 23.4 / 23 | ok |
| 0009 | 0009 | 0005 | 7.500 | 6 | – | 0 | 6 | 0 / 0 | 0% | 0% | – | 31.6 / 39 | ok |
| 0010 | 0010 | 0008 | 12.833 | 6 | – | 0 | 6 | 0 / 0 | 0% | 0% | – | – | ok |
| 0011 | 0011 | 0006 | 4.000 | 6 | – | 0 | 12 | 0 / 0 | 0% | 2% | – | 18.2 / 30 | ok |
| 0012 | 0012 | 0009 | 5.500 | 6 | – | 0 | 6 | 0 / 0 | 0% | 0% | – | – | ok |
| 0013 | 0013 | 0008 | 14.667 | 6 | – | 0 | 6 | 0 / 0 | 0% | 0% | – | – | ok |
| 0014 | 0014 | 0012 | 2.833 | 6 | – | 0 | 6 | 0 / 0 | 0% | 0% | – | – | ok |
| 0015 | 0015 | 0010 | 8.500 | 6 | – | 0 | 6 | 0 / 0 | 0% | 0% | – | – | ok |
| 0016 | 0016 | 0013 | 12.667 | 6 | – | 0 | 6 | 0 / 0 | 0% | 0% | – | – | ok |
| 0017 | 0017 | 0013 | 14.833 | 6 | – | 0 | 6 | 0 / 0 | 0% | 0% | – | – | ok |
| 0018 | 0018 | 0009 | 7.500 | 6 | – | 0 | 6 | 0 / 0 | 0% | 0% | – | – | ok |

`Jev→act` / `Astra→act`: fraction of posted actions preceded by a fresh System-1 / System-2 answer since the previous action (reported, not used for selection). Latency columns are the median per-episode p50 and the worst observed call: design timeouts and pipelining from these measurements, not from assumptions.

Best so far: `0017` (mean score 14.833)
Only candidates whose status is `ok` can be selected: a fatal, partial or erroneous evaluation cannot win, whatever its score. A used-up call budget or the deadline is not an error. `parent` is the candidate a proposal says it was derived from (`parent: NNNN` in proposal.md).

Validation (top candidates re-evaluated on unseen seeds; the final selection is the best train+validation mean among them):
- `0008`: validation 6.000 over 6 episodes; train+validation 9.667
- `0013`: validation 10.333 over 6 episodes; train+validation 12.500
- `0017`: validation 12.000 over 6 episodes; train+validation 13.417

Episodes (score, how it ended, wall time; ✗ = error):
- `0000-tier1_astra_naive`: seed 1100: 0.00 deadline 1800s · seed 1101: 8.00 deadline 1800s · seed 1200: -7.00 deadline 1800s · seed 1201: -8.00 deadline 1800s · seed 1300: -6.00 deadline 1800s · seed 1301: -14.00 deadline 1800s
- `0001`: seed 1100: 12.00 game_done 152s · seed 1101: 2.00 game_done 152s · seed 1200: -11.00 game_done 152s · seed 1201: -12.00 game_done 152s · seed 1300: -2.00 game_done 150s · seed 1301: 13.00 game_done 150s
- `0002`: seed 1100: 8.00 game_done 181s · seed 1101: -9.00 game_done 180s · seed 1200: 3.00 game_done 462s · seed 1201: -11.00 game_done 328s · seed 1300: -8.00 game_done 211s · seed 1301: 4.00 game_done 241s
- `0003`: seed 1100: 1.00 game_done 133s · seed 1101: -1.00 game_done 125s · seed 1200: 12.00 game_done 142s · seed 1201: 3.00 game_done 123s · seed 1300: 30.00 game_done 164s · seed 1301: 12.00 game_done 150s
- `0004`: seed 1100: 1.00 game_done 130s · seed 1101: -9.00 game_done 129s · seed 1200: 5.00 game_done 140s · seed 1201: 13.00 game_done 145s · seed 1300: 27.00 game_done 121s · seed 1301: 9.00 game_done 109s
- `0005`: seed 1100: 7.00 game_done 1s · seed 1101: 4.00 game_done 1s · seed 1200: -7.00 game_done 1s · seed 1201: -3.00 game_done 1s · seed 1300: 14.00 game_done 1s · seed 1301: 16.00 game_done 1s
- `0006`: seed 1100: 1.00 game_done 123s · seed 1101: 0.00 game_done 122s · seed 1200: 3.00 game_done 122s · seed 1201: 3.00 game_done 118s · seed 1300: 33.00 game_done 108s · seed 1301: 6.00 game_done 103s
- `0007`: seed 1100: 1.00 game_done 129s · seed 1101: 0.00 game_done 121s · seed 1200: 12.00 game_done 137s · seed 1201: 3.00 game_done 117s · seed 1300: 25.00 game_done 94s · seed 1301: -8.00 game_done 37s
- `0008`: seed 1100: 15.00 game_done 108s · seed 1101: 6.00 game_done 107s · seed 1200: 9.00 game_done 100s · seed 1201: -20.00 game_done 113s · seed 1300: 42.00 game_done 118s · seed 1301: 28.00 game_done 127s
- `0009`: seed 1100: 6.00 game_done 66s · seed 1101: -2.00 game_done 72s · seed 1200: 12.00 game_done 96s · seed 1201: -13.00 game_done 91s · seed 1300: 22.00 game_done 103s · seed 1301: 20.00 game_done 149s
- `0010`: seed 1100: 12.00 game_done 88s · seed 1101: 6.00 game_done 77s · seed 1200: 10.00 game_done 87s · seed 1201: -21.00 game_done 51s · seed 1300: 42.00 game_done 110s · seed 1301: 28.00 game_done 106s
- `0011`: seed 1100: 8.00 game_done 43s · seed 1101: -3.00 game_done 79s · seed 1200: 0.00 game_done 106s · seed 1201: -5.00 game_done 25s · seed 1300: 11.00 game_done 85s · seed 1301: 13.00 game_done 74s
- `0012`: seed 1100: 6.00 game_done 40s · seed 1101: -8.00 game_done 41s · seed 1200: 12.00 game_done 49s · seed 1201: -13.00 game_done 47s · seed 1300: 10.00 game_done 40s · seed 1301: 26.00 game_done 44s
- `0013`: seed 1100: 8.00 game_done 108s · seed 1101: -11.00 game_done 100s · seed 1200: 9.00 game_done 98s · seed 1201: -6.00 game_done 115s · seed 1300: 60.00 game_done 134s · seed 1301: 28.00 game_done 154s
- `0014`: seed 1100: -1.00 game_done 44s · seed 1101: -2.00 game_done 44s · seed 1200: 12.00 game_done 45s · seed 1201: -13.00 game_done 44s · seed 1300: 13.00 game_done 36s · seed 1301: 8.00 game_done 36s
- `0015`: seed 1100: 12.00 game_done 89s · seed 1101: -11.00 game_done 83s · seed 1200: -2.00 game_done 97s · seed 1201: -21.00 game_done 53s · seed 1300: 42.00 game_done 127s · seed 1301: 31.00 game_done 148s
- `0016`: seed 1100: 11.00 game_done 93s · seed 1101: -11.00 game_done 91s · seed 1200: 10.00 game_done 93s · seed 1201: -16.00 game_done 125s · seed 1300: 42.00 game_done 133s · seed 1301: 40.00 game_done 199s
- `0017`: seed 1100: 11.00 game_done 91s · seed 1101: -11.00 game_done 90s · seed 1200: 7.00 game_done 83s · seed 1201: -6.00 game_done 122s · seed 1300: 60.00 game_done 134s · seed 1301: 28.00 game_done 149s
- `0018`: seed 1100: 6.00 game_done 50s · seed 1101: -2.00 game_done 55s · seed 1200: 12.00 game_done 74s · seed 1201: -13.00 game_done 70s · seed 1300: 22.00 game_done 67s · seed 1301: 20.00 game_done 67s

System-2 latency ≈ 43.3 s + -35.0 ms × output tokens (fit over 176 archived calls, r = -0.14; slowest 242 s): long answers are slow — size timeouts from this, not from short baseline answers.

Score composition (mean per episode, from the adaptor's score_breakdown):
- `0000-tier1_astra_naive`: agent_vp 9.33, fallback_count 0.00, illegal_action_count 0.50, opponent_vp 13.83, truncated 0.00, vp_margin -4.50, win_score 0.00
- `0001`: agent_vp 33.33, fallback_count 0.00, illegal_action_count 0.00, opponent_vp 33.00, truncated 0.00, vp_margin 0.33, win_score 0.50
- `0002`: agent_vp 32.50, fallback_count 0.00, illegal_action_count 0.00, opponent_vp 34.67, truncated 0.00, vp_margin -2.17, win_score 0.50
- `0003`: agent_vp 35.17, fallback_count 0.00, illegal_action_count 0.00, opponent_vp 25.67, truncated 0.00, vp_margin 9.50, win_score 0.83
- `0004`: agent_vp 34.50, fallback_count 0.00, illegal_action_count 0.00, opponent_vp 26.83, truncated 0.00, vp_margin 7.67, win_score 0.83
- `0005`: agent_vp 36.17, fallback_count 0.00, illegal_action_count 0.00, opponent_vp 31.00, truncated 0.00, vp_margin 5.17, win_score 0.67
- `0006`: agent_vp 33.00, fallback_count 0.00, illegal_action_count 0.00, opponent_vp 25.33, truncated 0.00, vp_margin 7.67, win_score 0.92
- `0007`: agent_vp 32.83, fallback_count 0.00, illegal_action_count 0.00, opponent_vp 27.33, truncated 0.00, vp_margin 5.50, win_score 0.75
- `0008`: agent_vp 40.00, fallback_count 0.00, illegal_action_count 0.00, opponent_vp 26.67, truncated 0.00, vp_margin 13.33, win_score 0.83
- `0009`: agent_vp 36.00, fallback_count 0.00, illegal_action_count 0.00, opponent_vp 28.50, truncated 0.00, vp_margin 7.50, win_score 0.67
- `0010`: agent_vp 41.50, fallback_count 0.00, illegal_action_count 0.00, opponent_vp 28.67, truncated 0.00, vp_margin 12.83, win_score 0.83
- `0011`: agent_vp 34.33, fallback_count 0.00, illegal_action_count 0.00, opponent_vp 30.33, truncated 0.00, vp_margin 4.00, win_score 0.58
- `0012`: agent_vp 34.67, fallback_count 0.00, illegal_action_count 0.00, opponent_vp 29.17, truncated 0.00, vp_margin 5.50, win_score 0.67
- `0013`: agent_vp 40.50, fallback_count 0.00, illegal_action_count 0.00, opponent_vp 25.83, truncated 0.00, vp_margin 14.67, win_score 0.67
- `0014`: agent_vp 36.00, fallback_count 0.00, illegal_action_count 0.00, opponent_vp 33.17, truncated 0.00, vp_margin 2.83, win_score 0.50
- `0015`: agent_vp 35.83, fallback_count 0.00, illegal_action_count 0.00, opponent_vp 27.33, truncated 0.00, vp_margin 8.50, win_score 0.50
- `0016`: agent_vp 40.00, fallback_count 0.00, illegal_action_count 0.00, opponent_vp 27.33, truncated 0.00, vp_margin 12.67, win_score 0.67
- `0017`: agent_vp 41.17, fallback_count 0.00, illegal_action_count 0.00, opponent_vp 26.33, truncated 0.00, vp_margin 14.83, win_score 0.67
- `0018`: agent_vp 36.00, fallback_count 0.00, illegal_action_count 0.00, opponent_vp 28.50, truncated 0.00, vp_margin 7.50, win_score 0.67

Held-out evaluations (not used for selection):
- `0017` / heldout (Dominion (offline pyminion)): mean score 3.000 over 6 episodes
