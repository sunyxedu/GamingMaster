# Leaderboard

| # | candidate | mean score | episodes | stale ticks | S1 calls | S2 calls | failed S1/S2 | Jev→act | Astra→act | Jev p50/max s | Astra p50/max s | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0000 | 0000-tier2_jev_only | 248.333 | 6 | 65% | 994 | 0 | 0 / 0 | 100% | 0% | 0.2 / 1 | – | ok |
| 0001 | 0001-tier2_astra_only | 60.000 | 6 | 100% | 0 | 9 | 0 / 0 | 0% | 100% | – | 18.8 / 28 | ok |
| 0002 | 0002-tier2_idle | 60.000 | 6 | 100% | 0 | 0 | 0 / 0 | – | – | – | – | ok |
| 0003 | 0003 | – | – | – | – | – | – | – | – | – | – | not evaluated |
| 0004 | 0004 | – | – | – | – | – | – | – | – | – | – | not evaluated |
| 0005 | 0005 | – | – | – | – | – | – | – | – | – | – | not evaluated |

`Jev→act` / `Astra→act`: fraction of posted actions preceded by a fresh System-1 / System-2 answer since the previous action (reported, not used for selection). Latency columns are the median per-episode p50 and the worst observed call: design timeouts and pipelining from these measurements, not from assumptions.

Best so far: `0000-tier2_jev_only` (mean score 248.333)
Only candidates whose status is `ok` can be selected: a fatal, partial or erroneous evaluation cannot win, whatever its score. A used-up call budget or the deadline is not an error.

Episodes (score, how it ended, wall time; ✗ = error):
- `0000-tier2_jev_only`: seed 0: 410.00 game_done 33s · seed 1: 120.00 game_done 24s · seed 2: 340.00 game_done 36s · seed 3: 340.00 game_done 41s · seed 4: 170.00 game_done 26s · seed 5: 110.00 game_done 28s
- `0001-tier2_astra_only`: seed 0: 60.00 game_done 32s · seed 1: 60.00 game_done 32s · seed 2: 60.00 game_done 32s · seed 3: 60.00 game_done 32s · seed 4: 60.00 game_done 32s · seed 5: 60.00 game_done 32s
- `0002-tier2_idle`: seed 0: 60.00 game_done 32s · seed 1: 60.00 game_done 32s · seed 2: 60.00 game_done 32s · seed 3: 60.00 game_done 32s · seed 4: 60.00 game_done 32s · seed 5: 60.00 game_done 32s

Score composition (mean per episode, from the adaptor's score_breakdown):
- `0000-tier2_jev_only`: ale_score 248.33, deaths 3.00, ghost_mode 0.00, hud_score 248.33, lives_remaining 0.00, mazes_cleared 0.00, terminal 1.00, visible_collectibles_eaten 20.83, visible_collectibles_remaining 133.17
- `0001-tier2_astra_only`: ale_score 60.00, deaths 3.00, ghost_mode 0.00, hud_score 60.00, lives_remaining 0.00, mazes_cleared 0.00, terminal 1.00, visible_collectibles_eaten 6.00, visible_collectibles_remaining 148.00
- `0002-tier2_idle`: ale_score 60.00, deaths 3.00, ghost_mode 0.00, hud_score 60.00, lives_remaining 0.00, mazes_cleared 0.00, terminal 1.00, visible_collectibles_eaten 6.00, visible_collectibles_remaining 148.00
