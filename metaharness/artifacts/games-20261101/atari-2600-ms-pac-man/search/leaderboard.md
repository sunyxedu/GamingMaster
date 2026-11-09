# Leaderboard

| # | candidate | mean score | episodes | stale ticks | S1 calls | S2 calls | failed S1/S2 | Jev→act | Astra→act | Jev p50/max s | Astra p50/max s | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0000 | 0000-tier2_jev_only | 275.000 | 6 | 63% | 1231 | 0 | 0 / 0 | 100% | 0% | 0.2 / 0 | – | ok |
| 0001 | 0001-tier2_astra_only | 60.000 | 6 | 100% | 0 | 7 | 0 / 1 | – | – | – | – | ok |
| 0002 | 0002-tier2_idle | 60.000 | 6 | 100% | 0 | 0 | 0 / 0 | – | – | – | – | ok |
| 0003 | 0003 | 141.667 | 6 | 54% | 1478 | 0 | 0 / 0 | 100% | 0% | 0.1 / 0 | – | ok |
| 0004 | 0004 | 778.333 | 6 | 0% | 510 | 6 | 0 / 0 | 15% | 0% | 0.2 / 0 | 24.9 / 34 | ok |
| 0005 | 0005 | – | 0 | – | 0 | 0 | 0 / 0 | – | – | – | – | ✗ fatal: proposer failed: RuntimeError: claude -p failed (rc=1): API Error: Responses stream error: Upstream request faile |
| 0006 | 0006 | 0.000 | 6 | – | 0 | 6 | 0 / 0 | – | – | – | – | ✗ 6 episode error(s), first: AttributeError: 'Harness' object has no attribute 'override' |
| 0007 | 0007 | 1028.333 | 6 | 0% | 843 | 6 | 0 / 0 | 17% | 0% | 0.2 / 0 | 35.9 / 53 | ok |
| 0008 | 0008 | 1608.333 | 6 | 0% | 1117 | 6 | 0 / 0 | 20% | 0% | 0.2 / 0 | 39.2 / 44 | ok |
| 0009 | 0009 | 1725.000 | 6 | 0% | 1127 | 6 | 0 / 0 | 18% | 0% | 0.2 / 0 | 55.4 / 69 | ok |
| 0010 | 0010 | 1861.667 | 6 | 0% | 1099 | 6 | 0 / 0 | 18% | 0% | 0.2 / 0 | – | ok |
| 0011 | 0011 | 1785.000 | 6 | 0% | 1121 | 6 | 0 / 0 | 19% | 0% | 0.2 / 0 | – | ok |
| 0012 | 0012 | 2141.667 | 6 | 0% | 1322 | 6 | 0 / 2 (0 / 1 timed out) | 17% | 0% | 0.2 / 1 | 39.9 / 40 | ok |
| 0013 | 0013 | 1653.333 | 6 | 0% | 1162 | 6 | 2 / 2 (2 / 1 timed out) | 17% | 0% | 0.2 / 1 | 59.4 / 59 | ok |
| 0014 | 0014 | – | 0 | – | 0 | 0 | 0 / 0 | – | – | – | – | ✗ fatal: duplicate of candidates/0010 (the same code, comments aside): not evaluated |
| 0015 | 0015 | – | 0 | – | 0 | 0 | 0 / 0 | – | – | – | – | ✗ fatal: duplicate of candidates/0012 (the same code, comments aside): not evaluated |
| 0016 | 0016 | 4018.333 | 6 | 0% | 2755 | 16 | 0 / 5 (0 / 5 timed out) | 15% | 0% | 0.2 / 1 | 68.4 / 77 | ok |
| 0017 | 0017 | 23.333 | 6 | 0% | 273 | 6 | 0 / 0 | 9% | 0% | 0.2 / 0 | – | ok |
| 0018 | 0018 | 1556.667 | 6 | 0% | 318 | 8 | 0 / 1 (0 / 1 timed out) | 4% | 0% | 0.2 / 0 | 58.7 / 59 | ok |
| 0019 | 0019 | 3870.000 | 6 | 0% | 2564 | 16 | 0 / 4 (0 / 4 timed out) | 15% | 0% | 0.2 / 1 | 84.4 / 92 | ok |
| 0020 | 0020 | 2355.000 | 6 | 0% | 1421 | 11 | 0 / 1 (0 / 1 timed out) | 13% | 0% | 0.2 / 1 | 51.7 / 64 | ok |
| 0021 | 0021 | 1143.333 | 6 | 0% | 974 | 7 | 1 / 1 (1 / 1 timed out) | 15% | 0% | 0.2 / 1 | – | ok |
| 0022 | 0022 | 3865.000 | 6 | 0% | 2457 | 15 | 8 / 6 (8 / 5 timed out) | 15% | 0% | 0.2 / 2 | 49.8 / 79 | ok |
| 0023 | 0023 | 3945.000 | 6 | 0% | 2630 | 16 | 9 / 4 (9 / 4 timed out) | 15% | 0% | 0.2 / 2 | 59.6 / 78 | ok |

`Jev→act` / `Astra→act`: fraction of posted actions preceded by a fresh System-1 / System-2 answer since the previous action (reported, not used for selection). Latency columns are the median per-episode p50 and the worst observed call: design timeouts and pipelining from these measurements, not from assumptions.

Best so far: `0016` (mean score 4018.333)
Only candidates whose status is `ok` can be selected: a fatal, partial or erroneous evaluation cannot win, whatever its score. A used-up call budget or the deadline is not an error.

Episodes (score, how it ended, wall time; ✗ = error):
- `0000-tier2_jev_only`: seed 0: 340.00 game_done 51s · seed 1: 330.00 game_done 52s · seed 2: 110.00 game_done 26s · seed 3: 310.00 game_done 30s · seed 4: 280.00 game_done 33s · seed 5: 280.00 game_done 32s
- `0001-tier2_astra_only`: seed 0: 60.00 game_done 32s · seed 1: 60.00 game_done 32s · seed 2: 60.00 game_done 32s · seed 3: 60.00 game_done 32s · seed 4: 60.00 game_done 32s · seed 5: 60.00 game_done 32s
- `0002-tier2_idle`: seed 0: 60.00 game_done 32s · seed 1: 60.00 game_done 32s · seed 2: 60.00 game_done 32s · seed 3: 60.00 game_done 32s · seed 4: 60.00 game_done 32s · seed 5: 60.00 game_done 32s
- `0003` (refine of 0000-tier2_jev_only): seed 0: 160.00 game_done 35s · seed 1: 110.00 game_done 31s · seed 2: 110.00 game_done 37s · seed 3: 250.00 game_done 40s · seed 4: 110.00 game_done 35s · seed 5: 110.00 game_done 37s
- `0004` (rewrite of 0002-tier2_idle): seed 0: 620.00 game_done 31s · seed 1: 780.00 game_done 36s · seed 2: 760.00 game_done 36s · seed 3: 1070.00 game_done 50s · seed 4: 740.00 game_done 35s · seed 5: 700.00 game_done 40s
- `0006` (rewrite of 0001-tier2_astra_only): seed 0: 0.00 harness_error 0s ✗ AttributeError: 'Harness' object has no attribute 'override' · seed 1: 0.00 harness_error 0s ✗ AttributeError: 'Harness' object has no attribute 'override' · seed 2: 0.00 harness_error 0s ✗ AttributeError: 'Harness' object has no attribute 'override' · seed 3: 0.00 harness_error 25s ✗ AttributeError: 'Harness' object has no attribute 'override' · seed 4: 0.00 harness_error 0s ✗ AttributeError: 'Harness' object has no attribute 'override' · seed 5: 0.00 harness_error 0s ✗ AttributeError: 'Harness' object has no attribute 'override'
- `0007` (crossover of 0004, 0003): seed 0: 830.00 game_done 46s · seed 1: 940.00 game_done 40s · seed 2: 1450.00 game_done 85s · seed 3: 1480.00 game_done 87s · seed 4: 580.00 game_done 29s · seed 5: 890.00 game_done 40s
- `0008` (explore of 0004): seed 0: 1800.00 game_done 72s · seed 1: 1500.00 game_done 50s · seed 2: 1460.00 game_done 54s · seed 3: 1860.00 game_done 75s · seed 4: 1480.00 game_done 67s · seed 5: 1550.00 game_done 61s
- `0009` (crossover of 0008, 0007): seed 0: 1890.00 game_done 76s · seed 1: 1750.00 game_done 57s · seed 2: 1520.00 game_done 57s · seed 3: 1850.00 game_done 72s · seed 4: 1680.00 game_done 74s · seed 5: 1660.00 game_done 73s
- `0010` (explore of 0008): seed 0: 1780.00 game_done 64s · seed 1: 2470.00 game_done 90s · seed 2: 1960.00 game_done 71s · seed 3: 1500.00 game_done 62s · seed 4: 1880.00 game_done 63s · seed 5: 1580.00 game_done 62s
- `0011` (refine of 0008): seed 0: 1680.00 game_done 58s · seed 1: 1830.00 game_done 72s · seed 2: 1660.00 game_done 72s · seed 3: 1580.00 game_done 61s · seed 4: 2250.00 game_done 66s · seed 5: 1710.00 game_done 72s
- `0012` (explore of 0009): seed 0: 1890.00 game_done 70s · seed 1: 2330.00 game_done 71s · seed 2: 3990.00 game_done 158s · seed 3: 1460.00 game_done 65s · seed 4: 1410.00 game_done 58s · seed 5: 1770.00 game_done 82s
- `0013` (refine of 0010): seed 0: 1580.00 game_done 85s · seed 1: 1230.00 game_done 50s · seed 2: 1620.00 game_done 62s · seed 3: 1650.00 game_done 77s · seed 4: 1290.00 game_done 58s · seed 5: 2550.00 game_done 114s
- `0016` (rewrite of 0009): seed 0: 4320.00 game_done 232s · seed 1: 5010.00 game_done 265s · seed 2: 3150.00 game_done 144s · seed 3: 1790.00 game_done 127s · seed 4: 4880.00 game_done 237s · seed 5: 4960.00 game_done 215s
- `0017` (crossover of 0012, 0007): seed 0: 20.00 game_done 33s · seed 1: 20.00 game_done 32s · seed 2: 20.00 game_done 32s · seed 3: 20.00 game_done 32s · seed 4: 40.00 game_done 33s · seed 5: 20.00 game_done 32s
- `0018` (rewrite of 0013): seed 0: 1350.00 game_done 79s · seed 1: 1620.00 game_done 84s · seed 2: 1700.00 game_done 97s · seed 3: 1740.00 game_done 109s · seed 4: 1760.00 game_done 136s · seed 5: 1170.00 game_done 75s
- `0019` (crossover of 0016, 0013): seed 0: 7430.00 game_done 293s · seed 1: 2910.00 game_done 146s · seed 2: 3350.00 game_done 221s · seed 3: 3330.00 game_done 166s · seed 4: 3040.00 game_done 131s · seed 5: 3160.00 game_done 161s
- `0020` (explore of 0016): seed 0: 1650.00 game_done 96s · seed 1: 1730.00 game_done 88s · seed 2: 3790.00 game_done 200s · seed 3: 3430.00 game_done 170s · seed 4: 1750.00 game_done 85s · seed 5: 1780.00 game_done 109s
- `0021` (crossover of 0016, 0019): seed 0: 860.00 game_done 57s · seed 1: 1310.00 game_done 63s · seed 2: 870.00 game_done 59s · seed 3: 1590.00 game_done 110s · seed 4: 890.00 game_done 59s · seed 5: 1340.00 game_done 89s
- `0022` (explore of 0019): seed 0: 4720.00 game_done 248s · seed 1: 5220.00 game_done 224s · seed 2: 5060.00 game_done 252s · seed 3: 3370.00 game_done 155s · seed 4: 3160.00 game_done 131s · seed 5: 1660.00 game_done 88s
- `0023` (refine of 0016): seed 0: 4530.00 game_done 225s · seed 1: 3570.00 game_done 194s · seed 2: 3260.00 game_done 163s · seed 3: 1770.00 game_done 87s · seed 4: 5600.00 game_done 237s · seed 5: 4940.00 game_done 260s

System-2 latency ≈ 48.8 s + 24.2 ms × output tokens (fit over 47 archived calls, r = 0.12; slowest 92 s): long answers are slow — size timeouts from this, not from short baseline answers.

Score composition (mean per episode, from the adaptor's score_breakdown):
- `0000-tier2_jev_only`: ale_score 275.00, deaths 3.00, ghost_mode 0.00, hud_score 275.00, lives_remaining 0.00, mazes_cleared 0.00, terminal 1.00, visible_collectibles_eaten 27.50, visible_collectibles_remaining 126.50
- `0001-tier2_astra_only`: ale_score 60.00, deaths 3.00, ghost_mode 0.00, hud_score 60.00, lives_remaining 0.00, mazes_cleared 0.00, terminal 1.00, visible_collectibles_eaten 6.00, visible_collectibles_remaining 148.00
- `0002-tier2_idle`: ale_score 60.00, deaths 3.00, ghost_mode 0.00, hud_score 60.00, lives_remaining 0.00, mazes_cleared 0.00, terminal 1.00, visible_collectibles_eaten 6.00, visible_collectibles_remaining 148.00
- `0003`: ale_score 141.67, deaths 3.00, ghost_mode 0.00, hud_score 141.67, lives_remaining 0.00, mazes_cleared 0.00, terminal 1.00, visible_collectibles_eaten 14.17, visible_collectibles_remaining 139.83
- `0004`: ale_score 778.33, deaths 3.00, ghost_mode 0.00, hud_score 778.33, lives_remaining 0.00, mazes_cleared 0.00, terminal 1.00, visible_collectibles_eaten 77.83, visible_collectibles_remaining 76.17
- `0006`: ale_score 0.00, deaths 0.00, ghost_mode 0.00, hud_score 0.00, lives_remaining 3.00, mazes_cleared 0.00, terminal 0.00, visible_collectibles_eaten 0.00, visible_collectibles_remaining 154.00
- `0007`: ale_score 1028.33, deaths 3.00, ghost_mode 0.00, hud_score 1028.33, lives_remaining 0.00, mazes_cleared 0.00, terminal 1.00, visible_collectibles_eaten 99.50, visible_collectibles_remaining 54.50
- `0008`: ale_score 1608.33, deaths 3.00, ghost_mode 0.00, hud_score 1608.33, lives_remaining 0.00, mazes_cleared 0.00, terminal 1.00, visible_collectibles_eaten 126.50, visible_collectibles_remaining 27.50
- `0009`: ale_score 1725.00, deaths 3.00, ghost_mode 0.00, hud_score 1725.00, lives_remaining 0.00, mazes_cleared 0.00, terminal 1.00, visible_collectibles_eaten 144.83, visible_collectibles_remaining 9.17
- `0010`: ale_score 1861.67, deaths 3.00, ghost_mode 0.00, hud_score 1861.67, lives_remaining 0.00, mazes_cleared 0.00, terminal 1.00, visible_collectibles_eaten 138.50, visible_collectibles_remaining 15.50
- `0011`: ale_score 1785.00, deaths 3.00, ghost_mode 0.00, hud_score 1785.00, lives_remaining 0.00, mazes_cleared 0.00, terminal 1.00, visible_collectibles_eaten 130.83, visible_collectibles_remaining 23.17
- `0012`: ale_score 2141.67, deaths 3.00, ghost_mode 0.00, hud_score 2141.67, lives_remaining 0.00, mazes_cleared 0.00, terminal 1.00, visible_collectibles_eaten 163.83, visible_collectibles_remaining 15.83
- `0013`: ale_score 1653.33, deaths 3.00, ghost_mode 0.00, hud_score 1653.33, lives_remaining 0.00, mazes_cleared 0.00, terminal 1.00, visible_collectibles_eaten 143.67, visible_collectibles_remaining 36.00
- `0016`: ale_score 4018.33, deaths 3.00, ghost_mode 0.00, hud_score 4018.33, lives_remaining 0.00, mazes_cleared 0.00, terminal 1.00, visible_collectibles_eaten 342.17, visible_collectibles_remaining 40.17
- `0017`: ale_score 23.33, deaths 3.00, ghost_mode 0.00, hud_score 23.33, lives_remaining 0.00, mazes_cleared 0.00, terminal 1.00, visible_collectibles_eaten 2.33, visible_collectibles_remaining 151.67
- `0018`: ale_score 1556.67, deaths 3.00, ghost_mode 0.00, hud_score 1556.67, lives_remaining 0.00, mazes_cleared 0.00, terminal 1.00, visible_collectibles_eaten 133.00, visible_collectibles_remaining 21.00
- `0019`: ale_score 3870.00, deaths 3.00, ghost_mode 0.00, hud_score 3870.00, lives_remaining 0.00, mazes_cleared 0.00, terminal 1.00, visible_collectibles_eaten 319.67, visible_collectibles_remaining 38.33
- `0020`: ale_score 2355.00, deaths 3.00, ghost_mode 0.00, hud_score 2355.00, lives_remaining 0.00, mazes_cleared 0.00, terminal 1.00, visible_collectibles_eaten 200.83, visible_collectibles_remaining 4.50
- `0021`: ale_score 1143.33, deaths 3.00, ghost_mode 0.00, hud_score 1143.33, lives_remaining 0.00, mazes_cleared 0.00, terminal 1.00, visible_collectibles_eaten 99.33, visible_collectibles_remaining 54.67
- `0022`: ale_score 3865.00, deaths 3.00, ghost_mode 0.00, hud_score 3865.00, lives_remaining 0.00, mazes_cleared 0.00, terminal 1.00, visible_collectibles_eaten 323.50, visible_collectibles_remaining 33.83
- `0023`: ale_score 3945.00, deaths 3.00, ghost_mode 0.00, hud_score 3945.00, lives_remaining 0.00, mazes_cleared 0.00, terminal 1.00, visible_collectibles_eaten 317.17, visible_collectibles_remaining 40.17

Validation seeds (finalists only; the final pick ranks training + validation together):
- `0016`: validation mean score 3605.000 over 6 episodes
- `0019`: validation mean score 2998.333 over 6 episodes
- `0023`: validation mean score 4366.667 over 6 episodes

Final pick: `0023` (training 3945.0, validation 4366.666666666667).

Held-out evaluations (not used for selection):
- `0023` / heldout (Atari 2600 Ms. Pac-Man): mean score 3638.750 over 8 episodes

Ablation: `0023` without runtime System-1 / System-2 on its training seeds: 3341.667 (with them: 3945.000).
