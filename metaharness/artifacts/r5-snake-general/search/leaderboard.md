# Leaderboard

| # | candidate | mean score | episodes | stale ticks | S1 calls | S2 calls | Jev→act | Astra→act | status |
|---|---|---|---|---|---|---|---|---|---|
| 0000 | 0000-tier2_jev_only | 0.000 | 3 | 80% | 6 | 0 | 100% | 0% | ok |
| 0001 | 0001-tier2_astra_only | 0.000 | 3 | 100% | 0 | 3 | – | – | ok |
| 0002 | 0002 | 0.412 | 3 | 0% | 6 | 9 | 0% | 0% | ok |
| 0003 | 0003 | 0.770 | 3 | 0% | 169 | 11 | 3% | 0% | ok |

`Jev→act` / `Astra→act`: fraction of posted actions preceded by a fresh System-1 / System-2 answer since the previous action (reported, not used for selection).

Best so far: `0003` (mean score 0.770)

Held-out evaluations (not used for selection):
- `0003` / heldout (Snake): mean score 0.649 over 3 episodes
