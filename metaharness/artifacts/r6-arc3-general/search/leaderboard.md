# Leaderboard

| # | candidate | mean score | episodes | stale ticks | S1 calls | S2 calls | Jev→act | Astra→act | status |
|---|---|---|---|---|---|---|---|---|---|
| 0000 | 0000-tier1_astra_naive | 0.333 | 3 | – | 0 | 120 | 0% | 100% | ok |
| 0001 | 0001 | 4.333 | 3 | – | 0 | 97 | 0% | 16% | 2 episode errors |
| 0002 | 0002 | 4.333 | 3 | – | 0 | 87 | 0% | 16% | ok |

`Jev→act` / `Astra→act`: fraction of posted actions preceded by a fresh System-1 / System-2 answer since the previous action (reported, not used for selection).

Best so far: `0002` (mean score 4.333)

Held-out evaluations (not used for selection):
- `0002` / heldout (ARC-AGI-3): mean score 2.500 over 2 episodes
