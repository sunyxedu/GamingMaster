# Leaderboard

| # | candidate | parent | mean score | seeds median / min | episodes | stale ticks | S1 calls | S2 calls | failed S1/S2 | Jev→act | Astra→act | Jev p50/max s | Astra p50/max s | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0000 | 0000-tier1_astra_naive | – | -0.687 | -1.004 / -1.185 | 10 | – | 0 | 311 | 0 / 0 | 0% | 100% | – | 46.5 / 104 | ok |
| 0001 | 0001 | 0000 | -0.641 | -1.035 / -1.125 | 10 | – | 0 | 250 | 0 / 3 (0 / 3 timed out) | 0% | 70% | – | 44.5 / 177 | ok |
| 0002 | 0002 | – | -0.892 | -1.076 / -1.210 | 10 | – | 0 | 205 | 0 / 1 (0 / 1 timed out) | 0% | 65% | – | 46.6 / 175 | ok |
| 0003 | 0003 | – | -1.111 | -1.084 / -1.216 | 10 | – | 0 | 210 | 0 / 8 (0 / 8 timed out) | 0% | 64% | – | 49.4 / 178 | ok |
| 0004 | 0004 | 0001 | -0.893 | -1.090 / -1.215 | 10 | – | 0 | 183 | 0 / 1 (0 / 1 timed out) | 0% | 66% | – | 57.5 / 208 | ok |
| 0005 | 0005 | 0003 | -0.903 | -1.093 / -1.209 | 10 | – | 0 | 200 | 0 / 0 | 0% | 66% | – | 47.5 / 136 | ok |
| 0006 | 0006 | 0002 | -0.898 | -1.083 / -1.231 | 10 | – | 0 | 193 | 0 / 0 | 0% | 68% | – | 55.6 / 236 | ok |
| 0007 | 0007 | 0001 | -0.675 | -1.090 / -1.163 | 10 | – | 0 | 194 | 0 / 0 | 0% | 66% | – | 59.3 / 152 | ok |
| 0008 | 0008 | 0001 | -0.890 | -1.084 / -1.221 | 10 | – | 0 | 175 | 0 / 0 | 0% | 61% | – | 69.2 / 205 | ok |
| 0009 | 0009 | 0000 | -0.887 | -1.087 / -1.192 | 10 | – | 0 | 256 | 0 / 0 | 0% | 91% | – | 45.2 / 106 | ok |
| 0010 | 0010 | – | – | – | – | – | – | – | – | – | – | – | – | not evaluated |
| 0011 | 0011 | – | – | – | – | – | – | – | – | – | – | – | – | not evaluated |
| 0012 | 0012 | – | – | – | – | – | – | – | – | – | – | – | – | not evaluated |

`seeds median / min`: the median and the worst per-seed score — a mean over a few seeds is noisy and can be carried by one lucky seed; an improvement that lifts the weak seeds is more likely to hold on unseen seeds. `(= all seeds)`: every seed scored the same, which is what a scenario-independent script does; expect it to break on scenarios it was not written for. `Jev→act` / `Astra→act`: fraction of posted actions preceded by a fresh System-1 / System-2 answer since the previous action (reported, not used for selection). Latency columns are the median per-episode p50 and the worst observed call: design timeouts and pipelining from these measurements, not from assumptions.

Best so far: `0000-tier1_astra_naive` (mean score -0.687)
Only candidates whose status is `ok` can be selected: a fatal, partial or erroneous evaluation cannot win, whatever its score. A used-up call budget or the deadline is not an error. `parent` is the candidate a proposal says it was derived from (`parent: NNNN` in proposal.md).

Validation (top candidates re-evaluated on unseen seeds; the final selection is the best train+validation mean among them):
- `0000-tier1_astra_naive`: validation -0.489 over 10 episodes; train+validation -0.588
- `0001`: validation -1.133 over 10 episodes; train+validation -0.887
- `0007`: validation -1.112 over 10 episodes; train+validation -0.893

Episodes (score, how it ended, wall time; ✗ = error):
- `0000-tier1_astra_naive`: seed 104729: -0.10 deadline 1800s · seed 130363: -1.16 game_done 1544s · seed 155921: 0.01 deadline 1800s · seed 181081: 0.01 deadline 1800s · seed 206369: -1.00 game_done 1620s · seed 232003: -1.08 game_done 1290s · seed 257053: -0.06 harness_return 1733s · seed 282617: -1.13 game_done 889s · seed 308081: -1.17 game_done 787s · seed 333667: -1.18 game_done 1212s
- `0001`: seed 104729: 1.03 game_done 1687s · seed 130363: -1.01 game_done 1779s · seed 155921: -1.04 game_done 1738s · seed 181081: -1.11 game_done 1708s · seed 206369: 1.04 game_done 1680s · seed 232003: -1.12 game_done 1652s · seed 257053: -1.04 game_done 1662s · seed 282617: -1.11 game_done 1521s · seed 308081: -1.04 game_done 1457s · seed 333667: -1.02 game_done 1677s
- `0002`: seed 104729: -1.10 game_done 1586s · seed 130363: -1.18 game_done 1588s · seed 155921: -1.08 game_done 1624s · seed 181081: 1.04 game_done 1549s · seed 206369: -1.21 game_done 1539s · seed 232003: -1.03 game_done 1551s · seed 257053: -1.05 game_done 1560s · seed 282617: -1.13 game_done 1619s · seed 308081: -1.16 game_done 1490s · seed 333667: -1.04 game_done 1574s
- `0003`: seed 104729: -1.05 game_done 1643s · seed 130363: -1.08 game_done 1675s · seed 155921: -1.09 game_done 1635s · seed 181081: -1.12 game_done 1655s · seed 206369: -1.22 game_done 1652s · seed 232003: -1.16 game_done 1791s · seed 257053: -1.08 game_done 1682s · seed 282617: -1.15 game_done 1436s · seed 308081: -1.07 game_done 1630s · seed 333667: -1.08 game_done 1654s
- `0004`: seed 104729: -1.04 game_done 1543s · seed 130363: -1.09 game_done 1767s · seed 155921: 1.05 game_done 1573s · seed 181081: -1.02 game_done 1623s · seed 206369: -1.21 game_done 1581s · seed 232003: -1.09 game_done 1543s · seed 257053: -1.07 game_done 1601s · seed 282617: -1.18 game_done 1415s · seed 308081: -1.15 game_done 1567s · seed 333667: -1.13 game_done 1497s
- `0005`: seed 104729: -1.09 game_done 1588s · seed 130363: -1.16 game_done 1653s · seed 155921: -1.06 game_done 1627s · seed 181081: -1.08 game_done 1587s · seed 206369: 1.01 game_done 1539s · seed 232003: -1.09 game_done 1567s · seed 257053: -1.14 game_done 1611s · seed 282617: -1.21 game_done 1227s · seed 308081: -1.09 game_done 1481s · seed 333667: -1.11 game_done 1507s
- `0006`: seed 104729: 1.06 game_done 1619s · seed 130363: -1.12 game_done 1665s · seed 155921: -1.08 game_done 1435s · seed 181081: -1.11 game_done 1633s · seed 206369: -1.08 game_done 1455s · seed 232003: -1.15 game_done 1255s · seed 257053: -1.08 game_done 1588s · seed 282617: -1.07 game_done 1519s · seed 308081: -1.23 game_done 1559s · seed 333667: -1.10 game_done 1557s
- `0007`: seed 104729: 1.07 game_done 1543s · seed 130363: -1.05 game_done 1612s · seed 155921: -1.14 game_done 1564s · seed 181081: 1.02 game_done 1546s · seed 206369: -1.09 game_done 1601s · seed 232003: -1.13 game_done 1519s · seed 257053: -1.16 game_done 1550s · seed 282617: -1.09 game_done 1612s · seed 308081: -1.04 game_done 1546s · seed 333667: -1.12 game_done 1531s
- `0008`: seed 104729: -1.13 game_done 1536s · seed 130363: -1.12 game_done 1528s · seed 155921: -1.04 game_done 1691s · seed 181081: 1.03 game_done 1570s · seed 206369: -1.14 game_done 1567s · seed 232003: -1.03 game_done 1639s · seed 257053: -1.10 game_done 1569s · seed 282617: -1.08 game_done 1548s · seed 308081: -1.22 game_done 1610s · seed 333667: -1.06 game_done 1562s
- `0009`: seed 104729: 1.11 game_done 1713s · seed 130363: -1.15 game_done 1703s · seed 155921: -1.08 game_done 1719s · seed 181081: -1.09 game_done 1681s · seed 206369: -1.08 game_done 1043s · seed 232003: -1.09 game_done 1036s · seed 257053: -1.07 game_done 1243s · seed 282617: -1.10 game_done 1093s · seed 308081: -1.19 game_done 625s · seed 333667: -1.13 game_done 561s

System-2 latency ≈ 31.8 s + 21.0 ms × output tokens (fit over 2161 archived calls, r = 0.75; slowest 236 s): long answers are slow — size timeouts from this, not from short baseline answers.

Score composition (mean per episode, from the adaptor's score_breakdown):
- `0000-tier1_astra_naive`: bot_deadline_hits 0.00, bot_decisions 28.00, bot_max_seconds 0.99, bot_search_visits 8909700.00, bot_seconds 26.02, decisions 30.80, engine_seconds 26.67, hp_margin -0.35, opponent_hp_fraction 0.45, opponent_survivors 4.00, outcome -0.60, score -0.69, self_hp_fraction 0.11, self_survivors 1.10, turns 25.90
- `0001`: bot_deadline_hits 0.00, bot_decisions 34.70, bot_max_seconds 0.98, bot_search_visits 12047700.00, bot_seconds 31.81, decisions 35.50, engine_seconds 32.49, hp_margin -0.16, opponent_hp_fraction 0.19, opponent_survivors 1.60, outcome -0.60, score -0.64, self_hp_fraction 0.03, self_survivors 0.40, turns 30.50
- `0002`: bot_deadline_hits 0.00, bot_decisions 28.80, bot_max_seconds 0.99, bot_search_visits 10195300.00, bot_seconds 26.68, decisions 31.50, engine_seconds 27.35, hp_margin -0.37, opponent_hp_fraction 0.39, opponent_survivors 3.30, outcome -0.80, score -0.89, self_hp_fraction 0.02, self_survivors 0.10, turns 26.10
- `0003`: bot_deadline_hits 0.00, bot_decisions 28.10, bot_max_seconds 0.98, bot_search_visits 11139600.00, bot_seconds 26.11, decisions 31.50, engine_seconds 26.78, hp_margin -0.44, opponent_hp_fraction 0.44, opponent_survivors 3.40, outcome -1.00, score -1.11, self_hp_fraction 0.00, self_survivors 0.00, turns 25.50
- `0004`: bot_deadline_hits 0.00, bot_decisions 24.90, bot_max_seconds 0.99, bot_search_visits 9410400.00, bot_seconds 22.98, decisions 27.70, engine_seconds 23.61, hp_margin -0.37, opponent_hp_fraction 0.39, opponent_survivors 3.40, outcome -0.80, score -0.89, self_hp_fraction 0.02, self_survivors 0.40, turns 22.50
- `0005`: bot_deadline_hits 0.00, bot_decisions 27.60, bot_max_seconds 0.99, bot_search_visits 11001100.00, bot_seconds 25.40, decisions 30.40, engine_seconds 26.05, hp_margin -0.41, opponent_hp_fraction 0.41, opponent_survivors 3.50, outcome -0.80, score -0.90, self_hp_fraction 0.00, self_survivors 0.10, turns 25.10
- `0006`: bot_deadline_hits 0.00, bot_decisions 26.30, bot_max_seconds 0.99, bot_search_visits 11921300.00, bot_seconds 24.26, decisions 28.50, engine_seconds 24.91, hp_margin -0.39, opponent_hp_fraction 0.41, opponent_survivors 3.00, outcome -0.80, score -0.90, self_hp_fraction 0.02, self_survivors 0.20, turns 23.30
- `0007`: bot_deadline_hits 0.00, bot_decisions 27.30, bot_max_seconds 0.99, bot_search_visits 10385800.00, bot_seconds 25.22, decisions 29.40, engine_seconds 25.88, hp_margin -0.30, opponent_hp_fraction 0.33, opponent_survivors 2.70, outcome -0.60, score -0.67, self_hp_fraction 0.03, self_survivors 0.40, turns 24.20
- `0008`: bot_deadline_hits 0.00, bot_decisions 26.00, bot_max_seconds 0.99, bot_search_visits 10337600.00, bot_seconds 24.02, decisions 28.50, engine_seconds 24.67, hp_margin -0.36, opponent_hp_fraction 0.38, opponent_survivors 3.20, outcome -0.80, score -0.89, self_hp_fraction 0.01, self_survivors 0.20, turns 23.30
- `0009`: bot_deadline_hits 0.00, bot_decisions 26.10, bot_max_seconds 0.99, bot_search_visits 10045700.00, bot_seconds 24.13, decisions 28.10, engine_seconds 24.79, hp_margin -0.35, opponent_hp_fraction 0.39, opponent_survivors 3.20, outcome -0.80, score -0.89, self_hp_fraction 0.04, self_survivors 0.40, turns 23.40

Held-out evaluations (not used for selection):
- `0000-tier1_astra_naive` / heldout (Pokémon Showdown — Gen 9 Random Battle): mean score -0.901 over 10 episodes
