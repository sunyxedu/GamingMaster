# Leaderboard

| # | candidate | parent | mean score | seeds median / min | episodes | stale ticks | S1 calls | S2 calls | failed S1/S2 | Jev→act | Astra→act | Jev p50/max s | Astra p50/max s | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0000 | 0000-tier1_astra_naive | – | 0.516 | 0.500 / 0.375 | 8 | – | 0 | 30 | 0 / 0 | 0% | 100% | – | 43.1 / 217 | ok |
| 0001 | 0001 | 0000 | 0.578 | 0.500 / 0.500 | 8 | – | 0 | 0 | 0 / 0 | 0% | 0% | – | – | ok |
| 0002 | 0002 | – | 0.562 | 0.500 / 0.375 | 8 | – | 0 | 0 | 0 / 0 | 0% | 0% | – | – | ok |
| 0003 | 0003 | – | 0.562 | 0.500 / 0.375 | 8 | – | 0 | 0 | 0 / 0 | 0% | 0% | – | – | ok |
| 0004 | 0004 | 0001 | 0.578 | 0.500 / 0.375 | 8 | – | 0 | 0 | 0 / 0 | 0% | 0% | – | – | ok |
| 0005 | 0005 | 0002 | 0.547 | 0.500 / 0.500 | 8 | – | 0 | 0 | 0 / 0 | 0% | 0% | – | – | ok |
| 0006 | 0006 | 0003 | 0.641 | 0.500 / 0.500 | 8 | – | 0 | 0 | 0 / 0 | 0% | 0% | – | – | ok |
| 0007 | 0007 | 0006 | 0.594 | 0.500 / 0.500 | 8 | – | 0 | 0 | 0 / 0 | 0% | 0% | – | – | ok |
| 0008 | 0008 | 0004 | 0.531 | 0.500 / 0.500 | 8 | – | 0 | 0 | 0 / 0 | 0% | 0% | – | – | ok |
| 0009 | 0009 | 0004 | 0.578 | 0.500 / 0.375 | 8 | – | 0 | 0 | 0 / 0 | 0% | 0% | – | – | ok |
| 0010 | 0010 | 0006 | 0.594 | 0.500 / 0.500 | 8 | – | 0 | 0 | 0 / 0 | 0% | 0% | – | – | ok |
| 0011 | 0011 | 0003 | 0.562 | 0.500 / 0.375 | 8 | – | 0 | 0 | 0 / 0 | 0% | 0% | – | – | ok |
| 0012 | 0012 | 0001 | 0.578 | 0.500 / 0.500 | 8 | – | 0 | 0 | 0 / 0 | 0% | 0% | – | – | ok |
| 0013 | 0013 | 0006 | 0.609 | 0.500 / 0.500 | 8 | – | 0 | 0 | 0 / 0 | 0% | 0% | – | – | ok |
| 0014 | 0014 | 0011 | 0.578 | 0.500 / 0.500 | 8 | – | 0 | 0 | 0 / 0 | 0% | 0% | – | – | ok |
| 0015 | 0015 | 0011 | 0.578 | 0.500 / 0.500 | 8 | – | 0 | 0 | 0 / 0 | 0% | 0% | – | – | ok |
| 0016 | 0016 | 0006 | 0.578 | 0.500 / 0.500 | 8 | – | 0 | 0 | 0 / 0 | 0% | 0% | – | – | ok |
| 0017 | 0017 | 0011 | 0.531 | 0.500 / 0.500 | 8 | – | 0 | 0 | 0 / 0 | 0% | 0% | – | – | ok |
| 0018 | 0018 | 0013 | 0.609 | 0.625 / 0.500 | 8 | – | 0 | 0 | 0 / 0 | 0% | 0% | – | – | ok |

`seeds median / min`: the median and the worst per-seed score — a mean over a few seeds is noisy and can be carried by one lucky seed; an improvement that lifts the weak seeds is more likely to hold on unseen seeds. `(= all seeds)`: every seed scored the same, which is what a scenario-independent script does; expect it to break on scenarios it was not written for. `Jev→act` / `Astra→act`: fraction of posted actions preceded by a fresh System-1 / System-2 answer since the previous action (reported, not used for selection). Latency columns are the median per-episode p50 and the worst observed call: design timeouts and pipelining from these measurements, not from assumptions.

Best so far: `0018` (mean score 0.609)
Only candidates whose status is `ok` can be selected: a fatal, partial or erroneous evaluation cannot win, whatever its score. A used-up call budget or the deadline is not an error. `parent` is the candidate a proposal says it was derived from (`parent: NNNN` in proposal.md).

Validation (top candidates re-evaluated on unseen seeds; the final selection is the best train+validation mean among them):
- `0006`: validation 0.609 over 8 episodes; train+validation 0.625
- `0013`: validation 0.578 over 8 episodes; train+validation 0.594
- `0018`: validation 0.656 over 8 episodes; train+validation 0.633

Episodes (score, how it ended, wall time; ✗ = error):
- `0000-tier1_astra_naive`: seed 10: 0.50 game_done 127s · seed 11: 0.50 game_done 128s · seed 12: 0.50 game_done 191s · seed 13: 0.50 game_done 342s · seed 14: 0.50 game_done 160s · seed 15: 0.38 game_done 97s · seed 16: 0.50 game_done 214s · seed 17: 0.75 game_done 336s
- `0001`: seed 10: 0.50 game_done 0s · seed 11: 0.50 game_done 3s · seed 12: 0.50 game_done 1s · seed 13: 0.62 game_done 3s · seed 14: 0.50 game_done 0s · seed 15: 0.50 game_done 3s · seed 16: 0.50 game_done 1s · seed 17: 1.00 game_done 2s
- `0002`: seed 10: 0.50 game_done 0s · seed 11: 0.38 game_done 9s · seed 12: 0.50 game_done 0s · seed 13: 1.00 game_done 7s · seed 14: 0.50 game_done 0s · seed 15: 0.62 game_done 7s · seed 16: 0.50 game_done 0s · seed 17: 0.50 game_done 4s
- `0003`: seed 10: 0.50 game_done 0s · seed 11: 0.38 game_done 11s · seed 12: 0.50 game_done 1s · seed 13: 1.00 game_done 7s · seed 14: 0.50 game_done 0s · seed 15: 0.62 game_done 9s · seed 16: 0.50 game_done 1s · seed 17: 0.50 game_done 6s
- `0004`: seed 10: 0.50 game_done 0s · seed 11: 0.38 game_done 3s · seed 12: 0.50 game_done 1s · seed 13: 0.75 game_done 3s · seed 14: 0.50 game_done 0s · seed 15: 0.50 game_done 3s · seed 16: 0.50 game_done 0s · seed 17: 1.00 game_done 2s
- `0005`: seed 10: 0.50 game_done 0s · seed 11: 0.75 game_done 2s · seed 12: 0.50 game_done 1s · seed 13: 0.62 game_done 1s · seed 14: 0.50 game_done 0s · seed 15: 0.50 game_done 2s · seed 16: 0.50 game_done 1s · seed 17: 0.50 game_done 2s
- `0006`: seed 10: 0.50 game_done 0s · seed 11: 0.62 game_done 4s · seed 12: 0.50 game_done 1s · seed 13: 0.50 game_done 4s · seed 14: 0.50 game_done 0s · seed 15: 1.00 game_done 2s · seed 16: 0.50 game_done 0s · seed 17: 1.00 game_done 2s
- `0007`: seed 10: 0.50 game_done 0s · seed 11: 0.50 game_done 5s · seed 12: 0.50 game_done 1s · seed 13: 0.75 game_done 5s · seed 14: 0.50 game_done 0s · seed 15: 0.75 game_done 3s · seed 16: 0.50 game_done 0s · seed 17: 0.75 game_done 2s
- `0008`: seed 10: 0.50 game_done 0s · seed 11: 0.50 game_done 28s · seed 12: 0.50 game_done 0s · seed 13: 0.62 game_done 21s · seed 14: 0.50 game_done 0s · seed 15: 0.62 game_done 26s · seed 16: 0.50 game_done 1s · seed 17: 0.50 game_done 15s
- `0009`: seed 10: 0.50 game_done 0s · seed 11: 0.38 game_done 3s · seed 12: 0.50 game_done 1s · seed 13: 0.75 game_done 3s · seed 14: 0.50 game_done 0s · seed 15: 0.50 game_done 3s · seed 16: 0.50 game_done 1s · seed 17: 1.00 game_done 2s
- `0010`: seed 10: 0.50 game_done 0s · seed 11: 1.00 game_done 1s · seed 12: 0.50 game_done 1s · seed 13: 0.50 game_done 2s · seed 14: 0.50 game_done 0s · seed 15: 0.62 game_done 1s · seed 16: 0.50 game_done 0s · seed 17: 0.62 game_done 1s
- `0011`: seed 10: 0.50 game_done 1s · seed 11: 0.38 game_done 8s · seed 12: 0.50 game_done 1s · seed 13: 0.62 game_done 9s · seed 14: 0.50 game_done 0s · seed 15: 0.50 game_done 1s · seed 16: 0.50 game_done 0s · seed 17: 1.00 game_done 1s
- `0012`: seed 10: 0.50 game_done 0s · seed 11: 0.50 game_done 5s · seed 12: 0.50 game_done 1s · seed 13: 0.62 game_done 4s · seed 14: 0.50 game_done 0s · seed 15: 0.50 game_done 3s · seed 16: 0.50 game_done 0s · seed 17: 1.00 game_done 2s
- `0013`: seed 10: 0.50 game_done 0s · seed 11: 0.62 game_done 4s · seed 12: 0.50 game_done 1s · seed 13: 0.75 game_done 5s · seed 14: 0.50 game_done 0s · seed 15: 1.00 game_done 2s · seed 16: 0.50 game_done 0s · seed 17: 0.50 game_done 2s
- `0014`: seed 10: 0.50 game_done 0s · seed 11: 0.50 game_done 6s · seed 12: 0.50 game_done 0s · seed 13: 1.00 game_done 7s · seed 14: 0.50 game_done 0s · seed 15: 0.62 game_done 4s · seed 16: 0.50 game_done 0s · seed 17: 0.50 game_done 1s
- `0015`: seed 10: 0.50 game_done 0s · seed 11: 0.50 game_done 5s · seed 12: 0.50 game_done 1s · seed 13: 1.00 game_done 5s · seed 14: 0.50 game_done 0s · seed 15: 0.62 game_done 5s · seed 16: 0.50 game_done 1s · seed 17: 0.50 game_done 1s
- `0016`: seed 10: 0.50 game_done 0s · seed 11: 0.62 game_done 2s · seed 12: 0.50 game_done 2s · seed 13: 0.50 game_done 1s · seed 14: 0.50 game_done 0s · seed 15: 0.75 game_done 1s · seed 16: 0.50 game_done 0s · seed 17: 0.75 game_done 1s
- `0017`: seed 10: 0.50 game_done 0s · seed 11: 0.62 game_done 8s · seed 12: 0.50 game_done 1s · seed 13: 0.50 game_done 8s · seed 14: 0.50 game_done 0s · seed 15: 0.50 game_done 3s · seed 16: 0.50 game_done 0s · seed 17: 0.62 game_done 3s
- `0018`: seed 10: 0.50 game_done 0s · seed 11: 0.75 game_done 4s · seed 12: 0.50 game_done 0s · seed 13: 0.62 game_done 4s · seed 14: 0.50 game_done 0s · seed 15: 0.75 game_done 3s · seed 16: 0.50 game_done 0s · seed 17: 0.75 game_done 3s

System-2 latency ≈ 30.5 s + 33.5 ms × output tokens (fit over 30 archived calls, r = 0.97; slowest 217 s): long answers are slow — size timeouts from this, not from short baseline answers.

Score composition (mean per episode, from the adaptor's score_breakdown):
- `0000-tier1_astra_naive`: game/Mastermind-v0 0.27, game/TicTacToe-v0 0.25, literal_win 0.00, normalized_score 0.52, raw_reward 0.27, wrapper_truncation 0.00
- `0001`: game/Mastermind-v0 0.33, game/TicTacToe-v0 0.25, literal_win 0.12, normalized_score 0.58, raw_reward 0.33, wrapper_truncation 0.00
- `0002`: game/Mastermind-v0 0.31, game/TicTacToe-v0 0.25, literal_win 0.12, normalized_score 0.56, raw_reward 0.31, wrapper_truncation 0.00
- `0003`: game/Mastermind-v0 0.31, game/TicTacToe-v0 0.25, literal_win 0.12, normalized_score 0.56, raw_reward 0.31, wrapper_truncation 0.00
- `0004`: game/Mastermind-v0 0.33, game/TicTacToe-v0 0.25, literal_win 0.12, normalized_score 0.58, raw_reward 0.33, wrapper_truncation 0.00
- `0005`: game/Mastermind-v0 0.30, game/TicTacToe-v0 0.25, literal_win 0.00, normalized_score 0.55, raw_reward 0.30, wrapper_truncation 0.00
- `0006`: game/Mastermind-v0 0.39, game/TicTacToe-v0 0.25, literal_win 0.25, normalized_score 0.64, raw_reward 0.39, wrapper_truncation 0.00
- `0007`: game/Mastermind-v0 0.34, game/TicTacToe-v0 0.25, literal_win 0.00, normalized_score 0.59, raw_reward 0.34, wrapper_truncation 0.00
- `0008`: game/Mastermind-v0 0.28, game/TicTacToe-v0 0.25, literal_win 0.00, normalized_score 0.53, raw_reward 0.28, wrapper_truncation 0.00
- `0009`: game/Mastermind-v0 0.33, game/TicTacToe-v0 0.25, literal_win 0.12, normalized_score 0.58, raw_reward 0.33, wrapper_truncation 0.00
- `0010`: game/Mastermind-v0 0.34, game/TicTacToe-v0 0.25, literal_win 0.12, normalized_score 0.59, raw_reward 0.34, wrapper_truncation 0.00
- `0011`: game/Mastermind-v0 0.31, game/TicTacToe-v0 0.25, literal_win 0.12, normalized_score 0.56, raw_reward 0.31, wrapper_truncation 0.00
- `0012`: game/Mastermind-v0 0.33, game/TicTacToe-v0 0.25, literal_win 0.12, normalized_score 0.58, raw_reward 0.33, wrapper_truncation 0.00
- `0013`: game/Mastermind-v0 0.36, game/TicTacToe-v0 0.25, literal_win 0.12, normalized_score 0.61, raw_reward 0.36, wrapper_truncation 0.00
- `0014`: game/Mastermind-v0 0.33, game/TicTacToe-v0 0.25, literal_win 0.12, normalized_score 0.58, raw_reward 0.33, wrapper_truncation 0.00
- `0015`: game/Mastermind-v0 0.33, game/TicTacToe-v0 0.25, literal_win 0.12, normalized_score 0.58, raw_reward 0.33, wrapper_truncation 0.00
- `0016`: game/Mastermind-v0 0.33, game/TicTacToe-v0 0.25, literal_win 0.00, normalized_score 0.58, raw_reward 0.33, wrapper_truncation 0.00
- `0017`: game/Mastermind-v0 0.28, game/TicTacToe-v0 0.25, literal_win 0.00, normalized_score 0.53, raw_reward 0.28, wrapper_truncation 0.00
- `0018`: game/Mastermind-v0 0.36, game/TicTacToe-v0 0.25, literal_win 0.00, normalized_score 0.61, raw_reward 0.36, wrapper_truncation 0.00

Held-out evaluations (not used for selection):
- `0018` / heldout (TextArena (TicTacToe / Mastermind)): mean score 0.578 over 8 episodes
