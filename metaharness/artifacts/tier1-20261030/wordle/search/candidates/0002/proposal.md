parent: 0000-tier1_astra_naive

Hypothesis: the baseline's zero score is caused by an action-channel failure, while a solver that treats the public answer list and duplicate-aware history as executable state should solve without spending model calls. Restricting partition search to surviving answers should also be more reliable within the episode wall clock than asking Astra to perform candidate arithmetic.

What changed vs. the parent: the harness loads the rule-provided lists, parses history defensively, filters the exact surviving answer set after each turn, and chooses a bounded minimax/entropy guess in code. It always posts a complete `guess <word>` action. It checks `ctx.progress()`, turn advancement, and error fields after every action; one verified failure is handed to Astra for a legal recovery guess.

Evidence that would confirm or refute it: confirmation is nonzero solves across seeds with zero illegal actions and zero System-2 calls on the normal path. Refutation is a lost episode despite accepted turns, indicating candidate-only partitioning or the public-list interpretation needs broadening; a stale/rejected action should appear in recovery evidence.
