parent: 0001

# Hypothesis

The leader's randomized digit relabeling equalizes expected performance over every fixed hidden code, but it may add avoidable variance on a small fixed training split. A deterministic, symmetry-valid canonical policy should have the same uniform all-code mean, while producing repeatable actions and potentially a better realized training mean than sampling a new whole policy in each episode. This tests policy randomization itself, not another reward objective.

The seed-13 trace for leader 0006 shows a normal, verified execution: `[5 3 2 4]` receives `0 black, 3 white`, `[5 2 3 6]` receives `1 black, 2 white`, and `[4 2 3 5]` ends at 0.5. Seed 11 similarly follows its verified plan and scores 0.625. The remaining low outcomes are not failures of parsing, legality, or local feedback checks. Candidate 0006's proposal reports the same exhaustive mean as deterministic reward search (0.66493056) but its sampled root digit map changes the concrete plan episode to episode. This experiment removes that randomness and uses a fixed lexicographic tie order.

# Change

Start from candidate 0001's exact reward search and public TicTacToe minimax. Use canonical `[1 2 3 4]` as the Mastermind opener; every opening is equivalent under the documented digit/position symmetry. On turn two, evaluate all legal probes and all public feedback partitions, then choose the branch-optimal terminal guess by exact native expected reward. At every tie use the same stable lexical code order, with a possible-secret solve tie-break. No random source, seed, training lookup, or hidden state determines policy actions.

After every action, compare `ctx.progress()`, native move count, legality, board transitions or cumulative feedback, posterior consistency, and terminal score. A mismatch invalidates local control and sends the evidence to Astra for a single legal move; replanned actions pass through the same effect checks. Garbage model text and caught model errors fall back to a legal action.

# Evidence

Confirm the hypothesis if training mean or weak-seed performance improves over 0006 while TicTacToe remains a draw and no error or invalid action appears. The policy's own action trace should be identical across episodes with the same public state. Refute a score claim if the fixed policy merely trades outcomes among codes, or if the randomized leader remains better on the fixed split. Exhaustive all-code reward must match the known 0.66493056 exact-search mean; a gain there would indicate a search or tie implementation error rather than a real opportunity under a uniform code prior. Native training outcomes, abstract all-code results, and run-to-run randomized results must remain separate.

The hypothesis is deliberately limited: deterministic tie-breaking cannot improve the uniform prior's Bayes-optimal expected score. It can only reduce policy variance and may or may not help the small fixed seed sample. No validation or held-out results are used.
