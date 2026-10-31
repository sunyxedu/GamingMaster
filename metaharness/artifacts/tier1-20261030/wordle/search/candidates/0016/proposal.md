parent: 0015

Hypothesis: candidate 0015's quotient search is reasonable, but its fixed `soare` opener leaves score on the table. The archived 0015 traces show `soare` as the first action in all 12 episodes and a mean of 3.50 guesses (0.583 score). Leader 0012 used `slate` in every episode and averaged 3.25 guesses (0.625 score). This isolates a plausible structural cause: the first partition, before any target-specific evidence, determines whether the later exact search starts with a compact, useful candidate family. The public answer/guess lists are fixed and exposed by the rules, so testing a slate-first policy is scenario independent.

What changed versus parent 0015:

- Retain the parent’s duplicate-aware green-first feedback, full-history candidate filtering, public word-list loading and hash checks, bit-mask behavioral quotient search, expected finite-horizon reward objective, attainable fallback, and action/progress verification.
- Change only the deterministic opener order from `soare, raise, slate` to `slate, soare, raise`. It still selects the first available word from the evaluator’s public legal list and never reads seed or hidden state.
- Keep one action per observation and the bounded Astra recovery path. If parsing, legality, turn advancement, feedback membership, or `ctx.progress()` indicates a failed plan, the evidence is sent to Astra; its reply is filtered to one unused legal word and checked by the same verifier.

Evidence that would confirm the hypothesis is a lower mean guesses-used / higher score than 0015, especially more three-guess solves, with zero errors and no increase in rejected actions. A tie would suggest the continuation search dominates opener effects; a regression would refute slate-first on this pinned public list or show that the parent’s quotient search interacts badly with the changed first partition. Unseen seeds are needed because the opener is intentionally chosen without seed-specific tuning.
