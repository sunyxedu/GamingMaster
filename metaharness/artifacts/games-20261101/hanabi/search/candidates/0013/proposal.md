parent: 0010

# Hypothesis

The paired rollout policy is strongest while there is time to recover from a
bad posterior sample. Near deck exhaustion, however, a marginal play has a
large irreversible cost: one bad play consumes a scarce final action and can
remove a prerequisite card, while a discard or clue preserves the remaining
round. Candidate 0010's rollout estimate did not adequately reflect this
asymmetric late risk.

# Evidence

In `candidates/0010/trace_seed14.jsonl`, at turn 66 the public state had deck
size 3 and all three lives. The harness selected `PLAY` on slot 1 with
`p_success: 0.5` and a paired rollout advantage (`0.6875`, SE `0.1508`). The
actual revealed card was B3 while the B stack was already at 5, producing a
misplay. The episode then ended at 20 after the final draw, despite no earlier
life loss. This is a direct example where a noisy continuation advantage was
less valuable than avoiding an irreversible speculative action.

# Change from parent

Keep candidate 0010's exact multiplicity-aware beliefs, public discard-play
protocol, paired terminal rollouts, isolated seat sessions, timeout handling,
and legal-action validation. Add a deterministic horizon guard immediately
before action selection. If the deck has at most three cards, or the final-round
counter is active, an uncertain PLAY must have a higher posterior success
threshold (0.60 before the countdown, 0.72 in the last two final actions).
When it fails that threshold, select the highest-ranked non-PLAY candidate
already produced by the same public protocol. Certain plays remain unchanged;
the guard never invents an action or consults hidden faces.

# Confirmation and refutation

Support would be zero late misplays, fewer bomb-outs and fewer lost prerequisite
cards, with comparable early-game score and action count. A useful trace should
show the guard replacing a marginal late play with a legal discard/clue and the
next public state retaining all lives. Refute if the guard repeatedly discards
cards that were safely playable and lowers terminal score, or if it strands more
prerequisites during the final countdown than the parent. Compare paired
endgame states by deck size, final-turn count, lives, score, and action type;
preflight must also survive arbitrary model text without invoking this branch.

# Limits

The threshold is intentionally structural rather than tuned to a training seed;
`deck_size` and `final_turns_remaining` are public contract fields. It cannot
repair cards already lost or schedule a prerequisite chain that was never
clued. The guard may sacrifice an occasional point when a play's true identity
is favorable, so the evaluator should measure both prevented misplays and
ordinary late-play opportunities.
