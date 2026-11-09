parent: 0005

## Hypothesis

The score ceiling in 0005 is now dominated by inference availability and stale
context. Its best trace (seed 11) reaches 0.50 after finding and taking the
biography, but a final Astra call is still running when the 1,800-second
deadline arrives. The seed 2 trace spends the last part of the episode revisiting
blocked directions and ends at zero. Most seriously, seed 1 submits no action:
the first model request remains in flight until the episode deadline. Candidate
0005 also continues one Astra session across the episode, so hidden conversation
history grows even though its visible transcript is bounded.

The test is that fresh, compact calls with a finite per-call timeout will yield
more usable decisions before the deadline, while retaining short routes across
positive score changes will preserve the useful batching that produced 0005's
0.50 episode.

## What changed vs. the parent

0007 starts a fresh Astra session for every route. Each request contains a
bounded evidence packet: the opening text, the exact current observation, eight
recent records, score events, failed commands, repeated exact command/response
pairs, an unverified notebook, and budget. It asks for 4--10 parser commands,
then executes them one at a time. A suffix is cut on parser/precondition text,
clarification or menu text, hazards, negative score, empty output, terminal
flags, or a repeated exact pair; a positive score delta is retained as route
evidence.

Every model call has a 105-second upper timeout (and leaves six seconds of the
remaining wall budget). Timeouts, backend failures, malformed replies, and
illegal actions are caught. The decoder accepts strict/fenced JSON, Python
literal-shaped replies, tagged lines, and a bare command, while rejecting
lifecycle and compound inputs. If a reply is unusable, a small transcript-
derived probe queue keeps pre-flight and transient failures total without
introducing a story-specific route.

## Evidence that would confirm or refute it

Confirmation would show no episode spending its whole deadline in one model
request, lower and more stable prompt/context sizes, more actions per completed
Astra call, and at least 0005's final native score without more parser loops or
deaths. Seed 1 should produce actions even when the backend is slow, and seed 11
should have time for a post-biography plan. Refutation would be fresh calls that
forget clues and lower score, or repeated short timeouts that leave too little
planning time; in that case the timeout or packet size needs retuning and route
continuity needs a more explicit code ledger.
