parent: 0013

# Hypothesis

Candidate 0013's horizon guard reduces speculative plays late, but it still ranks ordinary discards by an expected-loss scalar. That scalar can hide a unique rank-5 card behind the value of drawing and restoring a clue. A late discard of a likely live five permanently removes one point and often its prerequisite chain; protecting that card only after it is fully identified is too late. A narrow, public-horizon discard guard should preserve likely live fives while leaving early play and communication unchanged.

# Evidence

The leader's seed-14 trace contains a concrete residual failure at turn 53: with deck size 11 before the discard (ten after its replacement draw), three lives, and R stack at 4, the acting seat selected `DISCARD 4` with `p_dead=0.5`, `p_critical=0.417`, and expected loss `0.811`. The accepted event revealed that slot as R5, and the final R stack remained 4. Candidate 0013 improved late speculative-play risk but did not alter this ordinary-discard path. The same class of loss appears in the leader archive for W5 (seed 11, turn 26) and B5 (seed 12, turn 10), although those earlier events are less clearly endgame-driven.

# Change from parent

I retain candidate 0013's exact finite-population beliefs, public promise/discard protocol, paired masked rollouts, late PLAY horizon guard, legal dictionary actions, isolated seat sessions, timeout handling, and deterministic fallbacks. Facts now include `p_five_live`, the posterior mass of non-dead rank-5 identities. When the deck has at most 12 cards, a DISCARD candidate with at least 0.15 live-five probability receives a strong ranking penalty, including after any free-play signal bonus. The guard only orders already legal candidates; if every action is risky, the harness still posts a legal action, and no hidden face enters shared memory. Dead-card cleanup, certain plays, clues, and all earlier-game behavior remain governed by the parent.

# Confirmation and refutation

Support would be fewer late discarded fives and fewer lost-live-last-copy events, with the R5-like terminal deficits repaired and no increase in misplays or bomb-outs. A useful trace should show the guard selecting another legal discard or a clue when a likely live five is present, then preserving that color through the final countdown. Refute if the guard over-saves probable fives that are actually dead, starves clue recovery, or leaves more prerequisite chains inaccessible; compare paired states by deck size, live-five posterior, chosen action, terminal score, and action count. Preflight must tolerate arbitrary Astra output and never rely on this branch for legality.
