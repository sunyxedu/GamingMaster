parent: 0010

# Hypothesis

The leader's paired rollout policy already protects most cards, but its late decisions still treat communication, discard-index signals, and immediate scoring as nearly interchangeable. The remaining score loss is a *turn-access* failure: a useful card can survive physically yet miss the few actor turns after the final draw, while a noisy discard can erase a live high card immediately before that phase. A strict public endgame scheduler should convert the final few draws into an execution phase: consume every certain play first, and only communicate when the recipient can still act on the information.

# What changed

I retained candidate 0010's exact literal-card accounting, persistent public promises, discard signal protocol, paired masked rollouts, legal dictionary actions, timeout handling, and separate seat sessions. The new scheduler wraps its candidate generator. When the final countdown is active, or the deck has at most six cards, certain own plays are moved ahead of every clue and discard. If no certain play exists, truthful clues are ranked by the number of newly executable partner slots and critical saves; in the one-action final state, discard signals are placed after plays and clues. This is a structural phase change rather than a parameter tweak, and all ordering uses public board/horizon facts plus the seat-local literal beliefs already used by the parent.

# Evidence

In the leader's seed-14 trace, R5 was discarded near the end despite a later need for the red chain, and Y2/Y3/Y4 consumed the final actions while Y5 remained held. The parent proposal also records a live R5 loss after a noisy save-vs-play comparison. These are consistent with insufficient execution scheduling, not malformed actions or misplays. The hypothesis is supported if late episodes show fewer nonproductive clues/discard signals, more prerequisite chains completed before the final countdown, and fewer live last-copy losses without increasing bomb-outs. It is refuted if the scheduler strands cards because an apparently certain play should have been deferred for a partner save, or if earlier `deck<=6` activation reduces scores on seeds where communication was still valuable.

The implementation is intentionally conservative outside this phase: normal-game choices and rollout evidence remain unchanged, so any score change can be attributed to endgame scheduling. Preflight must tolerate arbitrary Astra text and model failures through the inherited parser/fallback paths.
