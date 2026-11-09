parent: 0003

# Hypothesis

The leader's remaining losses come from a low-bandwidth, stateless instruction channel: fixed slot arithmetic wastes symbols on impossible actions, and color messages may recommend throwing away unique future points. Replace that channel with a public-belief-adaptive codebook and persistent noncritical-card certificates. Literal clues that already establish a safe play should take priority over arbitrary slot arithmetic, letting one clue enable several subsequent plays. Better clue efficiency and durable discard information should reduce last-copy losses when tokens reach zero.

# Archive evidence

In the leader's weakest training trace, `trace_seed14.jsonl`, turn 31 discards card 1, W5, while W is at 2. Turn 37 discards card 25, the second G3, with zero information tokens; G is at 1. This blocks G3–G5. The episode ends at 20 with all three lives. These are point losses rather than transport failures. The same trace spends turns 9–18 on five color-instruction/discard pairs, including future prerequisites, while the fixed rank channel cannot address many slots. Seed 12 loses unique G5 at turn 14 immediately after exhausting the clue supply. The leader's safe-action execution and exact literal beliefs are valuable and are retained.

# Changes

Retain finite-population hand inference, public card accounting, strict legal-action dictionaries, seat-isolated advice, and bounded model waits. Replace fixed five-slot signaling with a codebook computed solely from public masks, public protocol constraints, heights and remaining multiplicities. Rank instructions address only slots that can still be playable. Color instructions address slots that can be noncritical and certify that the indicated card was dead or had another surviving copy at transmission. The certificate is an explicit face-domain restriction, reconstructed from the public clue event, retained by stable card ID, and re-evaluated as the game changes; it is not permanent permission to discard a once-redundant card.

If a clue establishes any public guaranteed play, it carries literal information instead of a slot instruction. At eight tokens clues are literal, ensuring a legal neutral channel even if the recipient has no playable or expendable card. The sender checks every inferred promise against its permitted visible target hand. All public decoding is face-blind; private beliefs and model sessions remain seat-local. Astra may arbitrate a few close strategic options but cannot change the protocol.

# Predictions

Confirm: more scoring plays per clue, fewer arbitrary color instructions, no false certificates or instruction misplays, fewer discarded last copies, and improved terminal score relative to 21.5. Refute: literal-mode clues fail to produce usable extra plays, the compressed codebook reduces rather than improves safe symbol coverage, or excluding critical discard messages stalls draws. Record protocol mode and discard expected loss in decisions. No seed, deck order or hidden engine state is a policy input.

# Validation

Implementation and proposal are written before evaluation. Pre-flight, public-decoder audits, and at most two permitted dev episodes will be reported below. No training or held-out evaluation will be run by the proposer.

## First dev result and targeted refinement

The first completed dev episode (`dev/20261030-002705`, seed 201) scored **22/25** in 70 actions and 80.6 seconds, with all three lives and no error. One advisory request exceeded the 80-second outer wait; the caught timeout disabled further advice and autonomous execution completed. The evaluator records one submitted call and zero completed answers. There were no instruction misplays. A color certificate received at turn 3 survived an immediate scoring override and supported the turn-19 discard of Y4 with inferred critical probability zero. This is direct evidence for the persistent-information mechanism. This single episode did not produce a multi-play literal packet, so that part of the hypothesis remains unconfirmed.

The remaining loss includes the last B4, discarded on turn 37 with no clues. Inspection exposed a defect inherited from the parent's discard utility: dead identities already have zero expected loss, but an additional `p_dead` bonus rewarded that upside again. The policy preferred a mixed unknown with expected loss 0.549 and critical probability 0.364 over a known redundant G3 with expected loss 0.378 and critical probability zero. The final refinement removes the separate dead-probability bonus; discards with the same draw/token effect are ranked by expected card loss. No new risk-taking, delayed instruction, or hidden-view search was added.

Observation-only audits validated **811** sender/decoder messages on the parent's archived observations: all transmitted play promises were safe, every certificate contained the visible actual face, and replacing every face in decoder input with nonsense left decoding unchanged. One hundred randomized small arbitrary-domain beliefs matched exhaustive finite-population enumeration. Explicit fake-context checks exercised malformed output, out-of-range IDs, valid selection, backend failure and outer queue timeout; each executed a legal fallback. Both pre-flights completed naturally without an exception (21-point smoke-test score, not an evaluation result).

## Second dev result

The second permitted dev episode (`dev/20261030-003131`, seed 201) again scored **22/25**, in 71 actions with all lives and no error. The discard utility correction changed the ordering around uncertain cards but did not change this deal's terminal score. It therefore remains a correctness fix and a falsifiable dead end for this seed, rather than a claimed improvement. Across both dev traces, public message replay found 30 sender/recipient decodings identical and 23 guaranteed plays succeeding. The adaptive channel caused no rejected or misplayed action. Training selection remains the evaluator's responsibility.
