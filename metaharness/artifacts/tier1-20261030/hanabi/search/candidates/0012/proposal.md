parent: 0003

# Hypothesis

One-slot command policies discard information carried by the sender's choice
between commands. A deterministic sender can communicate negative evidence about
an entire hand: if it chose this clue, no more valuable legal command was
available. Public inverse-policy constraint propagation should preserve useful
cards and identify additional safe actions without buying separate clues for
all of them. This is a communication/inference experiment, not a refinement of
leader 0008's discard-index channel or a new rollout value function.

# Archive evidence

Leader 0008's weakest training episode (trace_seed14.jsonl) finishes at 21 with
three lives. Its public history discards R5/card 35 at turn 53; Y2/card 48 is only
played at turn 68, triggering the last draw. Y3 and Y4 score on turns 69 and 70,
but Y5 remains held. The final heights are R4/Y4/G3/W5/B5. Communication works,
yet valuable unknown cards and delayed prerequisites still cost points.

Parent 0003 finishes the same training episode at 20 with three lives. W5/card 1
is discarded at turn 31, then the last G3/card 25 at turn 37. Its logged estimates
are `own discard E_loss=0.542, p_dead=0.286` and
`own discard E_loss=0.605, p_dead=0.349`. The first G3 was discarded at turn 26.
Its rank commands reveal one safe action but its posterior ignores why the sender
preferred that command over all other truthful command choices.

# Changes from parent

Retain the rotating rank/play and color/discard command alphabet, exact
finite-population own-hand marginals, legal dictionary actions, and isolated
seat sessions. Replace free choice among clues with one canonical encoder:
score each truthful command by its encoded recipient card's public-board value,
then choose the maximum with a fixed tie break. The encoder uses only the target
hand and public board; it never uses the sender's hand or private reasoning.
Astra may choose whether to transmit the packet, but cannot alter its contents.

After the accepted public clue, rebuild the deductions from a face-erased
snapshot and the engine's touched IDs. Branch on the selected slot's possible
identity. Every competing message which would outrank it supplies a constraint:
its target slot is not better, OR its clue attribute is absent from the hand.
Propagate these constraints and public singleton card capacities to a fixed
point, then union the surviving branches. Store those public support sets by
stable card ID. Discarded correlations are explicitly an over-approximation;
there is no particle-based certainty or transfer of privately visible faces.

The canonical rank message remains a guaranteed playable certificate. Color
messages remain recommendations, yielding to scoring opportunities as in the
parent. Code recomputes all deductions against current stacks before action.
Astra receives at most two late ambiguous decisions, with separate seat sessions,
a short JSON choice, caught model failures, and bounded caller waiting.

# Confirmation and refutation

Confirm if inverse packets remove identities beyond literal clues and the
one-slot play promise, produce subsequent safe plays/dead discards, and reduce
last-copy losses while completing games. Refute if the additional constraints
are too weak, canonical message selection prevents valuable alternative clues,
or recommendation-driven discards still erase needed ranks. Any inferred domain
excluding its subsequently public actual face refutes correctness regardless of
score. Log public packet domains and the number of extra identities removed.

# Validation

The harness and this proposal were written before experiments. Required preflight,
public encoder/decoder audits, model-failure checks and up to two permitted dev
evaluations will be recorded below. No training or held-out evaluation is run.

## Verification and first dev result

Preflight passed with natural completion (69 actions, zero answered model calls;
its 21-point smoke-test score is not a performance estimate). The observation-only
`audit.py` tested 819 archived views from candidates 0003, 0007 and 0008. Erasing
both hands' faces left every decoding unchanged; every actual visible target
identity survived. The packets removed 5,661 identities beyond literal clues,
including 2,352 outside the commanded slot. All 305 rank certificates were safe.
An additional 400 small random fixtures exhaustively checked 1,237 compatible
hand assignments against the encoder; all their identities survived propagation.

Explicit fake contexts passed arbitrary text, invalid option numbers, backend
failure, outer queue timeout, and separate initial/resumed sessions for both
seats. The required preflight does not exercise those calls itself.

The first and only dev evaluation, `dev/20261030-011801`, completed at **24/25**,
70 actions, three lives, 70.6 seconds, one requested System-2 call, and no error.
The caller caught its 70-second queue timeout; automatic play then finished.
There were 28 packets and 176 extra identity exclusions, 67 outside the commanded
slot. No live last copy was discarded. G5 remained held after G4 scored on the
penultimate action, so the missing point was a turn-access loss.

A complete stateful replay reconstructed every public support set and verified
every revealed/visible actual face. A fixed-history inference ablation retained
only literal masks and persistent rank-play certificates: at turn 62 it assigned
Y2 a 0.5 play probability, while full inverse inference proved it playable. This
is direct evidence of one additional executable inference on the observed line;
it is not a counterfactual episode score comparison. The 24-point result is one
dev deal, not evidence of a training or held-out mean improvement.

No strategic parameter was changed after this dev result. A second dev run was
unnecessary. The uncertainty limitation remains: support propagation is sound,
but finite-population probabilities conditioned on the union of unary supports
omit the full encoder-induced correlations and selection weights.
