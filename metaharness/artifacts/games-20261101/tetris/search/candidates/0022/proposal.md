# Candidate 0022 proposal

## Parents and diagnosis

Parent: `candidates/0019/`, with `candidates/0017/` as the performance
reference.  I used the parent as a source of the verified real-time plumbing,
not as the scoring policy.

`0017` is still the best complete controller: every training episode placed
200 pieces, it had zero illegal/stale actions, and it averaged 78.17 lines.
The remaining failure is visible in its worst trace rather than in survival.
In `trace_seed4`, the controller finishes at 57,630 with 77 lines, one hole,
and `back_to_back: false`.  Its lock outcomes contain 67 single clears and
five doubles, but no Tetris or line-clearing T-spin; the I pieces are commonly
placed as ordinary horizontal/vertical pieces (for example the early I locks
at pieces 3 and 12) instead of being reserved for a repeatable four-line well.
The high-scoring seed 3 follows a different surface by chance and does obtain
line-clearing T-spin events, which explains why a safe, nearly identical
line total can score much more.

The two recent attempts to repair this by making spins more salient are not a
safe answer.  `0019`'s worst seed has 557 rotation ticks but only 122 locks,
then ends at 125 pieces with 38 lines and eight holes; its trace contains a
long clockwise-rotation run instead of progress.  `0020` improved local
recognition but mostly increased no-line spins, not line-clearing spins or
B2B.  Thus the new experiment does not add another rotation bonus or ask the
executor to manufacture a T-spin.

## Hypothesis

The missing score is primarily a **surface/queue problem**: the baseline
keeps a low flat stack, but does not condition its evaluator on preserving a
one-column I well.  It consequently spends I pieces on ordinary placements
and breaks the opportunity for a difficult clear.  A conditional 9-0-style
well, used only while the board is low and hole-free and an I is visible in
the current/preview/hold stream, should turn some of the existing safe
placements into vertical four-line clears.  No-line pieces do not break a
back-to-back chain, so separated I Tetrises can still sustain B2B while the
other six pieces fill the surface.

The test is deliberately conditional rather than “always build a right well”.
The code chooses the safer edge once, reserves it only while it remains a
shallow clean well, and turns the preference off for holes, dangerous height,
or a missing near-term I.  A stateful queue beam carries projected combo,
back-to-back, level, and line-clear value through the next two visible pieces;
this tests whether queue-conditioned well preparation, rather than a larger
single-piece line weight, is the bottleneck.  The immediate guardian and the
verified SRS executor remain in charge whenever the beam or a model is late.

## What changed in `harness.py`

* Reimplemented the exact symbolic board conversion, SRS shapes/kicks,
  placement enumeration, hold source handling, feature calculation, and
  one-action BFS router in a new module.  The engine remains authoritative for
  collision and scoring; all model output is still labels, never key presses.
* Added a persistent `well_side` chosen from the two edge columns.  A
  `well_signature` describes whether that edge is reserved, shallow, or ready
  for an I.  Reservation is enabled only with no holes, modest height, and an
  I in the current/hold/short queue horizon.  Non-I placements that fill the
  reserved edge are penalized; a safe I placement that actually clears four
  rows receives a substantial but bounded difficult-clear utility.
* Replaced the parent’s feature-only continuation with a bounded stateful
  continuation.  Each branch updates projected lines, combo, difficult-clear
  status, B2B, and level, and scores future well readiness and four-line
  opportunities.  It is a tie-breaker around the conservative board-quality
  rank, not a free pass for a tall or hole-making branch.  The foreground
  coarse guardian uses the same conditional well terms without waiting for
  recursion.
* Do not add a spin-route requirement.  A T piece is routed to a normal legal
  target; only an I Tetris gets the new difficult-clear signal.  This is an
  intentional isolation from `0019`/`0020`'s rotation-progress regression.
* Astra is an asynchronous mode reviewer.  It receives only buckets plus the
  current well status and may return `WELL`, `BALANCED`, or `RECOVER`; a token
  changes the future evaluator regime, never the current action.  Jev receives
  a small portfolio of code-generated `safe`, `clear`, `well`, and `recover`
  placements and can select a role only inside a code value envelope.  These
  model roles are advisory and cannot compromise the deterministic well
  safety gates.

## System-1 state, questions, and bridge

Every request is strict JSON rebuilt from the exact piece snapshot:

```json
{
  "ruleset": "guideline-marathon-v1",
  "policy": "BALANCED | WELL | RECOVER",
  "piece_instance": 42,
  "board": {
    "height": "low | moderate | high | critical",
    "holes": "hole-free | few | several | many",
    "surface": "flat | uneven | rough",
    "lines": 17,
    "level": 2,
    "combo": "active | inactive",
    "back_to_back": false
  },
  "active_piece": {"type": "I", "orientation": 0, "anchor_x": 3},
  "hold": "T",
  "can_hold": true,
  "next_queue": ["O", "J", "L", "S", "Z"],
  "well": {"side": "right", "status": "reserved | ready | off",
           "i_visible": true},
  "fallback_role": "safe",
  "roles": {
    "safe": "code-generated low-risk placement",
    "clear": "code-generated clean line clear",
    "well": "code-certified well-preserving or I-clear placement",
    "recover": "code-generated hole/height recovery placement"
  },
  "candidate_count": 9
}
```

The descriptions are derived from simulated cells and use semantic buckets;
Jev is explicitly told not to count cells, compare raw arithmetic, calculate
score, invent a destination, or output an action.  `safe` is the current
code-ranked plan.  `clear` is the best safe multi-line option.  `well` exists
only if it is a code-certified reserved-edge/I opportunity.  `recover`
minimizes holes and height.

One asynchronous `ctx.decide` call batches two independent `Choice` questions.
The second has the option order reversed to reduce first-option bias.  I accept
a result only if both values are valid and identical, the selected answer's
probability is at least 0.45 in both questions, the piece and hold state are
still compatible, and its rank is no more than 60 utility units below the
current deterministic plan.  A malformed answer, disagreement, timeout, or
stale snapshot is ignored.  A Jev answer can therefore choose a well
representative when it is genuinely close, but cannot turn an unsafe plan into
an I-well gamble.

Astra is called in the background once after the opening has enough board
facts, and at most once more after a real height/hole transition.  Its prompt
requires one leading token and includes the current well status, visible I
availability, and recent clear kind.  Only an exact leading token is accepted;
all other text is ignored.  Since Astra is slow, it is never awaited by the
game loop and its token only changes future ranking weights.

## Fallback and real-time safety

At each iteration there is an executable coarse guardian or emergency current
piece plan.  Search and Jev results are tagged with `(pieces, active.type)` and
are discarded if that identity changes.  Hold is posted at most once and only
when `can_hold` is true.  The router recomputes legal SRS moves from the latest
active pose; if a command does not change x/orientation on the next observation,
it hard-drops rather than repeating a blocked rotation.  If lock remaining is
at most 0.10 seconds, or active cells equal ghost cells, hard drop wins over
any pending plan.  The loop posts an action before waiting for the next tick,
so model latency cannot pause gravity.

The code uses `ctx.spawn` only for finite search work and model calls.  Search
futures are given a short bounded cleanup/cancel attempt on return; model
futures are left to the evaluator's cancellation contract.  All exceptions
from ordinary model timeout/runtime failures and garbage text are converted to
an unchanged deterministic policy.  Persistent `ModelUnavailable` is allowed
to propagate as required by the contract.

## Evidence that would confirm or refute the hypothesis

Confirmation would retain 200 pieces, zero illegal actions, and essentially
zero stale ticks while increasing score over 63,216.67 through more Tetrises,
line-clear points, combo/B2B points, or a higher level at similar survival.
The decisive traces should show `well_ready`/`well_clear` placements and
engine `clear_kind: "none"` between difficult I clears, rather than long
rotation-only spans.  Gains concentrated in 0017 seed-4/seed-5-like runs
would support the conditional-well explanation.

It is refuted if the well gates create holes/top-outs, reduce completed pieces,
or produce only extra no-line I placements and no additional four-line
clears/B2B.  If the well is rarely enabled because the surface is unsafe, that
is evidence that the 9-0 premise does not fit these streams; the next change
should remove the reservation rather than loosen the safety envelope.  If
ordinary lines and score are unchanged while the stateful beam changes many
placements, its projected economics are miscalibrated.  If rotation counts
again explode or lock progress falls, the experiment has reproduced the
`0019` execution failure and is rejected regardless of score.
