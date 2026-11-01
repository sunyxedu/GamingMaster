# Candidate 0025 — reverse the sweep at a confirmed life loss

## Parent and one targeted hypothesis

Parent: `candidates/0023/` (the calibrated temporal-evidence controller, tied
with 0016 at 405.625 mean). I retain its code-owned fresh-tick combined-FIRE
sweep, projectile ledger, conservative shield bookkeeping, paired Jev evidence
review, and single asynchronous Astra rubric call.

The one change is at a **confirmed life transition**: when the observation
reports fewer lives, the next sweep starts in the opposite horizontal direction
from the direction that was active before the hit. The direction is not changed
on ordinary scoreless ticks, decoder identity changes, shield chips, or a player
x jump without a life decrement. This is deliberately smaller than adding a new
bomb detector or changing firing cadence.

**Hypothesis:** the remaining low-tail failure is repeated exposure to the same
sweep phase immediately after a respawn. The parent correctly keeps firing, but
its reset clears projectile evidence while preserving `direction`; therefore a
respawn at the center resumes the same lateral route that was active when the
previous life was lost. If the unreported bomb/life transition is tied to that
route phase, immediately reversing gives the fresh life a different escape lane
without sacrificing the productive opening sweep.

## Trace evidence

The parent’s worst seed 1 ended at **145**. It recorded no `strong_dodges`, no
`chip_escapes`, and no `jev_changed_actions`. Its first loss was logged as:

```
life_evidence frame=1092 lives=2 previous_x=100.0 gap_start=964
```

and the second as `frame=1376`, `previous_x=80.0`, `gap_start=1248`. The tick
trace has a long `left_fire` run across the first transition (ticks 234–290),
then a `right_fire` run after the reset (291 onward); the controller never
explicitly chose a new post-loss route. Seed 5, the other low episode (**210**),
shows the same shape: losses at frames **1152** and **1556**, with
`previous_x=68.0`/`94.0` and `gap_start=1024`/`1432`, and its final summary
again has zero strong dodges, chip escapes, and action-changing Jev reviews.

The broad sweep itself is productive in the parent’s 520/625/560-point episodes,
so changing every route or adding a hidden-threat heuristic would confound the
test. The archived traces also show that empty sampled bomb lists and thick
shield chips do not establish a reliable immediate dodge cue; prior shield and
stall interventions produced escapes without demonstrated survival gains.
Candidate 0020 additionally tried a formation-relative patrol with a 52-frame,
parity-selected respawn recovery and fell to 280 mean, so this is not a repeat
of that broad phase rewrite: it tests only one reversal at the observed life
edge and leaves every normal leg unchanged. The life decrement is the strongest
available reset signal, so this candidate uses only that signal.

## Code/model division and reset rule

Code owns life accounting, direction, legal action mapping, the continuous
`left_fire`/`right_fire` sweep, fire lock by held FIRE, projectile identity,
shield masks, and all freshness/epoch checks. On `lives < previous_lives`, it
increments the life-loss counter, reverses `direction`, clears transient
projectile/chip evidence as before, and logs `respawn_sweep_reversal` with the
old and new direction. The next `action()` consequently posts the opposite
combined-FIRE action from the respawn center. No reversal is triggered by a
malformed observation, an inferred identity conflict, or a stationary period
alone.

Jev still classifies only supplied projectile/shield evidence. Astra still
chooses the `MOTION` versus `CORROBORATE` rubric once. Neither model is asked
to select the reversal or a joystick action; garbage, disagreement, timeout,
and low confidence leave the deterministic sweep in control.

## Jev state schema, questions, and thresholds

The state schema remains
`space_invaders_temporal_evidence_v3_respawn_reverse`. Its `control` object
contains the calibrated joystick polarity, `responsive` versus `motion_gap`,
and `post_life_direction` (`reversed` or `unchanged`). Candidate records carry
only the existing qualitative projectile/shield evidence: decoder label,
ascending/descending increments, own-launch association, same-lane top-chip
corroboration, continuity, and cover bucket. The rules state that y increasing
is incoming, y decreasing is outgoing, labels and an empty bomb list are not
proof, and Jev must not invent an object or choose movement.

For each candidate, two independent `Choice` questions use the same three
options, `INCOMING`, `OUTGOING`, and `UNRESOLVED`, with reversed ordering. An
answer is accepted only if both values agree, each selected probability is at
least **0.72**, and any supplied confidence is at least **0.30**. Answers older
than 36 emulator frames, from another life/wave epoch or Astra plan version,
or for a changed projectile identity are discarded. The live code rechecks
legality and collision risk before using an accepted classification. There is
at most one outstanding Jev request with a 1-second timeout.

## Astra scheduling and fallback

Astra is launched once with a 55-second timeout and its output is accepted only
as the exact token `MOTION` or `CORROBORATE`. A late, malformed, or failed call
never blocks the real-time loop. Jev/Astra futures are cancelled on return.
The fallback at every tick is the parent’s deterministic combined-FIRE sweep;
the only new fallback-state transition is the confirmed-life-loss direction
reversal.

## Confirmation or refutation

The hypothesis is supported if, relative to the parent, the candidate keeps the
same early kill throughput while low episodes show `respawn_sweep_reversal`
and either fewer losses, longer survival, more kills, or a wave clear. The
strongest confirmation is a life transition followed by movement away from the
prior sweep direction and a later survival/score continuation without any new
model-selected dodge.

It is refuted if reversals are never reached, if they alter high-scoring
episodes but do not improve survival, or if mean score falls while loss count
and post-loss survival are unchanged. If the score is unchanged and all
reversals occur only after an unavoidable terminal sequence, this isolates the
failure to an unseen threat rather than proving the route hypothesis.
