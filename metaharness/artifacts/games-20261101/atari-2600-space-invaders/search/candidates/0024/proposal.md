# Candidate 0024 — late outer-column coverage excursions

## Parent and hypothesis

Parent: `candidates/0011/`; the score-producing control is compared against the
current best `0016`/`0023` family.

The best traces establish a useful but incomplete mechanism: fresh
`left_fire`/`right_fire` sweeps score well, while the temporal reviewer never
changes an action. Candidate 0023's seed 1 reaches score **145** and still has
23 living invaders at frame 1000; its life signals arrive at frames **1092**
and **1376**, with no accepted dodge. Its terminal observation has 22 living
invaders whose right edge is around x=128, while the best harness keeps the
ordinary sweep between x=38 and x=116. Seed 5 similarly plateaus at 140–150
before its first two losses and ends with 19 living invaders. This does not
prove that the right edge caused a death, but it is concrete evidence that the
fixed interior sweep can leave late outer survivors outside the firing corridor.

**Hypothesis:** the remaining wave-completion gap is partly a coverage failure,
not a missing model target choice. The fixed x=38..116 sweep is productive in
the opening and middle, but after formation drift and kills, a surviving outer
column can remain outside it. One short, late excursion to the observed outer
edge should create additional firing opportunities without sacrificing the
baseline's continuous movement or turning the controller into a target lock.

This is deliberately narrower than the prior route/target experiments. The
normal sweep, firing cadence, and interior escape margin remain unchanged. The
new behavior is only an endpoint extension when a live outer edge is outside
the baseline, the formation is late or score has stalled, and the controller is
at that matching endpoint. No fixed training-seed coordinate is used.

## What changed

* Code keeps the proven fresh combined-FIRE actuator and the x=38..116
  interior sweep.
* Every valid observation computes the minimum and maximum x of living
  invaders. A left/right `OUTSIDE_WORKING_SWEEP` gap is a categorical fact;
  malformed or empty invader lists cannot create one.
* At a matching endpoint, code may arm one excursion to a bounded endpoint
  derived from the current formation: approximately x=27..34 on the left or
  x=120..136 on the right. The excursion returns to the ordinary corridor,
  has a 180-frame per-side cooldown, and is reset by a life/wave/teleport
  discontinuity. The code never parks, chooses a target column, or suppresses
  FIRE.
* The default `COMPLETE` posture allows an excursion when at most 26 aliens
  remain or score has been quiet for 120 frames. `SURVIVE` (if Astra returns
  it) requires at most 18 aliens or a 200-frame stall. A deterministic stronger
  gap fallback means arbitrary or late model output cannot disable the test.
* Explicit decoded bombs still get a code-owned short collision check. An empty
  bomb list remains `UNSEEN_NOT_SAFE`; it does not trigger an invented dodge.
  Respawn x is not stored as a death lane.

## Astra role and Jev bridge

Astra is launched once, off the actuator path, immediately after the first
posted action. It is asked for only `COMPLETE` or `SURVIVE`; garbage, prose,
timeouts, or cancellation leave the safe default `COMPLETE`. The result changes
only the late-extension threshold and increments a plan version. It never
supplies an action or coordinate.

When a late outer gap exists, the harness sends Jev a strict JSON state with
schema `space_invaders_outer_coverage_v1`:

* `episode`: qualitative life bucket, wave, and `OPENING`/`PRESSURE`/`CRITICAL`;
* `formation`: `MANY`/`SOME`/`FEW`, whether the left/right edge is
  `OUTSIDE_WORKING_SWEEP` or inside, and `RECENT`/`QUIET` progress;
* `sweep`: interior/edge position, heading, and whether an excursion is active;
* `plan`, qualitative threat (`VISIBLE_NEAR`, `VISIBLE_FAR`, or
  `UNSEEN_NOT_SAFE`), and shield-cover bucket.

One request batches two reordered `Choice`s (`KEEP`, `EXTEND_LEFT`,
`EXTEND_RIGHT`, `UNKNOWN`) and a `Noul` asking whether a bounded excursion is
worthwhile. Jev performs recognition over code-derived categories; it does not
count invaders, calculate coordinates, or output action strings. An extension
answer is accepted only when both Choices agree, each selected probability is
at least **0.66**, non-`None` confidence is at least **0.22**, the Noul is at
least **0.55**, the matching gap is still present, and the answer is no older
than 40 emulator frames in the same epoch and Astra plan. The accepted answer
is advisory until the matching endpoint; the latest observation and the code
safety check are re-evaluated before arming.

## Late answers and fallback

The actuator always has a legal fallback: the current interior sweep, with
combined FIRE, or a code-confirmed visible-bomb escape. Jev/Astra calls run in
single-call mailboxes and are never awaited by the tick loop. A stale,
malformed, disagreeing, low-probability, or failed answer is discarded. If no
Jev answer arrives, only the stricter code condition can arm a late extension;
thus a stand-in model returning arbitrary values cannot crash or stall the
episode. Deadline and episode-close exceptions are normal termination, and
unfinished model futures are cancelled in `finally`.

## Confirmation or refutation

The hypothesis is supported by logs showing `outer_coverage_gap` followed by
`outer_extension_armed` at a matching endpoint, increased late low/outer-row
kills, fewer scoreless late intervals, and ideally a first-wave clear without
more lives lost. A useful causal comparison is an extension immediately before
an outer survivor disappears while the ordinary sweep would have remained
inside.

It is refuted if gaps never occur, extensions are never reached, extensions
increase deaths or reduce the established early kill throughput, or the score
still comes entirely from the baseline with no late-kill benefit. If the mean
falls, the next iteration should retain the ordinary sweep and remove the
extension rather than adding more model target selection. The explicit primary
milestone remains clearing the first 36-invader wave while preserving lives.
