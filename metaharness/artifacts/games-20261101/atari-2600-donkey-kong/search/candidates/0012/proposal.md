# Candidate 0012 — calibrate the player edge, then close the loop on the RAM jump

## Parent and hypothesis

Parent: `candidates/0004/` as requested by the generation directive.  The
strongest positive control is `candidates/0009/`; `0010` is the relevant
negative comparison.

The hypothesis is that the remaining ramp failure is a coordinate-frame error,
not a need for more model calls: the ladder boxes expose their left edge, while
Mario's `player.x` is also a box edge and the usable overlap is offset by about
two pixels.  A second symptom of the same control error is treating a jump as a
fixed eight-tick action instead of synchronizing it to the observed jump-state
byte.  The test is therefore:

> Aim Mario's left edge at `ladder.x - 2` (106, 46, 106, 46, 106, 30), hold UP
> as a measured transaction until y has actually advanced, and hold directional
> FIRE only while raw RAM says the jump is active.

This is deliberately different from 0009's `x=47`/`x=107` deadband and from
0010's `x=48` target.  It is also different from switching to the second
same-row ladder after a short no-progress timeout.

## Evidence and diagnosis

0009's best trace repeatedly reaches the first upper walkway but stalls at the
second ladder.  In `trace_seed0`, its snapshots are `(105,167)` then `(105,154)`
for the first climb; it walks to `(49,149)`, posts `up`, falls back, and later
oscillates around `(81,152)` without reaching the next row.  The same pattern
appears in seeds 1, 2, 3, 4, and 5.  The terminal observations document the
static ladder at x=48, while the positive-control 0007 trace shows the successful
second climb at player x=46: at probes 395–410 it remains x=46 while y moves
145 → 139 → 132 → 126.  Thus `x=49` (0009/0010) is not equivalent to the
successful `x=46` alignment.

0009 also starts jumps from a hardcoded eight-quantum budget.  Its traces show a
jump begun at an immediate hazard followed by a transition only a few quanta
later; the controller has no observation-based end condition.  0010 improved
hazard direction but still uses a fixed FIRE count and targets x=48, and scores
133.3 versus 0009's 250.  No candidate combines the observed successful x=46
edge with a raw-RAM jump transaction.

## What changed

* The ramp is a finite player-edge route `(106, 46, 106, 46, 106, 30)`.  A
  phase changes only after the player has moved at least eight pixels upward and
  reached the next walkway.  Alignment is strict; after a long no-progress
  interval the controller retries one neighboring edge, but never abandons the
  current ladder for the alternate same-row box.
* A jump is a transaction driven by raw RAM index 5 (`$85` in the documented
  128-byte wrapped snapshot).  Directional FIRE is held while the byte is
  nonzero, followed by a five-quantum steering tail away from the meeting side,
  then eight UP recovery quanta before route control resumes.  The direction
  opposes measured hazard motion, while a still obstacle uses the route side.
  The row gate prevents a hazard on the adjacent girder from interrupting a
  climb; the hard jump trigger is `immediate` so a 36-pixel approach is not
  jumped too early and then allowed to arrive during the lockout.
* Idle, life loss, transition, and terminal state are code-owned.  FIRE is one
  start edge and sparse restart edges; no model can leave the actor blocked or
  spam FIRE.  The loop posts one valid action per real-time tick and waits on
  `ctx.wait_for_tick`, so no stale ticks are introduced by inference.
* The rivet controller is a raw-row, bottom-to-top serpentine fallback with
  committed jumps over observed byte changes.  If the run reaches it, Astra is
  spawned once in the background to choose only a first sweep side; Jev batches
  a legal side judgment and a separate hazard judgment.  Neither can override
  the route, gap, jump, or lifecycle guards.

## Confirmation or refutation

Confirmation is a trace that reaches the second ladder at player x≈46 and y≈126
without the 0009 fallback oscillation, then reaches a higher ramp phase or the
rivet screen.  A new jump reward while retaining lives, a rivet-byte change, or
screen completion is progressively stronger confirmation.  The alignment
hypothesis is specifically supported if the same action schedule at x=46 climbs
where x=49 does not.

Refutation is repeated no vertical progress while x is held near 46 with no
hazard interruption, or a trace showing the game accepts x=49 but rejects x=46.
If the ramp advances but rivet rows never change, that refutes only the later
rivet heuristic and leaves the ramp test informative.  A model failure is not a
refutation: all model outputs are advisory and malformed/late outputs are
ignored.

## Tier-2 System-1 interface and fallback

Jev is never on the actor's critical path.  Its strict JSON state contains only:
`stage`; player horizontal/vertical semantic bands and motion; the current ramp
phase or rivet row; the four raw row bytes; whether a climb/jump transaction is
open; one code-derived current-row hazard with `side`, `motion`, named distance
bucket, `closing`, and urgency; last action; lives; completed screens; and
rivets removed.  Code computes all coordinates, row status, distances, and
legality before sending the state.

One request batches two independent questions:

1. `sweep`: a `Choice` over `left`, `right`, `wait`.  It recognizes which legal
   side is consistent with the current rivet mask and serpentine route.  A
   returned side is consumed once and can only change the next rivet preference
   when no climb is open and that side still has a rivet.
2. `danger`: a `Noul` asking whether the named hazard is soon/immediate and
   closing.  It is logged implicitly by the trace but is not trusted for the
   hard jump guard; this avoids making Jev's ~0.15–0.2 s latency lethal.

Requests are spawned every 14 ticks, or every 5 ticks near a hazard, with a
0.65-second timeout.  A completed `sweep` Choice is accepted only when its
returned option probability is at least 0.55 (or when a stand-in provides no
probability field); otherwise it is treated as undecided.  Results are
collected only if the future is complete; a late, failed, garbage, or stale
answer leaves the deterministic fallback unchanged.  The fallback is FIRE in
idle, NOOP during transitions, calibrated route actions on the ramp,
RAM-synchronized directional FIRE for hazards/gaps, and the row-mask route on
the rivet screen.

Astra is started asynchronously once on entry to rivet play.  Its prompt
contains the same compact state and asks for exactly `{"first_side":"left"}`
or `right`; parsed output can affect only the first rivet sweep.  Astra is never
awaited, and a timeout/cancellation/garbage response is harmless.  It is not
called during the ramp because the archive's measured 49–177 s System-2 latency
cannot choose a four-frame action; if the calibrated route fails, the code's
alignment retry is the deliberate fallback rather than a blocking review.
