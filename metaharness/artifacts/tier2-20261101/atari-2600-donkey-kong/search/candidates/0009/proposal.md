# Candidate 0009 — settled ladder feedback with committed crossing jumps

## Parents and hypothesis

Parents: `candidates/0007/` and `candidates/0008/`.

The useful mechanism in 0007 is not its model usage (it used none): the fixed
`107,47,107,47,107,31` route actually moves Mario through multiple ramp levels
and earns 100 points.  Its failure is visible in `trace_seed0`: at `n=197`,
Mario is at `(82,152)` while a same-row obstacle is at `(33,157)` and moving
right; the controller posts `leftfire`, i.e. toward the closing obstacle.  It
then enters `transition` at `n=210`, loses a life, and later repeats the same
failure.  It also leaves the final life in transition after its 900-tick cap.

The useful mechanism in 0008 is measured phase feedback and explicit restart
handling.  Its trace confirms the first climb, but it changes to level 1 at
`y=156` and then holds `left` at `x=106,y=156` for about 100 ticks.  0007's
successful lateral trace begins after the avatar has settled near `y=154`.
The specific hypothesis tested here is therefore:

> The flat score is caused by two structural timing errors, not by a missing
> planner: changing route phase while Mario is still at the bottom of a ladder,
> and reacting to a closing obstacle without a committed head-on jump.  Require
> a settled, measured climb before lateral motion; use the cartridge's rising,
> direction-latched crossing jump with an early warning and cooldown; preserve
> the sparse FIRE restart state machine.

## What changed

* Kept 0007's action-grounded finite ramp route, but retained a route phase only
  after UP has produced vertical motion and Mario is at least the settled
  `next-row + 6` height.  This specifically rejects 0008's `y=156` transition
  and keeps UP for the extra quanta needed to reach the lateral walkway.  Split
  levels retain both public ladder candidates and switch only after bounded
  no-progress.
* Replaced 0007's late/repeated hazard reaction with an early-warning guard
  (a named `approaching` bucket and a 40-pixel vertical sprite gate) and a
  committed eight-quantum, direction-latched head-on jump.  A 22-quantum
  cooldown prevents FIRE from being retriggered while the same obstacle is
  crossing.  The guard owns the action during the jump; it is not re-planned
  from stale model output.  The wider warning is deliberately conservative,
  and the ramp phase remains code-owned so Jev cannot turn an upper-row slot
  into a persistent movement command.
* Added explicit idle start, stage/life route reset, and a FIRE retry at sparse
  transition ages.  The controller continues the last life instead of treating
  `lives_remaining == 0` as game over, and stops only on the reported terminal
  state/game over.
* Added a mask-driven rivet serpentine and gap-jump primitive so that a future
  ramp completion is actionable.  Row-byte changes, rather than guessed x
  crossings, arm a jump over the newly cleared hole.
* Jev is asynchronous and low-authority only after the rivet stage is reached.
  No Astra call is needed for the verified finite ramp topology: avoiding a slow
  System-2 call is itself part of the real-time hypothesis.  Jev's answer can
  choose only an already legal rivet sweep side and cannot suppress route,
  jump, or restart guards.  Garbage/timeout answers are ignored.

## Evidence that would confirm or refute it

A public-seed trial of the current controller reached 100--300 ALE points
from jump awards, versus 0007's 100, while retaining zero stale ticks; the
strongest short trial scored 300 before the last-life transition.  Confirmation
in evaluation is a trace that reaches a higher ramp waypoint than 0007 without
an immediate collision, earns another jump/rivet reward, or reaches `rivet`
and changes a row byte.  A completed screen and a post-death restart with low
stale ticks are stronger confirmation.  In particular, the first climb should
show UP continuing past `y=156` and lateral movement should resume near `y=154`.

Refutation is repeated death while the avatar is visibly settled, the obstacle
is still separated, and the controller commits a direction-latched jump; or
failure to move laterally even after the stricter settled threshold.  If the
ramp is completed but no rivet byte changes, the rivet x/row interpretation is
the next hypothesis to revise rather than weakening the ramp guard.

## Jev state, questions, thresholds, and fallback

The strict JSON state sent to Jev contains only: stage; player horizontal band;
ramp level/rivet row; the four raw rivet row bytes; whether a jump is committed;
one normalized same-row hazard (`present`, side, named distance bucket,
closing, urgency); last action; remaining lives; and completed screens.  Code
computes all coordinates, distances, masks, and legality.

Each request batches two independent judgments:

1. `side` is a `Choice` over `left`, `right`, and `wait`, asking which side is
   already supported by the current row mask and serpentine route.  A value is
   used only in the rivet stage, only if that side is still present, and only
   to select a legal horizontal sweep.
2. `danger` is a `Noul` asking whether the normalized hazard is an immediate
   same-row threat.  It is diagnostic; the deterministic guard always wins.

Requests are spawned at least 12 ticks apart with a 0.75-second timeout.  A
future is collected only when complete; a late, failed, malformed, low-
confidence, or stale answer leaves the code fallback unchanged.  The fallback
runs every tick: FIRE in idle, NOOP during the animation until a sparse restart
FIRE, phase-based ramp/rivet movement during play, and committed directional
FIRE for hazards/gaps.  No model call can block the real-time actor.
