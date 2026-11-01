# Candidate 0009 — interval-certified, risk-calibrated Frogger

## Parents and diagnosis

This is a crossover of `candidates/0006/` and `candidates/0008/`.

I kept the mechanism that made 0006 the best candidate: a code-owned,
time-expanded receding-horizon search to the median or an open home, durable
route goals rather than frame-by-frame model actions, and an explicit direction
pulse followed by neutral release.  I also kept 0008's useful motor discipline:
phase estimation from consecutive public observations, a bounded asynchronous
Jev adviser, and no waiting for a model before servicing the real-time clock.

The failure common to the parents is more specific than “the model was stale.”
Both infer river support from NUSIZ/width layouts.  In the 0006 seed-109 trace,
the frog is at `x=129` on river row 10 while the public carrier interval is
`[104,128]`; the observation still says `moving_with_platform=true`, but the
frog dies two ticks later.  The analogous failures are `x=137` beside
`[112,136]` (tick 5768) and `x=12` on the wrapped edge of a row-10 carrier
(tick 3054).  Their route certificates claim margins `1.25`, `0.5`, and `1.0`
because a guessed NUSIZ-7 width of 32 fills space that the adaptor's
`x_intervals` does not.  0008 has the same issue in its home-approach deaths:
its certificates at ticks 666, 1038, and 1492 claim margin `1.0`, and its first
river death is certified at margin zero.  Positive geometry margins therefore
do not mean support when the geometry has manufactured carrier ink.

The initial copy hypothesis also has a necessary correction.  Archived public
states show that `x_intervals` is sometimes a total span, not a list of every
ink copy: at seed 103 tick 105 the frog is at `x=24` on row 7 while NUSIZ-6
objects have bases `116` and `124` and reported spans `[116,148]` and
`[124,156]`.  Their separated copies include a wrapped copy around x=24.  A
controller using the span as solid ink incorrectly declares this state water.
Conversely, NUSIZ-7 at the 0006 death reports `[104,128]` or `[112,136]` and
behaves like a 24-pixel enlarged sprite, not the parent's guessed 32 pixels.
The final controller therefore combines the parents' copy expansion with the
public span as a width calibration: copy-bearing codes are expanded into
separate intervals, while NUSIZ-7 uses the observed 24-pixel width.  No gaps are
filled.

This also explains the score split.  0006's search and pulse executor can make
many homes (12 on its worst seed-109 episode and 20 on seed 103), while 0008's
very low stale fraction does not compensate for its over-strict/incorrect route
forecast (mean score 240).  The new test is an observation-model correction,
not another Jev prompt or a fixed route.

## Hypothesis

River support should use a *separated-copy* reconstruction, calibrated by the
adaptor's public intervals, rather than either extreme: a solid NUSIZ rectangle
or an x-interval-only decoder.  For copy-bearing NUSIZ values the controller
uses the cartridge spacing to create individual carrier intervals and preserves
their water gaps.  For single/enlarged sprites it uses the public wrapped
interval; NUSIZ-7 is calibrated to the observed 24-pixel width.  Thus the
x=129/[104,128] and x=137/[112,136] edge witnesses are rejected, while the
wrapped NUSIZ-6 copies that actually support x=24 or x=15 remain available.

A short phase-aware forecast and an empirical safety mode should preserve enough
of 0006's throughput to complete homes instead of turning corrected geometry
into a permanent wait.  False road clearance is handled more conservatively
than river support: road cars are checked against both the public interval and
the expanded NUSIZ hazard envelope.

## What changed

1. **Calibrated separated-copy river support.**  For copy-bearing NUSIZ codes,
   carrier checks reconstruct individual intervals from the code and base,
   preserving water gaps; for enlarged/single sprites they use the public
   wrapped interval, with NUSIZ-7's observed 24-pixel width.  The frog must
   overlap real carrier ink, not a point or the total span, and a new landing
   requires more overlap than an already-supported ride.  Unknown/diving
   carriers remain unusable; after a death the requirement tightens for the
   rest of the episode.
2. **Phase-aware short forecast.**  Consecutive public observations learn each
   lane's update phase and pixel step.  The transition checker evaluates every
   frame of a six-frame hop/release window, including source support, destination
   support, road collision, carrier drift, screen edge, and the asymmetric home
   opening.  Before a phase is known, both sides of the fractional motion
   prediction must be safe.
3. **Risk calibration from counterexamples.**  A death increments an
   episode-local risk level, increases the road guard, invalidates the Jev
   lease, and requires more river overlap.  At a carrier edge where the current
   observation claims support but the exact interval does not, the planner allows
   only a lateral escape toward verified ink; it never authorizes an UP hop from
   that state.  This is a recovery rule, not seed-specific memory.
4. **Bounded first-action-diverse beam.**  The planner searches separately for
   several currently open homes (or the median), keeps first-action diversity,
   and executes only the first action.  It ranks complete homes above partial
   progress but prefers a buffered route after failures.  If no long route is
   certified, it retains a one-hop hold/escape fallback rather than blindly
   repeating UP.
5. **Motor and model bridge.**  A direction is posted only until observed
   movement or a four-frame cap, then NOOP is posted for a release interval.
   Jev asynchronously chooses a durable route label from code-generated
   candidates; its reversed high-stakes Choice must agree with probability at
   least 0.62.  Astra reviews concrete forecast failures and may select only
   `balanced|strict`, home order, and safe/progress style.  Neither model can
   override a geometry rejection or block the frame loop.

## System-1 interface

Each Jev request carries strict JSON with:

- the frog zone and timer bucket;
- the current risk mode/level and whether a recent forecast failure exists;
- at most five code-certified candidates, each with a route label, first hop,
  complete-home versus median/forward versus hold/escape class, connectedness,
  interior/edge surface, and semantic clearance bucket (`wide`, `buffered`, or
  `edge`);
- the number of open homes and the explicit division of responsibility: code
  owns coordinates, geometry, legal actions, and rechecking.

The candidates are not raw NUSIZ data and Jev is not asked to count, subtract,
or invent an action.  The request batches three independent judgments:

1. `route`: a `Choice` over the candidate route labels plus `UNKNOWN`;
2. `verify`: the same Choice with reversed option order, guarding against option
   order bias;
3. `hold`: a `Noul` asking whether the leading edge candidate is explicitly
   unsafe.  It is diagnostic in this implementation; the executable planner
   still makes the final safety decision.

A route lease is accepted only if both Choices select the same non-UNKNOWN label,
the lesser selected-option probability is at least **0.62**, the response is at
most 90 ticks old, and epoch (death/home/level), zone, and plan version still
match.  The lease lasts 120 ticks, but it may only choose a route whose current
certificate remains within 45 value points of the leading route.  A stale,
ambiguous, malformed, or missing-probability answer is discarded.  Model output
is never parsed as an action.

Jev is requested no more often than every 75 ticks and has a three-second
request timeout, above the measured 0.2-second tail.  There is at most one
outstanding request.  A late answer does not pause the motor: the newest code
search supplies the leading route, and otherwise a verified NOOP or lateral
escape is used.  Every direction is rechecked by the latest public observation
before posting it.

## System-2 role and revision trigger

Astra is a background route-policy reviewer, not a controller.  Its prompt
contains the current public frog/home summary, recent forecast/death records,
risk level, and search timing.  It is asked for only the small JSON enum
`risk_mode`, `home_order`, and `style`; arbitrary prose or invalid values leave
the executable policy unchanged.  A slow call is not spent on a clean short
run: the first review is opened by a death after the initial motor warm-up (or
by a long clean run), and later reviews are separated by about 60 seconds.  A
policy change invalidates the Jev lease and increments the plan version; a level
change rejects a stale policy.

The hypothesis is confirmed if the interval-specific edge witnesses disappear,
complete-home routes remain frequent, homes per active frog increase, and search
latency does not produce a large stale fraction.  It is refuted if exact
intervals repeatedly reject observations that survive, if the controller stalls
on safe rows, or if deaths still occur while the frog body has a buffered
interval overlap.  Those outcomes would distinguish a decoder/coordinate
hypothesis from an unresolved input-sampling or home-hazard problem.

## Fallback and contract safety

The loop posts one legal action and waits for the next tick on every iteration.
It catches ordinary Jev/Astra timeout, transport, type, and garbage-output
failures; `BudgetExceeded`, `EpisodeOver`, and `ModelUnavailable` retain the
contract's lifecycle semantics.  JSON parsing is defensive, and stand-in model
answers cannot create an invalid action.  Direction pulses are neutral-rearmed,
state-8 death/home animation receives NOOP, and the loop returns only when the
game or context ends.  All state is derived from the supplied observation and is
fresh per episode.
