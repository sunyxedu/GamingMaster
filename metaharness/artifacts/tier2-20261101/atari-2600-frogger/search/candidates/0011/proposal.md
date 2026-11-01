# Candidate 0011 — adaptive river-support clearance

Parent: `candidates/0006/`. Mode: refine.

## Hypothesis

The parent's route search is largely correct, but its river support certificate is
systematically too optimistic at the moment a frog leaves the last river carrier
for a home.  A positive predicted margin is not enough when the frog is near the
edge of a moving carrier and the next joystick sample happens while the carrier
has advanced.  The parent already increases its road obstacle margin after a
witnessed road death, but it does not learn anything from a witnessed river
support failure: `river_pad` stays at 2.0 for the whole episode.

This is supported by the worst parent traces rather than by a stale-model
argument:

* In `candidates/0006/trace_seed109.jsonl`, the frog is at row 10, x=129 at
  tick 2530.  The row-10 carrier has a NUSIZ-7 turtle at x=104 (the frog is at
  the right edge of its 32-pixel copy).  The controller issues `UP` at tick
  2529 for `HOME_3`, with route margin **1.25**, yet a death is recorded at
  tick 2532.  The same episode dies at tick 3054 after `HOME_0` from row 10,
  with margin **1.0**, and at tick 3814 after another row-10 `HOME_3` route
  with margin **1.25**.
* The corresponding seed-101 trace has the same repeated witness: row 10,
  x=129, `HOME_3`, margin 1.25 at tick 2448 followed by death at 2452;
  row 10, x=17, `HOME_0`, margin 0 at tick 2967 followed by death at 2968.
* Other parent episodes show the same class at positive margins: seed 107
  dies at tick 2749 after a `HOME_1` route with margin 1.5, and seed 113
  dies at tick 2740 after a row-10 route with margin 0.5.  Thus the existing
  road-only feedback is aimed at the wrong part of the geometry.

The change tests one claim only: after a river death, shrinking the accepted
carrier interior by a small, episode-local amount will eliminate repeat edge
support deaths while still leaving broad logs/turtle groups routable.  It is
not a seed-specific bay rule and it does not change the motor clock or Jev
interface.

## Change

The 0006 controller is retained: copy-aware object expansion, body-aware
collision checks, time-expanded route search, explicit direction pulses followed
by NOOP, durable route goals, and asynchronous Astra policy remain unchanged.

The one new feedback path is in `update`: when a death is observed and the
previous public frog row is a river row (6--10), increase `river_pad` by 0.75,
up to 5.0, and log the new value.  `grid()` already applies `river_pad` by
shrinking every recognized carrier's safe frog-left interval, so the next
search automatically rejects the just-witnessed edge class and prefers a more
interior carrier/phase.  A later Astra policy application may request the
ordinary center/inset pad but cannot lower this episode-local learned floor.
The value resets for a new episode and is not applied to road or home-bay
geometry.  Existing road feedback remains as-is.  All routes are still
rechecked from the newest observation; no model answer can override this
certificate.

## Confirmation / refutation

Confirmation would be fewer deaths whose previous frog is row 6--10, especially
fewer repeated row-10 home-entry deaths, with at least as many completed homes
per episode and no increase in stale ticks or model errors.  The strongest
confirmation is that a seed which previously repeats the 1.0--1.5-margin
row-10 witness instead waits for/chooses a higher-clearance carrier and reaches
another home.

Refutation would be unchanged river-death witnesses after the pad increases,
or a large rise in `no_route`/timeouts that prevents the road and river from
being crossed.  If deaths fall but homes also fall, the hypothesis would be only
partly supported and the clearance increment is too aggressive; if deaths are
unchanged at median/road rows, that is a different failure mode outside this
isolated experiment.

## System-1 interface and control bridge

The System-1 interface is deliberately inherited from 0006 so this candidate
isolates geometry feedback.  Each request's strict JSON state contains:

* current phase (`road` or `river`), frog zone, timer urgency, homes and the
  current validated Astra policy;
* code-generated complete route candidates, each labelled by endpoint, shortest
  versus detour, widest versus narrower carrier support, comfortable versus
  tight clearance, and whether it is the current durable goal;
* explicit facts that sprite-copy gaps are water, unfamiliar/diving carriers are
  rejected, the home crocodile field is unknown, and recent prediction-failure
  evidence exists.

One request batches three independent typed judgments:

1. `goal`: a `Choice` among the code-certified route endpoints plus `UNKNOWN`;
2. `reverse`: the same Choice with reversed option order, used to reject option
   order bias;
3. `review`: a `Noul` asking whether the supplied recent prediction failure
   merits Astra policy review.

A goal is accepted only when both Choices agree, the selected option's
probability is at least 0.65 in both distributions, and the response is fresh
(no more than 90 game ticks old with matching life/home epoch, phase and policy
version).  It lasts at most 180 ticks and can only select among current routes
whose geometry is independently revalidated.  Missing/malformed probabilities,
UNKNOWN, stale answers, and disagreements are rejected.  Jev never computes
coordinates, timing, margins, or a joystick action.

While Jev or Astra is late, the deterministic search chooses the newest
code-verified complete route, then a verified forward route, then a verified
one-hop survival move, and finally NOOP.  A direction is held only until
observed movement (or a four-tick cap), then explicit NOOP is posted before the
next pulse.  The frame loop remains nonblocking and posts/waits every real-time
tick.  Astra is only an asynchronous policy reviewer and its garbage/timeout
leaves the executable policy unchanged.

## Expected diagnostic events

The harness logs the old life/home outcome, the route certificate, and a
`river_margin_increased` event containing the prior and new pad.  Evaluation can
therefore distinguish the proposed mechanism from merely issuing fewer hops:
compare row-10 deaths and `no_route` intervals before and after the first river
failure, as well as homes, deaths, stale ticks, and model-call counts.
