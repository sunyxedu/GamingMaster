# Candidate 0022 — learned sloped-walkway hazard geometry

## Parent and diagnosis

Parent: `candidates/0019/` (with `0018`/`0020` as the closest comparison and
`0017` as the high-scoring negative control).

The parent fixed the most obvious route-state defect: it confirmed levels from
observed y movement and reacquired a connector after a jump.  That mechanism
really does move Mario through the lower route.  In the representative
`candidates/0019/trace_seed3.jsonl`, it records `ramp_advance` at
`(105,154)` and `(46,126)`.  It then starts the first upper-ramp jump at tick
158, when the controller believes a same-row hazard is at distance 22, and
enters `transition` at tick 168.  The parent eventually reaches only score
300 and never reaches the rivet screen.  The same two-band progress followed
by a death appears in the other 0019 traces.

The relevant observations are not actually horizontal rows.  Immediately
after the parent jump, tick 165 reports Mario at `(87,118)` and a barrel at
`(86,132)` moving left.  The absolute top-edge separation is 14 pixels.  The
corresponding `0012` trace reports `(89,117)` and a barrel at `(90,131)` after
its first upper-ramp crossing.  The parent classifiers use an 18-pixel
vertical gate (and related candidates use 20--40 pixels), so a barrel on a
neighbouring sloped girder is eligible for a jump.  The fact that the sprite
is close in x does not prove it occupies Mario's walkway: on a ramp, the
walkway's y changes with x.

This is also consistent with the repeated score-farming pattern.  `0018` and
`0020` confirm the first two height bands, then either die at the same upper
hazard or fall back to repeatedly jumping near the lower upper band.  Their
traces show that better jump persistence and more accurate phase bookkeeping
changed the number of 100-point jumps, but did not establish that the object
which triggered the jump was on Mario's current sloped path.  No prior
candidate learns a local ramp line from observations; all use an absolute
player/hazard-y window or an unconditioned same-row approximation.

## Hypothesis

> The upper-ramp deaths are caused in part by a false same-row hazard
> classification.  If the actor estimates the current walkway's local y(x)
> line from Mario's own grounded horizontal motion, then compares each hazard
> with that line rather than with Mario's y alone, it will stop interrupting a
> climb or crossing for hazards on an adjacent girder.  When a hazard is
> genuinely on that learned plane, an open-side jump (away from the sprite)
> followed by route reacquisition should be safer than the parent’s jump toward
> the named hazard side.

This is a structural perception/control test, not another jump-distance
threshold.  The online line is discarded after a route or life boundary and
while Mario is airborne.  Thus it cannot carry a stale slope from a previous
walkway.  It is only used after at least six grounded horizontal samples over
18 pixels of x span with a small residual; before that, the fallback is a
narrow 10-pixel y test.  A hazard is eligible for the hard jump guard only if
it is close/closing and within 10 pixels of the fitted current-plane y.  A
nearby sprite that is not on the plane is retained in diagnostics but cannot
steal the route.

When the guard fires, 0022 deliberately latches the direction away from the
hazard (`left` if it is to the right, `right` if it is to the left).  This is a
new landing-corridor test: the jump's purpose is to clear the collision box,
not to steer into the named object.  The direction is held through the RAM
jump edge and four grounded-tail observations, then the route walks back to
the measured connector.  Rivet-gap jumps are separate and continue toward
the declared gap side.

## What changed

* Added `RampPlane`, a small bounded least-squares fit of grounded Mario
  samples for the current route phase.  It records only samples made while
  moving horizontally with no jump and no open climb.  It reports a learned
  slope, residual, and x span for trace diagnosis.
* Replaced the parent’s absolute-y hazard gate with `hazard_facts`.  On the
  ramp it computes `row_error` against the fitted local y(x) line; an
  uncertain fit falls back conservatively.  A current-plane close/commit
  object owns the hard guard.  A neighboring-plane object is ignored for
  jumping even if its x is close.
* Changed the ramp jump primitive to an open-side landing corridor.  FIRE is
  synchronized to the documented wrapped jump byte, with a short steering
  tail, cooldown, and a fresh plane after landing.  Route height still
  advances only after observed y gain; branch changes require bounded
  horizontal or vertical non-response.
* Kept the useful parts of 0019: calibrated player-edge connector graph,
  observed-height route transactions, explicit sparse FIRE restart edges
  including the zero-spare final Mario, and raw rivet-byte bookkeeping.
* Added a complete deterministic rivet continuation.  It follows a
  bottom-to-top serpentine route, detects exact 0/6/12/18 row changes, and
  commits a gap jump instead of walking through a cleared rivet.  Astra can
  advise only the initial side/route macro; it cannot override a plane guard,
  gap, jump lease, or lifecycle.
* Added a bounded event bridge.  Jev is asked only at a measured connector
  stall or during a rivet route, never synchronously on a four-frame hazard.
  Astra is spawned after rivet entry or repeated measured stalls and returns
  compact macro labels.  Invalid stand-in text, missing probabilities,
  timeouts, cancelled futures, and stale answers leave the deterministic
  action path intact.

No training-seed trace, emulator source, or held-out information is embedded;
the coordinates are the public fixed ladder prior and the fitted geometry is
learned anew in each episode.

## Confirmation and falsification

A confirming trace should show all of the following:

1. `plane="learned"` before an upper-ramp hazard, with a small `row_error` for
   a true current-plane hazard and a rejected/absent jump for a nearby
   off-plane object;
2. `ramp_advance` through more bands than 0019/0020, with fewer repeated
   `jump_begin` events at one phase;
3. a jump whose direction is the open side, followed by `jump_end`, connector
   reacquisition, and further measured y progress rather than immediate
   transition; and ideally
4. `stage="rivet"`, a nonzero `rivets_removed`, or a banked screen bonus.

The learned-plane hypothesis is supported even before screen completion if an
observation like `(87,118)` with a `(86,132)` hazard produces a large
`row_error` and no jump, while the actor continues to a higher connector.  It
is refuted if a stable fitted line repeatedly rejects hazards that visibly
share Mario's path and Mario dies without a hard guard, or if the fit remains
uncertain during ordinary horizontal traversal.  The open-side subhypothesis
is refuted if plane-confirmed hazards consistently require the parent’s
hazard-side direction to survive.  A ramp that advances but fails to change a
rivet byte would isolate a later rivet-route failure rather than refute the
plane classifier.

## Tier-2 System-1 contract

### State schema

Each request is strict JSON containing only derived semantic facts:

```json
{
  "game": "Atari 2600 Donkey Kong NTSC retail cartridge",
  "event": "ramp_stall|rivet_route",
  "stage": "ramp|rivet",
  "player": {
    "horizontal_band": "left|middle|right",
    "vertical_band": "bottom|lower|middle|top",
    "motion": "up|down|steady",
    "jump_committed": false
  },
  "route": {
    "phase": 2,
    "branch": 0,
    "options": 2,
    "climb_open": false,
    "plane": "learned|uncertain",
    "row": 0,
    "row_status": [0, 6, 12, 18],
    "gap_pending": false
  },
  "hazard": {
    "present": true,
    "side": "left|right|none",
    "motion": "left|right|still|none",
    "distance_bucket": "overlap|near|approaching|far|none",
    "relative_motion": "closing|opening|none",
    "closing": true,
    "urgency": "overlap|commit|soon|watch|near|clear"
  },
  "fallback_action": "up",
  "progress": {
    "lives_remaining": 1,
    "screens_completed": 0,
    "rivets_removed": 0
  },
  "last_action": "up"
}
```

Code computes coordinates, the fitted line, `row_error`, legal branches,
raw-rivet decoding, and action legality.  Jev is not asked to calculate
numbers, identify an action index, or invent a route.

### Questions and thresholds

A `ramp_stall` request batches:

* `branch`: a `Choice` over `hold`, `nominal`, and `alternate`.  `alternate`
  is accepted only when the route state explicitly has two connector options.
  `hold` is safe during a live climb.  An accepted non-hold answer requires
  option probability at least `0.60` and is applied only to the current
  `(phase, branch, route-failure)` stamp.
* `row`: a diagnostic `Noul` asking whether the supplied learned-plane facts
  establish a current-walkway hazard.  It never creates or suppresses a hard
  jump; code already made that decision.

A `rivet_route` request batches:

* `side`: a `Choice` over `left`, `right`, and `hold`.  A side changes only
  the future serpentine preference when that side is still legal for the raw
  row byte and no gap/climb transaction is active.  It requires probability
  at least `0.60` when probabilities are present.
* `danger`: a diagnostic `Noul` for an explicit current-row overlap/commit.
  The deterministic hazard guard remains authoritative.

Requests are asynchronous with a 0.60-second timeout and at most one
outstanding request.  Late, failed, malformed, low-probability, stale, or
budget-exhausted answers are discarded.  Pre-flight stand-in answers can
choose any option without causing an exception because every value is
validated and the code fallback is always executable.

### Astra bridge and fallback

Astra receives the same compact derived state after verified rivet entry or
after repeated measured connector stalls.  It is asked for optional JSON
labels `branch=nominal|alternate|adaptive`, `hazard=escape|climb`, and
`first_side=left|right`.  Only `first_side` can affect a fresh bottom rivet
row; a branch preference is advisory until a measured stall.  Astra never
posts an action and never blocks the actor loop.

The executable fallback is: one FIRE start edge; sparse FIRE transition edges;
line-aware current-plane hazard jumps; measured connector walking and UP
climbs; and raw-mask serpentine rivet movement with committed gap jumps.  The
actor waits on `ctx.wait_for_tick()` and posts one valid literal action per
quantum until the game/context ends.
