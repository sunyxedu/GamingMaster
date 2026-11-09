# 0018 — break the patrol/bomb phase, not the firing mechanism

**Parent:** `candidates/0015/` (rewrite as requested).  The strongest comparison is
`0016`; its continuous-fire fallback is retained, but its temporal-projectile
reviewer is not.

## Hypothesis

The useful part of the recent candidates is the actuator, not their model
interventions: a fresh combined left/right action with FIRE on every tick
produces most of the score.  The remaining failure is survival.  The best
candidate's movement is a nearly fixed triangular wave.  In its seed-1 trace,
for example, the action runs are `right_fire` for 73 ticks, then alternating
roughly 41-tick legs; after tick 291 it remains on a long `right_fire` run.  Its
controller reports life evidence at frames 1096 and 1380 at previous x=100 and
x=80, while `down_in_shot_field=0`, `strong_dodges=0`, and
`jev_changed_actions=0`.  The other 0016 episodes likewise report zero
accepted incoming-projectile promotions.  Thus the temporal sensor was not
reaching the actuator, and fixed-route movement did not prevent blind impacts.

The parent makes the same broad sweep and adds 89 inferred shield lanes / 63
escapes across its evaluation, yet still records 16 blind `life_loss_signal`
events.  Those lane cues are not established bomb tracks: early shield chips
repeat at x=106, 42, 44, 113, 45 in the 0016 traces and can be outgoing-beam
erosion.  More shield detours are therefore not the next experiment.

**Test:** preserve the proven full-rate firing, but replace the periodic
central triangle with long, changing-endpoint patrol legs.  Make a life loss a
phase reset: re-phase the endpoint generator and use a wider route rather than
restarting the same path.  This is a seed-independent way to test whether a
predictable patrol synchronizes with unseen bomb lanes.  Explicit decoded bombs
still get exact code-side collision avoidance.  If the route change increases
survival and reduces repeated life losses without destroying kill throughput,
the phase hypothesis is supported.  If scores fall with no survival gain, the
hypothesis is refuted and the simpler fixed sweep was doing the work.

## Structural change and division of labour

### Code owns the irreversible loop

`harness.py` uses only the public symbolic observation.

* Every tick posts a legal moving action with FIRE.  The Atari one-laser lock
  remains hardware-owned; no guessed cooldown suppresses the productive firing
  stream.
* It tracks player motion, life/wave epochs, formation center/phase, shield
  occupancy, shot launches, and short visible-bomb tracks.  Geometry,
  boundaries, action mapping, and collision prediction remain code-owned.
* A route is a long committed leg.  The default `WIDE_DITHER` endpoint is
  selected from safe interior/outer-margin choices and changes using current
  score, formation position, observed weak lane evidence, and the life epoch,
  not a training seed.  It is deliberately not a one-tick jitter or a target
  lock.
* After a life loss, old route/advice/projectile state is invalidated and the
  next long leg is re-phased.  The previous x is logged for diagnosis only; it
  is not memorized as a permanent death lane or confused with the respawn
  position.
* A visible bomb with a predicted collision temporarily overrides the route.
  An empty `bombs` list is encoded as `UNSEEN`, never safe.  Shield-mask damage
  is a weak, decaying lane feature used to describe posture; it is not promoted
  into a surviving projectile and cannot force a permanent avoidance wall.
* The fallback is always the current safe patrol direction plus FIRE.  Garbage
  observations, timeouts, stale advice, and malformed model output leave this
  controller executable.

This is intentionally a structural movement/scheduling test rather than a
parameter tweak to the parent erosion gate.

### Jev chooses a route posture, not an action

The state sent to Jev is strict JSON with schema
`space_invaders_patrol_posture_v1`:

* `episode`: categorical life stage, wave, opening/mid/late phase, and recent vs
  quiet score progress;
* `formation`: low/rising/high vertical pressure and wide/reduced geometry;
* `route`: current posture and side, new/committed age, responsive/stalled
  motion, and whether a boundary is near;
* `threat`: `NEAR_COLLISION`, `VISIBLE_APPROACH`, or `UNSEEN`, plus a categorical
  weak shield-lane side.  `UNSEEN_NOT_SAFE` explicitly explains empty bombs;
* `cover`: `LEFT`, `CENTER`, and `RIGHT` shield qualities (`GOOD`, `DAMAGED`,
  `OPEN`), computed from the nine-row masks;
* `programs`: code-computed preconditions for `WIDE_DITHER`, `REVERSE`, the
  three cover postures, and `KEEP`;
* `plan`: the current Astra posture and the rule that every tick moves and
  fires.

Each request batches two independent `Choice` questions, `posture_a` and
`posture_b`, with reversed option order.  Their options are:

* `WIDE_DITHER`: a long cross-field traverse with a changed endpoint;
* `COVER_LEFT`, `COVER_CENTER`, `COVER_RIGHT`: a future patrol around the
  named code-selected shield neighborhood, still moving and firing;
* `REVERSE`: choose the opposite long leg at the next boundary;
* `KEEP`: retain the committed deterministic posture;
* `UNKNOWN`: insufficient evidence.

The wording says explicitly that this is for the **next route boundary**, not
an immediate joystick command.  Jev is not asked to count aliens, compare
coordinates, predict a projectile, or invent a target.  Requests are launched
only at the first route / a life or wave reset / a late route boundary, with
one outstanding latest-only mailbox and a 0.90-second timeout.  This keeps
stale fractions at zero without putting inference on the tick path.

An answer is accepted only when both reordered Choices agree, the selected
probability is at least 0.62 in each, and any supplied confidence is at least
0.25.  It must be from the same life, wave, and epoch and be at most 100
emulator frames old.  It is consumed once when a new route begins.  Cover
choices are additionally code-gated to a recovery/last-life posture with
observed non-open cover; `REVERSE` is only meaningful at a boundary.  A late,
missing, disagreeing, low-confidence, or garbage answer does nothing.  Jev
can therefore alter a route program, but can never override the latest
visible-bomb safety check or create a parked cannon.

### Astra chooses the durable posture

One asynchronous Astra call starts immediately after the first observation,
while the code patrols.  It must return one short JSON object:
`{"patrol":"DITHER"}`, with allowed values `DITHER`, `WIDE`, and
`COVER_ROTATE`.  `DITHER` is the default and uses changed long endpoints;
`WIDE` uses the outer-margin endpoints; `COVER_ROTATE` permits use of the
healthiest shield neighborhood only after a life loss.  The answer applies at
the next route boundary and never blocks an action.  The prompt includes
current lives, formation phase, route count, weak shield-event count, and
visible-bomb evidence, and explains that the archived fixed sweep had no
accepted projectile interventions.  A 55-second timeout, cancellation at
episode end, invalid-token rejection, and the deterministic default handle
slow or arbitrary Astra output.  Only one call is used because the archived
30–50 second System-2 latency makes per-life replanning too late.

## Fallback and timing

The actuator performs: observe, update the epoch/ledger, poll completed model
work, commit or continue a route, post combined FIRE, then wait for the next
tick.  It never waits for Jev or Astra.  Direct visible-bomb risk is evaluated
for the planned direction, hold direction, and reverse direction on the latest
state.  If the planned path is predicted to collide, the least-risk legal
escape is held briefly and then the route resumes.  At an endpoint, a
`WIDE_DITHER` leg is reselected with a different endpoint; after a loss the
route and advice are invalidated before this selection.  Any action not in the
advertised action set falls back to `fire` or the declared noop.

## Confirmation and refutation

Confirmation requires more than a higher score: training traces should show
multiple non-identical long route endpoints, phase resets after life losses,
longer survival / fewer losses, and retained invader kills.  The strongest
confirmation is an improvement in the lower-tail episodes where 0016 died
with empty bomb observations, without a rise in stale ticks or model errors.
Jev changes should be route-posture applications, not emergency joystick
flips.

Refutation is a score/survival regression, repeated deaths at similar elapsed
phases despite endpoint changes, or a large reduction in kills because the
controller spends too long in cover programs.  If no model answer is ever
usable, that is not a failure of the primary phase test: the executable
`WIDE_DITHER` policy is the controlled intervention.  If dithered patrols
survive but kill fewer aliens, the next iteration should preserve the new
survival phase and add code-owned lowest-threat firing windows rather than
reintroducing the unvalidated shield-projectile inference.
