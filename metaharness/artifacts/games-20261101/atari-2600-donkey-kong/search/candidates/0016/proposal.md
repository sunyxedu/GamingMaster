# Candidate 0016 — adaptive connector graph with event-arbiter models

## Parent and hypothesis

Parent: `candidates/0006/` (rewrite directive).  I retained the useful
player-edge calibration and the positive route evidence from `0009`/`0012`, but
changed the route representation and the model boundary.  `0012` is the
strongest evaluated controller, so its traces are the main comparison.

The hypothesis is that the next failure is a **connector deadlock**, not a lack
of another hazard threshold.  The fixed route reaches the third ramp band, but
then can command a direction that has no horizontal effect while it is on the
sloped section.  In `0012` seed 0, for example, snapshots at ticks 400, 440,
500, 540 and 600 all show approximately `(x=87,y=111)`, `phase=2`, action
`right`, no climb transaction, and no score progress.  Seed 3 has the same
`(85,111)` plateau.  The target remains x=106, so the controller has no
observation that can make it change to the other connector at x=88.  The
proposal's deliberate test is:

> Treat each ramp band as a small connector graph.  Measure whether the last
> horizontal command actually moved Mario; on a bounded no-motion event mark
> that connector bad and try the other ladder on that band.  Preserve the
> demonstrated connector choice across lives.

A second, related hypothesis is that the jump landing tail should preserve the
latched direction.  `0012` starts a successful jump reward at tick 163/339 but
then reverses direction during its landing tail.  Candidate traces do not
establish that the reversal is safe; this controller keeps the original
left/right direction through a short non-FIRE tail and then locks out immediate
retriggering.  It also lets a newly aligned, genuinely progressing climb take
priority over a hazard warning on the old girder, rather than abandoning a rung
for every same-row warning.

## Evidence and diagnosis

The archive shows several distinct failures:

* `0004` had fresh actions but posted `fire`, `up`, `leftfire`, then `noop`
  through the transition.  Its problem was lifecycle and route ownership, not
  inference latency.
* `0009` established that the far-side route can earn jump points, but its
  second-ladder trace repeatedly falls back around x=49.  `0012` corrected the
  player-edge offset: x≈106 then x≈46 produces real height changes.
* `0012` confirms those first two connectors across all seeds and reaches
  phase 2.  In its best seed 2, the route continues to phase 3, but in seeds
  0 and 3 it has the long no-motion plateau at x≈85–87, y≈111 while still
  aiming at x=106.  That is not a model decision or a missed FIRE response;
  it is a missing horizontal-progress transition.
* In the same `0012` traces the first life commonly ends immediately after a
  close hazard jump (`jump_begin` around tick 157, score 100, transition near
  tick 166).  The controller can score the jump but has no explicit
  climb-versus-escape mode and reverses during the landing tail.  The new
  controller keeps the hard code guard, but does not let Jev or Astra delay it.
* All evaluated candidates still have `screens_completed=0` and
  `rivets_removed=0`.  The rivet controller is therefore retained as a
  verified-byte fallback, but this experiment spends its structural effort on
  the measured ramp deadlock that blocks reaching it.

## What changed

### Code owns the real-time machine

`harness.py` uses a `RAMP_GRAPH` with nominal and alternate player-edge
connectors: `(106)`, `(46,78)`, `(106,86)`, `(46,66)`, `(106)`, and `(30,74)`.
The coordinates are only the public fixed topology; every target is matched
against the visible ladder boxes before use.  A route phase advances only after
Mario has moved upward by at least seven pixels and reached the next band.

The new route transaction has two independent observations:

1. **Vertical transaction:** while `climb_open`, keep UP while y is improving,
   even if the sloped ramp changes x.  Only a long no-progress interval closes
   it.
2. **Horizontal transaction:** while walking toward a connector, count quanta
   in which the requested left/right action leaves x unchanged.  Eight such
   observations mark the current connector as a demonstrated deadlock.  The
   alternate connector is selected, and the failed connector is not retried on
   the next life unless the alternate also fails.  A successful climb records
   the connector as preferred for later lives.

This is intentionally different from `0012`'s strict x lock and its fixed
single choice at phase 2.  The route is a feedback graph, not a timer or a
nearest-ladder heuristic.

Hazards are normalized in code using the current-row y gate, relative motion,
and named distance buckets.  A closing object in the verified immediate/overlap window
gets a directional FIRE transaction; a farther warning is allowed to close
without spending the jump.  A climb that is already aligned
and showing vertical progress is allowed to finish unless the hazard is an
actual overlap.  FIRE is held until the raw `$85` jump byte is observed, then
the latched direction is held without FIRE for four quanta.  A fourteen-quantum
post-landing lock prevents retriggering on the same sprite.  These are hard
safety rules; a late model answer cannot suppress them.

Idle, transition, restart, terminal, and last-life transition handling remain
code-owned.  The loop posts an action and waits for the next real-time tick;
there is no synchronous model call in the actor path.  If a transition reports
zero lives, the program returns instead of spending 900 seconds posting noop in
a dead transition shell, which addresses `0012` seed 5.

The rivet fallback reads all four row bytes, infers the remaining side from
`0/6/12/18`, records which side was removed, jumps over a verified cleared gap,
and uses the alternating bottom-to-top ladder side.  It is not allowed to
walk through a recorded hole.

### Jev is an event arbiter, not a periodic joystick policy

A Jev request is spawned only for a meaningful event: a connector deadlock or an
immediate/overlap hazard.  Its answer is tagged with a signature containing
`context`, stage, lives, ramp phase, connector, rivet row, hazard slot,
distance bucket, hazard motion, and the deadlock count.  An answer is executable
only for the same signature and for at most three ticks; otherwise the
code-computed action wins.  Calls use `timeout_s=0.38`, above the measured warm
p90 while remaining off the critical path.

The strict JSON state sent to Jev is:

```json
{
  "game": "Atari 2600 Donkey Kong NTSC retail cartridge",
  "context": "hazard|deadlock",
  "stage": "ramp|rivet",
  "player": {
    "horizontal": "left|middle|right",
    "vertical": "low|middle|high",
    "motion": "up|down|level",
    "jump_committed": false
  },
  "route": {
    "fallback_action": "right|left|up|...",
    "ramp_phase": 2,
    "connector": 0,
    "connector_options": 2,
    "climb_open": false,
    "horizontal_stall": 8,
    "deadlocks": 1,
    "rivet_row": 0,
    "rivet_rows": [0, 0, 0, 0]
  },
  "hazard": {
    "present": true,
    "slot": 3,
    "side": "right",
    "motion": "left",
    "distance_bucket": "near",
    "closing": true,
    "urgency": "immediate"
  },
  "progress": {"lives": 1, "screens": 0, "rivets_removed": 0},
  "last_action": "right"
}
```

The batched questions are:

1. `tactic`: a `Choice` over `keep_route`, `try_other`, `jump_left`,
   `jump_right`, and `pause`.  In a deadlock context, `try_other` means only
   the already-listed alternate connector.  In a hazard context, a directional
   choice is meaningful only for the named immediate/overlapping same-row object.
2. `danger`: a `Noul` asking whether the code-derived hazard is explicitly an
   immediate/overlapping same-row closing object.  This is a recognition check and
   does not replace the numeric hard guard.

The bridge accepts a Choice only when its value is one of the declared options
and, when probabilities are supplied, its selected-option probability is at
least 0.55.  It can use `try_other` only in the matching deadlock signature and
can use a jump answer only while that same hazard remains present.  Missing
answers, arbitrary stand-in values, timeout, backend failure, low probability,
changed stage/life, or changed hazard all fall back to the adaptive code route.
Thus Jev can arbitrate an ambiguous connector without ever inventing an action
or holding a stale action over the real-time clock.

### Astra is a delayed macro reviewer

Astra is spawned after the first measured connector replacement, and once on
entry to rivet play (up to three calls total).  It receives the same compact
facts plus the reason for the event and is asked for only:

```json
{"connector":"nominal|alternate|adaptive",
 "hazard":"climb|jump", "first_side":"left|right"}
```

A valid response changes only future macro preferences: connector preference,
climb-versus-jump preference, and the first rivet sweep side.  It never emits a
joystick sequence, never blocks the actor, and is ignored if malformed or late.
The adaptive graph remains the executable fallback.  This reverses the earlier
division where Astra was asked for a route while the finite route code had no
way to detect a local deadlock: code supplies verified events, Jev handles the
fast ambiguous fork, and Astra reviews the route only after evidence says the
current macro is failing.

## Confirmation and refutation

Confirmation would be a trace where, after the familiar `(85–87,111)` plateau,
the controller logs `connector_replaced`, selects the x≈86 ladder, observes a
new upward band, and reaches phase 3 or the rivet screen without spending a
life on the plateau.  Stronger confirmation is any `screens_completed > 0`, a
changed rivet byte, or a successful restart followed by greater ramp progress.
A successful climb that preserves the branch in the next life would validate
the online graph memory rather than merely a lucky hazard trace.

The connector hypothesis is refuted if the controller repeatedly records no
horizontal progress, tries the visible alternate at x≈86/66/78, and still
cannot obtain vertical response while hazards are absent.  The landing
hypothesis is refuted if same-direction landing tails systematically lose a
life where the old reversal survived.  If the route reaches the rivet screen
but rows do not change, that is evidence against only the later rivet sweep,
not against the ramp experiment.  Model failures are not gameplay refutations:
all model output is optional and defensively parsed.
