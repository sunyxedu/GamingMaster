# Candidate 0014 — reachable trajectory reservations

## Parent and diagnosis

Parent: `candidates/0009/`.  The best reference is `candidates/0011/`, a refine of
0007.  I kept the demonstrated low-duty target firing, frame-local ticking, and
press/release FIRE handling, but changed the safety representation and the model
bridge.

The leader's remaining failures are not stale ticks.  In its seed-8 trace it posts a
`defense` at frame 387, then still sees `immediate` at frames 400 and 405 and loses a
ship at frame 410.  In seed 10 it defends at frame 279, remains `immediate` through
frames 288, 293, 298 and 303, and loses the ship at frame 304.  Similar windows occur
in seed 2 (immediate actions at 5664--5674 before the loss at 5688) and seed 4 (an
immediate window ending at frame 3690).  The trace shows the local emergency label is
already available, yet a static lane goal or a hyperspace pulse does not establish a
reachable escape.  0011's urgent cooldown bypass improved the mean dramatically, but
it still acts only after contact is essentially present; its positional lane can also
collapse to the current lane.

## Hypothesis

The missing mechanism is **early, reachable commitment**, not another model call.
A useful horizontal lane label is not a maneuver: the ship must rotate, thrust for a
finite number of frames, and keep its inherited velocity.  Waiting until `immediate`
leaves too little time for that sequence, while using DOWN at the first critical
forecast can teleport into another collision.  A controller that reserves a short
cardinal trajectory during `caution`, rechecks its predicted clearance every frame,
and uses hyperspace only when all reachable paths already overlap should preserve more
ships without repeating 0008's indiscriminate thrust.

This candidate tests that structural hypothesis.  It does not memorize seed starts or
import an arcade hyperspace probability.  It learns a bounded thrust component from
observed velocity differences, evaluates coast/current-nose/up/left/down/right paths
with wrap-aware relative motion, and executes at most a few thrust frames before
returning to the turret.  A path reservation expires quickly on a discontinuity, life
change, worsening forecast, or cleared danger.  Normal play remains target-aware
rotation plus code-owned FIRE pulses.

Evidence for the hypothesis would be later first/final life-loss frames and higher
score while `reservation_start` occurs mainly in caution and `thrust_frames` stays far
below the thrust-heavy 0008 run.  Refutation would be early losses during reservations,
substantially more edge commitment or thrust, or no survival improvement despite
critical windows still being dominated by FIRE/turn actions.  The useful trace fields
are `reservation_start`, `life_change`, `defense`, `danger`, `ttc`, `current_clear`,
`best_path`, and reservation cancellation/completion counts.  Official score remains
unclipped ALE reward.

## Division of labour

* **Code:** owns the emulator clock, immutable observation parsing, calibrated cardinal
  heading vectors, wrap-aware relative tracks, uncertainty inflation, finite trajectory
  scoring, target selection, acceleration estimation, action legality, FIRE release,
  life epochs, reservation expiry, defense gates, and fallback actions.  Code never
  treats a null enemy list as absence and never lets a model answer create an
  unvalidated thrust or DOWN.
* **Jev:** recognizes the regime and chooses among *named code-generated packets* in a
  compact state.  It no longer chooses a vague lane/posture that can be disconnected
  from reachability.  Its selected packet can start or prefer a reservation only when
  the latest local forecast still validates it; critical local safety is never vetoed
  by Jev.
* **Astra:** reviews aggregate evidence after the launch has been observed and selects
  durable policy (`reserve_early` versus `caution_only`, target versus sweep aiming,
  normal versus spaced fire, and overlap-only versus never defense).  It cannot post an
  action, choose a raw direction, or relax local safety.  Invalid or late prose is
  ignored.

## Jev state schema and questions

Every request is strict JSON and contains:

```json
{
  "rules": {
    "mode": 0,
    "fire": "press_then_release",
    "noop": "preserves_inertia",
    "down": "hyperspace_last_resort",
    "orientation": "LEFT increments, RIGHT decrements",
    "enemy_coverage": "unknown_not_absent"
  },
  "snapshot": {
    "frame": 1234,
    "lives": 3,
    "score": 530,
    "player_present": true,
    "position": "middle",
    "speed": "moving",
    "danger": "caution",
    "clearance": "limited",
    "impact": "soon",
    "asteroids": 7,
    "small": 3,
    "missiles": 1,
    "warning_streak": 4,
    "motion_discontinuity": false
  },
  "target": {"size": "small", "bearing": "left", "vertical": "above",
             "distance": "middle", "motion": "known"},
  "paths": [
    {"id": "left", "direction": "left", "clearance": "clear",
     "contact": "none", "bounded": true}
  ],
  "reservation": {"active": false, "path": "none",
                   "defenses_used_this_life": false,
                   "early_policy": "reserve_early"}
}
```

`mode_a` and `mode_b` are reversed-order `Choice`s over `turret`, the currently
exposed reachable path IDs, and `defend` only when the local critical state exposes
that candidate.  `risk` is an independent `Choice` over `clear/watch/caution/critical/
unknown`; `commit` is a `Noul` asking whether a validated caution reservation should
open now.  All arithmetic and comparison has already been done in code.  The two mode
answers must agree; a path/defense needs at least 0.36 probability in each answer,
ordinary turret advice needs 0.18.  Answers older than 44 observed frames, from a
previous life epoch, or referring to a packet no longer present are discarded.
`commit` only enables the watch-to-caution hint when its true probability is at least
0.72.  It cannot create a path, override a discontinuity, bypass cooldown, or lower
the overlap-only defense gate.

## Fallback and scheduling

The harness posts a code-owned action immediately and then waits on
`ctx.wait_for_tick()` for every native frame.  While Jev (2.3-second timeout) or Astra
runs, code continues with the current finite reservation or conservative turret
fallback, so model latency cannot hold thrust or hyperspace.  A missing player emits
NOOP.  A caution state can start a two/three-frame reservation; critical state can use
only a revalidated finite path or the final overlap gate.  A clear state uses target
rotation and one-frame FIRE presses with a local missile/cooldown check.

Jev requests are started about every 24 observed frames with at most one outstanding
request.  Astra gets one aggregate review after frame 360, when the controller has
seen the parameterized launch and some trajectories.  Its response is parsed as an
allow-listed JSON object; arbitrary stand-in text, timeout, cancellation, or malformed
output changes nothing.  The episode loop catches ordinary model failures and returns
cleanly on the contract's budget/episode closure conditions.  No background loop or
sleep remains after return.
