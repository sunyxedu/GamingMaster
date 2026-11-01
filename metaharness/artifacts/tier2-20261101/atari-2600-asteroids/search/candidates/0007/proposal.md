# Candidate 0007 — predictive lane relocation with bounded firing

## Parents and hypothesis

Parents: `candidates/0005/` and `candidates/0004/`.

`0005` established the strongest scoring mechanism so far: a frame-local controller
that fires in press/release pulses while making short aiming turns, with no stale
real-time ticks.  Its mean was 1320, but it still lost all four ships in every
training episode.  The failure is particularly concrete in the traces: in seed 8,
after the first respawn the player remains at approximately `(84,100)` with zero
velocity, and the final three life losses occur at ticks 1506, 1755, 2083, and
2094 while the controller mostly posts `NOOP`, `LEFT`, and `RIGHT`.  Seed 6 has the
same fixed respawn position and zero velocity; it scores 2250, then loses the last
three ships by ticks 2210, 3956, and 4196.  Thus the successful firing loop is
repeatedly defending the same horizontal lane rather than surviving the next
vertical crossing.

`0004` proved that deterministic per-frame ownership is essential, but its escape
mechanism is too late and too forceful.  In seed 8 it changes to long `UP`/`UPRIGHT`
stretches only after an emergency classification, loses ships at ticks 1435,
1729, 1988, and 2000, and scores only 380.  In seed 2 it reaches 1540, but its
emergency response still moves the player far from the center before the final
loss.  This is evidence against both open-loop model actions and panic thrust,
not against movement itself.

**Hypothesis:** the missing mechanism is an intermediate control scale.  A ship
that never changes position is eventually hit, while a thrust pulse selected only
at contact range is too late and can build uncontrolled drift.  A code-owned
predictive lane controller should improve survival while preserving `0005`'s
central, pulsed fire.  It forecasts swept-box crossings for the five central
horizontal lanes, changes lanes only when the current lane is threatened, uses at
most short thrust bursts with a braking pulse, and reserves hyperspace for a
computed imminent contact.  Jev recognizes the tactical regime and Astra selects
a durable risk/aim policy, but neither can hold a joystick level or bypass the
local safety forecast.

This is a structural crossover rather than a timeout or prompt tweak.  It does
not use seed labels, memorized schedules, or raw RAM rules specific to a launch.
All lane choices come from the current decoded objects and current player state.

## What changed

* `LanePilot` owns the real-time clock and posts one fresh legal action after every
  tick, including while both model calls are outstanding.  A malformed observation
  or model answer leaves the controller's release/firing fallback running.
* Each observed asteroid is represented with wrap-aware relative position, size,
  velocity-known status, and approximate swept geometry.  The controller samples
  contact over a short horizon for the current position and for five central lane
  centers (`45, 65, 85, 105, 125`).  It uses hysteresis so one flicker cannot
  cause a lane flip.
* Normal control retains controlled aiming and one-frame FIRE presses with release
  gaps.  When the current lane is predicted to be crossed, the controller turns
  toward the safest lane and uses no more than a two-frame thrust burst, followed
  by a cooldown; if measured drift would overshoot, it points back for one braking
  burst.  Urgent danger may combine turn and thrust for one frame, rather than
  starting an unbounded `UP` macro.
* A local collision gate can issue one `DOWN` hyperspace pulse per life only for a
  forecast contact within a few frames (or a fresh, high-confidence Jev emergency
  hint plus a near contact).  Defense is never used as braking and is reset after
  a life transition.  Missing velocity is treated as uncertainty, not as proof of
  zero motion.
* Jev receives a compact, derived JSON state and chooses only bounded semantic
  advice.  Two reversed-order danger and posture Choices reduce first-option bias;
  a lane Choice is only a suggestion and is rechecked against the numeric lane
  forecast.  Answers are discarded after 48 frames or a life epoch change.
* Astra is a delayed strategic reviewer, capped at two background calls.  It can
  select sweep versus target aiming, adaptive versus minimal patrol, cautious
  versus balanced risk, fire spacing, and last-resort versus never defense.  Its
  JSON is allow-listed; prose, stale output, and arbitrary preflight stand-ins are
  ignored.  Astra never posts an action.

## Jev state, questions, thresholds, and bridge

Every Jev request is strict JSON with these fields:

```json
{
  "rules": {
    "mode": 0,
    "fire": "level_press_then_release",
    "noop": "does_not_brake",
    "down": "mode_0_hyperspace",
    "heading": "left_increments_right_decrements",
    "enemy_objects": "unknown_not_absent"
  },
  "snapshot": {
    "frame": 1234,
    "lives": 3,
    "score": 530,
    "player_present": true,
    "position": "left|middle|right",
    "motion": "still|moving|unknown",
    "danger": "clear|distant|watch|immediate|unknown",
    "impact_window": "none|immediate|soon|later",
    "rocks": 7,
    "missiles": 1
  },
  "lanes": [{"id": "L45", "relation": "left_safe"}],
  "target": {"slot": 13, "size": "small", "bearing": "right_above",
             "motion": "known", "distance": "middle"},
  "controller": {"posture": "sweep_right", "lane_goal": "L105",
                  "defense_used": false, "frame_local": true},
  "plan": {"aim": "target", "patrol": "adaptive_lane",
           "risk": "cautious", "defense": "last_resort", "fire": "normal"}
}
```

Values are buckets or code-computed facts; Jev is not asked to count rocks,
compare distances, solve interception, or infer an action mapping. One batched
request contains:

1. `danger_a` and `danger_b`: reversed-order Choices over `clear`, `distant`,
   `watch`, `immediate`, and `unknown`. The controller records an emergency hint
   only when both agree and the selected probability is at least 0.45 for
   `immediate` (0.16 for lower-stakes labels). The local swept-box calculation
   remains the hard safety gate.
2. `posture_a` and `posture_b`: reversed-order Choices over `focus`,
   `sweep_left`, `sweep_right`, `hold`, and `avoid`. Agreement with selected
   probability at least 0.16 sets a semantic posture for 54 frames. It can alter
   aiming, never thrust duration or legality.
3. `lane`: a Choice among `stay` and the five precomputed lane ids. A selected
   lane is accepted only if it is one of the current candidates and the local
   controller rechecks it; it cannot create a lane outside the central set.
4. `defense`: a Noul. A true result is only a hint. It has no effect without a
   current local impact window and the once-per-life defense guard.

Requests are started about every 24 observed frames, with one outstanding request
and a 2.4-second timeout. A result is useful only if it is no more than 48 current
frames old and belongs to the same life epoch. Disagreement, low probability,
unknown answer shapes, timeout, and provider errors leave the local policy
unchanged. A new life clears tactical advice, lane goals, thrust state, and
hyperspace advice.

## Astra interface and timing

Astra is started in the background after roughly 300 observed frames and may be
started once more after a later review opportunity. It receives aggregate score,
lives, danger window, lane cost, rock count, current goal, and the current policy.
The requested response is a small JSON object with only:

```json
{"aim":"sweep|target", "patrol":"adaptive_lane|minimal",
 "risk":"cautious|balanced", "defense":"last_resort|never",
 "fire":"normal|spaced"}
```

The initial local policy is already valid; a 57–60 second Astra answer is never
on the native-frame path. Invalid JSON or values are ignored. The slow model is
therefore a strategic adaptation channel, not a delayed action channel.

## Fallback and confirmation criteria

With no usable model answer, the controller retains the current lane, follows a
code-computed target (or a bounded alternating sweep when the plan asks for one),
and emits a one-frame fire press only when the local release schedule allows. It does not thrust merely because a rock is far
away. A predicted watch/immediate crossing invokes lane planning; only an urgent
contact can permit a short combined turn/thrust or the once-per-life defense.
If the player is absent, it posts `NOOP` until a fresh player observation arrives.
All model futures are collected defensively, and arbitrary stand-in outputs in
pre-flight cannot escape the loop.

The hypothesis is supported if the candidate keeps the zero-stale property while
raising survival time and mean score above 1320, especially if losses no longer
cluster at the fixed `(84,100)` respawn lane. It is refuted if lane movement
causes early deaths or suppresses the demonstrated fire score without extending
life, or if the forecast flags many false contacts and spends hyperspace without
improving survival. The decisive diagnostics are life-loss frames, lane goals,
thrust/defense counts, score per life, and the fraction of time spent in the
pulsed aiming regime; official selection remains unclipped ALE score.
