# Candidate 0017 — risk-budget turret with an idle-derived safety firewall

## Parents and diagnosis

Parents: `candidates/0011/` and `candidates/0002-tier2_idle/`.

The score mechanism in 0011 is valuable and reproducible: the code, rather than a
slow model, owns every native frame; short FIRE press/release pulses and local
bearing turns produce thousands of points.  Its failure is also visible in its
traces.  It reports `danger=immediate` just before many losses but often posts no
movement that can alter the collision.  For example, in seed 8 the observation at
frame 3737 contains a small rock at `(89,98)` beside the player at `(84,100)`;
the controller reports `immediate` at frames 3728 and 3733, then the rock disappears
and the life is lost at frame 3748.  In seed 10 the player loses a life at frame
304 after the frame-279 defense, while the immediate window contains mostly FIRE.
Seed 4 has the same pattern around frames 5969–5986: immediate forecasts, a lane
change only at the end of the window, and a loss.  The five fixed horizontal lanes
are not enough when an asteroid is already crossing the ship's current two-dimensional
position.  Conversely, candidate 0002's robust mechanism is not making a clever
move: it never turns a stale or uncertain estimate into a long thrust commitment.

I therefore hypothesize that the missing mechanism is **risk budgeting**, not more
model calls or more movement: retain 0011's turret when the field is clear, but rank
short one-pulse escape trajectories in the full `(x,y)` plane, and make uncertainty
fall back to idle-like coasting.  The controller should move only when a fresh local
forecast says a bounded pulse improves clearance, and should invalidate that pulse
on a teleport, respawn, discontinuity, or stale answer.

## What changed

This is a new action architecture rather than a copy of the lane controller:

* `Radar` forecasts coast, current-heading thrust, and one-code-left/right
  turn-plus-thrust with wrap-aware swept boxes.  It compares the predicted clearance
  of those four candidates every frame.  It does not hold any thrust timer: an escape
  is one fresh pulse, followed by a 4–9 frame expiry and a new observation.
* A strict uncertainty firewall treats null velocity on the player or a *near*
  unknown object, large jumps, player absence, hyperspace, and life changes as
  recovery states.  Distant flicker does not paralyse a clearly improving escape;
  close uncertainty still releases rotation and thrust and allows only short FIRE
  pulses or NOOP, preserving the useful safety property of the idle baseline.
  Unknown enemy coverage is never converted to zero objects.
* A coast/escape/defense arbiter gives collision response priority over aiming.
  Defense is local and bounded: a second hyperspace is possible only after a
  120-frame (two-second) reserve interval, rather than the adjacent retries
  rejected in 0013.
  A defense also starts a fresh coast quarantine so old position and lane assumptions
  cannot steer the post-teleport ship.
* Outside a forecasted hazard, the controller keeps 0011's high-yield shape:
  code-owned target bearing, short alternating sweep options, and a level FIRE
  release timer.  Jev labels danger and posture but cannot request raw thrust or
  override the local gate.  Astra can select only an allow-listed durable preference
  (`strict/balanced` risk, firing cadence, target/sweep aim, and reactive/rare
  escape); malformed output is ignored.

## Evidence that would confirm or refute it

Confirmation would be a higher mean ALE reward with zero harness errors and longer
life-loss frames, especially in the immediate windows above, while escape-pulse
counts remain much smaller than the thrust-heavy 0008 controller.  Useful trace
signals are `escape_pulse` followed by a later or absent `life_epoch`, fewer losses
while `uncertain` is true, and a clear separation between `control` FIRE/turn pulses
and emergency thrust.  A surviving run should approach the 18,000-frame budget
rather than ending around 4,900–16,100 as 0011 did.

The hypothesis is refuted if one-pulse forecasts cause early drift deaths, if the
coast firewall removes firing opportunities without improving survival, if the
second spaced defense is harmful, or if `escape_pulse` events are repeatedly issued
without a changing forecast.  Official comparison remains unclipped ALE score;
survival and lives are diagnostics, not bonuses.

## Jev contract and bridge

Each batched request is strict JSON derived from the latest snapshot:

```json
{
  "rules": {"mode": 0, "difficulty": 0,
            "fire": "press_then_release", "noop": "coasts_not_brakes",
            "down": "mode_0_hyperspace", "enemy_coverage": "unknown"},
  "snapshot": {"frame": 1234, "lives": 3, "score": 530,
                "player": "present", "danger": "watch",
                "impact_window": "soon", "uncertain": false,
                "rocks": 7, "missiles": 1},
  "forecast": {"best_escape": "left_thrust",
                "candidates": {"coast": {"contact": 8,
                "clearance": "overlap"}}},
  "target": {"size": "small", "bearing": "right_above",
             "motion": "known"},
  "controller": {"coast_guard": 0, "defenses_this_life": 0,
                 "frame_local": true},
  "plan": {"risk": "strict", "fire": "normal",
           "aim": "target", "escape": "reactive"}
}
```

The one request contains four independent judgments:

1. `danger_one` and `danger_two` are reversed-order `Choice`s over
   `clear|distant|watch|immediate|unknown`.  They are used only when they agree;
   `immediate` requires at least 0.45 probability and other labels 0.18.
2. `posture` is a `Choice` over `focus|sweep_left|sweep_right|hold|coast`.
   A valid answer lasts 54 frames and changes only aiming posture.  `coast` can
   reinforce the fallback but never creates thrust.
3. `defense` is a `Noul`; it becomes a hint at probability 0.82, but code still
   requires an observed local contact and a defense budget/gap.  At actual overlap
   the local gate prefers defense over trusting a one-frame thrust forecast; before
   overlap it still requires no materially safer pulse.  A late or disagreeing
   answer is discarded.

Requests are launched about every 30 observed frames with a 2.2-second timeout,
while `wait_for_tick()` continues the game clock and the deterministic radar acts.
An answer is accepted only within 42 frames and the same life epoch.  Timeout,
provider garbage, malformed values, and stand-in arbitrary text are treated as a
missed judgment; the safe local policy continues.  No Jev answer can bypass the
uncertainty firewall or issue a sequence.

## Astra scheduling

One background Astra review starts after the launch/recovery evidence (frame 360).
Its prompt includes score, lives, danger, uncertainty, escape-pulse count and
spaced-defense count.  Only four allow-listed strings can change the durable plan;
Astra never posts an action and never changes the local collision thresholds.  Its
slow or cancelled result is harmless.  This tests whether System-2 can choose when
the strict coast policy should be relaxed without stopping the real-time loop.
