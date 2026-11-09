# Candidate 0015 — watch-state lane precommit

## Parent and targeted hypothesis

Parent: `candidates/0011/` (the urgent-relocation refinement of 0007).

I keep 0011's architecture: code owns every native frame, it predicts five bounded
central lanes, FIRE is pulsed, Jev supplies only short semantic judgments, and Astra is
an asynchronous policy reviewer.  The one change is when the existing lane planner
starts a relocation: a `watch` crossing can now reserve the currently safest precomputed
lane before the forecast becomes `immediate`.  The thrust action, one-frame urgent
bypass, ordinary cooldown, defense budget, model schedule, and fallback are otherwise
unchanged.

The hypothesis is that 0011 is often recognizing a collision correctly but too late to
move a drifting ship.  In the seed-8 trace, the controller reports `danger="watch"` at
frame 368, `immediate` at frame 383, uses hyperspace at frame 387, plans lane 105 at
frame 395, and loses a ship at frame 410.  The intervening controls include
`LEFTFIRE`, `UPLEFTFIRE`, and `LEFTFIRE`; the safety response begins only after the
contact window is already immediate.  The seed-10 trace shows the same ordering: a
`DOWN` at frame 279 with `threat_time=2`, then `FIRE` at frames 288 and 293 and
`UPFIRE` at 298, followed by the life change at frame 304.  These are fresh local
observations, not stale Jev actions.  They suggest that an earlier bounded relocation
could give inertia time to change the crossing, whereas waiting for `immediate` leaves
only defense or an ineffective turn.

This is deliberately not a general “move more” policy.  During `watch`, a goal is
created only when code's existing lane costs show a safer alternative (or the best
lane has no crossing while the current lane does).  The existing short action pulses
and cooldown still limit displacement.  A clear field retains the parent behavior,
and Jev cannot create a lane or authorize thrust.

## What changed

`LanePilot._refresh_lane_goal` now treats `watch` as a precommit state.  It compares
the five code-generated lane costs and first-crossing times, selects the safest lane
only when it has a measurable clearance advantage, and gives the goal a bounded
84-frame lifetime.  `immediate` keeps the parent's stronger goal and 100-frame
lifetime.  The action bridge is unchanged: `_lane_motion` still turns toward a goal,
allows only one-frame urgent thrust pulses, and applies the normal two-frame pulse and
22/28-frame cooldown outside urgency.  Hyperspace remains a one-shot per life and is
not used as braking.

The default plan and all arithmetic/bookkeeping remain deterministic and seed-agnostic.
Unknown enemy classes remain unknown; malformed model answers are ignored.  The
candidate therefore isolates an execution-timing hypothesis rather than changing the
Jev prompt or spending more model calls.

## Evidence that would confirm or refute it

Confirmation would be later life-loss frames and longer episodes, especially if the
seed-8 368→383→410 and seed-10 279→304 patterns move later, while thrust counts stay
well below the thrust-heavy 0008 controller.  It would also be useful if watch-time
lane goals are followed by a lower current lane cost without repeated oscillation.
Refutation would be earlier deaths from unnecessary lane changes, a substantial score
loss while survival is unchanged, or watch goals that repeatedly select a lane with no
real clearance.  The relevant diagnostics are `lane_plan` danger, current/best cost,
first crossing, goal transitions, urgent thrust count, defenses, life-loss frames,
and score per life; the official metric remains unclipped ALE reward.

## Jev state, questions, thresholds, and bridge

The state schema is retained from 0011 and contains only strict JSON derived facts:

```json
{
  "rules": {"mode": 0, "fire": "level_press_then_release",
             "noop": "does_not_brake", "down": "mode_0_hyperspace",
             "heading": "left_increments_right_decrements",
             "enemy_objects": "unknown_not_absent"},
  "snapshot": {"frame": 1234, "lives": 3, "score": 530,
                "player_present": true, "position": "middle",
                "motion": "moving", "danger": "watch",
                "impact_window": "soon", "rocks": 7, "missiles": 1},
  "lanes": [{"id": "L105", "relation": "right_safe"}],
  "target": {"slot": 13, "size": "small", "bearing": "right_above",
             "motion": "known", "distance": "middle"},
  "controller": {"posture": "avoid", "lane_goal": "L105",
                  "defense_used": false, "frame_local": true},
  "plan": {"aim": "target", "patrol": "adaptive_lane",
           "risk": "cautious", "defense": "last_resort", "fire": "normal"}
}
```

One request batches two reversed-order `Choice`s for
`clear|distant|watch|immediate|unknown`, two reversed-order posture Choices for
`focus|sweep_left|sweep_right|hold|avoid`, a Choice over `stay` and the five current
lane IDs, and a Noul asking whether last-resort hyperspace should be considered.
Danger agreement requires the selected probability to be at least 0.45 for
`immediate` and 0.16 for lower-stakes labels; posture/lane advice uses 0.16.  The
watch precommit does **not** depend on Jev: it is a local lane-cost rule.  Jev advice
is discarded after 48 frames or a life-epoch change.  One request is started about
every 24 frames with a 2.4-second timeout, and there is at most one outstanding
request.

Astra remains capped at two background strategic reviews and can only update the
allow-listed aim, patrol, risk, defense, and fire preferences.  It never posts an
action or overrides the watch/immediate safety gate.  If models time out or return
arbitrary preflight text, the code-owned controller continues.  If the player is
absent or geometry is malformed, the fallback posts `NOOP` until a fresh present
observation arrives; no model result can make that fallback thrust or hyperspace.
