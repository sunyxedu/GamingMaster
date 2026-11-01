# Candidate 0022 — acquire only on an empty laser

**Parent:** `candidates/0018/` (its working real-time plumbing and continuous
FIRE actuator). **Comparison:** `candidates/0016/`, the current best. I read the
parent and the best candidate's worst traces; this candidate does not modify
those directories.

## Hypothesis

The reliable mechanism is still a fresh lateral action with FIRE on essentially
every tick. The unresolved failure is the *launch schedule*. A sweep launches a
beam at whichever x happens to be under the cannon. That is good enough for
many aliens, but it can leave the last low aliens alive while the formation
gets faster and the run ends with all lives lost. The traces show a late
throughput problem rather than a demonstrated model-planning problem:

* In 0016 seed 1, the action runs are `right_fire` for ticks 1–72, then
  alternating 40/41-tick legs, and finally `right_fire` for ticks 291–422.
  It scored only 145 and ended after the two recorded life signals at frames
  1096 and 1380. Its final controller counts were
  `down_in_shot_field=0`, `strong_dodges=0`, and `jev_changed_actions=0`.
  Thus the best controller was not actually using its temporal reviewer; it
  was a sweep, including a very long final right run.
* In the stronger 0016 seed 3, score observations stop at 625 with 28 kills
  and one UFO, followed by game end at frame 2730. The controller again
  recorded no downward track or strong dodge. The high score confirms that
  continuous FIRE is productive, but not that random launch x finishes the
  wave.
* The 0018 seed-0 trace is a useful negative control for movement-only
  changes. It starts with a route marked `stalled` at frame 36 and then
  starts/finishes many four-frame routes (116 starts and 113 finishes in the
  episode), with no visible dodge. It loses lives at frames 600 and 856 with
  an empty bomb list and finishes at 75. Changing route endpoints and life
  phase did not repair the control problem.

**Test:** keep the proven sweep while a player laser is in flight, but, only
when the laser is empty and the formation is late or score has gone quiet,
briefly align to the current lowest surviving invader and fire. After that
launch, immediately return to the broad moving sweep. This is deliberately a
shot-window experiment, not a permanent target lock and not a model-selected
column intercept. The target is recomputed from the live object list in code;
Jev is not asked to count, calculate, or steer coordinates.

The key distinction from 0017 is timing and ownership. 0017 allowed Jev to
lease short target intercepts and its accepted intercept outcomes had zero
score delta. 0022 does not let a model pick a target or interrupt a shot. The
code opens an acquisition window only on `player_shot.active == false`, uses a
bounded 14-tick approach, and falls back to full-rate combined FIRE if the
window does not align. The experiment therefore tests whether synchronizing
launches to the one-shot lock helps, rather than whether a language model can
choose a route.

## Structural change

`ShotWindowController` owns:

* the six-action string mapping and a fresh `ctx.act`/`ctx.wait_for_tick`
  loop;
* a broad interior sweep, reversing only at the observed working lane
  limits (approximately x=38 and x=116), with combined FIRE on every sweep
  tick;
* the shot lifecycle from the symbolic `player_shot.active` field;
* live formation bookkeeping and a target chosen from the lowest-y surviving
  row, with an outer-side tie break in the current sweep direction;
* a short alignment attempt when the laser is READY. It moves without FIRE
  toward the target for at most 14 ticks, fires once when within a small
  alignment band, and then abandons aiming until that shot is no longer
  active. If it times out, it posts the ordinary `left_fire`/`right_fire`
  sweep action rather than suppressing throughput;
* exact legality/bounds checks and a short code-side escape for a currently
  decoded bomb close to the cannon. An empty `bombs` list remains
  `UNSEEN_NOT_SAFE`; it never creates a guessed hazard;
* life, wave, and teleport resets that invalidate a stale aim attempt and
  rephase only the direction, never memorizing a seed-specific death lane.

The acquisition mode is enabled by default only for `LATEWAVE` (fewer than
15 living invaders) or after a long score stall on a recovery/last life. Thus
the opening and middle of the episode retain the high-scoring 0016 actuator.
A 14-tick cap, a current-state recheck, and the visible-bomb override prevent
an ambiguous model result from parking the cannon or firing continuously at a
bad x.

## System-2 role

One asynchronous Astra request starts after the first observation. It must
return exactly `LATE_AIM` or `SWEEP_ONLY`. `LATE_AIM` is the safe default and
allows the late/score-stall shot windows; `SWEEP_ONLY` disables them and
reproduces the established baseline. The prompt contains only current
categorical evidence (wave, life stage, formation bucket, shot state, target
availability, and visible-bomb count) and explicitly says that code owns
geometry and the one-laser rule. The response is applied only if it contains
one of the two allowed tokens; malformed prose is ignored. The call is
asynchronous with a 55-second timeout and is never on the tick's critical
path, so a slow or cancelled Astra cannot create stale actions.

## Jev state and questions

The state sent on an asynchronous request has schema
`space_invaders_shot_window_v1` and is strict JSON. It contains:

* `episode`: wave, life stage, `OPENING`/`MIDWAVE`/`LATEWAVE`, and
  `RECENT`/`QUIET` score progress;
* `shot`: `READY` or `IN_FLIGHT`, plus the rule that only an empty laser may
  launch;
* `formation`: `MANY`/`SOME`/`FEW`, qualitative vertical pressure, target
  availability, target height, target side, and target relation
  (`ALIGNED`, near, or far); no raw grid is sent;
* `movement`: current sweep side and whether a bounded aim attempt is active;
* `threat`: `VISIBLE_NEAR_LEFT`, `VISIBLE_NEAR_RIGHT`,
  `VISIBLE_NOT_IMMEDIATE`, or `UNSEEN_NOT_SAFE`;
* `policy` and rules defining each program.

The request batches two independent reordered `Choice` questions, `mode_a`
and `mode_b`, over these programs:

* `AIM_TARGET`: use one short acquisition window for the supplied target,
  only if the laser is ready;
* `SWEEP_FIRE`: preserve the proven lateral combined-FIRE fallback;
* `KEEP`: do not change the current advisory mode;
* `EMERGENCY_LEFT`/`EMERGENCY_RIGHT`: a request for a visible-bomb escape,
  still subject to code verification;
* `UNKNOWN`: insufficient evidence.

Jev is explicitly forbidden to choose coordinates, count aliens, calculate
comparisons, invent bombs, or override a visible collision. A request is made
at most once per 96 emulator frames and only when a target exists. The game
loop launches it in the background after posting the current action.

An answer is accepted only when both reordered Choices agree, the selected
probability is at least 0.64 in both, and any supplied confidence is at least
0.25. It must have the same life/wave/epoch and be no more than 48 emulator
frames old. `AIM_TARGET` is still code-gated to a late/quiet shot window;
`EMERGENCY_*` is accepted only when the current bomb geometry independently
selects the same side. An accepted `SWEEP_FIRE` suppresses aiming for a short
mode lease, not forever. A disagreement, timeout, garbage answer, or stale
answer leaves the deterministic controller unchanged.

## Fallback and timing

The per-tick order is: observe; update formation, shot, and short bomb tracks;
poll completed model calls; choose the current action; post it; optionally
launch a fresh background request; wait for the next tick. There is never a
blocking Jev or Astra call. All spawned functions contain one model call and
stop when the context cancels them; `finally` cancels unfinished futures.

The fallback is always one of `left_fire`, `right_fire`, or `fire`, except for
a bounded target approach (`left`/`right`) or a code-confirmed visible-bomb
escape. Illegal or malformed actions fall back to `fire` or the advertised
noop. Observation/model parsing is defensive so arbitrary pre-flight model
text cannot escape `run_episode`.

## Confirmation and refutation

Confirmation requires more than a score bump: late traces should show
`shot_window_open` followed by `shot_window_fire`, fewer long scoreless late
runs, more low-row kills, and retained or improved survival. The strongest
support is a first-wave score near or above the 630-point wave value without a
rise in life losses or stale ticks. Jev should affect only shot-mode leases;
Astra should affect the durable policy, never individual actions.

Refutation is a reduction to the 0016 kill/score regime, repeated alignment
aborts, or late aiming that increases life losses without improving low-row
kills. If all score still comes from the sweep and the shot-window counters
remain zero, the hypothesis was not exercised; if counters are active but the
score falls, the next iteration should retain the sweep and reject target
acquisition rather than adding another route or shield heuristic.
