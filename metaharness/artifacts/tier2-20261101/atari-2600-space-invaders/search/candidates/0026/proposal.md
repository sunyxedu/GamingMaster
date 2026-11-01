# Candidate 0026 — recover in a different corridor after a life loss

## Parents and hypothesis

Parent: `candidates/0009/` (structural rewrite). The score-producing control
idea is retained from that lineage and from the current best `0016/0023`:
post a fresh combined `FIRE` action on every tick and traverse a broad lane.
This candidate changes the division of labour around the most reliable event in
the traces: a life decrement.

**Hypothesis:** the remaining score loss is not a missing Jev target label or
another projectile classifier. The controller repeatedly makes good early
progress, then loses lives during a long no-score interval and restarts the
same broad patrol from the same central geometry. A life loss is delayed in the
observation stream, but once the `lives` field changes it is reliable. A short
post-life corridor on the side away from the last moving x position should
avoid immediately revisiting the impact lane while preserving FIRE and then
return to the productive wide sweep. This is deliberately temporary, not a
permanent death wall or a selective firing policy.

## Concrete evidence from the archive

The current best `0016` and tied `0023` average 405.625, but all eight
episodes end with zero lives and zero waves cleared. In `0016/trace_seed1`:

* score reaches 145 by frame 1000, then remains 145 through the end;
* the first `life_evidence` is frame 1096 with `previous_x=100`,
  `last_mobile_frame=964`, and the second is frame 1380 with `previous_x=80`
  and `last_mobile_frame=1248`;
* the final controller has `strong_dodges=0`, `chip_escapes=0`, and
  `jev_changed_actions=0`, while the bomb list in the sampled observations is
  empty.

The same pattern is not just one seed. The best candidate records delayed
life signals at x=48/50, 86/94, 68/94, 98/90, 68/56, and 86/78 across other
seeds. Its own warning says the life flag may lag impact and intentionally does
not learn a death lane. The trace also shows that the temporal Jev reviewer
never produced a movement change in the worst run: it classified shield chips
as `UNRESOLVED`, correctly refusing to invent a bomb, but therefore could not
help the controller survive.

This evidence does **not** prove that the last x is the exact bomb coordinate.
It does establish a useful, testable event boundary: after a life decrement,
the next 96–180 frames are the only interval in which a recovery posture can
be applied without changing the high-throughput opening. The fallback uses the
opposite side of the last observed moving x; if that x is central, it uses the
healthier outer shield as a tie-break. The code never treats an empty bomb list
or an unexplained shield chip as a collision.

## What changed

### Code owns execution and exact facts

* A fresh combined-FIRE action is posted after every new observation. The
  single-laser lock remains ROM-owned; FIRE is never suppressed just because a
  shot is active.
* Code decodes only supplied symbolic fields and computes player motion,
  command polarity, life/wave epochs, shield cover buckets, living-count
  phase, and visible-bomb collision checks. It calibrates whether the command
  named RIGHT increases or decreases screen x, never learning from a respawn
  jump.
* Normal operation is a wide 40–116 screen-x sweep. A visible bomb near the
  cannon can temporarily choose the safer screen direction, but all such
  decisions are code-owned and immediately recomputed.
* On a life decrement, the previous mobile x is remembered, the current
  projectile/dodge state is discarded, and a short outer-half corridor is
  entered. The corridor is left or right, away from the last impact-side
  category, and lasts at most 180 emulator frames. It oscillates within that
  half rather than parking at a movement limit. Afterwards the normal wide
  sweep resumes. A wave reset cancels recovery.
* Missing, malformed, or contradictory observations leave the deterministic
  FIRE sweep active. Garbage model output cannot select an illegal action.

### Jev changes posture, not action geometry

Jev is no longer asked to classify ambiguous projectile identities or choose a
per-tick joystick action. Only a life restart triggers one asynchronous request.
The code sends three already-defined alternatives: `LEFT`, `RIGHT`, or
`FULL_SWEEP`. Jev recognizes the qualitative `death_zone`, shield cover
(`GOOD`/`DAMAGED`/`OPEN`), formation phase, and code fallback. It does not
calculate distances, count invaders, or invent a threat. A valid answer can
extend/change the temporary recovery corridor; it cannot change FIRE, the
screen bounds, or the collision projection.

A paired, reversed-order `Choice` is accepted only when both answers agree,
the selected probability is at least 0.62, and any supplied confidence is at
least 0.20. The answer must match the current life and wave epoch and be no
older than 48 emulator frames. A timeout, disagreement, low probability,
missing fields, or stale answer leaves the code fallback in force. Jev runs in a
single background mailbox with a 0.9-second timeout, so no tick waits for it.
There is no backlog: at most one life review is outstanding.

### Astra supplies a durable tie-break rule

One short asynchronous Astra prompt asks for exactly `AWAY` or `COVER`.
`AWAY` is the default and uses the side opposite the last impact lane;
`COVER` only changes the central-death tie-break to the healthier outer
shield. The answer is stored as a rule for a future life event, never applied
retroactively to an action. It is started after the first action with a
52-second timeout and has no effect on the real-time loop if it is late,
malformed, cancelled, or the episode ends. Unlike a late route plan, this is
small enough to be consumed by code without parsing prose.

## Jev state schema and bridge

Schema: `space_invaders_life_recovery_v1`.

* `event`, `life_epoch`, `lives`, and `death_zone` identify the fresh event;
* `fallback` is the code-selected side, while `cover.left/middle/right` are
  named cover buckets and `healthier_outer` is a code-derived tie-break;
* `formation.phase`, `lowest_band`, and `lowest_side` describe urgency without
  asking Jev to compare raw y values;
* `recent_control` distinguishes responsive movement from a post-impact gap;
* `rules` explicitly say that code owns coordinates and that the alternatives
  are temporary corridors, not target or bomb predictions.

The answer bridge maps `LEFT`/`RIGHT` to a temporary recovery side and
`FULL_SWEEP` to immediate normal patrol. It rechecks epoch, lives, wave, age,
and current state before applying the answer. The action bridge maps the
chosen *screen* direction through the measured command polarity and appends
`_fire`; it never trusts a model-provided action string.

## Falsification and diagnostics

The hypothesis is supported if traces show recovery corridors beginning after
life decrements, fewer repeated losses in the same broad phase, longer
survival, and higher score without reducing the opening kill rate. Strong
evidence is a high-score episode where the first recovery crosses the healthy
outer half and the next life is preserved, or a model-selected corridor that
beats the deterministic fallback on identical qualitative conditions.

It is refuted if recovery starts only after game over, if lives are lost at the
same rate while the corridor reduces kills, if the last mobile x is unrelated
to the impact lane, or if the best scores fall because half-lane coverage is
too restrictive. A high score with `recoveries=0` would only confirm the
retained sweep, not this intervention. `controller_summary`, `life_restart`,
`recovery_started`, `recovery_finished`, `life_review_applied`, visible-bomb
counts, calibrated polarity, and final life-loss/recovery counts make that
causal distinction explicit.

The module uses only standard-library and documented `metaharness.*` imports,
contains no polling loops in worker threads, catches ordinary model failures,
and returns cleanly on budget/episode closure. It was written before any new
evaluation; no training-seed rule is hard-coded.
