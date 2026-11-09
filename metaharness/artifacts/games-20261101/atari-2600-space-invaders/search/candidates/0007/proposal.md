# Candidate 0007 — empirical full-lane sweep after hidden deaths

## Parents and hypothesis

This is the requested crossover of `candidates/0004/` and `candidates/0005/`.
The score mechanism comes from 0005: code-owned aiming and fire locking,
shoot-and-relocate, a latest-state Jev worker, and an executable action every
tick. The robustness/interface mechanism comes from 0004: compact categorical
Jev judgments with reordered cross-checks, strict freshness gates, and Astra
writing a bounded policy rather than an action sequence.

**Hypothesis:** 0005 solved model-latency exposure but its relocation is still
locally target-shaped. It repeatedly returns to the same extreme lane, and its
bomb predictor cannot help when the decoder's sampled list is empty. The next
structural fix is to separate *firing alignment* from *survival movement*: aim
only for the launch tick, then sweep across the empirically observed cannon
lane for the whole beam flight and after an unobserved death. Remember the lane
of a death and do not deliberately dwell there on the next life.

The evidence is concrete. In `candidates/0005/trace_seed3.jsonl`, the score was
130 and all three lives were lost at frames 676, 956, and 1521. The sampled
observations at those life-loss transitions had `bombs: []`; the first two
showed the player at x=37/38 with an active shot, and the final loss returned
from x=74 to the reset x=90. The trace's controller summary records 93 accepted
Jev intents, 14 changed actions, but `safety_overrides: 0`; the accepted intent
was repeatedly `relocate`, while the action counts were still 165 left versus
96 right. Thus a no-stale local relocation policy did not provide a broad escape
when the bomb was absent from the sampled state. The same left-lane pattern is
visible in 0005 seeds 0, 2, 5, and 6: early loss observations cluster at x=37
or x=38 while no bomb is decoded.

0004 demonstrates the complementary timing/interface lesson: its seed-3
failure had score 0 after 355 ticks despite nearly zero stale ticks, while its
compact typed policy often selected a repeated left move and low fire support.
Simply asking Jev to choose a direction more often is not enough. Conversely,
0005's mean 198.125 and 16.62 kills show that its asynchronous code loop and
shooting geometry should be retained rather than replaced by a slow Astra
policy. `lessons.md` also explicitly reports that empty bomb observations are
not safety evidence and that Astra informed zero actions in 0005. This
candidate addresses both facts without putting either model on the actuator
path.

## What changed

* **Empirical legal lane.** The controller starts with the canonical observed
  player range x≈37..119 and widens it only if a later public observation proves
  otherwise. It never uses the full 0..159 screen as a cannon bound. Movement
  reverses with a margin at both learned limits.
* **Separate aim and survival.** The lowest living target in a selected column
  is aligned in code and fired at only when the one-shot lock is clear. As soon
  as a shot is launched, the controller moves laterally even while it is in
  flight, reversing at the lane limits or after a bounded segment. This is a
  full-lane triangle sweep, not 0005's nearby +/-18-pixel relocation. The
  target column can be reacquired after the beam exits; firing never retargets
  an active beam.
* **Death-conditioned blind mode.** A life loss with no recent decoded bomb
  marks the episode as `BLIND_AFTER_DEATH`, records the death x bucket, shortens
  the re-centering period, and disables deliberate dwelling in that lane. The
  bomb tracker extrapolates only a short 12-frame observation gap. An empty
  current bomb list never clears blind mode or cancels the sweep.
* **Exact safety before model advice.** Visible bomb tracks and formation depth
  can force a left/right dodge. Recovery, boundary clamping, fire cooldown,
  target validity, and all action-name checks are code-owned. A typed Jev
  direction can only select a current broad sweep or agree with a current exact
  dodge; it cannot steer into a newly visible danger.
* **Astra is now an applied policy author.** One initial asynchronous Astra
  call and at most one later revision after a life/wave event return a tiny
  validated policy: `target_bias` (`NEAREST`/`OUTER`), `posture`
  (`WIDE`/`AIM`), and `dodge` (`EARLY`/`NORMAL`). The policy is copied into
  the compact Jev state and changes target ranking/patrol posture. A malformed,
  late, or cancelled answer leaves the code defaults. It cannot emit actions.

## System-1 state schema and questions

Every Jev request is a strict JSON object with schema
`space_invaders_lane_sweep_v1`. It contains:

* `frame`, `wave`, `lives`, and a categorical `phase`;
* `player.lane` (`LEFT`/`CENTER`/`RIGHT`), `player.boundary`, current patrol
  direction, and `motion_policy` (`WIDE_SWEEP`/`TARGET_AIM`);
* `shot` (`ACTIVE`/`LOCKED`/`READY`);
* `target.side`, `target.row_band`, and `target.alignment`;
* `threat.bomb`, `threat.urgency` (`NONE`/`WATCH`/`URGENT`), exact-code
  `threat.escape`, and formation urgency;
* `exposure` (`BOMB_SAMPLES_SPARSE` or `BLIND_AFTER_DEATH`), formation bands,
  outer-column availability, the current Astra `plan`, and the last action.

The state deliberately omits the raw 36-object list and numeric comparisons.
Code computes geometry, nearest tracks, and legality; Jev recognizes a named
near-term situation.

One batched request contains two independent `Choice` questions over the same
seven intents: `AIM`, `SWEEP_LEFT`, `SWEEP_RIGHT`, `DODGE_LEFT`,
`DODGE_RIGHT`, `HOLD`, and `UNKNOWN`. The second question reverses option
insertion order to reduce first-option bias. A result is accepted only when:

1. both choices agree;
2. each selected option has probability at least **0.58**;
3. the tagged frame is at most **28 emulator frames** old and its lives, wave,
   and plan version still match.

`DODGE_LEFT/RIGHT` is translated to a game action only if the current exact
bomb calculation has the same escape direction. `SWEEP_LEFT/RIGHT` changes the
short-lived patrol direction (with boundary and threat checks); `AIM` leaves
code's target navigation in control; `HOLD` is not allowed to suppress the
post-death sweep. Unknown, disagreeing, low-probability, malformed, or stale
answers are discarded. Requests are pipelined with one future at a time and a
0.50-second Jev timeout, above the measured 0.136-second median and
0.182-second maximum. The main loop posts an action and waits for the next tick
while the request is in flight.

## Astra interface and fallback

Astra is prompted with the game description, current score/lives/wave, alive
count, recent life-loss evidence, compact Jev state, and the current policy. It
must return only the small JSON policy described above. The first valid policy
increments a plan version and invalidates old Jev advice; a revision after an
unseen death is forced to retain `posture=WIDE`. No Astra output is used as a
coordinate or a timed command, and no call blocks the game loop.

The no-model fallback is deterministic:

1. urgent visible bomb or urgent formation -> move to the code-computed safe
   side, with no fire;
2. after a life loss -> move toward the lane center and then start a wide
   triangle sweep;
3. with no active shot -> move toward the lowest living target, fire only when
   aligned and the decoder says the laser is free;
4. with a shot active or a blind death history -> keep sweeping, never press
   FIRE again until the lock is released;
5. reverse at empirical lane bounds and reject all illegal action names.

This fallback runs on every tick, including while Astra or Jev is late. A
future's exception is contained as an unavailable judgment; normal deadline,
budget, game-over, and backend termination still stop the episode through the
contract.

## Confirmation or refutation

The hypothesis is supported if the candidate keeps the near-zero stale rate of
0005 while reducing deaths at x≈37/38, increasing survival ticks and kills per
life, especially on the seed-3-like empty-bomb episodes. Useful trace evidence
would be `life_loss_evidence` followed by movement across both lane sides,
more right/left balance after the first blind loss, and shots launched from
multiple columns without fire-lock violations. Astra support is confirmed only
if a returned policy changes a later target/patrol decision before game end.

It is refuted if the full sweep reduces shot alignment enough to lower score,
if boundary estimates cause repeated edge stalls, if death-lane avoidance
merely trades one fixed lane for another, or if the extra Astra revision never
returns in time and adds no measurable policy change. The principal comparison
is mean ALE score, then stale ticks, then System-2 calls, with kills, survival,
life-loss lane, and effective Jev/Astra events used to diagnose the result.
