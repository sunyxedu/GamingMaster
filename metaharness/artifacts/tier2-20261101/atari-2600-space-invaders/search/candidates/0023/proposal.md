# Candidate 0023 — calibrate joystick polarity before using temporal safety

## Parent and one targeted hypothesis

Parent: `candidates/0016/`. I retain its useful architecture: a code-owned,
fresh-tick combined-FIRE sweep; a short-lived projectile ledger; conservative
shield-chip handling; paired asynchronous Jev evidence reviews; and one
bounded Astra rubric call.

The single change is an online **joystick-polarity calibration**. The sweep
still treats `+1` as the command named `right` and uses the same action choices
and bounds. Before code uses movement history for collision prediction or
motion-gap diagnostics, it learns whether that command makes decoded screen x
increase or decrease. The learned sign is then used in the projectile path
projection, speed estimate, and responsive/stalled classification. No target
order, firing cadence, shield threshold, or Astra policy is changed.

**Hypothesis:** 0016's temporal reviewer was unable to protect its low-score
runs partly because its geometric safety code assumes command sign equals
screen-x sign. In this ROM/adaptor trace, that assumption is unsafe: the early
trace has `right_fire` actions while the observation checkpoint moves from
player x=90 to x=51. The code in 0016 computes movement sign as
`-1 if "left" ... else 1 if "right"` and projects candidate paths by adding
that sign to x. Thus it can call a moving cannon a motion gap and, if a real
incoming track is finally visible, choose the wrong side of it. Calibration
repairs that interface mismatch without disturbing the score-producing sweep.

## Trace evidence motivating the change

The current best parent scored 405.625 overall, but its worst episodes ended
before a wave clear:

* Seed 1 scored **145** and lost its first two lives at trace events
  `life_evidence` frames **1096** and **1380**. The events report
  `last_mobile_frame=964`, `gap_start=964`, then `last_mobile_frame=1248`,
  `gap_start=1248`; the warning says the life flag may lag impact. Its final
  controller summary has `down_in_shot_field=0`, `strong_dodges=0`,
  `chip_escapes=0`, and `jev_changed_actions=0`.
* Seed 5 scored **210**. Its life evidence is at frames **1156** and
  **1560**, with motion last recorded at **1024** and **1432** respectively.
  Its summary likewise has no strong dodge, no chip escape, and no Jev action
  change. The trace's explicit bomb list is empty at its sampled score
  observations, so the parent never gets a verified threat to route around.
* The parent repeatedly logs `unattributed_chip` (for example C4 at frame
  220 and C6 at frame 252 in seed 1), but all of those cues are `thin=false`;
  its deliberately conservative chip policy therefore does not alter the
  productive sweep. This candidate does not broaden that refuted shield-only
  intervention.

These facts do not prove that every death is a bomb-decoder error. They do
show that the parent records long, apparently blind motion gaps while making
no safety intervention. The calibration is a narrow way to make the existing
safety mechanism testable rather than adding another unvalidated detour.

## Code / model division

Code owns the exact x displacement convention, action legality, player
movement samples, projectile matching, vertical direction, collision
projection, fire lock (by continuing to press FIRE), shield bookkeeping,
life/wave epochs, freshness and all fallbacks. The code calibrates polarity
only from a non-teleporting player displacement while a directional action was
previously posted. A life/respawn jump clears projectile evidence but does not
learn from the teleport. Two or more consistent displacement votes are needed;
the default sign is retained until evidence arrives.

The fixed sweep remains the fallback. If a verified bomb or two consecutive
downward observations create a collision risk, `path_risk` now projects the
candidate action through the calibrated sign, including a one-tick sticky
switch. If the risk is not established, the controller posts exactly the same
kind of fresh `left_fire`, `right_fire`, or `fire` action as the parent. The
old weak shield-chip detour remains unchanged and can only run under its old
thin/promoted conditions.

Astra still chooses one token, `MOTION` or `CORROBORATE`, asynchronously. Jev
still reviews evidence rather than choosing movement. This isolates the test:
any change in strong dodges or survival should come from the corrected
control/geometry bridge, not from a new target policy.

## Jev state schema and questions

The state schema is
`space_invaders_temporal_evidence_v2_calibrated_control`:

* `control.joystick_polarity` is the named fact
  `screen_x_follows_command` or `screen_x_opposes_command`, derived by code;
  `control.motion` is `responsive` or `motion_gap`;
* `candidates[id]` contains only a small projectile or shield-chip record:
  decoder label, qualitative up/down evidence, own-launch association,
  shield-lane corroboration, continuity, and cover bucket;
* `rules` explicitly say that y increasing is incoming, y decreasing is an
  outgoing laser, labels can alias, empty bombs do not prove safety, and Jev
  must not invent an object;
* `plan.rubric` carries Astra's current `MOTION`/`CORROBORATE` meaning.

For every candidate, one `Choice` is sent twice with reversed option order.
The options are `INCOMING`, `OUTGOING`, and `UNRESOLVED`. They are independent
classifications of supplied evidence, not numeric geometry or action
selection. Answers are accepted only when both orderings agree, the selected
probability is at least **0.72** in both, and any supplied confidence is at
least **0.30**. Answers older than 36 emulator frames, from another life/wave
epoch or plan version, or for a changed projectile identity are discarded.

An accepted `INCOMING` answer cannot override upward motion, an own launch, or
the absence of the required downward increment. The live collision check runs
again on the latest observation before it can alter an action. Timeouts,
malformed stand-in values, disagreement, or low confidence leave the
calibrated deterministic sweep in control. There is at most one outstanding
Jev call, with a 1.0-second timeout and no queued backlog.

## Astra scheduling and fallback

Astra is launched once, immediately after the controller begins posting
actions, with a 55-second timeout. Its output is accepted only if it is one
of the two exact tokens; arbitrary prose or garbage is ignored. A late or
failed call never blocks the real-time loop. Jev and Astra futures are
cancelled on return, and no background worker contains a Python polling loop.

## Confirmation or refutation

The hypothesis is supported if training traces show a stable polarity
calibration before the first life loss, motion-gap diagnostics that now agree
with observed x movement, and—when a real descending track is decoded—more
correct `projectile_escape` decisions, longer survival, and higher score
without reducing the parent’s early kill throughput. The strongest evidence
would be a calibrated escape immediately before a life that the uncorrected
projection would have selected into the track.

It is refuted if polarity never becomes observable, if decoded movement is
already aligned and the change is inert, if corrected predictions still never
see an incoming track, or if escapes increase while survival and score fall.
A high score with zero polarity-driven safety events would only confirm the
parent's sweep, not this hypothesis. Wave completion remains the explicit
milestone: all eight parent episodes ended with zero waves cleared.

The module uses only documented `metaharness.*` imports and tolerates malformed
observations and arbitrary model output without allowing a parse error to end
an episode.
