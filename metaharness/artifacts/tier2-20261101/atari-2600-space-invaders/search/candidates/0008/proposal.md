# Candidate 0008 proposal

## Parent and hypothesis

Parent: `candidates/0005/`.  I also compared its losing traces with the
published lessons and the earlier typed controller, but did not inspect game
implementation code or the unfinished sibling directories.

**Hypothesis:** candidate 0005's remaining deaths are partly caused by an
incorrect movement envelope, not only by sparse bomb observations.  It treats
x=12 as the left safety boundary and can choose relocation destinations as low
as x=16.  In the actual public observations of that run, the cannon repeatedly
stops at x=37.  The controller can therefore spend many ticks issuing `left`
while the cannon is already clipped at the movement limit, losing its escape
route.  A controller that infers the real limit from repeated no-motion,
reverses before it, and keeps a small interior margin should preserve the
parent's firing progress while reducing deaths at the boundary.

This is a new test of **feedback-calibrated boundaries and interior bounce**.
It is not a claim that the public decoder's empty bomb list is safe, and it is
not a claim that a particular seed has a fixed bomb trajectory.

## Concrete evidence

* In `candidates/0005/trace_seed3.jsonl`, the first life-loss observation is
  frame 676 with player x=37, an active shot at x=38, and `bombs: []`.  The
  preceding actions are `left` for ticks 159--169, including the action at the
  score/life transition.  The trace has repeatedly accepted Jev's `relocate`
  maneuver even when the candidate summary says the route is clear and the
  chosen movement is left.
* The same x=37 life-loss position occurs in the archived parent traces for
  seeds 0, 2, 5, and 6.  This is much more consistent with a clipped movement
  boundary than with a useful interior firing lane.  Other deaths occur after
  long one-direction runs, while the sampled bomb lists remain empty.
* Candidate 0005's code checks `px <= 12` and `px >= 147` before stopping a
  direction, and its relocation candidates are clamped to 16 and 143.  Thus
  observing x=37 does not by itself cause a reversal.  Its score was strong
  (198.125 mean, 16.62 kills), so the proposed change preserves its useful
  asynchronous tick pacing rather than returning to synchronous decisions.
* Jev's high-level answer is also a confound in the parent: seed 3 accepts
  `relocate` repeatedly and records many `jev_changed_control` events selecting
  it over `engage`; the action mix is 165 left, 96 right, 48 fire, and 56
  left_fire.  Candidate 0008 therefore removes `relocate` from the model's
  action interface.  Jev may alter the next target ordering, but code remains
  responsible for the movement feedback loop.

Candidate 0004 used a fixed conservative bound in a different, lower-scoring
controller (mean 95.625); it did not isolate online boundary inference from
its target/fire and Astra changes.  Candidate 0008 tests the narrower feedback
hypothesis on the best parent.

## What changed

1. **Adaptive boundary envelope.** The initial limits are deliberately broad.
   If the same left/right command is followed by three observations with
   negligible x motion near that side, code raises/lowers the corresponding
   limit.  It then uses `limit + 4` / `limit - 4` as the interior firing and
   escape margins.  A single sticky-action frame cannot teach a boundary.
   Observed x=37 while repeatedly commanding left consequently becomes a
   reason to reverse, not a reason to continue left.
2. **Closed-loop bounce.** When a laser is active, the cannon travels toward
   the interior and reverses from the observed limits.  A visible close bomb
   overrides this with the legal away direction.  When no shot is active, code
   aligns to a living target and fires only in a small lane tolerance.  It
   never assumes that an empty `bombs` list means no threat.
3. **Target lock and shot feedback.** A target column is retained during the
   shot.  A confirmed invader disappearance marks a hit; a shot ending without
   a disappearance marks a miss and permits a new target.  Confirmed hits make
   a small online correction to the decoder-to-laser lane offset.  The update
   is bounded and slow because the formation can move between launch and
   confirmation.
4. **Jev is a slow target-order hint, not an actuator.** One asynchronous,
   latest-only request is made roughly every 32 emulator frames.  Accepted
   advice changes only the policy used at a future target refresh.  It cannot
   request `left`, `right`, `relocate`, a fire command, or a long hold.  The
   main loop posts a legal action and waits for every tick regardless of Jev
   latency.
5. **No Astra call in this candidate.** This candidate intentionally isolates
   the code-level boundary hypothesis.  In 0005 the eight Astra requests
   informed zero posted actions, and the event-driven preset did not reach the
   actuator before those games ended.  Spending the slow budget on a second
   unvalidated preset would confound the boundary test and loses the
   fewer-System-2 tie-break.  There is consequently no Astra text to put into
   the Jev state; exact geometry, boundary learning, and one-shot bookkeeping
   stay in code.

## System-1 state schema and questions

Every Jev request carries a strict JSON object with schema
`space_invaders_boundary_feedback_v1`:

* `frame`, `lives`, `wave`, and a categorical `phase` (`OPENING`, `MID`,
  `LATE`);
* `player_lane` (`LEFT_MARGIN`, `RIGHT_MARGIN`, or `INTERIOR`) and `motion`
  (`LEFT_LIMIT_LEARNING`, `RIGHT_LIMIT_LEARNING`, or `BOUNCE_INTERIOR`);
* `shot` (`READY` or `ACTIVE`) and `outcome` (`NONE`, `HIT`, or `MISS`);
* `bomb_evidence` (`NEAR_CANNON`, `APPROACHING`, or
  `UNSEEN_OR_EMPTY`).  The last label explicitly means uncertainty, not safety;
* categorical boundary evidence and the preserved-escape-margin flag;
* the current target's column/side/lowest-row band, all living columns with
  side and lowest-row bands, and the current target policy.

The request batches two independent `Choice` questions over the same five
policies, with the second question's option order reversed:

* `LOWEST_THREAT`: choose the living column with the lowest surviving threat;
* `LEFT_OUTER` / `RIGHT_OUTER`: choose an alive outer column;
* `NEAREST_LANE`: choose the closest living lane;
* `KEEP`: retain the current alive target.

The instructions explicitly say that code performs coordinate arithmetic and
chooses joystick actions.  This avoids asking Jev to count, compare raw
numbers, infer the action mapping, or plan a wave.

An answer is accepted only when both reordered Choices return the same valid
policy, each selected option has probability at least **0.58**, and any
reported confidence is at least **0.25**.  Missing or malformed probabilities,
unknown values, disagreement, a changed life/wave, or a response older than 48
emulator frames is rejected.  Accepted policy is persistent until the next
shot/target refresh; it does not churn an already active shot.  This
cross-check also makes the arbitrary-first-option pre-flight safe.

## Fallback and timing

The executable fallback is always active:

1. classify a visibly close bomb in code and move away, respecting the learned
   interior limits;
2. if a shot is active, bounce toward the opposite interior side;
3. otherwise aim at the locked living target, fire only when the lane is within
   the bounded alignment tolerance and the command cooldown has elapsed;
4. if there is no target or the model is late, continue the feedback bounce.

Actions are posted once per observed game tick and followed by
`ctx.wait_for_tick()`.  Jev runs in at most one background future; late and
normal failed calls are discarded while the fallback continues.  The worker
returns only JSON-safe markers, the bridge validates every model value, and
all external actions are filtered through `action_spec.names`.  Budget,
EpisodeOver, and backend-unavailable termination are allowed to end the loop
normally; no plain Python worker loop survives `run_episode`.

## Confirmation or refutation

Confirmation would be a mean score above 198.125 with at least as many kills,
longer survival, and trace evidence that left/right commands reverse near the
inferred limits rather than accumulating at x=37.  A reduction in repeated
boundary commands and life losses without a large fall in fire/hit rate would
support the causal explanation.  A useful online `shot_lane_calibrated` event
would show that the feedback path is active, but score and survival—not the
event count—are the test.

Refutation would be unchanged boundary behavior despite inferred limits,
substantially fewer kills because the interior margin prevents valid outer
shots, or a lower score with similar survival.  If the best episodes still
lose lives in the interior while the boundary is never reached, the parent
boundary diagnosis was incomplete; the next iteration should revisit bomb
prediction or shield use rather than increase Jev complexity.  If accepted
Jev policies change no target refreshes, the target hint is unnecessary but
that does not by itself refute the boundary hypothesis.
