# Candidate 0010 proposal

## Parent and hypothesis

Parent: `candidates/0008/`, with the sustained-sweep mechanism from the best
candidate `0006` as a comparison.  I read the parent code/proposal and the
public worst traces; I did not inspect game implementation source or the
unfinished sibling directories.

**Hypothesis:** the biggest remaining failure is not only the learned movement
boundary.  Candidate 0008's safety-first alignment gate often turns its
controller into a patrol that moves but rarely launches a useful beam.  The
strongest archived controller, 0006, scored 280 mean and killed 19.88 invaders
while firing on every tick.  Its worst seed still died blind, but its broad
sweep had much more opportunity to put a shot through the formation.  Candidate
0008's worst seed 6 scored 10 with only two kills, despite zero stale ticks and
feedback code; its trace has a sparse `fire/right` rhythm and ends at frame
1155 with the player back at x=90 and no active shot.  Its own score breakdown
shows 2 kills and 0 UFO hits.  The parent therefore likely over-corrected from
wasted FIRE commands: it waits for a narrow target alignment, then patrols
through the firing lane without exploiting the ROM's harmless one-shot lock.

I test a careful structural combination: retain 0008's feedback-calibrated
interior boundary, target bookkeeping, bomb-track memory, and no-empty-bomb
assumption, but restore 0006's broad alternating traverse and hold FIRE on every
fresh tick.  Repeated FIRE cannot create a second beam, so while a beam is
active this changes no projectile state; it still permits movement.  When the
beam exits, a sweep tick is immediately eligible to launch if the cannon is
under a surviving invader.  This is a test of **fire opportunity throughput
under a closed-loop sweep**, not a claim that firing blindly is always safe.

## Evidence and why this is not a refuted repeat

* `0006` is the current best: mean **280.0**, **19.88** kills, **723.75**
  survival ticks, and 0% stale.  Its seed-4 episode reached 460 with 20 kills
  and 1 UFO; the action mix was 401 `right_fire`, 372 `left_fire`, and 48
  `fire`.  This is direct evidence that movement plus continuous FIRE can
  generate productive shots in this ROM.
* The same candidate's seed-2 failure is diagnostic: it remained fresh but
  died at frames 508 and 896 with empty decoded bomb lists.  Its trace reports
  `program_finished` routes 2--5 as `blocked`, then `life_loss` at frame 508
  with `previous_x=64`, and another at 896 with `previous_x=102`; it still
  reached only 15 points / 3 kills.  Thus I preserve the parent's interior
  boundary and bomb-memory protections rather than copying 0006 wholesale.
* `0008` improved the boundary mechanism but its best seed 2 was 315 and its
  worst seed 6 was 10.  The seed-6 evaluator reports 289 ticks, 2 kills, and
  no stale ticks.  The trace action pattern begins alternating `fire` and
  `right`, then spends long stretches on `right`; later it has only occasional
  `fire` commands.  No `boundary_inferred` event appears in the archived
  parent traces, so the proposed boundary learning was not exercised as a
  causal intervention.  A narrow alignment gate is therefore a plausible
  confound, not a solved failure.
* The lessons explicitly say sustained fire during lateral travel is the
  strongest supported score mechanism, while the parent 0008's score fell
  below 0005 despite its feedback additions.  This candidate isolates that
  mechanism while keeping the parent's timing and safety structure.

This is not a return to a stale synchronous policy: every tick is still posted
and followed by `ctx.wait_for_tick`, and Jev remains latest-only/asynchronous.
It is also not a model-selected movement rewrite: Jev only orders target
categories; code commits and reverses the sweep.

## What changed

1. **Broad committed sweep.** The controller computes living-column lanes,
   expands a narrow cluster into a meaningful left/right interior traverse,
   reverses at learned interior limits, and commits an endpoint until reached
   or expired.  A visible bomb track may override the sweep for the current
   tick; an empty bomb list never certifies safety.
2. **Hardware-rate FIRE.** The action bridge returns `left_fire` or
   `right_fire` on every posted tick.  It does not gate FIRE on a guessed
   cooldown or a single-frame alignment window.  Atari's one-player beam lock
   makes busy FIRE harmless; movement and beam travel continue together.  This
   directly tests whether the parent's missed shot opportunities caused its
   low-kill tail.
3. **Retained feedback controls.** Player displacement under a commanded side
   updates the learned left/right limits only after repeated no-motion
   observations.  The controller keeps an interior escape margin, tracks
   observed bombs across short decoder gaps, uses an away-direction emergency
   override, and resets its route after a life or wave transition.
4. **Retained shot/target calibration.** Living invaders are grouped by stable
   column.  The selected policy chooses the lowest threat/outer/nearest/kept
   column; a confirmed disappearance updates a bounded aim offset.  The
   sweep's purpose is to cover lanes, so exact model-side aiming is not a
   prerequisite for every shot.
5. **Simplified, bounded Jev bridge.** A single asynchronous request every 32
   emulator frames carries categorical phase, lane, sweep direction, shot
   state, bomb evidence, target columns, progress, and boundary evidence.  Two
   reordered `Choice` questions must agree before changing the target ordering.
   The accepted policy persists only as a future target-ordering hint; it cannot
   issue a joystick action, suppress FIRE, or invalidate safety checks.
6. **No Astra call.** The candidate intentionally isolates the fire-throughput
   hypothesis.  In 0006, 0007, and 0008, Astra either informed no actions or
   was not used; adding a slow policy would confound this test and increase the
   System-2 tie-break cost.  The deterministic controller is complete without
   it.

## System-1 state schema and questions

Each Jev state is strict JSON with schema
`space_invaders_fire_sweep_v1`:

* `frame`, `lives`, `wave`, `phase` (`OPENING`, `MID`, `LATE`),
  `player_lane`, and `sweep`;
* `shot` (`READY`/`ACTIVE`) and `shot_outcome` (`NONE`/`HIT`/`MISS`);
* `bomb_evidence` (`NEAR_CANNON`, `APPROACHING`, or
  `UNSEEN_OR_EMPTY`), where the last value explicitly means uncertainty;
* `progress`, categorical boundary evidence, and `escape_margin`;
* a compact list of living columns with side and lowest-row band, plus the
  current target and current policy.

The request batches two independent `Choice` questions over the same five
policies, with the option order reversed:

* `LOWEST_THREAT`: choose the living column with the lowest surviving alien;
* `LEFT_OUTER` / `RIGHT_OUTER`: choose a live outer column;
* `NEAREST_LANE`: choose the nearest live lane;
* `KEEP`: retain the current live target.

The instructions explicitly state that code performs geometry, arithmetic,
fire-lock handling, and action selection.  This follows the Jev contract: it
recognizes a categorical priority rather than counting or calculating.

An answer is accepted only if both Choices return the same valid policy, each
winning probability is at least **0.55**, and any present confidence is at
least **0.20**.  A missing answer, unknown value, disagreement, malformed
probability map, changed life/wave, or age outside 56 emulator frames is
ignored.  This gate is intentionally below 0008's 0.58 because target ordering
is reversible and low consequence; the movement/fire safety path is entirely
code-owned.  The arbitrary-first-option pre-flight remains safe because any
first-option response with no usable probability is ignored, while the
fallback sweep still acts.

## Fallback, timing, and failure handling

The executable fallback is always active and does not require Jev:

1. update movement limits from observed displacement;
2. if a tracked bomb is close, move away while clamping to the interior
   envelope;
3. otherwise continue the committed broad sweep, reversing before its learned
   limits;
4. append `_fire` to every legal lateral action, relying on the ROM's one-beam
   lock for duplicate-fire suppression.

The main loop posts one action before waiting and then one after each
`wait_for_tick`.  A late Jev result is polled and dropped by tag; it never
blocks or rewinds the game.  Futures are latest-only, and `stop()` cancels a
pending future reference when the episode closes.  Model exceptions are
contained inside the Jev request and treated as fallback events.  Context
budget/deadline termination ends the loop normally; legal-action filtering
prevents malformed stand-in actions from reaching the game.  No plain Python
worker loop remains after `run_episode` returns.

## Confirmation or refutation

Confirmation would be a mean above **280.0** or a comparable score with more
survival-adjusted kills, especially if the low-tail episodes show materially
more than 2--3 kills.  Trace evidence should show sustained `left_fire` /
`right_fire` actions, active shots overlapping sweeps, and `shot_lane_calibrated`
or score transitions without a large increase in boundary pushing.  A
reduction in the parent's no-shot intervals with unchanged stale fraction
would support the causal claim.

Refutation would be a score materially below 0008 or 0006 with comparable
survival, a sharp reduction in hit rate despite near-total FIRE coverage, or
life losses caused by the sweep repeatedly entering a visible bomb lane.  If
scores improve only when `boundary_inferred` fires, the result would support
feedback calibration rather than fire throughput alone.  If Jev policies are
accepted but never affect target refreshes, that is a bridge-utility failure,
not evidence against the deterministic sweep hypothesis; the primary test is
the code-owned fire/sweep change.
