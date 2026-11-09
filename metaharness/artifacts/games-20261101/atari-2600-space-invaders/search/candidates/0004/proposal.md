# Candidate 0004: a code-owned firing/safety loop with pipelined tactical judgments

## Parent and hypothesis

Parent: `candidates/0002-tier2_idle/` (the requested rewrite parent). I also used
`0000-tier2_jev_only` as the strongest comparison and `0001-tier2_astra_only` as
the slow-model comparison.

**Hypothesis:** the best candidate is not limited by Jev's ability to recognize
Space Invaders. It fails because a raw six-way action choice is being used for
facts that must be exact and because the synchronous request leaves the same
action applied for multiple real-time ticks. In particular, the controller has
no owner for the one-shot lock, target alignment, bomb escape, or the transition
from a hit to the next target. A code safety/navigation loop with a deterministic
fallback should turn the same model budget into more useful shots and longer
survival. Jev should judge only compact near-term ambiguities, while Astra should
choose a wave-level risk/sweep policy off the control path.

The worst trace is concrete evidence. In `0000` seed 1, 179 Jev calls produced
425 ticks, including 247 stale ticks, and the episode ended at score 105 with
11 invaders killed and zero lives. The responses were 62 `fire`, 94 `noop`, and
23 `right_fire` (no left action). The trace repeatedly sends `fire` while the
shot is already active, then holds a lane: at frame 388 the player is at x=37
with score 25, and after life losses the observations show x=90 at frames 744
and 1084. The final observation is frame 1697, `all_lives_lost`. Seeds 2 and 3
show the same shape (110 points, 11 kills, zero lives). Bomb lists are often
empty at request instants, so a controller that reacts only to visible bombs is
not enough; the small patrol and formation-depth margin are intentional.

The idle parent confirms that simply avoiding bad actions is insufficient: its
seed-1 episode reaches 693 ticks but score 0, while its breakdown still records
six passive kills before all lives are lost. `0001-tier2_astra_only` confirms
that putting the real-time loop behind Astra is worse: mean score 16.25, 100%
stale ticks, and one failed System-2 call. The new design therefore changes the
division of labour rather than increasing a timeout or changing an option order.

## What changed

* **Code owns exact mechanics.** It decodes defensively, selects an alive
  column and its lowest surviving invader, maps the observed invader x to the
  cannon lane (the canonical traces verify the approximately +11 pixel lane
  offset), clamps movement, and refuses to fire while `player_shot.active`.
  It uses the observed bomb coordinates when present, predicts a near-term
  horizontal escape with hysteresis, treats a low formation as an emergency,
  and patrols laterally in alternating short phases while a beam is in flight.
  It also resets target/recovery bookkeeping on a life or wave transition.
* **Jev is pipelined, not on the critical path.** A background request carries a
  small derived state rather than all 36 raw objects. The main thread posts a
  legal action every tick: a fresh Jev answer when it is current and consistent,
  otherwise the code policy. Thus a 150--190 ms Jev request cannot freeze the
  cannon for two or three ticks. Old answers are discarded after 40 emulator
  frames or a life/wave change, and safety overrides them.
* **Astra writes policy, not actions.** An initial Astra call is launched in the
  background, and at most one revision is launched after the first life loss
  (or a new wave). Its strict requested JSON is parsed defensively; malformed or
  partial text leaves the current `LEFT_TO_RIGHT`, `CAUTIOUS`, `NO`-bonus,
  `SURVIVE` defaults unchanged. Astra can choose sweep direction, risk level,
  whether a UFO shot is ever opportunistic, and survival versus score priority,
  but it cannot bypass the code fire lock or safety check.
* **Fallbacks are executable.** Missing System-1, timeout, arbitrary answer
  values, an invalid Astra response, or a late future all fall back to the
  symbolic target/dodge loop. Model worker exceptions are contained, while
  `ModelUnavailable` still makes the context's `done` state stop the episode as
  required by the contract.

## Jev state schema

Every request is a strict JSON object with schema
`space_invaders_tactical_v2`:

* `frame`, `wave`, `lives`, `phase`, `recent_action`, and `recovery`;
* `player.lane` and a named x bucket, never raw arithmetic for Jev to compare;
* `shot`: `READY` or `ACTIVE`;
* `target`: current column, `LEFT/CENTER/RIGHT` side, `TOP/MIDDLE/BOTTOM`
  row band, `ALIGNED/APPROACHING/FAR`, and `CLEAR/GOOD/DAMAGED/OPEN` shield
  lane;
* `threat`: `bomb` side, `NONE/WATCH/URGENT` urgency, the code-derived escape
  direction, and formation urgency;
* `formation`: alive count, lowest row/y bands, and whether either outer column
  still has a living invader;
* named left/center/right shield quality and the current Astra `plan`.

The state contains no screenshot, raw RAM, large invader list, or arithmetic
request. Code computes the comparisons and sends their named results.

## Questions and answer bridge

One batched request contains four independent typed questions:

1. `defense` is a `Choice` over `DODGE_LEFT`, `NO_DODGE`, and
   `DODGE_RIGHT`, based only on the derived bomb fields. Code still wins when a
   current bomb is `WATCH` or `URGENT`; Jev cannot veto an urgent escape.
2. `move` is a `Choice` over `LEFT`, `HOLD`, and `RIGHT`. It is accepted only
   when its probability for the selected option is at least 0.43 and its
   direction agrees with the exact target error/escape lane. Otherwise code
   moves toward the target.
3. `fire` is a `Noul`. A yes probability of at least 0.58 can produce `fire`
   (or a combined move/fire when within the small alignment margin); a value at
   most 0.34 vetoes a nonessential shot; the middle or a malformed value uses
   the code fallback. Code always rejects a fire component when the shot is
   active.
4. `target` is a `Choice` over `KEEP_CURRENT`, `LEFT_OUTER`, `RIGHT_OUTER`,
   and `LOWEST_THREAT`. A selected option must clear the same 0.43 probability
   gate before it affects only the next target lock; dead columns are rejected
   and an active shot cannot be retargeted.

Choice insertion order alternates between equivalent orderings to reduce the
known first-option bias. The request is submitted with a 0.45 s Jev timeout,
above the measured p90/max calibration enough to avoid turning normal tail
latency into failures. A result is tagged with its observation frame, lives,
and wave. It is usable only while it is no more than 40 emulator frames old and
those episode markers still match.

The no-model fallback is: urgent visible bomb -> move to the computed escape;
otherwise move toward the lowest living invader in the current outer-to-inner
sweep; fire only when aligned and the player shot is inactive; while a shot is
active, oscillate a short distance laterally and never press FIRE. It posts this
fallback at every tick, including while Jev or Astra is thinking.

## Astra revision trigger and falsification

Astra is prompted with the current derived state and the game description, and
must return one JSON policy object. The initial call supplies the opening
policy. The first life loss supplies the only normal mid-episode revision; a
wave reset can also request a revision if the initial one has completed. The
slow output is applied only after enum validation and is fed back into the next
Jev state as `plan` plus the code's sweep/risk thresholds.

The hypothesis is supported if `0004` materially reduces stale ticks and raises
score through more kills per life, especially in the seed-1/2/3 failure shape,
without System-2 errors. It is refuted if the deterministic controller still
loses lives at the same stationary-lane depth despite the patrol, if Jev's
compact judgments consistently veto useful aligned shots, or if Astra revisions
change the sweep/risk in ways that reduce survival. The useful diagnostics are
score, kills, lives at termination, survival ticks, fire-lock violations,
formation/bomb overrides, and the fraction of actions made from a fresh Jev
answer.
