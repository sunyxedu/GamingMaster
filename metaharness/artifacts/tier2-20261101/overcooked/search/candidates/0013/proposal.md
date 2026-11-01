# Candidate 0013 — detect the gate conflict before the turn is lost

## Parent and isolated hypothesis

Parent: `candidates/0006/`.  I retained its committed production executor,
recipe/phase extraction, BFS interaction poses, asynchronous Astra/Jev bridge,
and primitive fallback.  This is one targeted refinement of the parent's
cooking-pot traffic rule; it is not a new recipe policy or a model change.

The parent’s worst episodes are the two cramped-room runs at 180 points.  In
`candidates/0006/eval.json`, seed 3 has 108 collision ticks, 45 stuck ticks,
and 105 pot-idle ticks; seed 6 has 110 collision ticks and 104 pot-idle ticks.
The parent trace makes the lockout concrete.  At
`candidates/0006/trace_seed3.jsonl:647`, tick 394 shows the controlled chef at
`[2,1]` with a dish, the partner at `[3,1]` with a dish, a cooking pot with 10
ticks remaining, and `blocked_ticks: 3`.  The following actions are `north`,
`stay`, then `stay` through tick 400 (lines 649–654); the terminal state still
has both dishes and a pot cooking at tick 16 (line 655).

The preceding refinement, 0011, also supplies a useful negative control.  At
`candidates/0011/trace_seed3.jsonl:114`, tick 57 has the same decisive shape:
controlled `[2,1]`, facing east, holding a dish; partner `[2,2]`, holding a
dish; cooking remainder 8; `blocked_ticks: 3`.  Yet 0011 does not emit its
`stage_gate_release` event there: it first posts `north` at line 116 to turn
toward the pot, then posts `stay` at lines 118–125 until the partner/soup
transition.  Thus its two-snapshot pressure guard is not actually exercised
on this documented parent failure.  The lessons explicitly note that seed 3
produced zero release events, so this is not evidence that releasing the gate
is useless; it is evidence that the detector misses the only qualifying
snapshot.

**Hypothesis:** when a staged-dish chef has already reached the sole pot gate,
one current observation of a dish-carrying partner pressing that gate is enough
to justify a bounded release.  Requiring a second pressure observation, or
spending the first qualifying tick merely correcting orientation, converts a
recoverable handoff into a persistent reservation/collision.  Moving to a
nearby returnable floor cell for a few ticks lets the partner claim the soup
gate, while the controlled chef returns before a normal cooking window closes.

## What changed

Only the code-owned motor gained an early gate-pressure escape:

1. On a `stage_dish` job at its interaction pose, it checks the pressure before
   the ordinary orientation-turn branch.  The guard requires the existing
   parent traffic evidence (`blocked_ticks >= 3`), a partner carrying a dish,
   adjacency to the pose (or a predicted entry), a cooking pot, and at least
   eight cook ticks remaining.  A ready pot, a partner without a dish, an
   incomplete snapshot, or a last-second arrival keeps the parent reservation.
2. The first qualifying observation chooses a walkable neighboring parking cell
   with a BFS route back to the pose.  The release is attempted at most once
   for that committed job.  It lasts 3–6 ticks, waits if the partner has
   physically taken the gate, and returns as soon as the bounded interval ends
   and the gate is free.  If no safe parking cell exists, the parent motor is
   unchanged.
3. Recipe arithmetic, interaction legality, job invalidation, action names,
   Astra prompt, Jev questions, timeouts, and the always-live action fallback
   are unchanged.  No model response can authorize the release.

This isolates the question “did the parent miss a useful release because its
pressure detector was late?” from the broader, already-refuted idea of a
long-lived generic right-of-way lease.

## Confirmation and refutation

Confirmation would be a `gate_release_one_snapshot` event in the parent-like
seed-3 state around tick 57 (and analogous seed-6 states), followed by the
partner clearing the gate or delivering the ready soup; collision/stuck and
pot-idle ticks should fall, and score should exceed or at least preserve the
parent’s 244.444 mean.  The release should remain absent on clean cases where
the partner is not adjacent, the pot has fewer than eight ticks left, or no
returnable parking cell exists.

The hypothesis is refuted if the newly early release causes the partner and
controlled chef to swap/oscillate, makes the controlled chef miss ready soup,
increases pot idle time, or lowers score on the clean 0006 episodes.  It is
also refuted if the qualifying seed-3 state still produces no release event;
then the remaining issue is route/parking geometry rather than the missed
observation window.

## Tier-2 System-1 interface retained from 0006

Every Jev request is strict JSON derived from the current observation.  Its
schema is:

* `tick`, `layout`, and active `orders`;
* `self.position`, `self.facing`, and `self.held`;
* partner position/facing/held object, last action, and whether it is near the
  committed target;
* each pot’s position, phase, ingredients, missing ingredients, and derived
  cooking ticks remaining;
* the committed job kind/target/lock and exact `blocked_ticks`; and
* `traffic.yield_ticks`, the cached Jev mode, Astra’s accepted mode, and a
  short code-generated list of legal medium options.

The batched questions are unchanged:

* `phase: Choice` over `continue`, `delivery`, `supply`, `handoff`, and
  `yield`.  Code accepts only a closed-world value whose returned probability
  is at least `0.55` (or a closed-world stand-in answer with no distribution).
  The hint lasts 14 ticks or one job epoch and never emits a direction.
* `yield: Noul`, whose true value is used only when the controlled chef is
  empty-handed and the partner visibly carries soup/dish at the urgent gate.
  A true probability of at least `0.72` gives at most two `stay` ticks; it
  cannot interrupt a carried object or authorize the new release.

Requests are launched with `ctx.spawn`, one outstanding request, and a
one-second timeout.  Answers are tagged with the job epoch; late, malformed,
low-confidence, timed-out, or cancelled answers are ignored.  While Jev or
the one reset-time Astra call is pending, the code-owned BFS motor posts a
legal action every tick.  If any model output is arbitrary during pre-flight,
closed-world parsing and the deterministic fallback keep `run_episode` alive.
