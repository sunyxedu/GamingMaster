# Candidate 0011 — bounded release of a contested cooking gate

## Parent and isolated hypothesis

Parent: `candidates/0006/`.  This is a refinement that keeps its committed
production executor, BFS interaction poses, asynchronous Astra/Jev bridges, and
closed-world fallback policy.

The parent’s remaining failure is narrower than generic path planning.  Its
worst episode, seed 3, scored 180 and had 108 collision ticks, 45 stuck ticks,
and 105 pot-idle ticks (`candidates/0006/eval.json`).  The trace shows the
specific contention:

* At trace line 639 (`trace_seed3.jsonl`), the controller commits
  `stage_dish` at tick 388.  It then spends ticks 389–395 on
  `north, stay, north, east, east, east, north` before reaching the
  interaction pose.
* At the Jev snapshot around tick 394 (trace line 647), the controlled chef is
  at `[2, 1]` holding a dish, the partner is at `[3, 1]` also holding a dish,
  the pot is cooking with 10 ticks remaining, and the committed job has
  `blocked_ticks: 3`.
* The terminal observation at line 655 has both chefs holding dishes, the pot
  still cooking at tick 16, and score 180.  Seed 6 ends analogously at line
  658, with both chefs holding dishes and the pot cooking at tick 17.

Candidate 0006 intentionally returns `stay` whenever a staged dish has reached
the pot pose.  That protects the controlled chef from the old oscillating
traffic policy, but it also makes the controlled chef a permanent gate blocker
when the scripted partner has independently brought a dish.  The Jev `yield`
answer cannot fix this case: the parent only applies it when the controlled chef
is empty-handed, whereas this failure occurs while it carries a dish.

**Hypothesis:** persistent, observable contention by a dish-carrying partner at
a still-cooking pot is a distinct traffic state.  A short deterministic release
of the gate, followed by an early reclaim, will let the partner pass without
sacrificing the controlled chef’s ready-soup reservation.  A generic right of
way lease is not being tested: the release is limited to one staged-dish job,
requires several observed blocked ticks and adjacent partner pressure, and
ends before cooking completes.

## Change in 0011

The only new behavior is in the code-owned motor:

1. While the controlled chef is at a `stage_dish` pose, count gate pressure only
   when the partner is carrying a dish and is adjacent to the pose or predicted
   to enter it.
2. If the pot is cooking, at least 8 cook ticks remain, the parent’s existing
   traffic detector has recorded at least 3 blocked ticks, and pressure has
   lasted at least 2 observations, choose a nearby walkable parking cell with
   BFS paths both out and back.  If none exists, retain the parent behavior.
3. Release the pose for a single interval of 3–6 ticks (bounded by the
   remaining cook time), using `stay` while parked if necessary, then route
   back to the original pose.  The release is attempted at most once per
   staged-dish job.  A ready pot, a held soup, an absent partner dish, or a
   malformed/incomplete state never triggers it.
4. All interaction legality, cooking state, timer arithmetic, and the final
   primitive action remain code-owned.  The existing job epoch invalidation
   still handles a partner taking the soup or the pot changing state.

No model call, timeout, action vocabulary, recipe assumption, or Astra prompt
is changed.  The model-independent fallback remains live on every tick, so the
change cannot create stale ticks while Jev or Astra is in flight.

Confirmation would be: `stage_gate_release` events appear only in the
contested cooking cases; dish-to-pot travel and ready-soup pickup latency fall;
collision/stuck and pot-idle ticks fall on seeds 0, 3, and 6; and team score
improves or at least does not lose the parent’s 244.444 mean.  A particularly
strong confirmation would be the partner no longer ending with an unused dish
while the controlled chef reserves the same gate, or the partner delivering a
soup during the released interval.

The hypothesis is refuted if releases make the partner repeatedly steal a pot
without delivering, cause the controlled chef to miss a ready soup, increase
pot idle time, or lower score on otherwise clean episodes.  It is also refuted
if no release occurs in the parent’s documented both-dish/blocked states; that
would indicate the pressure detector or parking search is too strict rather
than supporting a broader lease.

## Tier-2 model interface retained from 0006

The System-1 interface is intentionally unchanged so this candidate isolates
the gate-release hypothesis.  Every Jev request is strict JSON containing:

* `tick`, layout, and active `orders`;
* controlled and partner position, facing, held object, partner last action,
  and whether the partner is near the current target;
* every pot’s position, phase, ingredients, missing ingredients, and derived
  cook ticks remaining;
* the committed job kind/target/lock and `blocked_ticks`; and
* traffic yield state, the accepted Astra mode, and code-generated legal
  medium options.

The batched questions are:

* `phase: Choice` over `continue`, `delivery`, `supply`, `handoff`, and
  `yield`.  It is accepted only for one of those keys and only when the
  selected probability is at least `0.55` (a stand-in answer without a
  distribution is accepted only if its value is closed-world).  It is cached
  for 14 ticks or one job epoch and never emits a direction or task ID.
* `yield: Noul`, true only when the partner has the urgent carried object or
  visibly owns the sole approach.  A true probability of at least `0.72` gives
  an empty-handed controlled chef at most two `stay` ticks.  It never
  interrupts a chef carrying soup, dish, or ingredient and does not authorize
  the new gate release.

Requests are launched with `ctx.spawn`, one outstanding request, and a one
second timeout.  Responses are tagged by job epoch; late, malformed, missing,
low-confidence, timed-out, or cancelled answers are ignored.  The deterministic
priority and BFS motor immediately post an advertised action while a response
is pending.  Astra remains one asynchronous reset-time call for a weak
`mode/side/traffic` hint; it cannot override the release guard or a ready-soup
invariant.
