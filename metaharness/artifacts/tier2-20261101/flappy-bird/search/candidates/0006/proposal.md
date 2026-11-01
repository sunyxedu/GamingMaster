# Candidate 0006 — proof-carrying actuator, incident-scoped model supervision

## Parent, evidence, and hypothesis

**Rewrite parent:** `0001-tier2_astra_only`. **Successful reference:** `0005`.
No sibling candidate, game implementation, private state, hidden seed sequence,
or held-out result is used.

The parent proves that the model cannot own the live control loop. In
`0001-tier2_astra_only/trace_seed0.jsonl:37`, the episode ends after **31 ticks**,
**31 stale ticks**, **zero posted actions**, and **zero flaps**. The final bird
state is `y=380.48, v=10`, the cause is `ground_collision`, and the sole Astra
request is cancelled. Its 1.233-second episode did not outlast even one useful
reasoning response. The only retained parent mechanism is managed, budgeted
Astra invocation; its blocking action bridge is replaced entirely.

The best candidate has **no training survival failure left to repair**. Its
seed-0 and seed-5 terminal trace records report **716 pipes**, **26,999 frames**,
and **deadline** rather than collision, with zero death penalty. The practical
score ceiling under the fixed clock/deadline has already been reached. This
proposal does not claim that a better flap pattern can increase the pipe speed
or extend the evaluator deadline.

The remaining concrete inefficiency is treating discarded, rapidly obsolete
model advice as evidence that a functioning controller needs slow replanning:

- `0005/trace_seed0.jsonl:459` requests Astra at frame **385** after 15 rejected
  judgments; line **3285** requests again at frame **2770** after 12 rejections.
- Lines **2963** and **4656** apply the same `CURRENT_GAP / margin_px=10`
  configuration already running. These are non-corrective revisions.
- Seed 0 has **1,845 Jev calls**, **1,258 discarded answers**, **587 accepted
  profiles**, but only **14 `jev_changes_action` events**. Physics is verified at
  frame 12 and the trace contains **no `physics_mismatch` event**.
- Seed 5 similarly has **1,843 Jev calls**, **1,240 discarded answers**, only **8
  action changes**, and two slow reviews, one of which times out. It still
  survives the deadline. There is no measured evidence that these calls caused
  the successful control.
- The seed-0 best trace has one stale tick (**23007**); seed 5 has none. Avoiding
  unnecessary work on the actuation thread is relevant to the second selection
  criterion, although scheduler jitter cannot be eliminated by a claim.

**Hypothesis:** preserve the validated reachability/pulse machinery but replace
per-route semantic preferences with **incident-scoped supervision of controller
integrity**. A cached *state-space certificate*, invalidated by observed geometry
rather than by every tick, should preserve deadline survival while reducing
actuator work. Jev distinguishes true control incidents from harmless inference
latency; Astra is invoked only for corroborated structural faults. The intended
improvement is equal maximum score, equally low or lower staleness, and fewer
System-2 calls—not a fabricated improvement over the deadline score ceiling.

## Structural redesign

### Code: compile a certificate, query it each tick

The public observation is parsed into bird geometry, known pipe rectangles,
frame number, velocity, and the **actually applied** action. Offscreen
`(x=screen_width, top=0, bottom=screen_height)` placeholders remain unknown.
Pipe identity uses observed `x + speed * frame` and its gap, never a seed.

Backward dynamic programming constructs all viable `(y, velocity)` states for
each future frame. Each of the twenty integer velocities has a bitset of safe
vertical positions. The horizon covers all observed pipe exits plus a coast
buffer, bounded to 56–100 frames. Pipe constraints use the full bird rectangle,
10 pixels of extra vertical clearance, and a 2-pixel horizontal buffer. A
terminal rising state must still be able to coast without hitting the ceiling.
These are the working reference's verified geometry and integration conventions,
not the failed short beam-search objective of 0004.

**New:** retain the entire time-indexed certificate rather than discarding all
but its first-action result and solving again every frame. Reuse requires:

1. the same public pipe identities and gaps;
2. the same bird/pipe/ground geometry and pipe speed;
3. the same observed fractional-y lattice and verified integration convention;
4. at least 56 frames remaining in the certificate.

Each tick re-observes the actual bird position and queries that position's
successor membership. No action sequence is replayed open-loop. Newly revealed
pipes, changed geometry, expired horizon, or changed integration convention
force compilation immediately. A model cannot waive any invalidation condition.

If the normal certificate has no continuation, rebuild with margins 6, 2, then
0. A relaxed certificate is never cached as a permanent policy. If even zero
margin has no path, use the clearly labeled recovery preference rather than
claiming safety. Among viable actions, use the successful reference's CRUISE
arc preference **without LIFT/DROP model perturbations**. This is also an explicit
ablation of the unproven routine route-profile contribution.

The actuator posts IDLE before startup work, observes every applied tick,
revokes an applied FLAP before compiling or parsing model output, and checks
that the observation frame did not advance during computation. It posts the
next action before discretionary logging/model processing and waits with
`ctx.wait_for_tick()`. No inference blocks the actuator.

### Jev: interpret a durable diagnostic incident

A request describes an integrity incident, not a position that will be obsolete
five ticks later. The strict-JSON schema is:

- `task`, `rules`: pipe-score objective, held-action semantics, independent model
  timing, and finite-horizon safety limitations;
- `plan`: revision, `CONTROL_INTEGRITY / DYNAMICS / CLOCK` focus, and the goal of
  retaining verified control unless evidence contradicts it;
- `evidence`: code-computed booleans `physics_disagreement`, `clock_disruption`,
  `continuation_missing`, `clearance_pressure`, `physics_verified`;
- `definitions`: complete meanings of those booleans in every request;
- `controller`: current continuation status, viable actions, and the fact that
  the bird is re-observed every tick;
- `recent_mismatch_examples`: at most six actual-versus-predicted transitions,
  present only when they are relevant evidence.

Code does all arithmetic and counting: three recent one-frame mismatches,
two missed observation frames, three missing zero-margin continuations, twelve
relaxed-margin frames, or twelve verified transitions set the respective flags.
The rolling diagnostic window is 120 simulation frames. Inference latency and
an uncertain answer are explicitly **not** physical failure evidence.

Each request batches three independent typed questions:

1. **Incident Choice:** KEEP, CLOCK_GUARD, AUDIT, UNKNOWN.
2. The same Choice with the options reversed to reject option-order bias.
3. **Structural-fault Noul:** whether repeated physics disagreement or missing
   continuation is explicitly established—not general severity or network delay.

Both Choices must agree on a non-UNKNOWN option with winning probability
**>=0.85** and supplied confidence **>=0.40**. Missing confidence requires
winning probability **>=0.92**, not invented certainty. All numeric fields and
model containers are validated; arbitrary stand-in text is ignored.

**Answer-to-control bridge:**

- KEEP retains the existing certified controller and schedules no Astra call.
- CLOCK_GUARD, only while missed-frame evidence still exists, temporarily prefers
  an action whose **first and second identical applications** each remain inside
  the viability certificate. If that subset is empty, ordinary viable control
  remains authoritative. This can change a posted action, but cannot introduce
  an uncertified one. `jev_guard_changes_action` records actual causal changes.
- AUDIT, only with live control evidence, disables certificate reuse for 120
  frames and recompiles from each observation. It does not lower safety margins
  or invent new physics.
- UNKNOWN, disagreement, bad data, timeout, or a stale response leaves the
  deterministic controller in charge.

At most one Jev request is pending. The first commissioning audit is after
frame 30; a healthy controller is audited at most once per 1,800 frames (one
minute), roughly 15 requests per complete episode. Incidents may request at
180-frame intervals. The timeout is **3 seconds**, above the successful
reference's observed 1.28–1.34-second tail and the leaderboard's approximately
2-second maximum; it never blocks actuation.

Responses attach to an **incident signature + interface revision**, not to a
pipe. They expire on any diagnostic flag change, revision change, or age over
120 frames. A CLOCK_GUARD lease lasts at most 180 frames and also requires
current clock-disruption evidence. Thus a cached causal interpretation can
outlive a bird coordinate without outliving its preconditions.

### Astra: a corroborated-fault reviewer, not a routine pilot

There is no unconditional startup call and no review triggered just by rejected
Jev answers. A review requires a fresh confident AUDIT consensus, a
structural-fault Noul of **>=0.95**, and local corroboration of repeated physics
errors or missing continuation. Clock jitter, clearance pressure, and model
uncertainty alone cannot launch it.

At most **one** Astra call is allowed, with **120-second** timeout and at least
135 seconds remaining. The prompt includes the public controller design,
observed prediction discrepancies, Jev latency samples, and the relevant
self-contained typed-question, probability/confidence, freshness, and
no-arithmetic usage contract. Astra may return only a bounded JSON configuration:

- focus `CONTROL_INTEGRITY`, `DYNAMICS`, or `CLOCK`;
- cache policy `VERIFIED_REUSE` or `REBUILD_ON_OBSERVATION`;
- a short reason.

A revision changes the diagnostic context of later Jev questions; a requested
rebuild policy has a bounded 180-frame lease and is honored only if the physical
fault still exists on receipt. It cannot change actions, collision geometry,
gravity, margin, the evaluator clock, budgets, or unknown pipe data. Text is
never executed. All background jobs contain exactly one managed model call;
there are no unbounded Python workers to outlive the episode.

## Confirmation and refutation

**Confirm:** retain 716 pipes / deadline survival across training seeds, no
harness errors, no increased stale fraction, a large majority of ordinary
frames served by certificate queries, and **zero normal-case Astra calls**.
This would support the claim that routine route judgments and rejected-answer
reviews were not necessary for the reference's success. A higher Jev→act rate
is not a goal, and fewer calls alone do not demonstrate better survival.

**Refute:** a collision, systematic transition mismatch, lost feasible action
relative to fresh reachability, or repeated missing continuations implicates
certificate invalidation or terminal-horizon assumptions. Increased stale ticks
refute the scheduling benefit. A valid Jev repair which changes execution must
be distinguished from acceptance without effect by the repair/action events.
Unseen-seed robustness remains an evaluator question; no held-out result is
claimed.

The protocol intentionally cannot guarantee survival through *unobserved*
future pipes or arbitrary scheduler stalls. Its certificate covers the known
finite horizon and is recompiled as new public evidence appears.

## Validation status

The complete harness and proposal are written before refinement. No live game,
model backend, hidden game source, or long experiment has been run. Next checks
are syntax/import restrictions, bounded stand-in contract tests, short
public-physics simulations, and consistency checks against archived
observations. Real training and unseen-seed evaluation remain the evaluator's
responsibility.
