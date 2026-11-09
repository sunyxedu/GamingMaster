# Candidate 0016 — effect-aligned skills and outcome-based recognition

## Parent, comparators, and hypothesis

Structural rewrite of **0006**. Comparator: eligible leader **0011**. I read both
controllers/proposals, the leader's evaluation, their worst-episode traces, and
`lessons.md`. This candidate preserves the part supported by the archive: a
code-owned real-time actuator with asynchronous inference and measured movement.
It changes the unit of model cooperation from a vague `advance/retreat` vote to
an **observed objective + a skill outcome diagnosis**.

**Hypothesis:** the apparent navigation/combat failures are partly an
input/effect alignment failure, not simply insufficient frontier memory. Posted
turns are not immediately reflected in the next observation. An instantaneous
bearing servo reacts to the old view, reverses before queued turns drain, and
changes targets before a shot or interaction has had its effect. Similarly,
`USE` immediately followed by an escape is not a test that a door failed: opening
and controller latency need time, followed by a same-facing crossing attempt.
An effect-aligned executor should make existing observations and model judgments
actually executable before adding another global route graph.

This is not a claim that latency correction alone discovers exits. It tests a
necessary lower-level precondition that the earlier frontier/contact experiments
did not isolate. Static exits and doors are not assumed to have actor labels.

## Concrete trace evidence

- In `0006/trace_seed1008.jsonl`, the first forward post is at controller tick
  1 / observation tic 18; the first observed displacement is only at tic 21.
  The first `turn_left` post is tick 4 / tic 21, yet yaw first changes at tic
  24. The parent treats an unchanged position at tic 19 as a blocked face and
  posts USE before the initial movement effect appears. This is a concrete
  example of measuring a command before its result arrives.
- In the leader's worst episode, `0011/trace_seed1012.jsonl`, the first
  `macro_escape` is at tic **10**, even though the first small forward
  displacement is just appearing. `turn_left` starts at controller tick 11 /
  tic 16; observed yaw remains zero through tic 18 and changes at tic 19.
  That is direct evidence for an approximately three-observation command gap;
  it is measured again online, not treated as a guaranteed engine constant.
- The final 350 actions of that episode are dominated by one-to-four-tic left /
  right reversals. At tics **2504, 2524, 2544, 2564** the position is repeatedly
  approximately **(757.25, -276.44, -64)** while yaw cycles around 9 and 355
  degrees. Jev repeatedly says `retreat`, but the code keeps aiming. It ends
  at tic **2742**, health **6**, **2 kills / 3 items**, with no exit.
- The parent's worst trace has **47/47** semantic intents `advance`, **13**
  blocked-face probes, and ends at health **7** / **2 kills / 3 items**. From
  tic **987** to **1009**, `backward_right` leaves it at approximately
  **(1198.76,-271.99,-128)**. Neither the macro nor the model judgment is
  evaluated as a completed or failed skill.
- The leader's final observations identify actual `DoomImpBall` missiles as
  category **Explosive**, while `BulletPuff` also has that category. A missile
  rule based only on category `Projectile` misses real hazards; treating every
  Explosive as incoming would instead dodge the agent's own hitscan impacts.
  The new taxonomy distinguishes named missiles from puffs and checks velocity
  and closest approach.
- Lessons already refute “more frontier bookkeeping” or “blind contact following”
  as sufficient routes: 0009/0010/0014 do not beat 0011 or produce exits. The
  rewrite therefore does not add an omniscient map, a remembered seed route,
  another global frontier graph, or another fixed wall-following side vote.

## New division of labour

### Code: effect identification, skills, and verification

`EffectModel` stores the recent posted action timeline. It scores delays 0–7
against observed angle changes, updates turn rate from sustained turns, and
predicts the yaw that is already queued but not yet visible. The initial
three-tic estimate comes from the quoted trace and is updated using only public
observations. Alignment is against the predicted settled view, not an old
current angle. Position/target-velocity extrapolation is deliberately short.

The executor owns five explicit skills:

1. **Fight:** lock a currently rendered monster ID; correct aim using queued-yaw
   prediction; give aligned fire a short execution window; shoot close blockers
   rather than converting low health into endless non-firing retreat. A failed
   sustained engagement expires and records its health/ammo/kill cost.
2. **Pickup:** approach a supplied useful same-height object, measure distance
   improvement, walk through overlap, and record disappearance separately from
   verified inventory or counter change. A below-floor key/pickup is not treated
   as a safe path into a pit. A stalled approach receives an ID cooldown.
3. **Transit:** preserve a heading for a finite measured traverse. Alternative
   headings are charged for actual departure attempts, not for looking at them.
4. **Contact:** settle the view, pulse USE once, wait for the effect/opening,
   then test forward traversal on the **same heading**. Displacement is logged
   as `passage_after_use`, explicitly not as proof of a door or map completion.
   Repeated failed faces are remembered per key-contact epoch.
5. **Recover:** a bounded backstep, predictive alignment, and committed
   translation with displacement verification. It is not a perpetual turn loop.

Every skill emits an outcome ledger entry with kind, duration, displacement,
health change, kill change, ammunition spent, and evidence. Only the game's
terminal observation can establish completion. No score/record is fabricated.

The movement collision window is longer than the measured input gap and startup
acceleration. Missile emergencies can interrupt skills, but the dodge side is
committed briefly rather than flipped every tic. Weapons use the **observed
slot-keyed dictionaries**, with a selection acknowledged or timed out before
another switch. Splash weapons are not chosen in unobserved close geometry.

### Jev: concrete objective recognition and outcome diagnosis

Each request contains:

```text
{
  task, rules,
  recognizer_focus: {focus, encounter, travel, interaction},
  facts: {blocked_after_sustained_motion, damaged_recently,
          new_ground_recently, usable_ammo, better_gun_owned, critical_health},
  skill: {kind, phase, target, target_currently_rendered}|null,
  candidates: [{option, skill, id, name, category, range, direction,
                height, hitscan, body_blocker, major,
                direct_approach_permitted, needed_capability}],
  last_outcome,
  obstruction_evidence: {close_monster, height_mismatch_target,
                         settled_USE_failed}
}
```

Derived facts and semantic buckets avoid asking Jev to do distance arithmetic.
Objects are only currently rendered candidates. No hidden map, model-generated
coordinate, or inaccessible prior context is included.

Three independent questions share one asynchronous request:

| Question | Role | Executable consequence |
|---|---|---|
| `objective: Choice` | Choose among explicit `enemy:<id>`, `pickup:<id>` skills and `keep_skill` | At probability >= **0.64**, store an expiring objective. It can replace a transit skill when the object is still rendered and its approach/threat preconditions hold. It cannot redirect to an unseen object. |
| `encounter: Choice` | `clear_blocker`, `preserve_travel`, `recover_capability`, `unknown` | At >= **0.68**, permits a supplied threat objective to interrupt transit; deterministic immediate-danger rules remain authoritative. |
| `obstruction: Choice` | `enemy`, `height`, `tested_barrier`, `unresolved_contact`, `unknown` | At >= **0.68**, classify the measured result. A current matching pickup can be invalidated for a rechecked height mismatch; an outcome diagnosis never invents a door. |

Non-null confidence below **0.15** rejects an answer. Missing probabilities,
unknown values, malformed objects, timeout and ordinary backend failure preserve
the current skill. Accepted candidate IDs are revalidated in the latest scene.
A result older than **48 tics**, or judged under an old Astra interface version,
is discarded. Objective hints expire after **45 tics**. Option order alternates.

Requests occur roughly every 22 tics with threats and 46 tics otherwise, only one
outstanding. Empty space without an outcome is handled by code instead of asking
Jev to guess a door. The timeout is **3 seconds**, above the archived tail; a
late request never blocks motor refresh.

### Astra: revise the recognition interface from outcomes

Astra runs asynchronously at most three times with a **240-second** timeout.
The prompt includes current semantic facts, measured lag/rate, the last 18 skill
outcomes, coverage/contact results, and the relevant local Jev contract. It does
not ask for a raw action, a seed-specific route, or hidden architecture.

Its bounded output revises:

- `focus`: capability / keys / exit / recover;
- `encounter`: blockers / secure;
- `travel`: persistent / radial;
- `interaction`: patient / quick.

The **focus** becomes the decision-specific recognition instruction on every
subsequent Jev objective question and changes deterministic pickup emphasis.
**Travel** changes departure persistence; **interaction** changes the
wait-before-crossing interval. Invalid text retains the old interface. Applying
a revision invalidates old objective hints but not an already valid motor skill.

The revision hypothesis is that repeated `approach_no_progress`,
`combat_no_kill`, and `blocked_after_settled_use` outcomes are actionable planning
evidence, while an old scene plus the vague choice `advance` is not. The ledger
lets later iterations distinguish lack of a route from lack of skill execution.

## Real-time and error contract

The harness posts an advertised action on every newly observed tic, then
launches/polls inference asynchronously and waits with `ctx.wait_for_tick()`.
No worker contains a persistent control loop; only bounded model calls use
`ctx.spawn`. The deterministic skill remains active through latency, rejected
answers and transient failures. Budget/episode exceptions are normal endings;
`ModelUnavailable` propagates. Garbage Astra text and absent/first-option Jev
answers do not alter the validated fallback. Only standard-library and
`metaharness.*` imports are used.

## Confirmation and refutation

Confirm the alignment part if sustained turn reversals fall, aim actually
produces shots/kills before retargeting, and measured skills improve their
postconditions rather than merely posting fresh actions. Confirm interaction
semantics if settled USE followed by the same-heading crossing produces more
new traversable space than immediate escape, without claiming this proves a
door. Confirm cooperation if `jev_skill_bridge` changes a **current valid**
objective and Astra revisions improve the following ledger, not just call
counts. **The decisive task result remains `completed_by_exit`.**

Refute if lag fitting is unstable, predicted aiming still oscillates, contact
waits cause lethal exposure without crossings, the same failed skill is restarted
indefinitely, or semantic choices almost always preserve the fallback. A higher
kill/item score without exits would validate at most the low-level execution
hypothesis, not the primary route objective. The archive's all-zero exit rate
makes that distinction essential.

## Validation status

Initial module and proposal are written before running real-time trials. Further
local validation/results will be appended below; no unrun test is claimed.
