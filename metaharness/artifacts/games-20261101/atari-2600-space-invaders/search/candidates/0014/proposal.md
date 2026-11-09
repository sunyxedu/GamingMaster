# 0014 — Outcome-accounted corridors, not repeated strategy labels

## Parent and hypothesis

**Parent:** `0009` (rewrite). Comparisons: best candidate `0006`, and `0010`/`0011` as recorded in the archive. Only this candidate directory is modified.

**Hypothesis:** the successful mechanism is continuous FIRE during lateral travel, but the current hierarchy cannot respond to evidence that a particular traverse is unproductive. Jev repeatedly endorses a generic policy rather than choosing between different executable interventions; the next-route bridge can also fail to consume accepted advice. Meanwhile, some supposed boundary stalls are interior periods of lost control preceding a life decrement, and some right endpoints are simply beyond observed travel. Repeating or reversing those same routes is not a useful model decision.

This rewrite keeps hardware-rate FIRE and a tick-local actuator, but replaces generic ordering hints with **outcome-accounted, reachable corridors**. Code supplies geometry and score evidence, Jev chooses a specific next corridor, and Astra selects the conditional rubric governing that choice. Choices are applied transactionally when a leg ends, with explicit logs of their consequences. This is a test of whether an operational feedback loop can improve useful firing coverage and first-wave progress, not a claim to recover unobserved bombs.

## Concrete archive evidence

* `0006/trace_seed2.jsonl`, lines 160–198: route 2 ends blocked after **45 pixels** at x=64; routes 3–5 each end blocked after **28 frames and 0 pixels**. Line 207 records the life loss at frame **508**, `previous_x=64`, `blind=true`. Another sequence at x=102 precedes the frame-896 loss. This is compatible with a frozen/death animation, not proof of an interior wall. It makes the direction-only repair tested by `0011` poorly targeted.
* The same trace ends with **349 actions**, **58 accepted Jev programs**, **zero effective Jev changes**, and **15 points**. Its `eval.json` confirms zero stale ticks and no error. More accepted generic judgments did not solve the low tail.
* `0009/trace_seed6.jsonl`, lines 144, 224 and 272, 354: rightward routes have endpoints **123.9** and **136.9** and last **268/280 frames** before expiry. At line 580, a score-change observation has cannon x=**119**. The left limit is inferred at x=37, but the final right limit remains **149**. This is asymmetric feedback: the right stall test requires proximity to a hypothesized far-right limit that observations do not reach.
* Parent seed 6 loses lives at frames **1320/1616** with previous x **100/88**, both marked `unseen_bomb=true`, and ends at **65 points**. I do not infer a bomb-free lane from the missing projectiles.
* Parent seed 0 demonstrates the preserved mechanism: **540 points**, **27 decoded kills**, including a UFO, with combined FIRE actions. However, its final controller event records just **one accepted Jev hint and zero effective changes**. The parent code consumes advice only if `route is None` before `action()`, while normal route finishing and recommit both occur inside `action()`. Thus ordinary boundaries need not expose the pending-advice application point at all.
* `0010` restores broad sweep/FIRE and reaches **264.375 mean** with **20.25 decoded kills**, but repeated `LOWEST_THREAT` hints do not change the sweep geometry. Its best seed 7 reaches **505**, while sampled shots and cannon positions verify sustained motion at roughly **0.5 pixel/frame**. Public observations repeatedly show x=119 on rightward travel.
* `lessons.md` explicitly refutes direction-only stalled-route repair and reports no wave clears. It also shows why I do **not** restore a narrow alignment/fire gate: `0007`/`0008` frequently moved without firing and lost substantial score. Empty bombs and disappearing sprites are not reliable ground truth; ALE score gain is the outcome signal here.

These observations support a travel-envelope prior and a control-gap distinction; they do not establish the exact physical cause of every freeze or death. The harness logs evidence needed to test those distinctions.

## Structural changes and division of labor

### Code: observations, executable alternatives, outcomes, actuator

* Always post `left_fire`, `right_fire`, or the bounded visible-bomb fallback `fire`. The ROM enforces its one-shot limit. No model can disable firing or change the environment.
* Start with the **observed 37..119** travel envelope and an interior margin, rather than an unreachable right goal. Expand the envelope only if this episode demonstrates a wider position. Near-extreme non-response may confirm a clip; interior non-response is recorded as a control gap, not learned as a wall.
* Keep a leg's direction through an interior control gap. Exclude frozen time from active timeouts and log the last controllable position before a later life decrement. This avoids manufacturing many zero-distance route outcomes or attributing the respawn center to the collision.
* Compute three concrete moving corridors: `SWEEP` covers the usable lane; `LEFT_WORK` and `RIGHT_WORK` concentrate moving fire on the corresponding live target lanes. Work corridors span approximately **52 pixels**, not a tiny left/right oscillation. Geometrically redundant or empty alternatives are unavailable.
* Estimate movement and formation velocity from observations; compute approximate interception lanes and shield-mask obstruction in code. These are geometry priors, not guaranteed hits. Keep visible projectile tracks briefly across decoder gaps and recheck collision projections every tick.
* Account for every completed leg using **ALE score gain, active elapsed frames, distance moved, and observed life loss**. Do not count sprite disappearance as a confirmed kill or use the conflicting study row-value tables as reward predictions.
* Consume Jev advice **inside the next-leg commit**, after finishing the previous leg. Recompute the alternatives and their availability at that instant. The chosen corridor changes actual frozen endpoints.
* Log `corridor_committed` with selected corridor, same-state code fallback, endpoints, direction, and whether Jev changed the selection. Log `jev_action_consequence` when a Jev-selected leg differs from a parallel wide-sweep direction reference. These are intervention diagnostics, not counterfactual score estimates.

### Jev: choose a specific intervention, not arithmetic or a default label

Jev chooses among the already computed corridors for the **next** leg. It sees categorical outcome and geometry evidence, and Astra's current rubric. A corridor choice is a real bounded change in coverage and revisit frequency. Code retains all immediate collision, boundary, legality, and freshness checks.

### Astra: revise the decision rubric, not an old snapshot action

An asynchronous startup call requests just **one validated word**:

* `BALANCED`: preserve broad coverage while productive; favor a distinct working corridor after repeated silent legs.
* `HARVEST`: concentrate moving fire on promising live/open lanes instead of repeatedly covering low-yield space.
* `ESCAPE`: favor broad lateral room and fewer reversals after losses or poor control.

The label changes both the next Jev state's literal rubric and the code fallback ranking. It is conditional strategy intended to remain meaningful after a life change, not a stale coordinate decision. A second call is allowed only after at least four completed outcomes, repeated evidence of failure, and at least 1,800 frames since the prior request. At most two calls are made. The prompt includes the relevant Jev local contract and the actual corridor consequences; Astra is not asked to generate executable code or unconstrained question text.

The hypothesis for returning to Astra is **rubric failure**: multiple silent traverses or repeated life losses despite current choices. One quiet shot or decoder silence is insufficient. A Jev review judgment must be supported by those code-side facts.

## System-1 interface

Each request carries strict JSON schema `outcome_corridor_v1`:

* `rules`: game-specific action/one-beam semantics and decoder uncertainty;
* `plan.profile` and `plan.rubric`: the actual current conditional strategy;
* `evidence`: movement response/control gap, repeated-silent/recent/quiet progress, loss bucket, visible/unobserved bombs, current corridor, sparse/multiple-column phase, and the last three categorical leg outcomes;
* `candidates`: each candidate's availability, all/partial/no reachable coverage, wide/moderate escape room, mostly-open/mixed/shield-blocked firing paths, code-ranked live-lane density, presence of low targets, untried/productive/slow/silent outcome, and observed-loss evidence.

The batched questions are:

1. `corridor`: `Choice(SWEEP, LEFT_WORK, RIGHT_WORK, UNKNOWN)` selects the next intervention under the rubric.
2. `crosscheck`: the same question with reversed option order. The first ordering also alternates between legs.
3. `review`: `Noul` asks whether repeated explicit outcome failures support strategic review. It does not ask Jev to count, calculate, detect an invisible bomb, or decide an immediate joystick dodge.

The two Choices must agree on an available known corridor, each selected probability must be **>=0.60**, and each supplied confidence must be **>=0.25**. Missing confidence is not fabricated; missing probabilities cannot pass. `UNKNOWN`, malformed answers, disagreement, and low probability retain the code fallback. Review requires yes-probability **>=0.80** plus an explicit repeated-failure predicate.

## Timing, freshness, and fallbacks

* The main loop posts an action before model scheduling, then uses `ctx.wait_for_tick()`. It never waits for a model future.
* Jev is requested ahead of a leg boundary, once per next-decision slot, and no more often than every 24 frames. Only one request may be outstanding. The timeout is **1.0 s**: archived warm medians are around 0.14–0.19 s, but parent seed 6 includes **0.2926 s**, and other parent episodes reach **0.3195 s**. A sub-tail timeout would add failures without making the actuator faster.
* Each answer is tagged by frame, life/control epoch, wave, plan version, and next-leg slot. It must arrive within **72 frames**, and expires at commit after **144 frames**. Life/wave/discontinuity or rubric changes invalidate old advice. These are future-corridor decisions, not permission to apply an old tick action.
* Actual Jev latency is measured with `ctx.clock.now()` and its episode median is logged. Astra's **60 s** timeout accommodates the archive's 27.4 s fixed-cost estimate and 42 s slow calls; one-word output limits additional latency.
* Fallback is a complete code controller: select a checked corridor, continue to its reachable endpoint, hold FIRE, and use current visible-bomb geometry for overrides. `BALANCED` starts broad and uses density/yield evidence after repeated silence. It remains functional without either model.
* Background tasks contain one context inference call only, never an unbounded Python worker. Ordinary model timeout/parse/runtime failures become fallback events; budget exhaustion and persistent model unavailability retain their normal context semantics. Pending futures are canceled on return, and the evaluator owns cancellation of in-flight model calls.

## Confirmation and refutation

**Confirm:** no errors, nearly zero stale ticks, FIRE retained, fewer unreachable-endpoint timeouts, and increased score/first-wave progress versus parent **208.75** and preferably best **280**. Mechanism-level evidence requires distinct corridor choices that change endpoints/actions and subsequent score gain—not merely accepted-answer counts. Reduced quiet active time and more productive revisits would support the targeting/feedback hypothesis. A returned Astra rubric must visibly reach Jev state and alter ranking when relevant.

**Refute:** narrower moving corridors reduce score or survival despite actual bridge activity; alternate corridors remain silent; control gaps are misclassified as boundaries; repeated local reversals replace long traverses; or Jev continues to select the same generic default and produces no intervention. If improvements come only from reachable endpoints with no model-dependent changes, that supports the mechanical fix, **not** the proposed division of labor. If Astra never returns in time, its contribution remains untested rather than credited.

No seed-specific action sequence, hidden game state, emulator inspection, alternate game settings, direct backend calls, or training-time persistent state is used. Development checks are short stand-in/fixture checks only; real scoring remains for the evaluator.
