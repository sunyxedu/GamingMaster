# Candidate 0005 — idle-default pulse executor with asynchronous route judgments

## Parents and hypothesis

Crossover parents: `0000-tier2_jev_only` and `0002-tier2_idle`. Both scored zero, so neither contributes a successful scoring policy. The useful mechanisms are different: the Jev parent demonstrates a working typed observation-to-action bridge; the idle parent demonstrates an inference-independent, contract-correct tick loop that never accidentally repeats an upward impulse. This candidate combines semantic model judgments with a continuously running actuator, not the parents' literal action streams.

**Hypothesis:** the flat leaderboard is primarily an execution-timescale and state-representation failure. A roughly five-frame model round trip is too slow to hold a one-frame FLAP command. A compact semantic question can remain useful over an arc or pipe transition, while public-physics reachability and explicit pulse release must operate every frame. Separating those timescales should pass pipes without requiring fast or perfect model answers.

## Concrete archive evidence

- `0000-tier2_jev_only/trace_seed0.jsonl`: the applied action runs are ticks **1–5 IDLE, 6–9 FLAP, 10–14 IDLE, 15–50 FLAP**. The last uninterrupted held impulse lasts 36 frames. The episode ends at frame 50 with `termination_cause="pipe_collision"`, bird `y=-71`, velocity `5`, score `0`, and 40 applied flaps. This is not merely insufficient inference throughput: at request frame 31, bird `y=-5`, velocity `-9`, Jev still chooses FLAP with probability `0.75`; the raw-state question also misrecognizes the situation.
- Across all six Jev-parent traces, measured request latency has median **0.1522 s** and maximum **0.2149 s**: about 4.6 and 6.4 game ticks. All six die at frame 50. The seed-0 terminal report records 39 stale ticks and 11 posted actions.
- `0002-tier2_idle/trace_seed0.jsonl`: IDLE is applied on all 31 ticks; termination is `ground_collision`, bird `y=380.48`, velocity `10`, score `0`. The other five idle episodes end identically. IDLE is a useful default between impulses, not a complete policy.
- Observations also resolve an ambiguity in the study prose: the Jev-parent seed-0 state moves from `y=244, v=-9` at frame 0 to `y=214, v=-4` at frame 5 under five IDLE ticks. The displacement is the sum `-8-7-6-5-4=-30`, supporting **increment velocity before position integration** for this build. The harness checks consecutive observed transitions again at runtime rather than trusting this finding indefinitely.
- No `lessons.md` was present in the permitted archive. Sibling candidates were not read or modified. No game implementation, private state, RNG, seed-specific policy, or held-out results are used.

## Structural change

1. **One actuator on the game clock.** It observes each tick, computes the next action, posts it, and calls `ctx.wait_for_tick()`. Before any expensive work after an observed FLAP tick, it posts IDLE to revoke the pulse. Inference is never awaited on this thread. A second observation guards against publishing an action computed across a tick boundary.
2. **Public-state reachability, not a learned physics guess.** The harness computes backward reachable sets over observed pipe rectangles and ballistic dynamics for a 56–100-frame horizon. Sets of vertical positions are Python integer bit masks, with one set for each velocity. A next action is eligible only if it has a collision-free continuation through that horizon. The horizon covers known pipes and a terminal coast buffer; unobserved offscreen pipe placeholders are excluded, never interpreted as enormous openings or privately filled in.
3. **Sparse, arc-shaped preference.** Among viable actions, the default prefers IDLE while rising, and a single FLAP near the lower-middle part of the current corridor. Reachability overrides preferences when a pipe transition requires it. Normal vertical clearance is 10 pixels; if the current state has no feasible continuation, retries use 6, 2, then 0 pixels. This relaxation is explicit and logged, not a change to game collision rules. If none is feasible, a bounded observation-based recovery preference runs.
4. **Asynchronous models with bounded influence.** Jev selects an arc/transition profile, never an unfiltered impulse. Astra is an evidence-triggered reviewer of a tiny interface configuration, not a mandatory startup delay or live per-frame pilot.

## System-1 interface

Every request carries a compact independent JSON object:

- `rules`: pipe-passage objective, downward-positive y convention, and one-frame pulse execution semantics;
- `plan`: version, CURRENT_GAP/NEXT_GAP focus, and a short goal;
- `route`: observable target identity, APPROACH/INSIDE phase, current/following direction labels, and the already-computed priority direction;
- `motion`: RISING or FALLING_OR_LEVEL;
- `safety`: locally computed legal-continuation actions, clearance margin, and the non-overridable reachability boundary;
- `diagnostics`: booleans for repeated one-step physics mismatch and persistent margin relaxation.

The target identity is derived from observed pipe position plus observed frame displacement, not the seed. UNKNOWN offscreen geometry remains unknown. Jev does not subtract coordinates, count frames, run search, or infer a missing gap.

One request batches three independent questions:

1. A Choice of **CRUISE / LIFT / DROP / UNKNOWN**. CRUISE preserves the normal arc, LIFT modestly advances safe upward preparation, DROP favors patient descent.
2. The same Choice with reversed option ordering, to reject order-sensitive answers.
3. A Noul asking whether explicit diagnostic evidence warrants a slow review. This is a yes/no evidence judgment, not an urgency scale or an action.

Both Choices must agree on a non-UNKNOWN profile. Each winning probability must be at least **0.72**, and each supplied confidence at least **0.35**. Missing confidence is not converted into certainty: the required winning probability rises to **0.85**. Malformed, disagreeing, uncertain, and stale answers leave the local CRUISE controller in charge.

A profile changes only a preference among reachability-approved actions. LIFT moves the flap anchor upward by 6 pixels and permits an earlier climb preference; DROP moves it downward by 5 pixels and avoids the early-climb preference. Events distinguish accepted judgments from cases where a Jev preference actually changes the posted action. Call count alone is not evidence of cooperation.

## Freshness, scheduling, and fallback

- At most one Jev request is in flight, launched no more often than every 12 frames (0.4 seconds). This fits approximately 2,250 calls into a full 900-second episode, below the 4,000-call budget.
- A two-second request timeout is deliberately above the observed latency tail, but never blocks actuation. A returned profile is accepted only within 24 frames, for the same target/phase/following-direction key and plan version. Its lease ends 30 frames after the request. Route or plan changes immediately invalidate it.
- A local safe-continuation preference supplies **every** action while requests are pending or invalid. FLAP is not a fallback held across model latency. Late answers do not retroactively execute their old action.
- Exceptions representing episode/budget termination and model unavailability propagate normally. Ordinary inference or parsing failures retain the controller. No Python background loop is created; spawned tasks contain only one managed model call and end within that call's bound.
- Runtime observed transition checks detect physics-order disagreement. Diagnostics record verified transitions, mismatches, safety overrides, relaxed margins, and accepted/rejected judgments. Model text is parsed as bounded JSON, never executed.

## Astra's role and revision trigger

There is no unconditional startup Astra call: the parent cannot survive even its initial latency. After at least 150 frames, repeated rejected semantic judgments or corroborated controller/model mismatches may trigger a background review. Reviews are at least 1,800 frames apart and limited to two per episode, with a 120-second timeout and at least 150 seconds of remaining wall budget.

Astra receives current public-state labels, the failure counters, measured runtime Jev latency, validated-transition counts, and the relevant typed-interface usage rules. It may return only a bounded configuration: focus on CURRENT_GAP or NEXT_GAP, and clearance margin 6/8/10/12. This changes which named direction Jev is asked to judge and the conservative reachability margin; it cannot change physics, actions, budgets, observations, or collision semantics. Valid revisions increment the plan version and invalidate cached judgments. Garbage or out-of-range output is ignored. There is no routine review when the initial interface works.

## What would confirm or refute the hypothesis?

**Confirm:** zero harness errors; verified one-step predictions; a high fraction of fresh actuator ticks; isolated rather than multi-frame-held flap impulses; first-pipe passages across training seeds; and a clear score improvement over both zero-score parents. Accepted profile events and `jev_changes_action` events should show whether model judgments add useful preference changes rather than simply consuming calls.

**Refute or narrow:** continued deaths before the first passage despite accurate one-step predictions and pulse release would implicate geometry/terminal-horizon assumptions or the preference policy. Frequent physics mismatches refute the dynamics model. A controller that survives only by relaxing to zero margin is not evidence that the conservative policy works. If model profiles almost never change an action, or their changes precede avoidable failures, useful model cooperation has not been established even if the code controller scores well. If pulse release still spans several ticks, scheduler latency rather than inference latency remains unresolved.

## Validation status

The implementation and proposal are written before refinement. No real evaluation, model backend call, or long game experiment has been run by the proposer. Short local syntax/contract and public-physics unit checks are planned; actual training and unseen-seed scores are left to the evaluator.
