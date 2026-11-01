# Candidate 0017 — bounded column intercepts inside the firing sweep

Parents: **0012 and 0003**. Mode: crossover.

## Hypothesis and concrete evidence

0012 earns its 346.250 mean primarily through tick-local broad lateral movement
and continuous combined FIRE, not demonstrated model adaptation. In its worst
seed-7 trace, `life_loss_signal` occurs at frames **1128** and **1744**, with
`previous_x` **86** and **88** and `recent_bomb=false` both times. Its final
controller event reports **190 points**, **7 unexplained chips**,
**0 Jev-changed actions**, and **0 effective Astra updates**. The trace records
shield escapes at **212, 292, and 532**, so the shield mechanism executes, but
its causal survival value is not established. Even seed 0, which scores 620,
ends without a completed wave and reports zero model-changed actions.

0003 contributes the useful *representation*: living target columns, lowest
surviving targets, and a typed target-to-action bridge. Its mean 188.125 beats
the raw-action baseline's 139.375, although that comparison does not isolate
target choice as the sole cause. Its synchronous controller and fire suppression
are not inherited. For example, the inspected seed-1 trace repeats FIRE through
tick 33, then alternates left/right at successive model completions; tick 102
finally reaches 5 points. The aggregate worst seed is seed 4 (95 points, per the
supplied archive summary), with 207 left, 173 right, 135 fire ticks and no
combined-fire actions. These are throughput costs, not arguments to stop firing.

**Test:** a broad sweep is a good default but may waste its one available beam
in empty lanes as the formation thins. Use 0003's typed column recognition to
insert a strictly bounded, one-shot intercept into 0012's continuous-fire loop.
The adjunct must produce actual alternative actions, not merely approve the
already-selected fallback. This is not a claim that the decoder is accurate
enough for unrestricted target chasing.

## What changes

- Keep fresh actions every observed tick and hold FIRE on every action. The ROM,
  not a model-side fire veto, enforces the one-projectile restriction.
- Keep broad sweeps between approximate observed operating lanes x=38 and 116,
  weak exposed-shield-erosion escapes, and stronger visible-bomb checks. Bounds
  are controller operating limits, not claimed physical movement limits.
- Add asynchronous column selection only when <=18 aliens remain by default.
  Code shortlists nearby columns within 20 pixels and rejects a predicted
  collision or an active weak erosion lane.
- Accepted advice leases a target for **16 emulator frames maximum**. A newly
  observed beam, target disappearance, life/wave change, safety veto, or expiry
  ends the lease. Continuous FIRE is preserved even while approaching.
- Resume the underlying sweep after the lease; impose a **72-frame cooldown**
  so a sequence of approvals cannot park the cannon under a persistent lane.
- Estimate a modest, clipped formation-motion lead from public observations,
  not from game source or emulator access. No boundary is inferred from stalls.
- Record accepted intercepts separately from actual action changes and outcomes.
  A short-window score delta is diagnostic, **not** proof the intercept caused it.

This does not retry the refuted blocked-route-extension hypothesis, restore
selective-fire starvation, memorize reset-position death lanes, or claim an
empty bomb list certifies safety. No seed-specific constants or trajectories
are used.

## Jev state, questions, and bridge

Schema: `bounded_column_intercept_v1`. Each request includes:

- `plan`: explicit eligibility policy and the continuous-FIRE/short-lease rule;
- `shot_window`: OPEN, NEAR_OPEN, or BUSY, calculated in code; BUSY is not queried;
- `formation`: SPARSE or SCORE_STALLED;
- `candidates.COLn`: ALIGNED/SHORT_APPROACH, LEFT/RIGHT/HERE, front-threat and
  outer-column flags, target-presence and collision facts;
- an explicit statement that no predicted collision is not proven safety;
- the last intercept outcome, if present.

One request batches two independently answered, reverse-ordered `Choice`s over
the shortlisted literal COLn labels, SWEEP, and UNKNOWN. They select a useful
short intercept rather than perform geometry. A `Noul` independently judges
whether the supplied shot window and eligibility support trying an intercept.

Both Choices must agree on a listed column, assign that choice probability
>=**0.60**, and have confidence >=**0.20** if confidence is supplied. Missing
confidence is not manufactured; the probability gate still applies. Noul must
be >=**0.65**. Answers older than **24 frames**, from a different life/wave
epoch, or from an earlier Astra policy version are discarded. Latest target
existence, approach distance, collision geometry, and lane cues are checked
again before execution. Unknown, malformed, disagreeing, or uncertain answers
leave the fallback unchanged.

There is at most one outstanding Jev request, with a **0.85-second timeout**,
no queue, and at least 20 frames between submissions. This exceeds the supplied
~0.14–0.19-second normal latency; a late completion still loses its freshness
lease. Transient failures back off without stopping per-tick control. No model
call is made directly or through a separately constructed client.

## Astra role and late decisions

One early background Astra call (after 240 observed frames) chooses only a
durable policy: SPARSE (default), SPARSE_OR_STALLED (also allow an intercept
after 180 scoreless frames), or SWEEP_ONLY. The prompt supplies the complete
bounded bridge and current public evidence. It does not ask Astra to generate
new questions or code. Its policy updates `plan` and eligibility for subsequent
Jev requests, invalidating pending old-policy answers.

The call has a 55-second timeout and never blocks actions. The early start
tests whether a ~30–50-second answer can affect the surviving tail, unlike
waiting for a late death before requesting adaptation. It may still arrive
too late: that is measured, not concealed. Arbitrary text leaves SPARSE intact.
There are no autonomous background loops; only bounded model calls run off-path.

## Confirmation and refutation

Confirmation requires `jev_changed_action` events followed by productive
`intercept_outcome` records, maintained near-zero stale ticks and full-rate
FIRE, and better mean score or wave completion than 0012 without additional
errors. Inspect low-score seeds rather than explaining the mean with a single
UFO. `astra_policy_applied` must actually change eligibility to count as useful
System-2 intervention.

Refutation: intercepts almost never pass the gate; advice changes no actions;
short stops increase blind deaths; scoreless intercepts consume time without
more kills; or the mean declines despite full firing throughput. If models
never intervene, any score movement cannot validate this target-selection
hypothesis. Shield-escape counts are retained as a confound, not credited as
validated bomb detection.

## Scope and validation

Written under the final write-now instruction from the inspected parent code
and traces and supplied lessons. No game-source inspection, backend calls,
held-out evaluation, or long experiments were performed. Python syntax is
checked locally; real stand-in pre-flight and training evaluation remain the
evaluator's responsibility.
