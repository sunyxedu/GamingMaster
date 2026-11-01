# Candidate 0011 — make Astra replans executable

## Parent and targeted hypothesis

Parent: `candidates/0006/`. I keep its frame-rate-owned reflected-intercept controller,
actuator-aware band braking, mixed/edge return schedule, asynchronous Jev supervision,
and all of its score/serve bookkeeping. The best reference is also 0006 (mean return
20.333), so this is a careful repair rather than a new controller.

The targeted hypothesis is that 0006's adaptive Astra loop was often not actually
adaptive: its response parser rejected the backend's repeated, concatenated JSON
objects. The useful replan evidence is visible in the worst training episode,
`candidates/0006/trace_seed1.jsonl`:

* At trace line 13932, Astra returned the same object repeatedly in one response,
  beginning `{"mode":"MIXED","attack_every":2,"bias_fraction":0.3,...}`.
  The following `astra_policy` event at line 13934 still records
  `attack_every: 3`, `bias_fraction: 0.28`, `mode: MIXED` — exactly the old plan.
* At line 15649, Astra returned a repeated object beginning
  `{"mode":"EDGE","edge_sign":"ALTERNATE","attack_every":2,"bias_fraction":0.34,...}`.
  Line 15651 again records the unchanged old `MIXED / 3 / .28` plan.
* That episode then scored 20:0 but reached the wall deadline at tick 53,999.
  The 19th point was at frame 51,357 and the 20th at 53,656, leaving only 343
  frames for the final point. Thus a late or ignored policy revision is especially
  costly: the controller was still using the parent's conservative schedule when
  the remaining time became decisive.

This tests whether making the already-requested Astra adaptation take effect converts
that near-win/deadline episode into a win, without changing the collision arithmetic.
It also tests a narrower claim than “FIRE is good”: no new feature behavior is added
and no model is put on the critical path.

## What changed

`harness.py` differs from 0006 in one functional place: `_extract_json` now tries each
flat `{...}` object found in Astra text before trying the whole first-to-last span.
Consequently, a response containing repeated short JSON objects uses the first valid
policy object. Fenced JSON and a normal single object still work as before;
unstructured prose without valid JSON, malformed JSON without a recoverable object,
arbitrary pre-flight text, and an empty response still produce `{}`.
The existing field validation and clamps remain the authority, so the parser cannot
turn model text into an illegal action or an unbounded control parameter.

The first valid object is intentional: duplicated objects in the trace are equivalent,
and choosing the first makes the behavior deterministic. This patch does not alter
Astra call limits, replan triggers, Jev cadence, local-controller logic, feature
gates, or fallbacks. The expected observable difference is `astra_policy` showing the values
Astra requested (for example `attack_every=2`, then `mode=EDGE`) rather than the old
plan.

## System-1 interface (unchanged from the parent)

Every Jev request is strict JSON assembled from code-owned facts, not raw pixels or
unbounded history. Its schema is:

* `game` and `rules`: right-paddle orientation (`RIGHT` means smaller screen y,
  `LEFT` larger y), automatic serves, and FIRE never being needed to serve;
* `scenario`: variation, difficulty, and right-paddle size;
* `score` and `observation`: reward-derived scores, frame, phase, serve status, and
  contact count for the current rally;
* `ball`: `RIGHT`/`AWAY`/`UNKNOWN` approach, reliability, and ETA bucket;
* `paddle`: current y, visible collision band, predicted y after settling, and the
  code-computed direction; and
* `control`: requested center/edge style, target relation, contact-window flag,
  uncertainty, and whether Astra has preapproved a legal feature experiment.

The batched independent questions are:

1. `style_a` and `style_b` are reordered `Choice`s over `CENTER`, `EDGE_UP`,
   `EDGE_DOWN`, and `HOLD_CENTER`. They recognize the supplied return style; Jev
   does not calculate a trajectory. A style is accepted only if both values agree
   and the selected probability after the confidence cap is at least **0.56**.
2. `feature` is a `Choice` over `USE` and `NO`. Its instructions require a
   contact window, preapproval, Robot Pong 1/2, and a safe paddle action. The state
   supplies preapproval inside `control`. Acceptance requires **0.62**. Execution
   rechecks the plan, rally/contact identity, current ETA window, rightward
   approach, Robot Pong variation, nonzero attack bias, and unused pulse budget.
   The inherited gate does not separately prove current ball/paddle overlap; this
   repair adds no FIRE authorization or new safety mechanism.

Requests run through `ctx.spawn` with a **0.9 s** timeout and are tagged with
`(rally_id, contact_count, frame)`. Responses from another rally or more than
80 frames old are rejected on receipt. Accepted style and feature records carry
rally/contact identity; edge-sign and feature use require those identities to
match. Style agreement uses the smaller of the winning probability and a supplied
confidence; missing confidence does not manufacture probability. Missing
probabilities mean zero. The inherited HOLD_CENTER guard additionally holds
movement on an unreliable trajectory; it is not separately freshness-tagged in
`_effective_direction`. No new freshness or interface behavior is introduced here.
Jev never computes or substitutes a numeric trajectory.

## Fallback and safety

The local loop always computes a reflected target and uses the visible paddle band plus
measured short-term paddle velocity. It holds/brakes when its target is covered,
uses a short trajectory estimate through occlusion and then center coverage, posts
a legal action before `wait_for_tick`, and continues if either model is absent.
During serve or unreliable perception, the inherited Jev HOLD_CENTER guard can
hold instead of recentering. FIRE remains one-frame, contact-gated, variation-gated, Jev-gated,
and disabled after its per-rally budget is used. Automatic serve frames never press
FIRE. Astra parse failure—including arbitrary stand-in text in pre-flight—leaves the
existing mixed fallback intact. Model exceptions are caught in their spawned workers,
and the loop stops when `ctx.done` or the game ends.

## Confirmation or refutation

Confirmation would be a score above 0006's 20.333 mean, especially seed 1 reaching
21 before the deadline, with near-zero stale ticks and no episode errors. Its trace
should show `astra_policy` values matching the non-concatenated first JSON object and
subsequent return behavior using those clamped values.

Refutation would be unchanged policy events despite valid duplicated responses (a
remaining parser/trace-shape problem), or policy updates with no improvement while
collision and stale-tick metrics remain comparable. If the plans apply but seed 1
still stalls at 20:0, the hypothesis that ignored Astra adaptation caused the
deadline failure is weakened; the next change should address scoring throughput
or return physics, not repeat this parser repair.

## Astra revision policy (unchanged)

Astra remains the macro-policy author, not a runtime question generator. It is
called at startup and at most twice more, after eight scoreless right contacts
or four point outcomes. It receives the public description, compact observation,
current policy, and recent verified point/contact evidence. Its 58-second timeout
remains asynchronous. The fixed Jev questions then recognize the requested style
derived from the clamped Astra policy. The intervention tests whether that existing
Astra-to-policy-to-Jev bridge works once duplicated output is decoded, not whether
a different question schema or a longer timeout would improve play.

## Local validation completed

Only archived training traces and synthetic contract-shaped stand-ins were used.
No emulator experiment, live model call, held-out result, or game implementation
was accessed for these tests.

* Syntax compilation passed. AST comparison establishes that `_extract_json` is
  the only changed function: all other parent logic is identical.
* Replayed all six successful archived Astra responses from parent seeds 0/1/2.
  The repaired extractor recovers all six, including three responses the parent
  rejected; the three formerly valid results are unchanged.
* Applied those six replies through the actual `_apply_plan` method. Every
  supplied policy field matches the decoded object. In the seed-1 replay the
  effective final plan becomes `EDGE / attack_every=2 / bias_fraction=.34`, with
  FIRE still disabled, rather than the parent's unchanged `MIXED / 3 / .28`.
* Ten parser cases passed, covering normal JSON, concatenated duplicates, fenced
  output, surrounding prose, malformed text, empty input, and non-object scalars.
* Five synthetic 180-tick loop tests passed: first-option Jev answers plus arbitrary
  Astra text, malformed answer containers, model timeouts, pending futures, and
  repeated Astra objects. Each posted 180 legal actions and no FIRE during serves.
  These test control plumbing, not game performance or the evaluator's official
  virtual-clock pre-flight.

These checks confirm the parser failure and its repair, not a score improvement.
The real evaluation must determine whether the newly executable policies help
scoring throughput or instead worsen return quality.
