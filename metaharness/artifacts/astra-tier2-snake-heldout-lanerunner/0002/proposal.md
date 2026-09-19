# Candidate 0002: compiled reactive strategies

Parent: `0000-tier2_jev_only`. This is one structural hypothesis test, combining
a background rule compiler with classification of reusable strategies.

## Diagnosis from the archive

I read both prior harnesses, proposals, evaluations, and all six episode traces.
Every baseline episode scored zero and ended at tick 6 with `"death": "wall"`.
There are two distinct failures:

* **The classifier is not doing the reasoning implicit in its action question.**
  All 16 System-1 responses across the three parent episodes selected `keep`.
  In `0000-tier2_jev_only/trace_seed1.jsonl`, the initial request has
  `"head": [6, 6]`, `"food_delta": [-4, 4]`, and
  `"danger": {"up": false, "down": false, "left": true, "right": false}`.
  The answer is `"choice": "keep"`, with probability **0.58**. The next answer
  raises that probability to **0.69**. Its criterion is simply
  `"keep": "keep current heading"`; none of the criteria explains when an
  action is desirable. This is consistent with an inertia/label-matching bias,
  rather than reliable comparison of reward progress and danger. It does not
  prove the internal cause of that bias. The final request in seed 0 is more
  explicit: at tick 5, heading is `right` and `danger.right` is `true`, yet
  `keep` still wins with probability **0.41**, above `up` at **0.33**.
* **Both baselines put inference latency on the action path.** In that same
  episode the final call starts from observation tick 4, takes
  **0.6381032467 seconds**, and finishes after ticks 5 and 6 have already used
  `keep`. Mean System-1 latency is **0.6127408981 seconds** against a 0.5-second
  tick. Worse, in `0001-tier2_astra_only/trace_seed2.jsonl`, ticks 1–6 all have
  `"fresh": false`; Astra's answer `"text": "up"` arrives after
  **8.8178169727 seconds**. Correctly choosing a move eventually cannot save an
  episode that ended roughly three seconds after it started.

The parent has 27.78% stale ticks, while the Astra baseline has 100%. Yet the
parent also dies with mostly fresh actions: improving freshness alone will not
fix its action semantics.

## Hypothesis and intervention

Slow reasoning should produce a reusable state-to-action policy, and the fast
classifier should recognize which strategy applies. Neither model should have
to deliver a newly reasoned movement command before every deadline. A strategy
can remain useful after its originating observation expires if its action is
resolved again against the current observation.

The new harness implements that division as follows:

1. **Compile asynchronously.** Astra receives `game.describe()`, the complete
   action specification, and recent observations. It returns JSON containing
   ordered global features, per-action features, an immediate safety predicate,
   and one to three strategies. Each strategy supplies a classification
   criterion and a numeric action-utility expression. The first strategy is a
   general survival-and-reward default. The prompt asks for future viability as
   well as immediate progress, and for correct handling of ignored/default
   actions. These are instructions to the model, not guarantees of correctness.
2. **Interpret a restricted expression language.** Python AST validation admits
   indexing, arithmetic, Boolean expressions, comprehensions, a small set of
   functions, and dictionary `get`. It excludes statements, imports, arbitrary
   attributes/calls, lambdas, and exponentiation. Expressions and feature counts
   have size limits; `range` is capped. Validate on the initial and current
   observations before installing. This is a restricted evaluator, not a
   process sandbox or proof that arbitrary nested comprehensions are cheap.
3. **Classify strategy from computed facts.** After compilation, Jev receives
   Astra's rule summary, current derived features, safe candidate action facts
   and utilities, and a `Choice` over strategy IDs. Its criteria are Astra's
   observable applicability conditions. A typed answer selects a strategy;
   the tick loop chooses that strategy's highest-utility currently safe action.
   Thus Jev classifies context while arithmetic and immediate safety are
   recomputed locally. No new request is launched while one is outstanding.
4. **Bootstrap without waiting for Astra.** Before a policy exists, use an
   explicit safety-and-progress action question with opaque option IDs. Discover
   candidate Boolean masks by their structure: dictionaries containing at
   least two action-name keys with Boolean values. Jev classifies each as
   true-means-unsafe, true-means-safe, or unrelated. Only polarity answers with
   probability at least 0.7 are retained. Re-read confirmed masks every tick,
   preferring explicitly known-safe actions over unspecified ones. No field
   spelling or Boolean polarity is built in. When no mask is recognized, this
   phase necessarily has weaker safety guarantees.
5. **Keep ticking independently of inference.** Post a freshly resolved action
   each tick and call `ctx.wait_for_tick()`. Reject selections from an older
   policy revision or more than four tick intervals old, using the game clock.
   While a call is pending, re-evaluate the last unexpired strategy; otherwise
   use strategy 0. Bootstrap rankings have the same expiry and are filtered
   against current masks. If all actions are unsafe, the harness logs the
   condition and makes a best-effort choice. A small virtual-mode scheduler
   yield prevents the driver from starving model workers.
6. **Repair and reuse.** Request Astra at cold start and after parse/validation
   or expression-evaluation failures, with at most three calls per episode and
   budget checks before dispatch. Repairs include the error and recent
   observations and reuse Astra's session when available. There is no periodic
   replan for an otherwise valid policy. Cache validated JSON and learned masks
   under a hash of rules, action descriptions, and observation shape. This
   avoids transferring a policy to a different described game. Drain outstanding
   calls at episode end so a late cold-start compilation can benefit the next
   seed and tracing finishes before the runner closes its sink.

The harness contains no Snake observation keys, directions, board dimensions,
collision logic, target-distance formula, or route solver. Any such semantics
must appear in a traced model response derived from public game information.
The same compilation and classification protocol is used for held-out rules.

## What would confirm or refute it

The primary confirmation is positive mean reward over the same seeds, followed
by held-out-game reward. Useful intermediate evidence is survival beyond tick 6,
successful `policy_installed` events, movement that changes with the current
state while Jev is pending, and fewer late-command failures. Compare cold-start
seed 0 with warm-cache seeds: a gain confined to cached episodes would identify
bootstrap survival as the remaining bottleneck. Trace `mask_learned`,
`selection_received` (including response age), `reactive_action`, compiler and
runtime errors, and safety disagreements to distinguish these outcomes.

The hypothesis is weakened if installed, error-free policies still score zero;
if Jev selects inappropriate strategies despite computed facts; or if reduced
staleness accompanies stationary or reward-free survival. Frequent compiler
failures would instead implicate the policy representation. Incorrect mask
classification or an inadequate safety expression can still cause immediate
death. A syntactically valid but strategically weak policy is not repaired
automatically. Warm-cache performance cannot establish first-episode quality.

Freshness is partly mechanical here because the loop posts every tick. It is
not independent evidence of better decisions. Also, if Astra produces only one
strategy, Jev's strategy choice is trivial; record the number of installed
strategies before attributing an improvement to System-1 classification. This
single candidate changes representation and scheduling together, so successful
evaluation would motivate later ablations rather than establish which component
caused the gain.

## Local validation

No real model calls or Snake/held-out evaluation runs were performed. Syntax,
expression rejection, comprehension scoping, Boolean-mask discovery, and
fresh-state strategy resolution were checked locally. Short counter-game
fixtures with different observation and action names exercised the real runner
and context with fake models. Wall-mode checks covered background installation
and cache reuse. Final virtual checks used 0.5-second ticks, 0.75-second fake
System-1 latency, and 2-second fake System-2 latency: cold, cached, and
malformed-first-response repair fixtures each completed 24 ticks, achieved
positive fixture reward, and had zero stale ticks and no runner errors; they
used 1, 0, and 2 System-2 calls respectively. These validate control plumbing,
not expected game scores or model-generated policy quality.
