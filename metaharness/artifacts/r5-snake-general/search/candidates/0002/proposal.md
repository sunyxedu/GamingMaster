# 0002 — Compile a controller, lease a strategy

## Hypothesis and archive evidence

Parent: `0000-tier2_jev_only`. Both parents score zero on all three seeds. The
primary failure is architectural: neither has executable control between model
replies. Asking a better next-action question cannot remove that dependency.
My hypothesis is that runtime-generated local control, with asynchronous semantic
review, can survive the opening and earn score while both models are busy.

Concrete evidence from `../0000-tier2_jev_only/trace_seed2.jsonl`:

- The first request observes tick 0, head `[4,5]`, heading `DOWN`, target `[6,6]`.
  Its recorded `latency_s` is **0.6251630783081055**, or 3.13 movement intervals.
  Ticks 1–3 all record `"action": "NOOP", "fresh": false` before it returns.
- The reply selects `NOOP`, probability **0.49**, confidence **0.37**. The next
  request observes tick 3 with head `[4,8]`; its reply again selects `NOOP`
  (probability **0.38**, confidence **0.23**) after another **0.5763468742370605 s**.
  Tick 5 has already occurred. The episode ends with score 0, five steps,
  four stale ticks and only one posted action.
- Thus latency and the decision formulation both fail: an old, weakly supported
  straight-ahead choice is posted without a current-state legality check.

In `../0001-tier2_astra_only/trace_seed2.jsonl`, all five ticks are stale before
Astra returns `DOWN` after **5.627825975418091 s**. There are zero posted actions.
Across all archived traces Jev calls take 0.559–0.625 s and Astra calls take
5.125–5.628 s. These are measurements from the parents, not latency predictions
for this candidate's larger synthesis and batched interpretation requests.

The archived manifest also gives a **900 s wall budget**. A controller that merely
survives or traverses very slowly can still leave substantial completion unearned,
especially on larger boards. This motivates progress-preserving shortcuts when a
runtime policy can justify them, rather than treating low stale ticks as success.

## The structural change

The main thread observes, computes, posts, and waits for the next tick. Inference
runs through at most one outstanding `ctx.think` and one `ctx.decide`, both using
`ctx.spawn`. Posting precedes prompt construction. A late reply never directly
posts an action. Every spawned future is drained before returning, including on
`ModelUnavailable`, which propagates.

There are three control stages:

1. **Opening bridge.** The harness extracts signed one-cell x/y displacements
   from action descriptions. If public prose establishes head-first ordering and
   the observation contains exactly one ordered coordinate chain, it derives the
   current heading and briefly cycles perpendicular/forward/opposite headings in
   two-tick legs. This bounded turning pattern buys time for initial interpretation.
   It is a heuristic, not a safety proof; cramped starts or long chains can defeat
   it. Unsupported interfaces use the declared noop until a controller is ready.
2. **Model-bound geometric reflex.** One batched Jev request identifies coordinate
   extents, the controlled occupied chain and an optional collectible, and checks
   the required movement semantics against public rules. After a confident binding,
   code checks bounds and occupied cells, flood-fills free space for each move,
   and breaks ties by target distance. It conservatively blocks the entire old
   chain; it makes no unverified assumption about tail release or growth. The next
   observed head position checks its motion prediction. This is temporary survival
   assistance, not the final solver.
3. **Astra-compiled policy.** Astra receives the description, action spec, current
   observation, bootstrap binding, recent evidence, remaining budget and full Jev
   usage guide. It writes `policy(observation, memory)` plus helpers. The prompt
   requests exact rules, legal-action checks, current-state invariant validation,
   reliable progress and explicit recovery. Covering traversals are suggested only
   where the discovered geometry supports them. The study's core lesson—put exact
   simulation and bookkeeping in executable code—is applied at runtime, without
   embedding a game-specific completion algorithm in the submitted module.

The generated interface returns candidate strategy IDs, recomputed actions,
descriptions, safety flags, a default, exact facts, goal, preconditions, a regime
key, a Jev question, and optional deterministic next-observation predictions.
Only currently safe candidates with valid action names can execute. Predictions
are compared only across one observed tick; skipped ticks invalidate the comparison.
A contradiction disables the policy and requests repair. Safety flags are produced
by the generated implementation, not an independent formal verifier: incorrect
generated semantics remain an important risk.

Generated modules accept only function definitions, with restricted builtins and
no imports, I/O, private attributes, context access, or exception handlers. Each
invocation has a 25 ms / 60,000 trace-event work guard. This catches ordinary
runaway Python loops; it is not a security sandbox or a hard preemption guarantee
for arbitrary expensive native operations. Invalid text, syntax, schemas or runtime
behavior falls back instead of crashing. Policy installation discards dry-run memory.

## Jev schema, questions and action bridge

Bootstrap state contains `rules`, `actions`, `noop`, `notes`, and the actual
`observation`. Five independent Choice questions identify `extent_x`, `extent_y`,
`trail`, `target`, and whether the combined geometry assumptions are supported.
Field options are enumerated from observed value shapes, never fixed game keys.
Each option explains its actual field and value. Every question includes unknown.
Bindings need selected probability >=0.65 and non-null confidence >=0.2, followed
by structural checks. These are conservative initial thresholds, not calibrated
results. A missing optional target does not prevent survival control.

Policy review state additionally contains `plan_version`, `goal`, `preconditions`,
`facts`, candidate descriptions and actions, the default, and five recent control
records. Two independent Choice questions are batched:

- **strategy:** Astra's focused question, supplemented with instructions to compare
  the supplied strategy evidence instead of recalculating geometry. A supported
  candidate ID changes which *current* candidate is executed.
- **audit:** keep, revise, or unknown. A concrete contradiction or missing progress
  strategy can request Astra revision; ordinary travel is explicitly not failure.

Strategy acceptance requires probability >=0.65 and confidence >=0.2. Audit revision
requires probability >=0.75 and confidence >=0.2. Missing confidence or ambiguity
keeps the local default. A strategy lease lasts at most 20 ticks from the REQUEST,
and requires the same policy version, matching regime key, and a currently safe
candidate. Audit requests undergo the same version/regime/age check. Consequently
a cached semantic preference may remain useful while its directional action changes.
Review runs at most once per 16 ticks; it never gates a tick. Logs distinguish lease
acceptance, actual Jev-caused action changes, invalidation and revision requests.
Worker events measure this candidate's actual inference latency in addition to
the standard model traces.

Astra revises after parsing/execution failure, exact prediction contradiction,
supported Jev audit, or 256 ticks without score change. Requests have a ten-second
minimum spacing; stagnation additionally requires sixty seconds since compilation.
There are at most four compilation attempts. A valid old policy continues during
strategic revision; contradicted or broken policies fall back. The hypothesis is
that revisions should change a reusable interface when evidence refutes it, rather
than ask Astra for another overdue move.

## Generality and limits

The module contains no fixed observation field names, action identifiers, board
sizes, game-name branches, precomputed game route, or stored inter-episode policy.
Rules enter through `game.describe()`, the public spec and observations. The
optional bootstrap supports a class of explicitly described grid/chain controls;
other interfaces go directly to runtime synthesis with a noop fallback. That
fallback may be inadequate for an unfamiliar game with a lethal opening deadline.
The renamed-field test verifies binding mechanics, not held-out-game performance.
No hidden implementation, future randomness, external backend, or other archive
was consulted.

## Evidence that would confirm or refute the hypothesis

Confirm: observed tick-local posting continues during multi-tick model calls;
episodes survive beyond tick 5; the compiled policy installs before the bootstrap
fails; deterministic checks remain consistent; and mean completion exceeds zero
across seeds. Stronger confirmation is substantial completion within the wall
budget, not just a lower stale fraction. `jev_changes_action` should establish
whether semantic review ever improves control; merely counting Jev calls cannot.

Refute or narrow: bootstrap failures precede usable compilation; generated policies
are repeatedly rejected or contradicted; a local loop survives but fails to eat;
stale ticks remain high despite asynchronous inference; or Jev leases cause worse
choices than the certified default. A useful next controlled ablation would keep
compilation and defaults but remove semantic leases. This iteration implements
only the one proposed harness and leaves evaluation to the evaluator.

Validation performed: syntax compilation and short, synthetic stand-in tests for
garbage/empty Astra output, forbidden imports, infinite loops, valid policy
installation, renamed fields/actions, actual Jev-caused action changes, prediction
mismatch invalidation, ModelUnavailable propagation, and joining delayed workers
after termination. These passed. No real model calls, game evaluation, long runs,
or claimed completion measurements were made.
