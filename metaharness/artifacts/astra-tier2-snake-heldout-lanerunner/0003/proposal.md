# Candidate 0003: make the reactive compiler finish

Parent: `0002`. This is a focused repair of the parent's compilation path, not
a new game solver. The hypothesis is that the parent never tested its intended
architecture because its compiler deadline and synthesis scope were mismatched.
A smaller policy that actually installs should outperform indefinite bootstrap
classification while preserving the parent's successful real-time scheduling.

## Diagnosis and trace evidence

I read the three prior harnesses, proposals and evaluations, the baseline traces,
and the parent's traces, concentrating on its worst episode, seed 0.

The crucial distinction is between runner success and compiler success.
`0002/eval.json` reports no runner errors, scores **1, 2, 3**, 300 ticks per
episode, and **zero stale ticks**. However, across its three traces there are
**nine unsuccessful System-2 calls, nine compiler errors, zero
`policy_installed` events, and 900 actions with `"source": "bootstrap"`**.
The successful-System-2 latency summary has `"n": 0`; this does not mean no
System-2 calls were attempted.

Concrete evidence from `../0002/trace_seed0.jsonl` (one-based JSONL lines):

* Lines **244, 500, 758** record the three System-2 calls with `"ok": false`,
  `"response": null`, and latencies **40.0144, 40.0087, 40.0087 seconds**.
  Lines **246, 503, 760** log `"event": "compiler_error"` and an error
  starting `"TimeoutExpired: Command ..."`, at episode clocks **40.5052,
  83.0117, 125.5104**. The parent explicitly passes `timeout_s=40.0`.
  Each retry asks for the same substantial feature/strategy compilation; no
  partial policy survives the timeout.
* Line **7** logs `"polarity": "unsafe_true", "probability": 1.0` for the
  discovered Boolean action mask. Line **10** records
  `"tick": 3, "action": "up", "score": 1.0`. Reward never increases again
  during the remaining **297 ticks**. Thus the mask and tick loop maintain
  survival, but the intended reward policy is absent.
* Line **894**, a System-1 request from observation tick **294**, shows
  `"head": [9, 5]`, `"food": [9, 3]`, `"heading": "left"`, and
  `"currently_permitted": ["up", "down", "left"]`. Its answer is
  `"choice": "a4"` (keep), with probability **0.47**; left gets **0.43**
  and up **0.06**. The fallback filters out keep and follows the remaining
  probabilities, favoring left despite the target being directly above.
  Across this episode **106 of 150** answers select a4. This is evidence of
  a weak fallback, not evidence that the uninstalled compiled policy failed.

The other parent seeds show the same three 40-second failures each. The older
baselines support retaining asynchronous execution: all their episodes end at
tick 6 with wall collisions. In `0001-tier2_astra_only/trace_seed2.jsonl`, ticks
1-6 are all `"fresh": false`; line 8 returns `"text": "up"` after
**8.8178 seconds**, after the game has ended. In contrast, the parent survives
every tick without waiting for a model reply.

The traces establish timeout and absence of installation. They do **not** expose
Astra's unfinished reasoning or prove that prompt complexity caused the delay;
backend overhead or availability could also explain it. That uncertainty is
what this intervention tests.

## Intervention

Keep the parent's expression interpreter, safety-mask bootstrap, asynchronous
Jev/Astra workers, fresh-observation action resolution, and cache design.
Change the compiler contract and recovery policy:

1. **Ask for a minimal useful policy.** The prompt requests a direct JSON answer
   without tools, research, tests, or prose. It targets **2,500 characters**, at
   most **eight global and eight per-action features**, and **one or two modes**,
   instead of allowing 24 features in each list and three modes. Begin with
   one reward-seeking mode. A second mode and a cheap escape cue are optional;
   search, route planning, and elaborate simulation are explicitly out of scope.
   Send one current observation instead of a three-observation history. The
   validator enforces the feature/mode limits and an **8,000-character** policy
   ceiling; 2,500 characters is a prompt target, not a parser requirement.
2. **Allow useful background work to finish.** Increase the compiler timeout
   from **40 to 90 seconds**. Continue posting actions every tick while it runs.
   Allow at most **two attempts per episode**, respecting remaining context
   budgets. A valid cached policy needs no new call. This spends less on repeated
   restarts but can spend longer waiting for a single result.
3. **Make the one repair simpler and informative.** After a timeout, parse,
   validation, or runtime failure, ask for just one mode, immediate safety and
   simple reward progress, targeting 1,800 characters. Include a rejected policy
   when available and preserve useful parser/runtime errors. Summarize
   `TimeoutExpired` rather than returning its command string, which embeds the
   original prompt and obscures the actual error. Retain the session if a
   response supplied one. There is no periodic strategic replan for an otherwise
   valid policy: this experiment tests compiler delivery first.
4. **Validate and measure installation.** Evaluate returned expressions on the
   initial, recent, and current observations before installing. Log request
   attempt, prompt size and timeout; installation frame, output size, mode count
   and whether the episode is still active; failure type; and total compiled
   versus bootstrap frames. A compilation finishing after terminal state can
   still enter the cache, but is explicitly distinguishable from active play.

The model interface remains explicit. Astra supplies ordered feature
expressions, an action-safety predicate, mode applicability criteria, and numeric
utilities. Each tick evaluates these against the current observation. Jev sees
the summary, computed global facts, safe candidates' per-action facts and
utilities, and the current mode. A typed `Choice` asks which mode's observable
criterion applies. Its answer chooses a reusable mode; the controller maximizes
that mode's utility over actions safe **now**. If there is only one mode, Jev's
compiled-stage choice is trivial, and any gain should not be attributed to
nontrivial System-1 strategy selection.

There is at most one pending call per model. New Jev requests launch when its
previous call finishes; local safety and action resolution still run every tick.
Replies from an earlier policy revision or older than **four tick intervals**
are ignored for action/mode selection. While inference is pending, reuse an
unexpired mode, otherwise mode 0. Before installation, retain the parent's raw
observation question, opaque action IDs, learned Boolean-mask polarity, expiring
probability ranking, and freshly applied masks. `ctx.wait_for_tick()` paces the
loop. No late model reply directly posts an old action.

## Generality and limitations

The module contains no game names, coordinate rules, observation-key constants,
action-name constants, target-distance formula, board size, or prewritten game
policy. Semantics must come from `game.describe()`, the public action
specification, observations, and traced model responses. The cache is keyed by
rules, action names and observation shape, with a new candidate-version prefix;
policies are revalidated before reuse. No game implementation or held-out game
source was used to design the controller.

Simplification trades strategic depth for availability. A greedy policy may
collect more reward but eventually enter a trap. Immediate safety expressions
can be wrong even when they parse; the learned mask is also model-derived. A
90-second failure still leaves this candidate in essentially the same fallback
as its parent. A second late call may only help a later episode, and draining
pending calls can extend wall time after play. The inherited restricted
expression evaluator limits syntax and sizes but is not a process sandbox or
a strict execution-time bound on nested comprehensions.

## Confirmation and refutation

The primary outcome remains mean reward, with freshness and System-2 calls as
tie-breakers. Compare against **2.0 mean reward**, and separately report cold
and cached episodes and held-out-game performance.

* The delivery hypothesis is supported if System-2 succeeds and installs during
  play, compiled-frame coverage becomes substantial, and later episodes reuse
  the policy with fewer calls. Continued near-zero stale ticks verifies that
  the scheduling behavior was preserved.
* The full hypothesis requires reward above the parent's result as well as
  successful delivery. A syntactically valid policy with high active coverage
  and no reward improvement points to inadequate utilities or mode selection,
  rather than a compiler timeout. Inspect the returned policy before proposing
  a richer planning representation.
* Repeated 90-second timeouts refute this compiler-delivery fix. Successful
  responses dominated by parse/runtime errors implicate the expression
  contract. Policies installed only after terminal state do not establish
  cold-start quality. Gains only in warm episodes identify startup as the
  remaining bottleneck. Loss of held-out performance would undermine the
  claimed reusable architecture.

This is one candidate changing compiler scope and its deadline together. A
positive result cannot separate the contribution of shorter synthesis from
the larger timeout; that would require later ablations. Freshness alone is
not evidence of decision quality because posting every tick is mechanical.

## Local validation performed

No real model calls, Snake evaluations, or held-out evaluations were run.
Python parsing/import and expression rejection were checked. Short synthetic
dial-control fixtures, with different observation/action names and true-means-safe
masks, exercised the actual runner/context with fake models. All ten fixture
episodes completed their specified ticks with no runner error and zero stale
ticks:

* Cold and cached 24-tick virtual episodes used **1 and 0** System-2 calls and
  **20 and 24** compiled frames. A changed rules description triggered a new
  compilation rather than reusing the cached entry.
* Malformed-policy and timeout fixtures each repaired with exactly **2** calls
  and then produced positive fixture reward. Repair feedback preserved the
  Boolean-validation error and summarized the timeout without its command.
* A **45-second simulated compilation** installed during a 120-tick episode,
  producing **30 compiled frames** without stale ticks. This checks the
  asynchronous path beyond the parent's deadline; it does not predict real
  Astra completion time.
* Three-second-old Jev mode replies expired while fresh policy actions
  continued. Compilation after a four-tick episode was marked inactive and
  successfully reused in the next episode.
* A short wall-clock episode also completed all 24 ticks, used 20 compiled
  frames, and earned positive fixture reward while both fake models ran in
  background workers.

These checks validate control plumbing and repair/cache behavior, not the
quality or latency of policies produced by the evaluator's models.
