# Candidate 0004: audited lookahead over a model-compiled transition system

Parent: `0003` (mean score 18.667). One structural hypothesis: the successful
reactive compiler now needs a representation of **multi-step consequences**.
A local exit count cannot reliably distinguish an escape corridor from a trap.
Keep the parent's small reactive compilation and asynchronous control loop;
add a separately compiled, empirically checked transition model and generic
bounded search. No game-specific solver is embedded in the harness.

## Diagnosis and archive evidence

I read the archived harnesses, proposals, evaluations and model/tick traces,
concentrating on `0003` seed 1, its lowest-scoring and shortest episode.

The previous delivery problem is resolved. `0002` has nine failed compiler
calls across its three episodes, each timing out at approximately 40 seconds,
and scores 1, 2, 3 despite surviving all 300 ticks. `0003/trace_seed0.jsonl`
line **184** installs its policy at **frame 60**, following a **29.6323-second**
System-2 call. Its output is only **973 characters**, with **one mode**. Cached
seeds need no new System-2 calls. `0003/eval.json` reports scores **19, 16, 21**,
zero stale ticks, no runner errors, and **self-collision in all three episodes**.
This supports changing strategic representation rather than another timeout fix.

The installed policy at seed 0 line 184 explicitly scores actions with:

```text
-x['distance']-20*int(x['exits']==0)+0.1*x['exits']
```

Its exit feature counts adjacent cells outside the **current** occupied body;
it does not update occupancy through future transitions. Its safety predicate
checks the immediately effective direction only. The sole Jev criterion is
`"m0": "True"`, so Jev cannot provide a strategic alternative.

Concrete evidence from `0003/trace_seed1.jsonl` (one-based JSONL lines):

- Line **409** records tick **137**, score **16.0**. Lines **410–418** then
  record four consecutive down actions/advances without further reward.
- Line **423** contains the decision facts from frame **140**. Left has
  `"pos": [7, 6], "distance": 6, "exits": 1, "utility": {"m0": -5.9}`;
  right has `[9, 6]` and the **same distance, exit count and utility**.
  Line **420** chooses left from `"allowed": ["left", "right"]`.
- Lines **422** and **426** have `"allowed": ["up"]`. Line **430** exposes
  the latter decision's only candidate: `"pos": [7, 4]`, `"exits": 0`,
  `"utility": {"m0": -28.0}`. The dead-end penalty recognizes the problem
  only after the controller has no alternative within its safety filter.
- Line **428** logs `"safety_disagreement_or_no_safe_action"` at frame 143.
  Line **431** is tick **144**, with `"fresh": true`; line **433** confirms
  `"death": "self"`. All **144** ticks were fresh and all **144** frames
  used the compiled policy.

Seeds 0 and 2 show the same final pattern: an increasingly constrained action
set, then no safe action and a fresh collision (seed 0 lines **788–797**;
seed 2 lines **583–591**). This evidence identifies a limitation of the
available decision features. It does **not** establish that right at frame 140
would have saved seed 1, or where its last recoverable decision occurred. The
new candidate tests for earlier distinguishable continuations; it does not
hard-code a correction to that particular trace.

## What changes

**Two-stage synthesis preserves availability.** The first Astra prompt and
reactive expression contract remain those of `0003`. Once that policy works,
a second background call asks only for a compact one-step transition model,
not search code. It receives public rules, action descriptions, a current
observation, the working reactive policy, and three recent observed transitions.
Both stages retain a 90-second timeout. Each has at most two attempts per
episode, checking the remaining budget; normally cold start takes two calls,
and a cached episode takes none. The existing Astra session is reused.

The additional JSON contains `encode`, ordered state/action features, `safe`,
`next`, `reward`, `value`, and `project` expressions. `encode` creates simulated
state. `next` updates all modeled dynamic components. `reward` represents
actual immediate reward; `value` is a bounded progress potential. `project`
selects predictable physical state for checking against observations. All use
the parent's restricted AST evaluator. New model JSON has a 12,000-character
ceiling, at most eight features per list, and the same expression syntax/size
limits. No generated imports, functions or unrestricted Python execution.

The compiler is instructed to distinguish simulated dynamic state from the
frozen root observation, resolve ignored/default actions, update resources and
occupancy where relevant, and avoid inventing future random events. A known
objective must pay only once: rollout bookkeeping records its consumption,
then predicts survival without inventing the next objective. A fresh observation
starts a new search. Astra can explicitly return `unsupported`, leaving the
reactive controller active; that decision is cached for these rules.

**Check the model before trusting it.** Validate syntax and multi-step execution,
and replay up to 16 recently observed one-tick transitions before installation.
Compare projected predicted state with projected encoded actual state. Continue
this audit after each unambiguous tick, and require at least two matched
transitions before search affects actions. Cached models must earn those two
matches again. Tick identity comes from the runner's public `game.ticks`
metadata, never from a hard-coded observation key. Intervals spanning multiple
ticks are skipped. If that metadata is absent, this implementation conservatively
continues reactive play without enabling unaudited search.

A model parse/runtime failure or transition mismatch disables lookahead
immediately, preserves the working reactive policy, and requests the one
available model repair with the rejected output and concrete error. Neither
one-step agreement nor the model's own projection proves correctness: an
incomplete projection or an unobserved action effect can evade these checks.

**Search continuations, not model-written routes.** Every current safe first
action gets its own beam, retaining at most ten distinct simulated states per
completed depth. Search reaches at most 14 transitions, with a 6,000-successor
work threshold and a wall-time allowance of `min(60 ms, 0.12 * tick_s)`.
Only complete equal-depth layers replace the previous result, so running out of
time halfway through a layer does not favor whichever root was expanded first.
These checks occur between node expansions, so one expensive expression can
still exceed the allowance; this is not a hard execution sandbox.

First prefer actions with the longest witnessed surviving continuation. Among
those, select predicted discounted reward plus bounded leaf potential, with the
reactive utility breaking ties. Rewards are discounted by 0.97 per transition;
leaf potential is clamped to [-0.5, 0.5]. Deduplicate successor states within
each root's beam. Replan from the real observation every tick; execute only the
first action. A beam that runs out of continuations is absence of a witness,
not proof of inevitable loss. Search shorter than three completed layers, or
without a usable root, leaves the reactive choice in control.

**Give Jev a meaningful, latency-tolerant choice.** Before model readiness the
parent's bootstrap/mode protocol remains. During lookahead Jev sees the rule
and model summaries, derived facts, per-root search depth/return/beam width,
completed-search statistics, audit count, surviving candidate set and reactive
utilities. A typed `Choice` selects between two reusable scoring rules:

- `plan`: use predicted return among the longest surviving continuations.
- `progress`: use the reactive reward utility within that **same** continuation
  set when modeled returns are tied or omit important unpredictable rewards.

Default to `plan`. Jev selects a scoring rule, never an old action or fixed
route. The current safety filter and all search facts are recomputed even if
its reply is pending. Replies from old revisions or older than four tick
intervals are ignored. Astra never blocks action selection; `ctx.act` and
`ctx.wait_for_tick` preserve the parent's clock discipline. Outstanding calls
are drained at episode end and useful late compilations can enter the cache.

The new versioned cache key includes the supplied rules/action specification
and observation shape. The harness embeds no game names, observation-field
names, directional geometry, board sizes, target-distance formula, collision
rules or game-specific search. It reads no archive, game implementation,
held-out source, filesystem artifact or external service during evaluation.
Game semantics must arrive through traced model responses to public inputs.

## Confirmation, refutation and limitations

Primary confirmation requires mean reward above **18.667** on the same seeds,
with maintained freshness and held-out-game performance. Separate cold and
cached episodes: synthesis can finish too late to improve the first seed.
Normal successful synthesis costs one extra System-2 call versus the parent.

Inspect `world_requested`, `world_installed`, `world_audit`, `world_rejected`,
`lookahead`, and `policy_coverage` events. Lookahead logs its reactive alternative,
selected action, survival candidates, search depths/returns, elapsed time and
whether search actually controlled that frame. Supporting evidence would be
validated models with substantial search coverage, choices that reject shorter
continuations before the live safety set collapses, and later collisions or
more reward. Freshness alone is mechanical and does not establish intelligence.

The hypothesis is weakened if accurate, actively used models and meaningful
search depths still yield similar traps/reward, or if longer survival merely
trades away too much reward. Frequent model mismatches implicate synthesis or
observation ambiguity; repeated timeouts implicate the extra compiler contract;
shallow deadline-limited searches implicate computational cost. Such failures
would leave the central lookahead hypothesis under-tested, rather than show
that a reliable multi-step model cannot help. A held-out regression refutes
practical generality even though the implementation contains no fixed game
semantics. Unpredictable environments may legitimately stay in reactive mode.

Finite horizon and beam pruning can miss delayed traps and valuable detours.
A mistaken reward expression can distort search even when physical projections
match. The inherited restricted expression evaluator does not preempt expensive
nested comprehensions. The new synthesis, auditing, search and two-way Jev
interface form one candidate; improvements would motivate later ablations to
separate their contributions.

## Local validation

No real model calls, Snake evaluations, held-out evaluations or long experiments
were run. Syntax/import and forbidden-expression rejection were checked.
A small graph fixture gave both first actions immediate safety, but one branch
ran out after three transitions while the other reached depth 14; search ranked
the longer continuation above the tempting immediate reward.

Eight short synthetic dial-game episodes exercised the actual runner/context
with fake models and different observation/action names. All completed 24 ticks
without runner errors or stale ticks. Seven used virtual time; one used a
0.03-second wall-clock tick. The checks covered:

- Cold and cached runs: two and zero System-2 calls, with 15 and 22 search frames.
- Changed rules: a new compilation, not reuse of an incompatible cache entry.
- Incorrect transition model: rejection and successful repair with three calls.
- Three-second-old classifier replies: expired while local search kept acting.
- Explicit unsupported-model response: uninterrupted reactive play.
- A deliberately surprising live transition: immediate model rejection and
  continued reactive play when the same incorrect repair failed validation.
- Wall-clock asynchronous compilation, audits and search: 19 search frames.

These checks establish control plumbing and the intended trap distinction in a
synthetic model, not expected game scores or Astra's future synthesis quality.
