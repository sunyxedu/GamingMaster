# Candidate 0003 — give the teammate right of way during Astra latency

## Parent and isolated hypothesis

Parent: `candidates/0001-tier2_astra_only/` (the best evaluated candidate).
Mode: **refine**. Retain Astra as the sole decision model, with the parent's
full-observation prompt, session continuity, one primitive per answer, one
outstanding request, default timeout, and tolerant action parser. No Jev,
production planner, task allocator, opening script, recipe policy, or
seed/layout-specific rule is added.

**Hypothesis:** most of the parent's zero-score episodes are stationary-chef
traffic deadlocks during model latency, not evidence that the partner cannot
cook. Replacing the passive wait/stale-action policy with a minimal live
right-of-way shim should recover partner deliveries even when Astra produces
no answer before the 400-tick horizon.

## Evidence from the archive

- Parent `trace_seed2.jsonl`, line 405: the sole S2 call has
  `"latency_s": 66.95045471191406`, `"ok": false`, and
  `"error": "ModelCallCancelled: model call cancelled: game over"`.
  Line 406 reports `"score": 0.0`, `"acts": 0`,
  `"collision_ticks": 198.0`, `"pot_idle_time": 800.0`.
  The final observation leaves our empty-handed chef at `[2,1]` and the
  onion-carrying partner at `[1,1]`. All 400 tick records apply stale `stay`.
- Parent `trace_seed3.jsonl`, line 406: `"collision_ticks": 190.0`,
  `"acts": 0`, and `"score": 0.0`. The final soup is already
  `"is_ready": true`, but our chef remains at `[1,2]`, the only floor
  approach to the dish dispenser. The partner is `[1,1]`, oriented `[0,1]`,
  with `"partner_action_last": "south"`. The teammate finished cooking;
  a blocked dish approach prevents delivery.
- Parent `trace_seed5.jsonl`, line 406: `"collision_ticks": 195.0`,
  `"acts": 0`, and `"score": 0.0`. The randomized-start controlled
  chef is `[1,2]`; the partner is `[1,1]`, facing south into that occupied
  floor cell. This supports a pose-derived rule, not a fixed opening.
- Parent seed 0 is the useful contrast: line 107 records Astra's `north`
  after 17.2095 seconds; tick 104 is the only fresh action. The final score
  is 120, entirely `sparse_agent_1`, with `sparse_agent_0` equal to zero.
  The controlled chef merely moved away from the dish approach.
- The idle baseline also scores 120 on all three asymmetric-advantages
  episodes. This is consistent with preserving a capable partner, rather
  than interpreting a high team score as proof of controlled-agent cooking.
- `lessons.md` is absent from this archive; no additional recorded refutation
  was available. The Jev-only baseline's mean 15.556 does not support replacing
  this parent with raw-grid per-tick model classification.

These are observations and a causal hypothesis, not a claim that the trace
reveals the partner's hidden intended action at every tick.

## The one change: a live waiting policy

The parent's sequential S2 calls run in a `ctx.spawn` future so the main
thread can perform the following narrow waiting policy once per observed tick:

1. Read only the public terrain and the two player poses, using
   `controlled_agent_index` rather than assuming player zero.
2. When the teammate faces our adjacent floor cell, give it right of way.
   Move to a walkable cardinal neighbor that is not the teammate's cell.
   Prefer a side step, then more available exits, with a deterministic tie
   break. Do not interact or choose a production goal.
3. With no such obstruction, post `stay` while Astra is pending.
4. Treat any posted primitive as a one-tick lease. At the next tick re-read
   the observation and renew a necessary yield or replace the action with
   `stay`, rather than accidentally running down a corridor for a whole call.

Because cardinally adjacent grid cells have no common cardinal neighbor,
this egress cannot produce a same-target collision with a one-step teammate
move; excluding its present cell also prevents swaps. If there is no free
exit, the shim stays rather than inventing an illegal move. It does not
presume the partner keeps its last action or follows a known private policy.

The asynchronous placement is the implementation of the waiting-policy
change, not a new planning hierarchy: there is still precisely one sequential
Astra action-selector and no other model. No background Python loop is left
running; the only future holds `ctx.think`, which the runner can cancel.

## Model bridge, freshness, thresholds, and fallbacks

**System 2:** retain `game.describe()`, the advertised actions, and the full
public observation, asking for exactly one action name. Parse with the parent's
`parse_action` helper. Garbage becomes the advertised noop; transient
`TimeoutError`/`RuntimeError` retries on the next tick. `ModelUnavailable`
propagates; budget exhaustion and episode closure end normally.

A reply is not a permanently reusable policy. A movement answer is admissible
only in the same episode and with the same controlled-player pose and held
object as its request; it must not target the teammate's current cell.
An `interact` answer additionally requires the same observation tick, because
an old pot/counter snapshot is not sufficient evidence for an irreversible
interaction. The traffic shim can override an otherwise valid answer.
These guards and single-tick leases are necessary to keep the new fallback
from executing an old action after it has already moved the chef.

**System 1 state schema:** none; `ctx.decide` is deliberately never called.
**System 1 questions / confidence thresholds:** none. Introducing Jev or a new
Astra-authored semantic interface would confound this refine experiment.
The fallback consumes exact derived facts (controlled pose, partner pose and
orientation, floor membership) and uses Boolean geometry, not model confidence.
No meaning is inferred from seed identifiers or layout names.

**Astra revision trigger:** the parent still asks again after each completed
answer (or transient failure), using the new public observation. It does not
author Jev questions or revise a task policy in this candidate. If traffic
clears but production still fails, that would justify a separate later
interface/production-planning hypothesis, not an unreported addition here.

The measured S2 p50/max of 33.1/53 seconds among completed calls, and the
cancelled approximately 67-second first calls, make it essential that the
fallback never waits for an answer. There is no sub-tick S2 timeout assumption.
Jev's measured latency is not relevant to this deliberately S2-only ablation.

## Falsifiable expectations and limitations

Confirm the hypothesis if previously zero-score connected-layout episodes
produce partner deliveries, collision counts fall materially from 188–198,
and the asymmetric-advantages partner scores are preserved. The strongest
confirmation is an episode with no completed S2 call that nonetheless scores
because logged `traffic_yield` events release blocked approaches.

Refute or weaken it if collision counts fall but sparse reward remains zero,
if repeated yielding prevents partner service, or if scoring depends on
Astra doing new production work rather than the evacuation shim. Merely
lowering stale ticks by reposting `stay` does **not** confirm improvement.
A lower Astra-to-action fraction is expected because fresh fallback actions
are now posted; it must not be described as greater model participation.

Expected limitation: this deliberately cannot solve recipes or forced
cross-counter supply on the controlled chef's behalf. No results from
held-out games are inspected, and no claims about their scores are made.

## Validation status

The two required files are written before further refinement. Only short
syntax, invariant, and stand-in-context checks will be run locally; real game
experiments and model latency/score evaluation are left to the evaluator.
