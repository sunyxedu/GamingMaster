# Candidate 0007 — checkpointed search with semantic arbitration

## Parents and hypothesis

Parents: `candidates/0003/` (best candidate, descended from the Jev-only controller) and `candidates/0005/` (the odometric/semantic crossover).

The common failure is now more specific than “the model needs a better plan.” Both parents can make useful progress, but neither has a persistent *route-search state*. Candidate 0003's watchdog breaks a fixed-coordinate action loop, yet it often returns to the same exposed lane. In its seed-1008 trace, after the first recovery the player reaches `(-208,-271)` and then spends long runs on `forward`/`forward_left` without displacement; later it oscillates around `(273..304,240)` while health falls from 90 to 3. Candidate 0005 measures motion and has a better combat/collection interface, but its direct steering to visible landmarks is too open-loop: seed 1001 posted **161 `turn_left` and 127 `turn_right`** actions, only 54 forward ticks, ended at `(581,62)` with health 1 and zero kills/items; seed 1008 reached 8 items and 4 kills but spent **2246/2075** ticks turning left/right and still died without an exit.

**Hypothesis:** the missing mechanism is a map-agnostic, finite search procedure that treats failed movement as a checkpoint and changes the route branch, rather than repeatedly asking Jev to choose a button or steering toward a far visible pickup. A bounded wall-following/search branch with measured odometry and explicit USE tests should turn the partial progress of both parents into actual exit attempts. It should preserve 0003's safe held macros and 0005's code-level aiming/resource bookkeeping, while preventing a model or a stale landmark from owning the motor indefinitely.

The test is deliberately structural: the controller does not memorize maps, coordinates, or seeds. It only uses rendered labels, the player's public pose/angle, counters, and measured motion.

## What changed

1. **Finite motor checkpoints.** Actions are posted as bounded macros. A straight segment is held long enough to traverse an ordinary corridor, but 8 tics of negligible displacement mark a measured collision. A collision creates a queue: one USE pulse (if legal and not recently tried), a roughly quarter-turn, and a forward test. If that branch does not leave the cell, a second opposite-side branch and then a short observed-trail backtrack are tried. The queue is finite and alternates/records heading attempts; it cannot become an unbounded turn-in-place loop.
2. **Search rather than pickup chasing.** Rendered keys, weapons, and urgently needed health/armor are candidates only when they are nearby, same-level, and recently visible. A goal is abandoned and deferred after 1.8 seconds without distance progress. Far labels and labels above/below the player no longer pull the controller into the 0005 steering oscillation. Once a goal is rejected, bounded forward/search segments resume.
3. **Interaction and exit behavior.** A currently rendered door, switch, or exit is aligned with a finite turn and receives a one-tic USE pulse. Unseen exits are not invented; collision checkpoints also provide conservative door/exit probes. The exit objective therefore has an executable path even though the native observation does not expose a map graph or exit label in the archived starts.
4. **Measured combat safety.** Threats are recognized only from the public `Monster` category and current labels, with dead/corpse names excluded. Code ranks close/active/hitscan/urgent evidence, aims from current world bearings, switches only to an owned/ammo-bearing slot when needed, fires while strafing, dodges close projectiles, and retreats to a breadcrumb when health/damage makes exposure unsafe. This retains 0005's code-level combat bridge without letting a semantic vote target a stale ID.
5. **Two model speeds with narrow authority.** Jev receives a compact derived state and votes on a short-lived intention, target, side, USE suggestion, and retreat suggestion. Two reversed mode Choices must agree and meet a probability threshold; otherwise code searches or fights by measured rules. Astra runs in the background at most twice and returns only a four-field policy (`wall_side`, combat posture, near/moderate resource radius, and door-probe preference). Invalid, late, or slow output leaves the finite motor unchanged. A policy revision invalidates semantic votes, not safety checks.

## Evidence and falsification

The parent traces support both sides of the design. 0003 demonstrates that a position watchdog can change position, but its recovery is only a short nudge and the model can immediately reselect the old direction. 0005 demonstrates that stale-tick reduction alone is not effectiveness: it had near-zero stale ticks yet its motor repeatedly reversed turn direction, while its best episode collected substantial resources without activating an exit. The new logs distinguish `checkpoint_recovery`, `visible_use`, `jev_vote`, and `astra_policy` events.

The hypothesis is supported if traces show: (a) recovery branches followed by new cells or a changed view rather than the same coordinate, (b) fewer long alternating turn runs and fewer multi-second zero-displacement forward runs, (c) combat occurring while health remains materially above the one-hit floor, and especially (d) a nonzero `completed_by_exit` rate or longer safe search after a key/door interaction. It is refuted if finite branches repeatedly revisit the same cells without widening measured coverage, if near-only goal selection misses useful capability and causes early death, or if combat still consumes health while semantic votes never alter a valid local decision. A lower stale-tick percentage by itself is not confirmation.

## Jev interface and bridge (tier 2)

Each request is a strict JSON object, rebuilt from the latest observation:

- `rules`: exit objective and the actor-only/visible-label limitation;
- `policy`: the accepted Astra fields;
- semantic health, armor, selected-ammo and recent-damage buckets;
- player cell, facing sector, moving flag;
- up to four visible monster records: stable label id, name/category, range bucket, side, height, active-motion evidence, hitscanner and urgent-name flags;
- up to two projectile records;
- up to four nearby resource records with id/category/name, range/side/height and a code-computed `approachable_now` flag;
- `motion`: last action, blocked flag, stalled seconds, visited-cell count, known-trail retreat, and recovery count;
- bounded kill/item/secret counts and time since count progress;
- the current semantic vote, if it has not expired.

Code computes distances, bearings, buckets, legality, collision evidence and ammo facts. Jev is not asked to count, infer a wall, generate an action string, or plan a map.

Questions, all in one request:

| ID | Type | Use |
|---|---|---|
| `mode_a`, `mode_b` | Choice over `advance`, `fight`, `evade`, `interact`, `search`, `hold` | The reversed pair counters option-order bias. Only agreement with chosen-option probability at least **0.56** is adopted, for 1.6 seconds. |
| `side` | Choice: `left`, `right`, `straight` | May bias the next recovery/dodge only when the code has measured a block; it never directly selects a button. |
| `target` | Choice over current visible threat IDs plus `none` | Accepted only for an ID still visible in the current observation; otherwise code's threat ranking wins. |
| `retreat` | Noul | Accepted at probability/value **>= 0.72**, but code still forces retreat for critical health, recent large damage, or a close projectile. |
| `use` | Noul | Accepted at **>= 0.72**, but code only pulses USE for a currently close interaction or measured collision. |

A Jev result is tagged with tic and policy version. Answers older than 2.4 seconds, answers from a superseded Astra policy, malformed typed objects, low-confidence Choices, and disagreement are discarded. One request is spawned at a time; normal pacing is about 0.42 seconds in danger and 0.85 seconds in search, with a 2.6-second per-call timeout. While a call is in flight, the current finite macro continues. `TimeoutError`/ordinary `RuntimeError` logs a fallback and leaves code in charge; budget, episode, and model-unavailable exceptions follow the harness contract.

## Astra interface revision

Astra is given the compact state plus the last ten measured history entries, not a screenshot or hidden map. It is asked for only:

```json
{"wall_side":"left|right", "combat":"cautious|normal|aggressive", "resource_radius":"near|moderate", "probe_doors":true}
```

The first review is spawned while the initial finite search is running; a second is eligible only after repeated measured collision recovery and a long separation. The parser accepts only those enumerated fields and ignores prose/garbage. Astra cannot issue a raw action or a coordinate route. `near` is the default specifically to test the parent-0005 far-landmark failure; a revision may widen it only from evidence. No Astra output can suppress emergency retreat, legality checks, collision checkpoints, or terminal detection.

## Late-decision fallback and robustness

- The initial action is the advertised `noop`; the first finite search macro starts without waiting for a model.
- Between decisions the last legal macro remains held. It is refreshed only at macro expiry, while close damage/projectiles and measured blocks interrupt immediately.
- USE and weapon selection are one-tic pulses; cooldowns prevent pulse spam. Movement and fire are never left to an unbounded model response.
- If System-1 is unavailable, the same deterministic checkpoint/search/combat controller runs without model calls. If model outputs are arbitrary stand-in text or missing fields, all parsers fall back to legal code choices.
- Background work contains only one bounded inference call per future and stops naturally when `ctx.done`; no custom worker loop survives `run_episode`.
