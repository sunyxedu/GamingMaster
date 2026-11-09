# Candidate 0026 — deadline-checked physical transactions

## Parent and concrete diagnosis

**Requested parent: 0010. Strongest evaluated control: 0017.** This is a
structural rewrite of the scheduler and model bridge, not a threshold change.
I retain public-observation recipe bookkeeping, explicit cooking starts,
interaction poses, BFS execution, and nonblocking inference. I remove the
parent's pot-ownership tokens and persistent time-expanded route reservations.

Evidence inspected in the archive:

* `0010/trace_seed8.jsonl`: the last `ownership_job` is `fill_pot [4,1]` at
  **tick 83**, then **every action at ticks 84–400 is `stay`**. The episode ends
  at 20 points with the controlled chef holding an onion at `[1,3]` and its
  partner holding soup at `[1,2]`; pot-idle time is 683. A physically legal task
  and an apparently valid route/ownership claim did not establish progress.
* `0017/trace_seed0.jsonl`: `fetch_dish` at tick 27 becomes `wait_dish` at 30;
  **ticks 32–45 are all `stay`**. Long dish waits recur at 70–83, 113–126,
  153–164, and 192–203. The final state has nine deliveries/180 points, a
  controlled-held dish at `[2,2]`, a partner-held onion at `[3,1]`, and a pot
  cooking at 15/20. Actions were fresh: this is not model latency.
* `0017/trace_seed3.jsonl`: counter retrievals at **215, 256, 298** lead to
  counter deposits at **225, 268, 310**, rather than useful fills. Ingredients
  can arrive after the partner has already filled the pot.
* `0023/trace_seed0.jsonl`: peripheral dish-wait targets repeatedly alternate
  between `[1,1]` and `[1,2]`; the final score is still 180. I also inspected
  0022's implementation and traces. The lessons already refute unconditional
  parking, role locks, and assuming that ingredient staging alone fixes the
  cramped-room bottleneck.

These observations support preserving production invariants while replacing
indefinite logical claims with phase deadlines and observed progress.

## Hypothesis, including the refuted first attempt

The initial hypothesis was that putting a dish on a counter during cooking
would free the chef's hand for useful next-cycle work. **Unconditional dish
buffering was refuted by local trials.** The first version scored 140 on seed
0. After fixing wait preemption, the next trial scored 80/320/20 on seeds
0/1/2. Seed 0 repeatedly recommitted retrieval of a dish at `[3,0]` behind the
partner; seed 2 parked on the partner's route while its small back-and-forth
movements falsely counted as continuing progress. Counter ownership had
recreated the failure of pot ownership.

The final, narrower hypothesis is:

> Acquire and approach with a dish according to observed cooking slack. Release
> an occupied hand only for an existing useful production operation. No waiting
> or buffered-object commitment may suppress a newly available productive task,
> and movement counts as progress only when it beats the previous best distance.

This is intentionally not another fixed parking-coordinate change. A buffered
dish is replaceable by a fresh dispenser dish. Counters are a recovery/optional
production mechanism, not a mandatory extra stage of every soup.

## Final structure and division of labour

### Code: physical obligations and a continuously executable motor

* Reconstruct accepted recipes, compatible ingredients, pot phases, observed
  cooking remainder, counters, held objects, and interaction preconditions.
  Walkability is exactly the observed space cells. Numeric decoders reject
  semantic strings such as `cook` and `start`.
* Generate only object-compatible medium jobs. Held soup is delivered; a held
  dish picks up ready soup; an empty hand starts a full idle pot explicitly.
  Compatible pot filling and ordinary supply retain deterministic fallbacks.
* **Time the dish transaction:** compute conservative travel/turn/interaction
  estimates to get a dish and reach the pot. A held dish begins its final
  `approach_ready` shortly before readiness. The motor still refuses a pickup
  interaction until the current observation actually says ready; it never
  predicts the last cooking increment early.
* **Condition hand release:** stash a redundant/premature dish to free a hand
  for a full idle pot or needed production. Optional release for a partial pot
  requires a complete estimated detour to fit within observed cook slack.
  One-item lookahead additionally requires a physically staged dish, available
  counter space, no already-staged or partner-held ingredient, and enough time.
* **Preempt waiting immediately:** cooking completion, a newly empty pot,
  explicit-start work, or a changed hand invalidates the old logical wait.
  Jobs also have bounded ages; the primitive motor checks the current partner
  cell and a short movement prediction on every step.
* **Check real progress:** a partner handoff is reset by a new target/held
  object, and renewed by a new best route distance—not by oscillating between
  previously visited cells. A partner-held dish can cover at most one ready
  soup. Other pots remain independent work.
* **Check the parking destination itself:** a zero-length route does not make
  a floor cell safe if the partner's inferred route passes through it. Route
  corridors constrain parking, not all productive movement.
* **Keep alternatives:** a buffered dish and dispenser dishes are generated
  together. A retrieval requiring a path through the partner is ranked behind
  a clear dispenser route, preventing a counter-specific indefinite claim.

### Astra: bounded counter protocol, not motor commands

One asynchronous opening request receives observed terrain, player/pot facts,
stable geometry-derived counter IDs, free counters, and current legal jobs.
It returns a closed JSON object:

```json
{"dish_buffer":"buffer_x_y or none","lookahead":"one or none","protocol":"counter or handheld"}
```

Code validates the ID against the observed geometry. Buffer preference breaks
close route-cost ties; it cannot force an occupied/unreachable counter.
`handheld` disables optional counter-release work, while mandatory hand release
needed to resume production remains legal. `lookahead` controls the optional
one-item buffer. The accepted protocol is explicitly included in subsequent
Jev states and in their freshness signature.

The interface is deliberately fixed and code-owned, rather than asking slow
Astra to generate Python or arbitrary questions at runtime. Given the archive's
30–67 second typical Astra latencies in a roughly 67-second game, this version
uses **one opening review, not repeated in-episode interface rewrites**. A future
revision should be triggered by repeated counter retrieval without pickup or
repeated stale handoff trust—not merely by another task boundary. The local
refutation above already drove that interface revision during development.

### Jev: short-lived operating mode and handoff assessment

Jev does not choose primitive directions, invent coordinates, count ingredients,
or calculate arrival times. It judges whether to continue direct production,
use a physically offered buffer job, yield briefly to a supported handoff, or
rebuild an actually stalled commitment. Every answer is constrained by the
latest code-generated jobs and interaction checks.

## Exact System-1 interface and answer bridge

Each request is strict JSON containing:

| Field | Contents |
|---|---|
| `self` | Position and held object |
| `partner` | Position, held object, inferred intent/target, `near/far/unknown` distance bucket, and `unknown/at_goal/stalled/advancing` progress label; raw counts are diagnostics |
| `pots` | Position, phase, missing ingredients, and `ready/soon/later/none` time bucket |
| `buffer` | Remembered dish counter, observed dish presence, free counters, optional lookahead policy |
| `current` | Job kind, age, blocked count |
| `work` | Code-computed direct-work/buffer availability, route-stalled flag, offered job kinds |
| `astra_protocol` | Accepted protocol, lookahead setting, preferred buffer |
| Other | Tick, public layout label, corridor cells, request trigger |

One request batches three independent questions:

1. `mode_a`: Choice over `finish`, `buffer`, `yield`, `rebuild`, `unknown`.
2. `mode_b`: identical instructions and meanings, reversed option order.
3. `handoff`: Noul assessing the carried object, target, progress evidence, and
   pot phase together; waiting at a cooking target can be legitimate, but
   oscillation is not route progress.

A mode is accepted only when both choices agree and **both selected
probabilities are at least 0.52**. `unknown`, absent distributions, malformed
values, and disagreements leave code in control. Confidence is not substituted
for winning probability. A handoff answer must be finite and in `[0,1]`;
**≥0.75** is positive, **≤0.25** negative, otherwise undecided. A missing answer
is not interpreted as a negative answer.

Modes apply small ranking preferences to existing jobs. `rebuild` can invalidate
an observed blocked job earlier, but cannot invent a replacement action.
Handoff advice can shorten a verified stale-dish grace period from six to three
ticks, or modestly extend it to seven; current held-object and route checks
remain mandatory. `jev_handoff_apply` and `jev_rebuild_apply` log actual bridge
effects separately from merely receiving a model label.

Requests have one outstanding future, an **18-tick minimum interval**, and a
52-tick heartbeat, with a **1.0-second timeout**. Replies are tagged by hand
contents, pot phases/ingredients, physical dish-buffer presence, and Astra's
protocol. A changed signature or age over **12 ticks** discards the reply.
Accepted modes last at most ten ticks; handoff labels at most eight. Astra's
90-second timeout is asynchronous and its call is cancellable at game end.

The always-live fallback is exact production plus current-state BFS/motor
checks. `ctx.act` posts one advertised primitive before `ctx.wait_for_tick`.
Timeouts, arbitrary stand-in text, unknown options, and cancelled calls cannot
turn a wait for a model into stale action ticks. There are no background Python
loops: spawned work consists only of bounded context model calls.

## Confirmation, refutation, and limits

Confirmation of the narrowed hypothesis requires more sparse deliveries on the
cramped cases, a final approach beginning before readiness but a pickup only
after readiness, and removal of persistent ready-yield or blocked-counter
retrieval tails. Independent multi-pot production must remain active.

The hypothesis is refuted by extra counter operations lengthening the critical
path, a parking job surviving a phase transition, or renewed handoff trust based
only on back-and-forth motion. Counter-buffer benefits specifically require
observed useful consumption; merely logging `stash_dish` is not confirmation.

The final changes were refined together, not independently ablated. Any score
increase supports the combined scheduler, **not a causal claim that dish
buffering alone improved throughput**. Generalization to forced handoffs, mixed
recipes, reversed control, and held-out randomized starts remains untested.
No held-out games/results or game implementation source were inspected.

## Validation

`check_harness.py` uses **237 recorded training observations** from 0010/0017,
plus public-rule mutations. It checks legal primitive output, explicit starts,
ready-only pickup, immediate phase preemption, buffered-dish replacement,
nonrenewing route oscillations, missing/malformed model outputs, protocol
freshness, and full control-loop completion on three recorded streams with
stand-in models. This is a regression check, not a simulator score estimate or
the evaluator's official preflight. Compilation and all these checks passed.

The narrowed scheduler's preliminary live Jev-only runs scored
**200/360/200/200/360/240/200/360/220** on seeds 0–8, all at 400 ticks without
errors or stale ticks. A subsequent frozen-source confirmation is recorded
below. These are local training tries, not the official selection evaluation.

An additional live Astra+Jev seed-0 check scored **200** at 400 ticks, with zero
stale ticks, no failed calls, and no pending tasks. Astra answered in **7.38 s**
and selected `handheld`, `lookahead=none`; its policy was accepted at tick 45.
That verifies the live bridge, not a causal Astra score gain.

### Frozen-source confirmation

The final sweep used live Jev with Astra off, as in the default `try` command.
Every episode completed all 400 game ticks; the two-minute local wall limit
never cut a game short.

| Seed | Sparse score | Trace directory |
|---|---:|---|
| 0 | 200 | `tries/seed0-221907/` |
| 1 | 360 | `tries/seed1-222112/` |
| 2 | 200 | `tries/seed2-221907/` |
| 3 | 200 | `tries/seed3-222015/` |
| 4 | 360 | `tries/seed4-222015/` |
| 5 | 240 | `tries/seed5-222016/` |
| 6 | 200 | `tries/seed6-222123/` |
| 7 | 360 | `tries/seed7-222123/` |
| 8 | 220 | `tries/seed8-222124/` |

**Mean: 260.000**, with **0 errors**, **0/3,600 stale ticks**, **183 Jev
calls**, and **0 failed calls**. Across these calls, Jev latency was **0.178 s
median / 0.403 s maximum**. Mean diagnostics were 13 deliveries, 21.11 collision
ticks, 41.00 stuck ticks, and 204.11 pot-idle ticks. Only sparse reward enters
the score.

This is +108.889 over parent 0010's archived mean and +4.444 over 0017's archived
255.556, but is a **local, Astra-off comparison**, not a new official leaderboard
result or a statistically established gain. All three cramped seeds gained a
delivery relative to 0017; asymmetric total reward was unchanged, while ring
seed 2 lost one delivery. The remaining trade-off is real, not hidden by the mean.

A separate **frozen-source Astra+Jev seed-0 check** in
`tries/seed0-222112/` also scored 200 with zero errors/stale ticks, 21 Jev calls,
one successful Astra call, and no pending tasks. Astra answered in **8.32 s**;
its `counter`, `lookahead=none` protocol was accepted at tick 50. Together with
the earlier live check, this validates malformed/late/live policy handling,
not an Astra contribution to the score improvement.

The final sweep logged 71 accepted mode labels, 51 accepted handoff labels,
and **seven changes to the handoff-trust gate**; there were no Jev-triggered
rebuilds. These are not seven proven changed primitive actions or a model
ablation. Code remains the dominant executor, and the score gain is not
attributed to model reasoning alone.

`local_validation.json` contains the full per-episode metrics, provenance, and
source SHA-256:
`c477287e54b5e936989ff3c37c4285615cdad1f5657b56a327d6eb5084964aaa`.
One concurrent pair of CLI tries chose the same second-resolution seed-1 output
directory; that interleaved log (`tries/seed1-221907/`) was excluded and seed 1
was rerun independently for the table above. No ambiguous trace contributes to
these statistics. No other candidate was modified.
