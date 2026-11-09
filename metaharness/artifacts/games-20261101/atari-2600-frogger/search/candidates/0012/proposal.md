# 0012 — phase-diversifying retries for deterministic Frogger failures

Parent: `0007` (as requested), with the evaluated route-search and sprite-copy geometry from `0006`/`0010` rebuilt rather than copied. This candidate is deliberately a structural experiment: it adds a temporal retry controller, not another global clearance constant or semantic action lease.

## Hypothesis

The residual deaths are partly **phase-locked failures**. A route can be geometrically valid at the current RAM snapshot yet fail repeatedly because a carrier/home approach is sampled at an unlucky phase. The clearest evidence is 0010 seed 109: four consecutive deaths at ticks 4270, 4900, 5552, and 6192 all occurred at row 10 while pressing `UP` toward `HOME_0`, with the same `x≈11–12`, zero reported route margin, and the same bay still open. The same controller later succeeded with the same nominal home route at tick 7144, when the traffic/carrier phase had changed. Thus the 0010 contact book correctly diagnosed “entry” but only narrowed the x window; it did not change the temporal phase and retried the same failure four times.

A second concrete witness is 0010 seed 109 tick 12022: the median death happened after a long partial route whose forecast margin was `0.75`, while the observation still marked the median `safe_pause=true`. The controller’s later bank diagnosis stopped some unsafe dwelling, but only after the life was lost. A retry policy should treat death as evidence about the next attempt’s start phase and should make the post-death behavior different, while remaining bounded so it cannot turn into an idle controller.

I expect a bounded wait on the start sidewalk after a river/home failure, plus temporary preference for a different open bay, to reduce repeated losses more than a universal `river_pad` increase. The waiting cost is small relative to losing a life and can expose a different moving-gap phase. It is not a fixed seed route: the delay is selected from failure count/type and current observations, and all subsequent routes are rebuilt from live tracks.

## What changed

* Retained code-owned, neutral-rearmed pulses and a time-expanded beam search over separated NUSIZ copies. Only the first action of a route is posted; no stale multi-hop script is executed.
* Added an episode-local failure automaton. On each death it classifies the last public state into `HOME_ENTRY`, `RIVER`, `MEDIAN`, or `ROAD`, increments a failure/target record, schedules a bounded sidewalk phase delay (30–144 frames), and clears the old route/press. The delay grows for repeated failures to the same home but is capped.
* Temporarily quarantines a failed bay only when other open bays exist, with expiry. If it is the sole bay, the route remains possible and temporal diversity is used instead of declaring the unknown crocodile unsafe.
* For median deaths, suppresses long median `NOOP` dwelling for 240 frames; it still permits traversal and immediate departure. This addresses the 0010 bank witness without making the median unreachable.
* Keeps local carrier-form contact bans and the 163/170 stretched-tail width rule, but does not apply a global pad increase after every failure. Home-entry failures narrow only that bay’s entry window; river failures can add a shape-local band.
* Re-observes and independently checks the first action immediately before posting it. Malformed/late model results never affect legality or crash the loop.
* Astra selects only bounded enums (`route`, `risk`, `tempo`), including whether future repeated failures justify a larger phase shift. It cannot choose an action. Jev selects a durable verified route goal and a retry preference; it cannot override the live guard.

## System-1 interface and bridge

Every request sends a strict compact JSON state containing:

* `phase`, frog row/x, timer urgency, current durable goal, and code-generated route candidates;
* each candidate’s label, whether it is complete/verified, short vs long travel, comfortable vs tight clearance, and current-goal flag;
* `retry.recent_failure`, repeated failure count for the same target, whether a phase shift is already scheduled, and whether another open-home route is present;
* explicit rules that code performs arithmetic, collision geometry, freshness, and legal-action checking; unknown home crocodile status is not treated as safe.

The batched questions are:

1. `goal`: Choice among the code-generated home/median goals plus `UNKNOWN`.
2. `goal_reverse`: the same Choice with reversed option order, to reduce first-option bias.
3. `retry`: Choice among `KEEP_PHASE`, `SHIFT_PHASE`, `USE_ALTERNATE`, and `UNKNOWN`.
4. `retry_reverse`: the reordered retry Choice.
5. `replay_evidence`: Noul asking only whether the supplied retry facts explicitly establish a repeated timing/home-entry failure.

A goal is adopted only if both Choices agree, the selected option is in the request’s candidate set, both probabilities are at least `0.62`, and the answer is no more than 120 ticks old with unchanged epoch, phase, and revision. It lasts at most 180 ticks and only biases near-best routes. `SHIFT_PHASE` is accepted only on the same agreement/freshness gate and only while the frog is on the sidewalk; the code still sets the mandatory post-death delay. `USE_ALTERNATE` is advisory logging because the route finder itself removes an unexpired failed bay when an alternate exists. Any missing answer, garbage stand-in answer, disagreement, low probability, timeout, or stale result is ignored.

The posted action bridge is deterministic: wait during a scheduled sidewalk delay; otherwise search live geometry, select the best current route (with only near-tie durable-goal/Astra preferences), run a fresh one-hop safety guard, post one legal action, hold it for at most three frames until motion is observed, then post `NOOP` for recovery. If search or the live guard finds no safe action, post `NOOP` and retry; never post an invalid model string.

## Astra scheduling and revision

Astra runs asynchronously after the first failure and at most once per 3000 game ticks, or sooner only if the evaluator’s normal budget permits the already-scheduled request. It receives recent failure records, successful entry samples, open-bay occupancy, level and policy. It must return only JSON enums. A policy is applied only if its level still matches; application increments the interface revision and clears the durable goal. `tempo=shift_small|shift_large` changes only the bounded next-frog delay. Repeated localized failures are evidence for temporal revision; a single failure is not enough to inflate all margins. Astra is never on the critical frame path.

## Confirmation / refutation

Confirmation would be: fewer repeated same-target HOME_ENTRY deaths, fewer deaths per five-home cycle, higher homes and score than 0010/0006, and successful attempts after a prior failure at visibly different tick/carrier phase, without large `no_route` or sidewalk-time losses. The phase mechanism should help especially in traces like seed 109 while preserving the 0010 ability to reach levels 4–5.

Refutation would be unchanged repeated deaths at the same target despite the scheduled delay, a large increase in timer deaths/no-route stalls, or a broad score drop from quarantining too many bays. Stale-tick reduction alone is not evidence: the relevant measurements are deaths, homes, repeated-target retries, and score/time bonus.

The module is defensive against arbitrary Astra text and stand-in Jev answers, imports only the standard library plus `metaharness.*`, and always returns from `run_episode` only when `ctx.done` is true.
