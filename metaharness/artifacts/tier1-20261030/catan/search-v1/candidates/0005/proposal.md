parent: 0003

## Hypothesis

Independent placement maximization can buy high pip totals without a functioning economy. Candidate 0003's worst episode, `trace_seed103.jsonl`, places at vertex 17 (wheat/wood/wheat) and vertex 14 (wood/sheep/wood). The starting pair has neither brick nor ore production. It remains at 2 VP through step 46 and finishes at 5 VP after step 125: the terminal observation reports `"winner": "ORANGE"` and `"termination": "lost"`. Steps 62–63 exchange `['WHEAT', 'WHEAT', 'WHEAT', 'WHEAT', 'WOOD']` and then `['WOOD', 'WOOD', 'WOOD', 'WOOD', 'WHEAT']`, as resolved from the preceding legal lists. A weak starting resource portfolio forces the later greedy controller into costly conversions.

This candidate tests **joint opening portfolio search**, rather than adjusting the leader's normal-turn priorities. Each first settlement is valued together with several legal, distance-compatible second placements; the second placement is scored against the actual existing settlement. Resource production, ports, expected time to useful build bundles, and plausible expansion access determine portfolio value. The hypothesis is that a coherent initial economy improves the worst seeds even with the same inexpensive normal-turn executor.

## What changed versus the parent

The setup policy searches pairs rather than choosing each vertex's independent pip maximum. It discounts a first pick whose complementary second pick has no good substitutes. Setup roads target reachable settlement sites rather than simply the adjacent endpoint's production. All planned actions are resolved against the current legal list; no queued action ID is replayed after a step.

Normal-turn action priorities remain those of 0003 to keep the experiment focused on opening planning. The controller makes no model calls, does not inspect engine state, does not consume the engine RNG, and uses only the public observation. It catches action failures and rereads the current observation.

## Confirmation and refutation

Confirmation: fewer opening portfolios missing both brick and ore without a compensating efficient port, earlier third VP/city/first development card, fewer bank conversions before the first expansion, and improved minimum terminal VP without losing the leader's rapid episode completion. Log the first and second opening portfolio selections so later iterations can inspect predicted build delays. Refutation: coherent portfolios lose too much raw production, the preferred second locations vanish during the snake, or the remaining greedy trade/road policy still dominates losses. The 8 training seeds are too few to establish broad superiority; dev outcomes are exploratory only.

## Implementation and limits

For each compatible pair, sum dice-weighted production by resource and divide by 9 to obtain expected cards per four-roll round. A bounded binary search estimates when that mean production could fund (1) a road plus settlement, (2) a city, and (3) a development card, with surplus exchanged at observed specialized port rates. Rank the resulting production portfolios, including the three best second-pick substitutes and a nearby expansion opportunity. This is a **continuous production surrogate**, not an exact expected waiting time or a simulator: it ignores variance, discards, temporary robber blocks, bank depletion, and competing construction. Initial resource coverage is only a small tie-breaker.

The observed vertex field uses `port: null` both at ordinary vertices and at generic-port locations; generic ports are only unambiguous once owned (`self.ports: [null]`). The opening planner therefore grants a 2:1 rate only for an explicitly identified specialized port and otherwise assumes 4:1. It does not infer unobserved port locations. A portfolio missing brick and ore can still be coherent when a matching 2:1 port monetizes strong production; missing-resource counts alone are insufficient to judge it.

The parent's previous-ID alternate-choice guard is bypassed during setup so it cannot override the search winner merely because an unrelated prior action had the same revision-local ID. The ordinary-turn executor, including its known weaknesses, is retained. No queued ID is replayed after a step.

## Validation performed

- Required pre-flight: passed, `game_done`, 10 VP, 97 controlled actions, zero System-1/System-2 calls. Its no-System-2 warning is intentional.
- Opening choices were replayed against all eight archived training opening observations only to verify current-list legality and inspect computed portfolios; no training games or full evaluation were run.
- One permitted dev-eval invocation, outputs in `dev/20261030-062504/`: seed 301 won with 10 VP in 68 controlled actions; seed 302 won with 11 VP in 87 controlled actions. Mean dev score 10.5, zero model calls, no errors, about 4 seconds each. Digests and traces were inspected.
- Both dev episodes selected the forecast first/second pair: `[9,23]` on 301 and `[19,23]` on 302. Seed 301's third settlement appeared at step 22; seed 302 took Largest Army at step 29 and finished by taking Longest Road at step 87. These are positive functional checks, not a paired estimate of improvement over 0003. No second dev evaluation was needed.
