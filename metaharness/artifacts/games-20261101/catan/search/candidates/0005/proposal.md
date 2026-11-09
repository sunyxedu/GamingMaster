parent: 0002

## Hypothesis

Candidate 0002 has a useful economic executor, but its two losses show a specific failure: it can remain perfectly legal while never converting a weak opening into a new production source. In seed 102 it ended at true VP 3 after repeatedly rolling and ending turns; in seed 103 it ended at VP 4. The final visible states show the opponent winning immediately after a maritime trade and settlement, while our own recent actions contain no settlement or city conversion. The old executor only paid a route step when the selected project's expected ETA was at most three rounds. That gate turns a low-production or disrupted opening into an end-turn loop.

The structural test is an adaptive **frontier commitment** policy: once a legal road is the verified first step toward an open productive vertex, build it even when the expected resource ETA is long. If the portfolio route disappears, recompute a bounded frontier edge from the live graph. This tests whether changing the executor's commitment horizon matters more than adding another Astra planning call.

## What changed vs. the parent

The portfolio model, exact live-ID execution, maritime-trade search, setup scoring, robber/discard handling, delayed Astra calls, and post-action verification are inherited from 0002. The route executor changed in two ways:

* A selected settlement route is continued whenever its next road is legal; the previous `eta <= 3` condition is removed.
* When a project route is unavailable, code scores current legal road edges by the newly reachable productive settlement sites and commits to the best bounded frontier edge. Every action is still revalidated against the current legal menu and `ctx.progress()`; rejection clears the project and returns to replanning.

This keeps the model responsible for strategic mode/project choices while making the local compiler willing to invest in expansion under adverse production. It does not pursue Longest Road as an independent objective.

## Evidence that would confirm or refute it

Confirmation would be a reduction in VP-2-to-4 losses, more ordinary settlements/cities before termination, and wins on boards where 0002 spent many turns rolling without a conversion. Refutation would be a rise in disconnected or wasted roads, more award-only races, or unchanged losses despite the executor reaching new legal settlement sites. A useful trace signature is whether every paid road has a reachable open endpoint and whether the next few turns contain a settlement/city rather than another end-turn loop.
