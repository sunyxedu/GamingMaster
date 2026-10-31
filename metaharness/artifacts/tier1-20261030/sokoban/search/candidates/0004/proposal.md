parent: 0001

## Hypothesis

The deterministic solver is the right structural idea, but the parent still ranks frontier
states with Manhattan distance and a 24,000-state/12-second budget. That heuristic treats a box
as close to a goal even when walls make the push direction impossible, so it can spend the
search budget on misleading assignments. Candidate 0003 shows the cost of falling back to
unguided control: seed 103 alternated `left`/`right` through step 2,000 and ended at -21.5,
while seed 106 made 99 actions and timed out. The leader solved all ten seeds, so the safer
test is to retain its local-first architecture and make the search ordering more geometry-aware.

## What changed vs. the parent

I compute a reverse-push BFS distance map for every goal. The A* lower bound is now an exact
minimum bipartite matching over those maps, including unreachable box/goal pairs as a large
cost. The search budget is expanded to 60,000 states/20 seconds, still far below the 1,800
second episode deadline. The existing reachability checks, useful-cell guard, compact Astra
fallback, transition validation, timeout catches, and garbage-output fallback remain in place.
The fallback now prefers a changed, unseen legal state and gives a safe push a small priority,
which prevents repeated two-action oscillations if local search ever hands off.

## Evidence that would confirm or refute it

Confirmation would be equal or higher solved/boxes-on-goals scores than 0001 with fewer pushes
or actions on the same seeds, and traces showing the local solver completing before any Astra
call. A refutation would be a score drop on simple levels, a search timeout on levels the parent
solved quickly, or a transition mismatch; those would indicate that reverse-push distances or
the larger frontier are too expensive or that the canonical player-region key is over-pruning.
