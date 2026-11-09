# Sokoban strategy prior

Sokoban has no opening book or opponent. Strong play is a sequence of safe pushes selected with full knowledge of the static board. The search literature emphasizes reachability, deadlock pruning, and assignment-aware lower bounds ([Sokoban Wiki: Solver](http://sokobano.de/wiki/index.php?title=Solver), [Deadlocks](http://sokobano.de/wiki/index.php?title=Deadlocks), Junghanns & Schaeffer [DOI](https://doi.org/10.1016/S0004-3702(01)00109-6), and the Rolling Stone project [page](https://webdocs.cs.ualberta.ca/~games/Sokoban/)).

## Tactics and ownership

| Tactic | What a strong player does | Put it in |
|---|---|---|
| Parse and bookkeeping | Keep immutable walls/goals, current player coordinate, box set, pushes, and reachable player region. Treat a walk path as cheap only after verifying it does not cross a box needed as a gate. | **Code**: canonical state, flood fill, counters, undo/replay, and deterministic seed mapping. |
| Taboo/dead squares | Precompute non-goal corners and wall pockets where a box can never reach any goal. Extend this with reverse pull reachability: a cell is useful only if a box could be pulled backward from a goal through empty geometry. Never push into a taboo cell. | **Code** for exact masks; **Jev question** can label a proposed destination `safe/taboo/uncertain` for a visual state. Sources: [Deadlocks](http://sokobano.de/wiki/index.php?title=Deadlocks), [How to detect deadlocks](http://sokobano.de/wiki/index.php?title=How_to_detect_deadlocks). |
| Freeze and wall deadlocks | Reject states where a box (or a mutually blocking cluster) is immobile off goals, or where boxes fill a wall segment while goals remain mismatched. Check after every push, including 2x2 freezes and corridor closures. | **Code**: deterministic deadlock predicates and pruning; ask Astra to explain an uncertain cluster. |
| Player reachability | Before considering a push, flood-fill from the player with boxes blocked; a push requires access to the square behind the box. Recompute after each push because one box can open or close a region. | **Code**. Jev may answer a typed `can_player_reach(square)` query, but must not invent a route. |
| Goal assignment | Match boxes to goals using feasible reverse-push distances, not nearest Manhattan distance. Reserve constrained goals and boxes first; detect bipartite conflicts when two boxes compete for one goal. | **Code** for distance matrix/matching lower bound; **Astra** chooses among equivalent assignments and explains ordering. |
| Goal order and corridors | Fill deep goals before shallow goals in one-cell corridors and rooms. Put a box on a goal only when it will not block the player's standing square or the route needed by another box. Keep transit squares and bottleneck goals open until their dependencies are satisfied. | **Astra prompts** for dependency planning; **code** checks resulting reachability and deadlocks. |
| Tunnels and macro-actions | In a one-cell tunnel with no branch, repeated forced walking/pushing can be compressed into a macro, but stop before a goal, junction, or irreversible push. | **Code** for safe forced-walk compression; **Astra** approves macros containing a push. |
| Push-first search | Optimize pushes first, then walking. Use BFS/A* over push states (player reachable region plus box coordinates), transposition tables, deadlock pruning, and a lower bound from box-goal matching. Iterative deepening A* or a well-pruned A* is a practical exact solver for small levels; hard levels may need long search. | **Code** for search and bookkeeping; **Astra** sets a subgoal and replans when search hits a frontier. See [Solver](http://sokobano.de/wiki/index.php?title=Solver) and [Junghanns & Schaeffer](https://doi.org/10.1016/S0004-3702(01)00109-6). |
| Execute conservatively | After planning a push sequence, send one primitive action at a time, verify the returned JSON, and stop/undo on an unexpected transition. A walking typo is recoverable; an irreversible bad push often is not. | **Astra prompts**: request a short next-action plan and a stated invariant; **code** validates action legality and keeps snapshots. |
| Recognition questions | For fast typed judgments, ask Jev independent yes/no or choice questions over the JSON board: “Is the proposed push legal?”, “Is destination a known taboo square?”, “Which of these two boxes is more constrained?”, “Is the level solved?”, “Did the last action push or walk?”. Include candidate coordinates and require `yes/no/uncertain` or an enum. | **Jev questions** only. Jev should not perform multi-push planning, arithmetic, or choose among unenumerated actions. |

## Common mistakes and counters

* **Pushing into a non-goal corner or dead wall end.** A visually close box can become permanently useless. Static taboo masks and reverse pull analysis prevent this.
* **Filling an easy goal too early.** The box may occupy a standing square, seal a corridor, or prevent a different box from reaching a deep goal. Plan goal dependencies first.
* **Using Manhattan distance as the plan.** Walls, push direction, and player access dominate; use legal reverse-push distances and matching.
* **Assuming a box on a goal is always fixed.** Some goals are transit cells; allow moving a solved box only when the assignment and deadlock checks prove it is required.
* **Ignoring the player's side of a box.** A geometrically possible box route fails if the player cannot get behind the box at each turn. Recompute reachable regions after every push.
* **Continuing after an irreversible error.** Keep snapshots and request an undo/restart when an invariant fails. In a benchmark, restarting the same seed is preferable to spending the remaining action budget in a dead state.
* **Confusing walking and pushing.** Count both for the local budget, but optimize pushes first and avoid long exploratory walks when the push state is unchanged.

## Prompt guidance for Astra

Give Astra the complete JSON board plus the static goals/walls and counters. Ask it to (1) identify taboo squares and bottlenecks, (2) assign boxes to goals, (3) state the next push dependency and the required player standing square, and (4) emit a short primitive action sequence whose every push has a legality reason. After each action or macro, provide the new JSON and ask it to compare the observed transition with the prediction. When no safe push is apparent, have Astra revise the assignment or search horizon instead of blindly walking.

## Benchmark implications

Held-out seeds should select different fixed levels, not randomize walls or give an opponent. Report solved levels, boxes on goals, pushes, walks, and actions. A level solved with fewer pushes is stronger, but a solver that leaves a box on a goal while another goal is empty has made no progress toward completion. Preserve the corpus revision and level ID so a strategy result is reproducible.
