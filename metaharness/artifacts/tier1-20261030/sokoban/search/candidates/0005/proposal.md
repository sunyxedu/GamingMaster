parent: none

# Hypothesis: backward certificates and the real score objective

The best candidate has no unsolved training episodes to repair. Its remaining
weakness is an abstraction mismatch: it minimizes pushes on player-component
states and uses walking as a tie-break. That abstraction preserves push
reachability, but it does not preserve action cost, and mutable component
parents can reconstruct a different walking route from the one costed.
Candidate 0001's worst episode, `trace_seed107.jsonl`, ends with
`"pushes":20,"walk_steps":25,"total_actions":45`, score 10389.55, and zero
System-2 calls. Seed 101 finishes with 14 pushes and 44 actions. The trace
score changes verify a walk costs 0.01 and a push costs 0.51 including its
action, so the search objective should be 50 * pushes + total_actions.

This is a fresh design, not a refinement of 0001's forward push-A* recipe.
It tests whether meeting a forward action search with a backward search from
solved states gives exact, inexpensive solution certificates and lower total
score cost. Inverse pushes are entirely internal planning operations; the
live game receives only the four public action strings.

# Concrete archive evidence and execution failure modes

* Candidate 0002, seed 100, posts a local-search prefix, then discards the
  certified suffix at its first push and hands control to Astra. It finishes
  with only `"pushes":5,"actions":15,"boxes_on_goals":1` after 1800 seconds.
  A verified plan should survive every expected push.
* Candidate 0003, seed 103, ends with `"pushes":3,"walk_steps":1997` and
  `"action_cap_terminated":1.0`, score -21.5. Fallback walking creates no
  progress and actively destroys the score.
* The latest archive reports 227 seconds for the slowest successful Astra
  call. This candidate uses 360-second timeouts and catches every failed
  call; no short calibration timeout or uncaught TimeoutError is retained.

# What changes

1. Dense floor IDs and box bitmasks encode `(all_boxes, exact_player)`.
   Forward edges are primitive walks and pushes. Backward edges are their
   exact inverses, rooted at goals filled and every legal player square.
2. Two static assignment bounds produce a balanced bidirectional potential:
   matching distance to goals and matching inverse distance to initial boxes.
   They guide search without erasing the player's actual location. The
   frontier lower bounds provide an explicit optimality certificate.
3. Every frontier join reconstructs and replays a complete solution before
   it becomes an incumbent. An anytime incumbent survives a time/node limit.
   Execution predicts and checks each returned player/box transition, keeping
   the verified suffix through pushes and abandoning it only on a mismatch.
4. Astra is consulted only if bounded local search has no solution. Its strict
   compact answer is an untrusted full-solution witness, not an executable
   partial policy. The parser accepts explicit ACTIONS or a JSON array and
   rejects prose. Only a fully replayed solved sequence is accepted.
5. When no certificate can be obtained, one finite, replayed partial path is
   allowed only if it improves the observed goal/cost score. The harness then
   returns. It never loops on arbitrary legal movement or drains the cap.

No seed, level ID, corpus file, game implementation, external solver, or
cross-episode cache is consulted. Geometry and initial states come solely
from the live observation. Standard-library computation and ctx model calls
are the only runtime dependencies.

# Confirmation and refutation

Confirm if all ten training puzzles still solve without errors or Astra,
with equal or lower 50 * pushes + actions, especially the leader's weak
episodes, and trace logs certify optimal search termination. Refute if the
primitive state graph saturates before obtaining a witness, if backward
states fail independent forward replay, or if the exact objective yields no
score improvement despite extra search time. Dev seeds test generalization
and interface behavior; archived observations are useful only for replay
and solver regression, not for stored answers or seed-dependent policy.

# Validation

Preflight passed (`game_done`, score 10395.25, 25 actions, no errors).
On archived seed-start observations 100-109, all ten plans replay to solved
and the search proves an optimum within 0.24 s per level. The new integer
costs versus candidate 0001 are: 475=475, 742<744, 686=686, 786=786,
475=475, 729<731, 688=688, 1044<1045, 489<492, and 526<527.
Across these starts this saves nine score-cost units, equivalent to +0.009
mean score at the traced score weights. No per-seed answers are embedded in
the harness; each search started from the archived public observation.

One dev evaluation solved both independent seeds with no calls, errors, or
timeouts: seed 300 scored 10395.30 (9 pushes, 20 actions) and seed 301 scored
10393.09 (13 pushes, 41 actions). An independent primitive Dijkstra on 50
small synthetic boards agreed with the bidirectional cost and proof result,
including unsolvable cases. Parser probes reject garbage and prose while
accepting marked UDLR or a strict JSON actions array. The preflight warning
that no System-2 answer was sampled reflects the local solver finishing the
stand-in level before any fallback call.
