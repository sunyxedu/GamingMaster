# Candidate 0011 proposal — bounded endgame food rescue

## Parent and evidence

Parent: `candidates/0008/`. I kept its model-free, validated Hamiltonian-cycle
controller, including the same cycle construction, order check, exact
transition, one-step shortcut ranking, and area fallback. The targeted change
is only an endgame exception for food that the cycle proof rejects.

The worst parent trace gives a concrete reason to test this. In
`trace_seed1.jsonl`, the observation at tick 3962 says:

```
head=[10, 8], direction=UP, food=[10, 11],
safe_directions_now=["RIGHT", "LEFT"], score=118, free_cells=23
```

The next tick posted `LEFT`; ticks 3963 through 4000 continued around the
cycle without changing the score. The final observation still had
`food=[10,11]`, `score=118`, `free_cells=23`, and
`terminal_reason="time_limit"`. Under the public transition rules, the exact
three-step route `RIGHT, RIGHT, RIGHT` from that tick is collision-free and
eats that food. The parent refused its first move because it breaks the fixed
cycle-order proof, even though the immediate move is listed safe. This is
conservative survival behavior, but it leaves a reachable point on the table
when only 38 ticks remain.

The same pattern is visible in the parent’s high-scoring seed 0 endpoint:
score 126, 15 free cells, a still-present food at `[5,9]`, and a time-limit
ending. These traces motivate a narrow endgame test rather than changing the
successful early-game invariant.

## Hypothesis

The cycle-order invariant is the right default, but it is too conservative in
the last part of a fixed episode: a short, exactly collision-free route can
reach food before `max_ticks` even when its first move is not compatible with
the selected cycle. Spending a finite amount of safety margin on that route
will increase food score, while keeping the parent’s zero-collision behavior
for almost the entire run.

This should be confirmed by a higher mean score and harness trace events
`endgame_food_rescue`, with the normal cycle selected and no increase in stale
ticks. It is refuted if the rescue rarely reaches food, produces early
self/wall collisions, or leaves the mean unchanged while shortening survival.
A rescue after which the snake later collides is still useful evidence only if
it earns a strictly higher score; score remains primary, while survival is the
final tie-break diagnostic.

## One targeted change in `harness.py`

* The normal policy is unchanged until the final `ENDGAME_WINDOW = 80` ticks.
* In that window, code runs a bounded breadth-first search over complete body
  states, up to 1800 states and at most the remaining ticks. Every edge uses
  the public exact transition: reverse requests are omitted, non-food moves
  vacate the tail, and food moves retain the tail and grow. No model is asked
  to do arithmetic or path planning.
* A route is used only if its first move is legal but fails the currently
  active cycle-order check. A route that still satisfies the parent’s proof is
  left to the parent controller, isolating the hypothesis that the proof is
  overly conservative rather than changing ordinary shortcut behavior.
* The chosen route is attached to the observed food and to the predicted full
  body/heading. Each next action is revalidated against the fresh observation;
  a mismatch, food change, malformed route, or illegal move discards it and
  immediately returns to the cycle/area fallback. If no route is found, the
  search is retried only every eight ticks, so a failed search cannot occupy
  the real-time tick path repeatedly.
* The controller reads `max_ticks` when present, then the public `max_steps`
  or game attribute; the documented profile values are a defensive fallback
  because the archived parent observations omit that field. No seed-specific
  food, body, or action is embedded.

The search is synchronous but small; it is the only added work and is confined
to the final 80 ticks. The harness still posts one symbolic action per loop
and never waits for Astra or Jev.

## System-1 and System-2 interface

This candidate intentionally makes **zero Astra and zero Jev calls**, as did
0008. The System-1 state schema is none, the question set is none, and there
is no answer-to-action bridge or confidence threshold. Exact state parsing,
body simulation, route search, freshness, and the cycle invariant are more
reliable in code than in a typed judgment, and a Jev response would be slower
than several game ticks.

The fallback hierarchy is: (1) an already validated endgame route when its
freshness checks pass; (2) the parent’s cycle-preserving action or a newly
validated reflected cycle; (3) the parent’s exact one-step reachable-area
fallback; and finally a currently advertised heading/action for malformed
stand-in state. A future Astra revision would be justified only if traces
show that the finite rescue is unsafe or that a different, code-verifiable
cycle/entry invariant is needed; it should not be inserted into the 0.10 s
control path.
