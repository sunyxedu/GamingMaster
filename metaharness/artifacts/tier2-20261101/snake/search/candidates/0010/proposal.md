# Candidate 0010 proposal — adaptive Hamiltonian portfolio

## Parent and concrete failure

Parent: `candidates/0008/`, the verified constant-time Hamiltonian controller.
It is the only prior controller that completed every episode without a collision,
but it averaged 121.167 food and always hit `time_limit` at tick 4000. The late
trace makes the failure concrete: in seed 1, the snake reached score 118 at tick
3962 with head `[10,8]` and the next food `[10,11]`, then the fixed-cycle policy
made 38 safe ticks without eating; at tick 4000 the head was `[6,2]`, the food
was still `[10,11]`, and 23 cells remained free. Seed 0 similarly ended with
15 free cells and food `[5,9]` after score 126. Thus the invariant prevents
collisions, but the single chosen cycle's forward order can leave the next food
behind the head and spend the remaining budget making a full lap.

## Hypothesis

A fixed Hamiltonian cycle is safe but unnecessarily one-dimensional. A
**portfolio of independently validated Hamiltonian cycles**, with the cycle
chosen afresh whenever the current body is ordered on it, can retain the same
collision invariant while changing which free cells are ahead of the head. The
new controller generates alternatives by joining verified two-row perimeter
cycles at variable seam columns, not by assuming a diagram is valid. It switches
only to a cycle for which the complete observed head-first body is in strict
cyclic order, and posts only a one-step successor (or one-step shortcut) whose
simulated post-move body remains ordered on that same cycle. If no portfolio
member is compatible, it falls back to an exact-safe action rather than guessing.

This tests a structural throughput hypothesis, not a threshold or model prompt
tweak: the old controller's safety proof is preserved, while the controller
gets more than one legal notion of “forward.” A useful cycle is ranked by the
cycle distance to the current food, with a small hysteresis preference for the
currently active cycle to avoid needless switching. High occupancy does not
relax the invariant; it makes the portfolio more valuable because food behind
the current order is expensive to wait for.

Evidence confirming the hypothesis would be a material increase over 121.167
(mean and especially the late-game score), with zero wall/self collisions and
all ticks fresh. It would be refuted by a seam-generation bug causing a
collision, by no compatible alternative cycles in observed states, or by
unchanged late food waits despite successful switching. Trace events
`cycle_portfolio` and `cycle_switch` expose whether the experiment actually
uses alternatives.

## What changed

* The base serpentine cycles and their reflections/reversals remain validated
  for bounds, uniqueness, and every closing adjacency edge.
* `_strip_cycle` starts with disjoint two-row perimeter cycles and joins
  adjacent strips at variable seam columns. Each resulting degree-two graph is
  accepted only after walking every vertex and checking that it is one
  Hamiltonian cycle. A bounded deterministic seam-pattern family produces a
  deduplicated portfolio; failure to generate a variant merely leaves the
  known base cycle active.
* At each observation, `_cycle_options` tests every portfolio member whose
  current body is ordered. For each candidate direction it applies exact wall,
  reverse, tail-vacating, and growth semantics, then rechecks order on the
  candidate cycle. It favors eating now and the shortest forward cycle distance
  to food; the active cycle receives a small tie preference.
* The loop remains entirely synchronous and bounded: no Astra or Jev request
  can delay a tick. A malformed observation or a portfolio miss uses the exact
  legal-area fallback. Model garbage therefore cannot fail pre-flight.

## System-1/System-2 interface and fallback

This candidate intentionally makes no Astra or Jev calls. The state schema is
none, the question set is none, and there are no model thresholds or
answer-to-action bridge. The public body is exact enough for code to decide
among mechanically verified alternatives; model latency would only recreate
the stale-tick failure seen before candidate 0008. The cycle portfolio is the
code-level analogue of a small set of prevalidated candidates that Jev would
otherwise rank.

The fallback ladder is: (1) a current-cycle successor or shortcut that passes
exact simulation and order validation; (2) a newly selected compatible cycle
successor; (3) the one-step legal move with largest reachable area and food
progress. Every posted action is recomputed from the immediately observed
body, and no stale prediction is reused. A future Astra revision would be
justified only if traces show that the portfolio is too small or that a
particular seam family systematically strands food; Astra should then choose a
verified cycle family, never emit per-cell actions.
