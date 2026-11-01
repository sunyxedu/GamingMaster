# Candidate 0008 proposal — make the safety invariant the whole controller

## Parent and concrete failure

Parent: `candidates/0005/` (with the same study-derived exact Snake rules, but a
very different control architecture).

The best prior candidate scored 6.0 on the six training seeds, yet every run
ended in a wall collision.  Its traces show why the safety shield did not fully
solve the real-time problem.  In seed 0, for example, the controller observed
`tick=30`, head `[5, 11]`, direction `RIGHT`, and
`safe_directions_now=["UP", "DOWN"]` immediately after eating its third food.
The next trace event was tick 31 with `action="RIGHT", fresh=false`, followed
by a wall collision at `[5, 12]`.  There was no new action between those two
observations: the synchronous `_food_plan`/fallback computation was still on
the control thread while the game clock advanced.  The same shape appears in
all six final traces: the last action is a boundary-facing heading and the
next tick repeats it.  The exact transition code was correct, but it could not
post its correction before the deadline.

Candidate 0005 already contained a Hamiltonian-cycle helper, but only invoked
it after occupancy reached one half of the board.  That condition is reached
too late to protect the expensive per-tick search, and the cycle never became
the primary policy in these short runs.

## Hypothesis

On the specified reference game, a validated Hamiltonian cycle should be used
from the first tick, not as a late fallback.  If the initial body is in
head-to-tail order on that cycle, then the next cycle edge is a constant-time,
collision-free baseline.  A local grid neighbor may also skip ahead toward the
visible food, but only when exact simulation leaves the resulting body in the
same cycle order.  This retains the safety guarantee while reducing the
roughly 5,000 expected ticks of blindly waiting for uniformly placed food.
Every food remains reachable under the invariant, and filling the board is the
terminal success condition.  The score objective therefore favors removing
A*/BFS and model latency from the tick path entirely over asking a model to
choose among equivalent safe turns.

This is a structural test of **invariant-first control**, not a threshold tweak.
The harness validates a rectangular cycle, checks the complete observed body
order, selects an orientation/reflection compatible with the current heading,
and then evaluates only the three exact one-step transitions.  A candidate
shortcut is accepted only if the post-move body still has a non-wrapping
head-to-tail cycle order; otherwise the controller takes the ordinary cycle
edge.  A cycle may have gaps between body cells, but it may not wrap the
occupied span or reverse into the current heading.  If a state ever violates
that invariant, the harness falls back to an exact one-step legal-area policy
rather than blindly trusting the cycle.

Evidence that would confirm the hypothesis is near-complete-board score (or at
least a large increase over 6), no self/wall collisions while the cycle is
active, and essentially no stale ticks despite zero model calls.  It would be
refuted by an early collision from a cycle-order bug, a low score despite a
valid cycle, or initial openings that cannot enter any generated/reflected
cycle.  Such a result would mean that this build's opening or transition
semantics differ from the public contract, and would justify a separately
validated entry planner.

## What changed in `harness.py`

* `_even_row_cycle` constructs a Hamiltonian cycle for an even row count and
  any column count.  Its transpose covers the case where only the column count
  is even.  Every candidate is checked for in-bounds cells, uniqueness, and
  orthogonal adjacency including the closing edge.
* Reflections and both traversal orientations are tried.  The complete
  head-first body is checked against the cycle's one-dimensional order.  The
  current heading must be able to accept the next cycle direction; an
  immediate reverse is never posted.
* Once selected, the cycle is retained and checked on each fresh observation.
  Each of the at most three non-reverse directions is simulated once; a
  cycle-order check rejects unsafe shortcuts, and the best forward food
  distance wins.  This remains a bounded local operation followed by
  `ctx.act`, with no synchronous Astra or Jev call and no unbounded search
  between ticks.  It directly addresses the 0005 trace where a correct action
  was computed too late.
* For odd-by-odd boards or malformed/non-cycle states, a small exact fallback
  applies tail-vacating and growth collision rules for each one-step candidate,
  then ranks only legal moves by food-now, reachable area, food distance, and
  heading retention.  It is a recovery path, not an assumption that a
  Hamiltonian cycle exists.

## Astra and Jev interface decision

This candidate deliberately has an empty model interface.  No Astra prompt is
issued, no Jev state is sent, and no answer is allowed to stand between an
observation and its next action.  That is the experiment: on a fully specified
12x12 board with an exact symbolic body, model reasoning adds latency but no
information needed by the cycle invariant.  It also avoids spending the
System-1 budget on judgments whose candidate set is already mechanically
ordered.

Consequently the Jev state schema is **none**, the question set is **none**,
and there are no answer thresholds, freshness checks, or answer-to-action
bridges in the active policy.  Freshness is instead enforced at the game
boundary: each loop observes the current body, validates the retained cycle
against that observation, and posts only a direction whose exact next cell is on the validated cycle
(or a one-step shortcut whose post-move body passes the same order check).  The
fallback has the same freshness property because it is recomputed from the
observation immediately before posting.

A future Astra revision would be appropriate only if traces show repeated
cycle-entry failures on a declared board shape, an obstacle/wrap rule not
present in the public contract, or a body-order violation caused by an
external controller.  Astra could then design a new *code-verifiable*
cycle/entry invariant.  It should not be called to output per-cell actions.
Likewise, Jev would become useful only on a board with no validated cycle,
and then should receive a compact set of code-legal fallback candidates (food
safe / tail accessible / area bucket) and choose among them, never discover
coordinates.  Neither contingency is enabled here because it would reintroduce
the failure mode being tested.

## Fallback and limits

The main policy does not claim that a Hamiltonian cycle exists on every
rectangle: the generator returns no cycle on odd-by-odd boards.  It also does
not force a cycle after an observed invariant break.  In those cases it uses
exact bounds, reverse rejection, simultaneous tail-vacating on non-food moves,
and retained tail on food moves, all implemented in code.  A malformed model
answer cannot crash pre-flight because this candidate makes no model calls at
all.
