# Candidate 0009 proposal

Parent candidates: `0007/` and `0008/`.

## Hypothesis

The high score comes from 0007's verified named-placement executor, while the
queue beam proposed by 0008 is useful only if planning cannot replay a previous
hard drop.  The concrete 0008 failure is a timing failure: its trace starts with
hard drops at ticks 8 and 9 while a fresh beam is being computed, and the first
nine locks build a stack with four holes.  In contrast, 0007 posts a `noop` at a
piece boundary and never records an illegal action, but its trace still has
stale noop runs (ticks 1--4 in seed 4, and repeated two-to-three tick runs) while
`_plans` evaluates a future placement for every first candidate.  The hypothesis
is that a **staged, bounded queue lookahead** can retain 0007's line/height
quality while shortening the piece-boundary computation: rank all legal first
placements immediately, evaluate the next one or two pieces only for a small
shortlist, and always replace an old drop with a noop before doing that work.

This tests the structural part of 0008 without repeating its unsafe execution
boundary.  The code owns all geometry, scoring proxies, legality checks, and
finesse.  Jev remains an optional, non-blocking tie-breaker among the shortlist;
its answer can never be required for an action.  Astra is not called: the prior
Astra calls were slow and never preceded an action, and zero System-2 calls is a
better test of the fast controller than an unobserved policy review.

## Changes from the parents

* Keep 0007's SRS-shaped placement simulation, hold bookkeeping, verified
  one-action-per-tick executor, same-piece freshness checks, and hard-drop
  fallback.
* Replace its expensive all-first-candidates future evaluation with a staged
  search.  Immediate metrics rank every legal current/hold destination; only
  the strongest diverse first candidates receive bounded queue continuation,
  and the second continuation is expanded only for a few top placements.  A
  small hold penalty and a large hole/top-out penalty remain in the code.
* Keep the boundary `noop`: after a piece count changes, post it before search
  and re-observe before selecting a target.  No search or model call may sit
  behind a pending hard drop.  Use the deterministic top plan if Jev times
  out, disagrees, emits garbage, or arrives for a different piece.

## Confirmation / refutation

Confirmation would be 200 pieces with roughly the 0007 line rate and zero
illegal actions, plus fewer stale ticks and less wall time spent at each piece
boundary.  A gain from a two-ply continuation with very few accepted Jev
choices would isolate the code search as the cause.  It is refuted if the
staged shortlist loses lines or creates holes like 0008, if it still repeats
hard drops across a piece boundary, or if geometry/hold validation produces any
illegal action.  If score is unchanged but stale ticks fall, the timing part is
confirmed and future work should tune the search only; if accepted Jev choices
hurt, raise the gate or make Jev purely diagnostic.

## Jev interface and tier-2 timing policy

Each background request receives strict JSON with only derived facts:

```text
{
  ruleset, policy: "BALANCED", piece_instance,
  board: {height bucket, holes bucket, surface bucket, lines, level,
          combo active/inactive, back_to_back},
  active_piece: {type, orientation, anchor_x},
  hold, can_hold, next_queue,
  candidates: {C0: named placement description, ...},
  fallback: "C0"
}
```

A description says whether hold is used, target orientation and anchor, clear
kind, projected height/hole/surface buckets, and bounded future support.  Jev is
not asked to count, calculate, invent SRS inputs, or output primitive actions.
Two independent `Choice` questions use the same candidates in opposite orders.
A response is usable only when both values are an existing, equal candidate and
both selected-label probabilities are at least `0.40`; confidence is not used
as a probability.  The answer is tagged with piece count and active type, and
is discarded after a lock, hold replacement, or incompatible active piece.

The controller posts one verified action and waits for the next tick on every
loop.  At a new piece it posts `noop` first, computes the deterministic staged
plan, re-observes, and then starts Jev in the background.  While Jev is pending,
the code target executes.  If a movement/rotation does not change the observed
pose, the target is abandoned for a current-orientation emergency plan rather
than repeating a blocked action.  If a target is unreachable or malformed,
`hard_drop` is the final valid fallback.  Timeout, ordinary runtime failure,
invalid stand-in answers, budget closure, and stale futures never escape
`run_episode`; persistent `ModelUnavailable` is allowed to propagate as the
contract requires.  All model and future exceptions are treated as advisory
unless the contract explicitly marks the backend unavailable.
